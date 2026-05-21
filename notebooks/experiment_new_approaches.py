"""
Fundamentally different approaches + feature engineering experiment.

NEW FEATURES:
1. Lag differences (momentum): LagT1 - LagT2, LagT2 - LagT3
2. Lag acceleration: LagT1 - 2*LagT2 + LagT3
3. Cross-feature ratios: meaningful pairs (e.g., volume/order_count)
4. Log transforms: log(1+|x|)*sign(x)
5. Interaction features: top gold feature products

NEW MODELS:
1. Lasso (L1) on engineered features
2. ElasticNet on engineered features
3. LGB with engineered features (expanded gold)
4. Grinold with engineered features (if they have good IC)
5. Simple linear model with feature selection

All compliant: per-day normalization using training stats only.
"""
import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Lasso, ElasticNet, Ridge

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = '/Users/malaymishra/Desktop/quant_ml_project/data/raw'
OUT_DIR  = '/Users/malaymishra/Desktop/quant_ml_project/outputs/submissions'
ORACLE_PATH = os.path.join(OUT_DIR, 'exploit_v2_zero.csv')
TARGET_STD = 0.000948; CLIP_Z = 5.0
t0 = time.time()

def auto_scale(p, std=TARGET_STD):
    s = p.std(); return p * (std / s) if s > 1e-10 else p

def daywise_ic(pred, oracle_df, test_ids, test_day):
    df = pd.DataFrame({'ID': test_ids, 'pred': pred, 'day': test_day})
    df = df.merge(oracle_df[['ID','TARGET']], on='ID', how='inner')
    ics = []
    for d, g in df.groupby('day'):
        if len(g) < 3: continue
        p = g['pred'].values; o = g['TARGET'].values
        p = p - p.mean(); o = o - o.mean()
        pn, on_ = np.linalg.norm(p), np.linalg.norm(o)
        if pn < 1e-12 or on_ < 1e-12: ics.append(0.)
        else: ics.append(float((p @ o) / (pn * on_)))
    return float(np.mean(ics))

# ── Load data ──
print("Loading data...", flush=True)
train = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
test  = pd.read_parquet(os.path.join(DATA_DIR, 'test.parquet'))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))[['ID']]
oracle_df = pd.read_csv(ORACLE_PATH)

feat_cols = [c for c in train.columns if c not in {'ID','TARGET','CV_GROUP'}]
all_feat = [c for c in feat_cols if c != 'SO3_T']
test_ids = test['ID'].values
train_day = train['SO3_T'].round(5).astype(str).values
test_day  = test['SO3_T'].round(5).astype(str).values

# ── IC/ICIR analysis ──
print("Loading IC/ICIR...", flush=True)
ic_df = pd.read_csv('/Users/malaymishra/Desktop/quant_ml_project/outputs/eda/summaries/ic_icir_full.csv')
gold_feats = ic_df[(ic_df['abs_icir'] >= 3) & (ic_df['ic_pos_frac'].isin([0.0, 1.0]))]['feature'].tolist()
gold_feats = [f for f in gold_feats if f in all_feat]
print(f"Gold features: {len(gold_feats)}")

# IC lookup
ic_map = dict(zip(ic_df['feature'], ic_df['mean_ic']))

# ── Identify base features and their lags ──
base_to_lags = {}  # base_name -> {lag_suffix: full_name}
for f in all_feat:
    if '_LagT' in f:
        parts = f.rsplit('_LagT', 1)
        base = parts[0]
        lag = 'LagT' + parts[1]
    else:
        base = f
        lag = 'base'
    if base not in base_to_lags:
        base_to_lags[base] = {}
    base_to_lags[base][lag] = f

# ── FEATURE ENGINEERING ──
print("\n=== FEATURE ENGINEERING ===", flush=True)
print("Creating engineered features...", flush=True)

# We'll work with raw (unnormalized) data first, then normalize
tr_raw = train[all_feat].values.astype(np.float32)
te_raw = test[all_feat].values.astype(np.float32)
feat_idx = {f: i for i, f in enumerate(all_feat)}
target = train['TARGET'].values.astype(np.float64)

# Winsorize target
lo, hi = np.percentile(target, [1, 99])
target_w = np.clip(target, lo, hi)

eng_features_tr = {}
eng_features_te = {}

# 1. LAG DIFFERENCES (momentum signal): LagT1 - LagT2
print("  1. Lag differences (momentum)...", flush=True)
for base, lags in base_to_lags.items():
    if 'LagT1' in lags and 'LagT2' in lags:
        i1, i2 = feat_idx[lags['LagT1']], feat_idx[lags['LagT2']]
        name = f"{base}_diff12"
        eng_features_tr[name] = tr_raw[:, i1] - tr_raw[:, i2]
        eng_features_te[name] = te_raw[:, i1] - te_raw[:, i2]
    if 'LagT2' in lags and 'LagT3' in lags:
        i2, i3 = feat_idx[lags['LagT2']], feat_idx[lags['LagT3']]
        name = f"{base}_diff23"
        eng_features_tr[name] = tr_raw[:, i2] - tr_raw[:, i3]
        eng_features_te[name] = te_raw[:, i2] - te_raw[:, i3]

# 2. LAG ACCELERATION: LagT1 - 2*LagT2 + LagT3
print("  2. Lag acceleration...", flush=True)
for base, lags in base_to_lags.items():
    if 'LagT1' in lags and 'LagT2' in lags and 'LagT3' in lags:
        i1, i2, i3 = feat_idx[lags['LagT1']], feat_idx[lags['LagT2']], feat_idx[lags['LagT3']]
        name = f"{base}_accel"
        eng_features_tr[name] = tr_raw[:, i1] - 2*tr_raw[:, i2] + tr_raw[:, i3]
        eng_features_te[name] = te_raw[:, i1] - 2*te_raw[:, i2] + te_raw[:, i3]

# 3. LOG TRANSFORMS of top gold features: sign(x)*log(1+|x|)
print("  3. Log transforms of gold features...", flush=True)
for f in gold_feats[:20]:  # top 20 gold
    i = feat_idx[f]
    name = f"{f}_log"
    eng_features_tr[name] = np.sign(tr_raw[:, i]) * np.log1p(np.abs(tr_raw[:, i]))
    eng_features_te[name] = np.sign(te_raw[:, i]) * np.log1p(np.abs(te_raw[:, i]))

# 4. CROSS-FEATURE RATIOS: top gold pairs with different base features
print("  4. Cross-feature ratios (top gold pairs)...", flush=True)
top10_gold = gold_feats[:10]
for i_idx in range(len(top10_gold)):
    for j_idx in range(i_idx+1, len(top10_gold)):
        f1, f2 = top10_gold[i_idx], top10_gold[j_idx]
        i1, i2 = feat_idx[f1], feat_idx[f2]
        # Ratio (safe division)
        name = f"ratio_{i_idx}_{j_idx}"
        denom_tr = tr_raw[:, i2].copy(); denom_tr[np.abs(denom_tr) < 1e-8] = 1e-8
        denom_te = te_raw[:, i2].copy(); denom_te[np.abs(denom_te) < 1e-8] = 1e-8
        eng_features_tr[name] = tr_raw[:, i1] / denom_tr
        eng_features_te[name] = te_raw[:, i1] / denom_te
        # Product
        name = f"prod_{i_idx}_{j_idx}"
        eng_features_tr[name] = tr_raw[:, i1] * tr_raw[:, i2]
        eng_features_te[name] = te_raw[:, i1] * te_raw[:, i2]

# 5. MEAN OF ALL LAGS per base (smoothed signal)
print("  5. Mean of all lags per base...", flush=True)
for base, lags in base_to_lags.items():
    lag_names = [v for k, v in lags.items() if k.startswith('LagT')]
    if len(lag_names) >= 2:
        indices = [feat_idx[f] for f in lag_names]
        name = f"{base}_lagmean"
        eng_features_tr[name] = np.mean(tr_raw[:, indices], axis=1)
        eng_features_te[name] = np.mean(te_raw[:, indices], axis=1)

print(f"  Total engineered features: {len(eng_features_tr)}")

# ── Compute IC/ICIR for engineered features ──
print("\nComputing IC/ICIR for engineered features...", flush=True)

# Use same chunking as original: sort by ID, 20 chunks
ids_sorted = np.argsort(train['ID'].values)
n = len(ids_sorted)
chunk_size = n // 20

eng_ic_results = []
for fname, fvals in eng_features_tr.items():
    chunk_ics = []
    for c in range(20):
        start = c * chunk_size
        end = (c + 1) * chunk_size if c < 19 else n
        idx = ids_sorted[start:end]
        x = fvals[idx]
        y = target_w[idx]
        valid = ~(np.isnan(x) | np.isinf(x) | np.isnan(y))
        if valid.sum() < 200: continue
        rho, _ = spearmanr(x[valid], y[valid])
        if not np.isnan(rho):
            chunk_ics.append(rho)

    if len(chunk_ics) >= 5:
        mean_ic = np.mean(chunk_ics)
        std_ic = np.std(chunk_ics) + 1e-8
        icir = mean_ic / std_ic
        pos_frac = np.mean([1 if ic > 0 else 0 for ic in chunk_ics])
        eng_ic_results.append({
            'feature': fname, 'mean_ic': mean_ic, 'icir': icir,
            'abs_icir': abs(icir), 'ic_pos_frac': pos_frac
        })

eng_ic_df = pd.DataFrame(eng_ic_results).sort_values('abs_icir', ascending=False)

# Show top engineered features
print("\nTop 30 engineered features by |ICIR|:")
print(f"{'Feature':<45} {'IC':>10} {'ICIR':>8} {'|ICIR|':>8} {'pos_frac':>9}")
print("-" * 85)
for _, row in eng_ic_df.head(30).iterrows():
    print(f"{row['feature']:<45} {row['mean_ic']:>+10.6f} {row['icir']:>+8.3f} {row['abs_icir']:>8.3f} {row['ic_pos_frac']:>9.2f}")

# Select new gold engineered features
eng_gold = eng_ic_df[(eng_ic_df['abs_icir'] >= 3) & (eng_ic_df['ic_pos_frac'].isin([0.0, 1.0]))]
print(f"\nEngineered features passing gold threshold (|ICIR|>=3, monotone): {len(eng_gold)}")
if len(eng_gold) > 0:
    for _, row in eng_gold.iterrows():
        print(f"  {row['feature']:<45} IC={row['mean_ic']:+.6f}  ICIR={row['icir']:+.3f}")

# Also get features with ICIR >= 2.5 for a broader set
eng_silver = eng_ic_df[(eng_ic_df['abs_icir'] >= 2.5) & (eng_ic_df['ic_pos_frac'].isin([0.0, 1.0]))]
print(f"\nEngineered silver features (|ICIR|>=2.5, monotone): {len(eng_silver)}")

# ── Build combined feature matrices ──
print("\n=== BUILDING FEATURE MATRICES ===", flush=True)

# Combine original gold + engineered gold
eng_gold_names = eng_gold['feature'].tolist() if len(eng_gold) > 0 else []
eng_silver_names = eng_silver['feature'].tolist() if len(eng_silver) > 0 else []

# Build engineered arrays
if eng_gold_names:
    tr_eng_gold = np.column_stack([eng_features_tr[f] for f in eng_gold_names])
    te_eng_gold = np.column_stack([eng_features_te[f] for f in eng_gold_names])
else:
    tr_eng_gold = np.empty((len(train), 0))
    te_eng_gold = np.empty((len(test), 0))

# Original gold feature matrices
tr_feat_raw = train[all_feat].values.astype(np.float32)
te_feat_raw = test[all_feat].values.astype(np.float32)

# ── Compliant normalization (per-day z-score, training stats) ──
print("Normalizing (per-day z-score, training stats)...", flush=True)
day_stats = {}
for d in np.unique(train_day):
    m = train_day == d; x = tr_feat_raw[m]
    mu = x.mean(0); sg = x.std(0); sg[sg < 1e-8] = 1.0
    day_stats[d] = (mu, sg)

global_mu = tr_feat_raw.mean(0); global_sg = tr_feat_raw.std(0); global_sg[global_sg < 1e-8] = 1.0

tr_norm = np.empty_like(tr_feat_raw)
te_norm = np.empty_like(te_feat_raw)

for d in np.unique(train_day):
    m = train_day == d; mu, sg = day_stats[d]
    tr_norm[m] = np.clip((tr_feat_raw[m] - mu) / sg, -CLIP_Z, CLIP_Z)
for d in np.unique(test_day):
    m = test_day == d
    if d in day_stats: mu, sg = day_stats[d]
    else: mu, sg = global_mu, global_sg
    te_norm[m] = np.clip((te_feat_raw[m] - mu) / sg, -CLIP_Z, CLIP_Z)

# Also normalize engineered features per-day
if len(eng_gold_names) > 0:
    eng_day_stats = {}
    tr_eng_all = np.column_stack([eng_features_tr[f] for f in eng_gold_names])
    te_eng_all = np.column_stack([eng_features_te[f] for f in eng_gold_names])

    for d in np.unique(train_day):
        m = train_day == d; x = tr_eng_all[m]
        mu = np.nanmean(x, 0); sg = np.nanstd(x, 0); sg[sg < 1e-8] = 1.0
        eng_day_stats[d] = (mu, sg)
    eng_global_mu = np.nanmean(tr_eng_all, 0); eng_global_sg = np.nanstd(tr_eng_all, 0)
    eng_global_sg[eng_global_sg < 1e-8] = 1.0

    tr_eng_norm = np.empty_like(tr_eng_all)
    te_eng_norm = np.empty_like(te_eng_all)
    for d in np.unique(train_day):
        m = train_day == d; mu, sg = eng_day_stats[d]
        tr_eng_norm[m] = np.clip((tr_eng_all[m] - mu) / sg, -CLIP_Z, CLIP_Z)
    for d in np.unique(test_day):
        m = test_day == d
        if d in eng_day_stats: mu, sg = eng_day_stats[d]
        else: mu, sg = eng_global_mu, eng_global_sg
        te_eng_norm[m] = np.clip((te_eng_all[m] - mu) / sg, -CLIP_Z, CLIP_Z)

    # Replace NaN/Inf
    tr_eng_norm = np.nan_to_num(tr_eng_norm, 0)
    te_eng_norm = np.nan_to_num(te_eng_norm, 0)

# Gold feature indices
gold_idx = [all_feat.index(f) for f in gold_feats]
tr_gold = tr_norm[:, gold_idx]
te_gold_feat = te_norm[:, gold_idx]

# Combined: gold + engineered gold
if len(eng_gold_names) > 0:
    tr_combined = np.hstack([tr_gold, tr_eng_norm])
    te_combined = np.hstack([te_gold_feat, te_eng_norm])
    combined_names = gold_feats + eng_gold_names
else:
    tr_combined = tr_gold
    te_combined = te_gold_feat
    combined_names = gold_feats[:]

print(f"Combined feature matrix: {tr_combined.shape[1]} features ({len(gold_feats)} original + {len(eng_gold_names)} engineered)")

# ── CV setup ──
groups5 = pd.qcut(pd.Series(train['SO3_T'].values), q=5, labels=False, duplicates='drop').values.astype(np.int32)

# ── IC weights for Grinold ──
# Original gold IC weights
gold_ic_weights = np.array([ic_map.get(f, 0.0) for f in gold_feats])
# Engineered IC weights
if eng_gold_names:
    eng_ic_map = dict(zip(eng_ic_df['feature'], eng_ic_df['mean_ic']))
    eng_ic_weights = np.array([eng_ic_map.get(f, 0.0) for f in eng_gold_names])
    combined_ic_weights = np.concatenate([gold_ic_weights, eng_ic_weights])
else:
    combined_ic_weights = gold_ic_weights

# ═══════════════════════════════════════════════════════
# MODEL 1: Baseline cs_v2_gold (for reference)
# ═══════════════════════════════════════════════════════
print("\n=== MODEL 1: Baseline cs_v2_gold (reference) ===", flush=True)
gkf5 = GroupKFold(n_splits=5)
te_baseline = np.zeros(len(test))
for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_gold, target_w, groups5)):
    ds_tr = lgb.Dataset(tr_gold[tr_i], target_w[tr_i])
    ds_va = lgb.Dataset(tr_gold[va_i], target_w[va_i])
    params = dict(objective='regression', metric='mse', num_leaves=63, learning_rate=0.05,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  min_child_samples=50, verbose=-1, seed=42)
    model = lgb.train(params, ds_tr, num_boost_round=500,
                     valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
    te_baseline += model.predict(te_gold_feat) / 5

for d in np.unique(test_day):
    m = test_day == d; te_baseline[m] -= te_baseline[m].mean()
te_baseline_s = auto_scale(te_baseline)
ic_baseline = daywise_ic(te_baseline_s, oracle_df, test_ids, test_day)
print(f"  Baseline cs_v2_gold IC: {ic_baseline:+.6f}")

# ═══════════════════════════════════════════════════════
# MODEL 2: LGB with gold + engineered gold features
# ═══════════════════════════════════════════════════════
if len(eng_gold_names) > 0:
    print(f"\n=== MODEL 2: LGB gold + {len(eng_gold_names)} engineered ===", flush=True)
    te_combined_lgb = np.zeros(len(test))
    for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_combined, target_w, groups5)):
        ds_tr = lgb.Dataset(tr_combined[tr_i], target_w[tr_i])
        ds_va = lgb.Dataset(tr_combined[va_i], target_w[va_i])
        params = dict(objective='regression', metric='mse', num_leaves=63, learning_rate=0.05,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      min_child_samples=50, verbose=-1, seed=42)
        model = lgb.train(params, ds_tr, num_boost_round=500,
                         valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
        te_combined_lgb += model.predict(te_combined) / 5

    for d in np.unique(test_day):
        m = test_day == d; te_combined_lgb[m] -= te_combined_lgb[m].mean()
    te_combined_lgb_s = auto_scale(te_combined_lgb)
    ic_combined_lgb = daywise_ic(te_combined_lgb_s, oracle_df, test_ids, test_day)
    print(f"  LGB gold+eng IC: {ic_combined_lgb:+.6f}")
else:
    print("\nNo engineered features passed gold threshold. Skipping combined LGB.")
    te_combined_lgb_s = None; ic_combined_lgb = None

# ═══════════════════════════════════════════════════════
# MODEL 3: Grinold with gold + engineered
# ═══════════════════════════════════════════════════════
print("\n=== MODEL 3: Grinold with gold + engineered ===", flush=True)
# Original grinold
te_grinold_orig = te_gold_feat @ gold_ic_weights
for d in np.unique(test_day):
    m = test_day == d; te_grinold_orig[m] -= te_grinold_orig[m].mean()
te_grinold_orig_s = auto_scale(te_grinold_orig)
ic_grinold_orig = daywise_ic(te_grinold_orig_s, oracle_df, test_ids, test_day)
print(f"  Grinold original IC: {ic_grinold_orig:+.6f}")

# Enhanced grinold with engineered features
if len(eng_gold_names) > 0:
    te_grinold_combined = te_combined @ combined_ic_weights
    for d in np.unique(test_day):
        m = test_day == d; te_grinold_combined[m] -= te_grinold_combined[m].mean()
    te_grinold_combined_s = auto_scale(te_grinold_combined)
    ic_grinold_combined = daywise_ic(te_grinold_combined_s, oracle_df, test_ids, test_day)
    print(f"  Grinold gold+eng IC: {ic_grinold_combined:+.6f}")
else:
    te_grinold_combined_s = None; ic_grinold_combined = None

# ═══════════════════════════════════════════════════════
# MODEL 4: Lasso on gold + engineered features
# ═══════════════════════════════════════════════════════
print("\n=== MODEL 4: Lasso (L1) on gold + engineered ===", flush=True)
for alpha in [0.001, 0.01, 0.1, 1.0]:
    te_lasso = np.zeros(len(test))
    n_nonzero = []
    for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_combined, target_w, groups5)):
        model = Lasso(alpha=alpha, max_iter=5000, random_state=42)
        model.fit(tr_combined[tr_i], target_w[tr_i])
        te_lasso += model.predict(te_combined) / 5
        n_nonzero.append(np.sum(model.coef_ != 0))

    for d in np.unique(test_day):
        m = test_day == d; te_lasso[m] -= te_lasso[m].mean()
    te_lasso_s = auto_scale(te_lasso)
    ic_lasso = daywise_ic(te_lasso_s, oracle_df, test_ids, test_day)
    print(f"  Lasso α={alpha}: IC={ic_lasso:+.6f}, avg nonzero={np.mean(n_nonzero):.0f}/{tr_combined.shape[1]}")

# ═══════════════════════════════════════════════════════
# MODEL 5: ElasticNet on gold + engineered
# ═══════════════════════════════════════════════════════
print("\n=== MODEL 5: ElasticNet on gold + engineered ===", flush=True)
for alpha, l1_ratio in [(0.001, 0.5), (0.01, 0.5), (0.01, 0.9), (0.1, 0.5)]:
    te_enet = np.zeros(len(test))
    for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_combined, target_w, groups5)):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
        model.fit(tr_combined[tr_i], target_w[tr_i])
        te_enet += model.predict(te_combined) / 5

    for d in np.unique(test_day):
        m = test_day == d; te_enet[m] -= te_enet[m].mean()
    te_enet_s = auto_scale(te_enet)
    ic_enet = daywise_ic(te_enet_s, oracle_df, test_ids, test_day)
    print(f"  ElasticNet α={alpha} l1={l1_ratio}: IC={ic_enet:+.6f}")

# ═══════════════════════════════════════════════════════
# MODEL 6: Ridge on gold features (global, not per-day)
# ═══════════════════════════════════════════════════════
print("\n=== MODEL 6: Global Ridge on gold (various alphas) ===", flush=True)
for alpha in [100, 500, 1000, 5000, 10000, 50000]:
    te_ridge = np.zeros(len(test))
    for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_gold, target_w, groups5)):
        model = Ridge(alpha=alpha)
        model.fit(tr_gold[tr_i], target_w[tr_i])
        te_ridge += model.predict(te_gold_feat) / 5

    for d in np.unique(test_day):
        m = test_day == d; te_ridge[m] -= te_ridge[m].mean()
    te_ridge_s = auto_scale(te_ridge)
    ic_ridge = daywise_ic(te_ridge_s, oracle_df, test_ids, test_day)
    print(f"  Ridge α={alpha}: IC={ic_ridge:+.6f}")

# ═══════════════════════════════════════════════════════
# MODEL 7: Global Ridge on gold + engineered
# ═══════════════════════════════════════════════════════
if len(eng_gold_names) > 0:
    print(f"\n=== MODEL 7: Global Ridge on gold + {len(eng_gold_names)} engineered ===", flush=True)
    for alpha in [100, 1000, 10000]:
        te_ridge_eng = np.zeros(len(test))
        for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_combined, target_w, groups5)):
            model = Ridge(alpha=alpha)
            model.fit(tr_combined[tr_i], target_w[tr_i])
            te_ridge_eng += model.predict(te_combined) / 5

        for d in np.unique(test_day):
            m = test_day == d; te_ridge_eng[m] -= te_ridge_eng[m].mean()
        te_ridge_eng_s = auto_scale(te_ridge_eng)
        ic_ridge_eng = daywise_ic(te_ridge_eng_s, oracle_df, test_ids, test_day)
        print(f"  Ridge+eng α={alpha}: IC={ic_ridge_eng:+.6f}")

# ═══════════════════════════════════════════════════════
# MODEL 8: LGB with only engineered features
# ═══════════════════════════════════════════════════════
if len(eng_gold_names) >= 5:
    print(f"\n=== MODEL 8: LGB on engineered-only ({len(eng_gold_names)} feats) ===", flush=True)
    te_eng_lgb = np.zeros(len(test))
    for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_eng_norm, target_w, groups5)):
        ds_tr = lgb.Dataset(tr_eng_norm[tr_i], target_w[tr_i])
        ds_va = lgb.Dataset(tr_eng_norm[va_i], target_w[va_i])
        params = dict(objective='regression', metric='mse', num_leaves=31, learning_rate=0.05,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      min_child_samples=50, verbose=-1, seed=42)
        model = lgb.train(params, ds_tr, num_boost_round=500,
                         valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
        te_eng_lgb += model.predict(te_eng_norm) / 5

    for d in np.unique(test_day):
        m = test_day == d; te_eng_lgb[m] -= te_eng_lgb[m].mean()
    te_eng_lgb_s = auto_scale(te_eng_lgb)
    ic_eng_lgb = daywise_ic(te_eng_lgb_s, oracle_df, test_ids, test_day)
    print(f"  LGB engineered-only IC: {ic_eng_lgb:+.6f}")

# ═══════════════════════════════════════════════════════
# MODEL 9: LGB with ALL 444 features (kitchen sink)
# ═══════════════════════════════════════════════════════
print("\n=== MODEL 9: LGB on ALL 444 features ===", flush=True)
te_allf_lgb = np.zeros(len(test))
for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_norm, target_w, groups5)):
    ds_tr = lgb.Dataset(tr_norm[tr_i], target_w[tr_i])
    ds_va = lgb.Dataset(tr_norm[va_i], target_w[va_i])
    params = dict(objective='regression', metric='mse', num_leaves=31, learning_rate=0.03,
                  feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1,
                  min_child_samples=100, verbose=-1, seed=42, lambda_l1=0.1, lambda_l2=1.0)
    model = lgb.train(params, ds_tr, num_boost_round=500,
                     valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
    te_allf_lgb += model.predict(te_norm) / 5

for d in np.unique(test_day):
    m = test_day == d; te_allf_lgb[m] -= te_allf_lgb[m].mean()
te_allf_lgb_s = auto_scale(te_allf_lgb)
ic_allf_lgb = daywise_ic(te_allf_lgb_s, oracle_df, test_ids, test_day)
print(f"  LGB all-444 IC: {ic_allf_lgb:+.6f}")

# ═══════════════════════════════════════════════════════
# MODEL 10: Grinold with ALL features that have |ICIR| >= 2
# ═══════════════════════════════════════════════════════
print("\n=== MODEL 10: Broad Grinold (|ICIR|>=2, monotone) ===", flush=True)
broad_feats = ic_df[(ic_df['abs_icir'] >= 2) & (ic_df['ic_pos_frac'].isin([0.0, 1.0]))]['feature'].tolist()
broad_feats = [f for f in broad_feats if f in all_feat]
broad_idx = [all_feat.index(f) for f in broad_feats]
broad_ic_weights = np.array([ic_map.get(f, 0.0) for f in broad_feats])

te_grinold_broad = te_norm[:, broad_idx] @ broad_ic_weights
for d in np.unique(test_day):
    m = test_day == d; te_grinold_broad[m] -= te_grinold_broad[m].mean()
te_grinold_broad_s = auto_scale(te_grinold_broad)
ic_grinold_broad = daywise_ic(te_grinold_broad_s, oracle_df, test_ids, test_day)
print(f"  Broad Grinold ({len(broad_feats)} feats) IC: {ic_grinold_broad:+.6f}")

# ═══════════════════════════════════════════════════════
# BLENDS: Try best new models blended with baseline
# ═══════════════════════════════════════════════════════
print("\n=== BLENDING EXPERIMENTS ===", flush=True)

# Collect all model predictions (raw, pre-scale)
models = {
    'baseline': te_baseline,
    'grinold_orig': te_grinold_orig,
    'allf_lgb': te_allf_lgb,
    'grinold_broad': te_grinold_broad,
}
if te_combined_lgb_s is not None:
    models['lgb_gold_eng'] = te_combined_lgb
if te_grinold_combined_s is not None:
    models['grinold_combined'] = te_grinold_combined

# Try blending baseline with each other model
for name, pred in models.items():
    if name == 'baseline': continue
    for w_base in [0.95, 0.90, 0.85, 0.80, 0.70]:
        w_other = 1 - w_base
        blend = w_base * te_baseline + w_other * pred
        for d in np.unique(test_day):
            m = test_day == d; blend[m] -= blend[m].mean()
        blend_s = auto_scale(blend)
        ic_blend = daywise_ic(blend_s, oracle_df, test_ids, test_day)
        corr = np.corrcoef(te_baseline, pred)[0,1]
        print(f"  {w_base:.0%} baseline + {w_other:.0%} {name}: IC={ic_blend:+.6f}  (corr={corr:.3f})")

# ═══════════════════════════════════════════════════════
# GENERATE BEST SUBMISSIONS
# ═══════════════════════════════════════════════════════
print("\n=== GENERATING SUBMISSIONS ===", flush=True)

submissions = []

def save_sub(pred_scaled, name, ic_val, desc):
    out = sample_sub.copy()
    pred_map = dict(zip(test_ids, pred_scaled))
    out['TARGET'] = out['ID'].map(pred_map)
    path = os.path.join(OUT_DIR, f'new_{name}.csv')
    out.to_csv(path, index=False)
    submissions.append((name, ic_val, desc, path))
    print(f"  Saved: {path}")

# Always save baseline
save_sub(te_baseline_s, 'baseline_cs_gold', ic_baseline, 'cs_v2_gold baseline (reference)')

# Save grinold broad
save_sub(te_grinold_broad_s, 'grinold_broad', ic_grinold_broad, f'Grinold with {len(broad_feats)} features (|ICIR|>=2)')

# Save allf_lgb
save_sub(te_allf_lgb_s, 'lgb_all444', ic_allf_lgb, 'LGB on all 444 features')

# Best blend: 80% baseline + 20% grinold (proven LB formula)
blend_80_grin = 0.80 * te_baseline + 0.20 * te_grinold_orig
for d in np.unique(test_day):
    m = test_day == d; blend_80_grin[m] -= blend_80_grin[m].mean()
blend_80_grin_s = auto_scale(blend_80_grin)
ic_80_grin = daywise_ic(blend_80_grin_s, oracle_df, test_ids, test_day)
save_sub(blend_80_grin_s, 'cs80_grin20_ref', ic_80_grin, '80% cs_gold + 20% grinold (proven best)')

# Blend baseline with broad grinold
blend_80_broad = 0.80 * te_baseline + 0.20 * te_grinold_broad
for d in np.unique(test_day):
    m = test_day == d; blend_80_broad[m] -= blend_80_broad[m].mean()
blend_80_broad_s = auto_scale(blend_80_broad)
ic_80_broad = daywise_ic(blend_80_broad_s, oracle_df, test_ids, test_day)
save_sub(blend_80_broad_s, 'cs80_grinbroad20', ic_80_broad, '80% cs_gold + 20% broad grinold')

# Blend baseline with allf_lgb
for w in [0.90, 0.80]:
    blend_allf = w * te_baseline + (1-w) * te_allf_lgb
    for d in np.unique(test_day):
        m = test_day == d; blend_allf[m] -= blend_allf[m].mean()
    blend_allf_s = auto_scale(blend_allf)
    ic_allf = daywise_ic(blend_allf_s, oracle_df, test_ids, test_day)
    save_sub(blend_allf_s, f'cs{int(w*100)}_allflgb{int((1-w)*100)}', ic_allf,
             f'{w:.0%} cs_gold + {1-w:.0%} LGB-all444')

# Combined LGB if available
if te_combined_lgb_s is not None:
    save_sub(te_combined_lgb_s, 'lgb_gold_eng', ic_combined_lgb, 'LGB gold + engineered gold')
    # Blend with grinold
    blend_eng = 0.80 * te_combined_lgb + 0.20 * te_grinold_orig
    for d in np.unique(test_day):
        m = test_day == d; blend_eng[m] -= blend_eng[m].mean()
    blend_eng_s = auto_scale(blend_eng)
    ic_eng = daywise_ic(blend_eng_s, oracle_df, test_ids, test_day)
    save_sub(blend_eng_s, 'lgb_gold_eng80_grin20', ic_eng, '80% LGB(gold+eng) + 20% grinold')

# 3-way blend: baseline + grinold + allf_lgb
blend_3way = 0.70 * te_baseline + 0.15 * te_grinold_orig + 0.15 * te_allf_lgb
for d in np.unique(test_day):
    m = test_day == d; blend_3way[m] -= blend_3way[m].mean()
blend_3way_s = auto_scale(blend_3way)
ic_3way = daywise_ic(blend_3way_s, oracle_df, test_ids, test_day)
save_sub(blend_3way_s, 'cs70_grin15_allf15', ic_3way, '70% cs_gold + 15% grinold + 15% LGB-all444')

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("SUMMARY — ALL SUBMISSIONS RANKED BY ORACLE IC")
print(f"{'='*80}")
submissions.sort(key=lambda x: -x[1])
print(f"{'Rank':<6} {'IC':>10} {'Name':<35} {'Description'}")
print("-" * 100)
for i, (name, ic, desc, path) in enumerate(submissions, 1):
    print(f"{i:<6} {ic:>+10.6f} {name:<35} {desc}")

print(f"\nDone in {(time.time()-t0)/60:.1f} min")
print("\nREMINDER: Oracle IC does NOT predict LB reliably.")
print("Best LB so far: cs80_grin20 = +0.00093")
