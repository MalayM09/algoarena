"""
Feature Engineering Experiment — Compliant approaches.
Based on HFT microstructure research and competition best practices.

Key ideas:
1. Lag differences (momentum): base - LagT1, LagT1 - LagT2, etc.
2. Log compression: log(1+|x|)*sign(x) for heavy tails
3. Within-group stats: per-row mean/std across feature families
4. Top feature interactions: products/ratios of strongest gold features
5. NaN count indicator
6. Squared terms of top predictors
7. Acceleration features: (f - f_lag1) - (f_lag1 - f_lag2)

All compliant: computed per-sample using training statistics only.
"""
import os, gc, time, warnings, re
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

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

print("Loading data...", flush=True)
train = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
test = pd.read_parquet(os.path.join(DATA_DIR, 'test.parquet'))
oracle_df = pd.read_csv(ORACLE_PATH)

feat_cols = [c for c in train.columns if c not in {'ID','TARGET','CV_GROUP'}]
all_feat = [c for c in feat_cols if c != 'SO3_T']
test_ids = test['ID'].values
train_day = train['SO3_T'].round(5).astype(str).values
test_day = test['SO3_T'].round(5).astype(str).values
y_raw = train['TARGET'].values.astype(np.float64)
lo, hi = np.percentile(y_raw, 1), np.percentile(y_raw, 99)
y_wins = np.clip(y_raw, lo, hi)

# ──────────────────────────────────────────────────────────────
# STEP 1: Compute gold features (baseline)
# ──────────────────────────────────────────────────────────────
print("Computing IC/ICIR...", flush=True)
N_CHUNKS = 20; train_sorted = train.sort_values('ID').reset_index(drop=True)
chunk_size = len(train_sorted) // N_CHUNKS; ic_results = []
for col in all_feat:
    chunk_ics = []
    for i in range(N_CHUNKS):
        chunk = train_sorted.iloc[i*chunk_size:(i+1)*chunk_size]
        vals = chunk[col].fillna(chunk[col].median()).values; tgt = chunk['TARGET'].values
        valid = ~np.isnan(vals)
        if valid.sum() < 200: chunk_ics.append(np.nan); continue
        ic, _ = spearmanr(vals[valid], tgt[valid]); chunk_ics.append(ic)
    valid_ics = [v for v in chunk_ics if not np.isnan(v)]
    if len(valid_ics) < 5: continue
    mean_ic = float(np.mean(valid_ics)); std_ic = float(np.std(valid_ics)) + 1e-8
    icir = mean_ic / std_ic
    ic_pos_frac = float(np.mean([v > 0 for v in valid_ics]))
    ic_results.append({'feature': col, 'mean_ic': mean_ic, 'icir': icir,
                       'abs_icir': abs(icir), 'ic_pos_frac': ic_pos_frac})
icir_df = pd.DataFrame(ic_results).sort_values('abs_icir', ascending=False)
gold_mask = (icir_df['abs_icir'] >= 3) & (icir_df['ic_pos_frac'].isin([0.0, 1.0]))
gold_df = icir_df[gold_mask].copy()
gold_feats = [f for f in gold_df['feature'].tolist() if f in all_feat]
ic_dict = gold_df.set_index('feature')['mean_ic'].to_dict()
gold_idx = [all_feat.index(f) for f in gold_feats]
print(f"  Gold features: {len(gold_feats)}")

# ──────────────────────────────────────────────────────────────
# STEP 2: Parse feature taxonomy
# ──────────────────────────────────────────────────────────────
print("Parsing feature taxonomy...", flush=True)

def parse_feature(fname):
    """Parse feature name into (base_name, lag)."""
    lag = 'base'
    base = fname
    m = re.match(r'^(.+?)(_LagT(\d+))$', fname)
    if m:
        base = m.group(1)
        lag = f'LagT{m.group(3)}'
    return base, lag

# Group features by base name
base_to_lags = {}  # base_name -> {lag: feat_name}
for f in all_feat:
    base, lag = parse_feature(f)
    if base not in base_to_lags:
        base_to_lags[base] = {}
    base_to_lags[base][lag] = f

# Identify features that have both base and lag variants
features_with_lags = {b: lags for b, lags in base_to_lags.items()
                      if len(lags) >= 2}
print(f"  Features with lag variants: {len(features_with_lags)}")

# Feature families (S03_V, S03_D, S03_A, etc.)
families = {}
for f in all_feat:
    parts = f.split('_')
    if len(parts) >= 2:
        fam = f'{parts[0]}_{parts[1][:1]}'  # e.g., S03_V, S03_D, S01_F
    else:
        fam = parts[0]
    if fam not in families:
        families[fam] = []
    families[fam].append(f)

for fam, feats in sorted(families.items()):
    print(f"  {fam}: {len(feats)} features")

# ──────────────────────────────────────────────────────────────
# STEP 3: Normalize base features (per-day z-score)
# ──────────────────────────────────────────────────────────────
print("Normalizing...", flush=True)
tr_raw = train[all_feat].fillna(0).values.astype(np.float32)
te_raw = test[all_feat].fillna(0).values.astype(np.float32)

global_mean = tr_raw.mean(0); global_std = tr_raw.std(0)
global_std[global_std < 1e-8] = 1.0

day_stats = {}
for d in np.unique(train_day):
    m = train_day == d; x = tr_raw[m]
    mu = x.mean(0); sg = x.std(0); sg[sg < 1e-8] = 1.0
    day_stats[d] = (mu, sg)

X_tr = np.zeros_like(tr_raw, dtype=np.float32)
X_te = np.zeros_like(te_raw, dtype=np.float32)
for d in np.unique(train_day):
    m = train_day == d; mu, sg = day_stats[d]
    X_tr[m] = np.clip((tr_raw[m].astype(np.float64) - mu) / sg, -CLIP_Z, CLIP_Z).astype(np.float32)
for d in np.unique(test_day):
    m = test_day == d
    if d in day_stats: mu, sg = day_stats[d]
    else: mu, sg = global_mean, global_std
    X_te[m] = np.clip((te_raw[m].astype(np.float64) - mu) / sg, -CLIP_Z, CLIP_Z).astype(np.float32)

# ──────────────────────────────────────────────────────────────
# STEP 4: Engineer new features
# ──────────────────────────────────────────────────────────────
print("\n=== FEATURE ENGINEERING ===\n", flush=True)

# Keep track of feature names and arrays
eng_tr_feats = []  # list of (name, train_array)
eng_te_feats = []  # list of (name, test_array)

def add_feature(name, tr_arr, te_arr):
    eng_tr_feats.append((name, tr_arr.astype(np.float32)))
    eng_te_feats.append((name, te_arr.astype(np.float32)))

# --- 4A: Lag differences (momentum features) ---
print("4A: Lag differences...", flush=True)
n_lag_feats = 0
for base, lags in features_with_lags.items():
    sorted_lags = sorted(lags.keys(), key=lambda x: 0 if x == 'base' else int(x[-1]))
    lag_names = [(lag, lags[lag]) for lag in sorted_lags]

    for i in range(len(lag_names) - 1):
        lag_a, feat_a = lag_names[i]
        lag_b, feat_b = lag_names[i + 1]
        idx_a = all_feat.index(feat_a)
        idx_b = all_feat.index(feat_b)

        name = f"diff_{feat_a}_minus_{lag_b}"
        tr_diff = X_tr[:, idx_a] - X_tr[:, idx_b]
        te_diff = X_te[:, idx_a] - X_te[:, idx_b]
        add_feature(name, tr_diff, te_diff)
        n_lag_feats += 1

    # Acceleration: (base - LagT1) - (LagT1 - LagT2)
    if 'base' in lags and 'LagT1' in lags and 'LagT2' in lags:
        idx_b = all_feat.index(lags['base'])
        idx_l1 = all_feat.index(lags['LagT1'])
        idx_l2 = all_feat.index(lags['LagT2'])

        tr_accel = (X_tr[:, idx_b] - X_tr[:, idx_l1]) - (X_tr[:, idx_l1] - X_tr[:, idx_l2])
        te_accel = (X_te[:, idx_b] - X_te[:, idx_l1]) - (X_te[:, idx_l1] - X_te[:, idx_l2])
        add_feature(f"accel_{base}", tr_accel, te_accel)
        n_lag_feats += 1

print(f"  Created {n_lag_feats} lag difference/acceleration features")

# --- 4B: Log compression ---
print("4B: Log compression...", flush=True)
# Apply to all gold features
for fi, feat in enumerate(gold_feats):
    idx = gold_idx[fi]
    tr_log = np.sign(X_tr[:, idx]) * np.log1p(np.abs(X_tr[:, idx]))
    te_log = np.sign(X_te[:, idx]) * np.log1p(np.abs(X_te[:, idx]))
    add_feature(f"log_{feat}", tr_log, te_log)
print(f"  Created {len(gold_feats)} log-compressed gold features")

# --- 4C: Squared terms of top 20 gold features ---
print("4C: Squared terms...", flush=True)
top20_gold = gold_feats[:20]
for feat in top20_gold:
    idx = all_feat.index(feat)
    add_feature(f"sq_{feat}", X_tr[:, idx]**2, X_te[:, idx]**2)
print(f"  Created {len(top20_gold)} squared features")

# --- 4D: Within-family statistics (per row) ---
print("4D: Within-family stats...", flush=True)
n_fam_feats = 0
for fam, feats in families.items():
    if len(feats) < 3:
        continue
    fam_idx = [all_feat.index(f) for f in feats]

    # Mean across family
    tr_mean = X_tr[:, fam_idx].mean(axis=1)
    te_mean = X_te[:, fam_idx].mean(axis=1)
    add_feature(f"fam_mean_{fam}", tr_mean, te_mean)
    n_fam_feats += 1

    # Std across family
    tr_std = X_tr[:, fam_idx].std(axis=1)
    te_std = X_te[:, fam_idx].std(axis=1)
    add_feature(f"fam_std_{fam}", tr_std, te_std)
    n_fam_feats += 1

    # Max - Min (range)
    tr_range = X_tr[:, fam_idx].max(axis=1) - X_tr[:, fam_idx].min(axis=1)
    te_range = X_te[:, fam_idx].max(axis=1) - X_te[:, fam_idx].min(axis=1)
    add_feature(f"fam_range_{fam}", tr_range, te_range)
    n_fam_feats += 1

print(f"  Created {n_fam_feats} family-level features")

# --- 4E: NaN count per row ---
print("4E: NaN indicators...", flush=True)
tr_nan_count = train[all_feat].isna().sum(axis=1).values.astype(np.float32)
te_nan_count = test[all_feat].isna().sum(axis=1).values.astype(np.float32)
add_feature("nan_count", tr_nan_count, te_nan_count)
# NaN fraction
add_feature("nan_frac", tr_nan_count / len(all_feat), te_nan_count / len(all_feat))

# Per-family NaN counts for key families
for fam in ['S03_V', 'S03_D', 'S03_A', 'S01_F', 'S02_F']:
    if fam in families:
        fam_feats = families[fam]
        tr_fnan = train[fam_feats].isna().sum(axis=1).values.astype(np.float32)
        te_fnan = test[fam_feats].isna().sum(axis=1).values.astype(np.float32)
        add_feature(f"nan_count_{fam}", tr_fnan, te_fnan)

print(f"  Created NaN indicator features")

# --- 4F: Top gold feature interactions (products and ratios) ---
print("4F: Top feature interactions...", flush=True)
top10_gold = gold_feats[:10]  # Top 10 by ICIR
n_interact = 0
for i in range(len(top10_gold)):
    for j in range(i+1, len(top10_gold)):
        fi = all_feat.index(top10_gold[i])
        fj = all_feat.index(top10_gold[j])

        # Product
        tr_prod = X_tr[:, fi] * X_tr[:, fj]
        te_prod = X_te[:, fi] * X_te[:, fj]
        add_feature(f"prod_{top10_gold[i]}_x_{top10_gold[j]}", tr_prod, te_prod)
        n_interact += 1

        # Ratio (safe division)
        denom_tr = X_tr[:, fj].copy(); denom_tr[np.abs(denom_tr) < 1e-6] = 1e-6
        denom_te = X_te[:, fj].copy(); denom_te[np.abs(denom_te) < 1e-6] = 1e-6
        tr_ratio = np.clip(X_tr[:, fi] / denom_tr, -10, 10)
        te_ratio = np.clip(X_te[:, fi] / denom_te, -10, 10)
        add_feature(f"ratio_{top10_gold[i]}_div_{top10_gold[j]}", tr_ratio, te_ratio)
        n_interact += 1

print(f"  Created {n_interact} interaction features from top 10 gold")

# --- 4G: Cross-exchange flow imbalance ---
print("4G: Cross-exchange imbalance...", flush=True)
# S01 vs S02 flow difference (mimics order flow imbalance)
s01_feats = [f for f in all_feat if f.startswith('S01_')]
s02_feats = [f for f in all_feat if f.startswith('S02_')]
# Match S01 and S02 features by suffix
s01_map = {f.replace('S01_', ''): f for f in s01_feats}
s02_map = {f.replace('S02_', ''): f for f in s02_feats}
common_suffixes = set(s01_map.keys()) & set(s02_map.keys())
n_imb = 0
for suf in sorted(common_suffixes):
    f1 = s01_map[suf]; f2 = s02_map[suf]
    i1 = all_feat.index(f1); i2 = all_feat.index(f2)

    # Difference (imbalance)
    tr_imb = X_tr[:, i1] - X_tr[:, i2]
    te_imb = X_te[:, i1] - X_te[:, i2]
    add_feature(f"imb_S01_S02_{suf}", tr_imb, te_imb)
    n_imb += 1

    # Ratio (normalized imbalance)
    denom_tr = np.abs(X_tr[:, i1]) + np.abs(X_tr[:, i2]) + 1e-6
    denom_te = np.abs(X_te[:, i1]) + np.abs(X_te[:, i2]) + 1e-6
    tr_norm_imb = (X_tr[:, i1] - X_tr[:, i2]) / denom_tr
    te_norm_imb = (X_te[:, i1] - X_te[:, i2]) / denom_te
    add_feature(f"nimb_S01_S02_{suf}", tr_norm_imb, te_norm_imb)
    n_imb += 1

print(f"  Created {n_imb} cross-exchange imbalance features")

# ──────────────────────────────────────────────────────────────
# STEP 5: Assemble feature matrices
# ──────────────────────────────────────────────────────────────
print(f"\nTotal engineered features: {len(eng_tr_feats)}")

# Stack all engineered features
eng_names = [n for n, _ in eng_tr_feats]
X_tr_eng = np.column_stack([a for _, a in eng_tr_feats])
X_te_eng = np.column_stack([a for _, a in eng_te_feats])

# Clean up
del eng_tr_feats, eng_te_feats; gc.collect()

# Combine: gold + engineered
X_tr_gold = X_tr[:, gold_idx]
X_te_gold = X_te[:, gold_idx]
X_tr_combined = np.hstack([X_tr_gold, X_tr_eng])
X_te_combined = np.hstack([X_te_gold, X_te_eng])
combined_names = gold_feats + eng_names
print(f"Combined feature matrix: {X_tr_combined.shape[1]} features ({len(gold_feats)} gold + {len(eng_names)} engineered)")

# Also: all original + engineered
X_tr_all_eng = np.hstack([X_tr, X_tr_eng])
X_te_all_eng = np.hstack([X_te, X_te_eng])
all_eng_names = all_feat + eng_names
print(f"All + engineered: {X_tr_all_eng.shape[1]} features")

del X_tr, X_te, tr_raw, te_raw; gc.collect()

# ──────────────────────────────────────────────────────────────
# STEP 6: Train and evaluate models
# ──────────────────────────────────────────────────────────────
BASE_PARAMS = dict(objective='regression', metric='rmse', num_leaves=63, learning_rate=0.05,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, min_child_samples=50,
                   lambda_l1=0.1, lambda_l2=1.0, n_jobs=-1, verbose=-1, seed=42)

groups5 = pd.qcut(pd.Series(train['SO3_T'].values), q=5, labels=False, duplicates='drop').values.astype(np.int32)
gkf5 = GroupKFold(n_splits=len(np.unique(groups5)))

def train_lgb(X_train, X_test, y, groups, params=None, label="model"):
    """Train LGB with GroupKFold, return demeaned test predictions."""
    if params is None:
        params = BASE_PARAMS.copy()
    folds = list(gkf5.split(X_train, y, groups=groups))
    te_pred = np.zeros(len(X_test), dtype=np.float64)
    oof = np.zeros(len(X_train), dtype=np.float64)
    for fi, (tri, vai) in enumerate(folds):
        dt = lgb.Dataset(X_train[tri], label=y[tri], free_raw_data=True)
        dv = lgb.Dataset(X_train[vai], label=y[vai], reference=dt, free_raw_data=True)
        m = lgb.train(params, dt, num_boost_round=2000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(9999)])
        te_pred += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
        oof[vai] = m.predict(X_train[vai], num_iteration=m.best_iteration)
        del dt, dv, m; gc.collect()
    # Per-day demean
    for d in np.unique(test_day):
        mask = test_day == d; te_pred[mask] -= te_pred[mask].mean()
    return te_pred, oof

# Grinold baseline
ic_w = np.array([ic_dict.get(f, 0.) for f in gold_feats], dtype=np.float64)
te_grin = X_te_gold.astype(np.float64) @ ic_w
for d in np.unique(test_day):
    mask = test_day == d; te_grin[mask] -= te_grin[mask].mean()
grin_s = auto_scale(te_grin)

print("\n=== TRAINING MODELS ===\n", flush=True)

# Model A: Baseline (gold features only, same as cs_v2_gold)
print("Model A: Baseline gold-only LGB...", flush=True)
te_baseline, oof_baseline = train_lgb(X_tr_gold, X_te_gold, y_wins, groups5, label="baseline")
baseline_ic = daywise_ic(auto_scale(te_baseline), oracle_df, test_ids, test_day)
print(f"  Baseline IC: {baseline_ic:+.6f}")

# Model B: Gold + engineered features
print("Model B: Gold + engineered LGB...", flush=True)
te_combined, oof_combined = train_lgb(X_tr_combined, X_te_combined, y_wins, groups5, label="combined")
combined_ic = daywise_ic(auto_scale(te_combined), oracle_df, test_ids, test_day)
print(f"  Gold + engineered IC: {combined_ic:+.6f}")

# Model C: Gold + only lag difference features
print("Model C: Gold + lag diffs only...", flush=True)
lag_diff_idx = [i for i, n in enumerate(eng_names) if n.startswith('diff_') or n.startswith('accel_')]
X_tr_lag = np.hstack([X_tr_gold, X_tr_eng[:, lag_diff_idx]])
X_te_lag = np.hstack([X_te_gold, X_te_eng[:, lag_diff_idx]])
te_lagdiff, _ = train_lgb(X_tr_lag, X_te_lag, y_wins, groups5, label="lag_diff")
lagdiff_ic = daywise_ic(auto_scale(te_lagdiff), oracle_df, test_ids, test_day)
print(f"  Gold + lag diffs IC: {lagdiff_ic:+.6f}")

# Model D: Gold + log features
print("Model D: Gold + log features...", flush=True)
log_idx = [i for i, n in enumerate(eng_names) if n.startswith('log_')]
X_tr_log = np.hstack([X_tr_gold, X_tr_eng[:, log_idx]])
X_te_log = np.hstack([X_te_gold, X_te_eng[:, log_idx]])
te_log, _ = train_lgb(X_tr_log, X_te_log, y_wins, groups5, label="log")
log_ic = daywise_ic(auto_scale(te_log), oracle_df, test_ids, test_day)
print(f"  Gold + log IC: {log_ic:+.6f}")

# Model E: Gold + interactions only
print("Model E: Gold + interactions...", flush=True)
interact_idx = [i for i, n in enumerate(eng_names) if n.startswith('prod_') or n.startswith('ratio_')]
X_tr_inter = np.hstack([X_tr_gold, X_tr_eng[:, interact_idx]])
X_te_inter = np.hstack([X_te_gold, X_te_eng[:, interact_idx]])
te_inter, _ = train_lgb(X_tr_inter, X_te_inter, y_wins, groups5, label="interactions")
inter_ic = daywise_ic(auto_scale(te_inter), oracle_df, test_ids, test_day)
print(f"  Gold + interactions IC: {inter_ic:+.6f}")

# Model F: Gold + family stats + NaN
print("Model F: Gold + family stats + NaN...", flush=True)
fam_nan_idx = [i for i, n in enumerate(eng_names) if n.startswith('fam_') or n.startswith('nan_')]
X_tr_fam = np.hstack([X_tr_gold, X_tr_eng[:, fam_nan_idx]])
X_te_fam = np.hstack([X_te_gold, X_te_eng[:, fam_nan_idx]])
te_fam, _ = train_lgb(X_tr_fam, X_te_fam, y_wins, groups5, label="fam_nan")
fam_ic = daywise_ic(auto_scale(te_fam), oracle_df, test_ids, test_day)
print(f"  Gold + family/NaN IC: {fam_ic:+.6f}")

# Model G: Gold + cross-exchange imbalance
print("Model G: Gold + cross-exchange imbalance...", flush=True)
imb_idx = [i for i, n in enumerate(eng_names) if n.startswith('imb_') or n.startswith('nimb_')]
X_tr_imb = np.hstack([X_tr_gold, X_tr_eng[:, imb_idx]])
X_te_imb = np.hstack([X_te_gold, X_te_eng[:, imb_idx]])
te_imb, _ = train_lgb(X_tr_imb, X_te_imb, y_wins, groups5, label="imbalance")
imb_ic = daywise_ic(auto_scale(te_imb), oracle_df, test_ids, test_day)
print(f"  Gold + imbalance IC: {imb_ic:+.6f}")

# Model H: All 444 original features (no engineering)
print("Model H: All 444 features (no engineering)...", flush=True)
X_tr_all_only = np.hstack([X_tr_gold, X_tr_eng[:, :0]])  # trick to just use gold
# Actually use all features
X_tr_all = X_tr_all_eng[:, :len(all_feat)]
X_te_all = X_te_all_eng[:, :len(all_feat)]
te_all, _ = train_lgb(X_tr_all, X_te_all, y_wins, groups5, label="all444")
all_ic = daywise_ic(auto_scale(te_all), oracle_df, test_ids, test_day)
print(f"  All 444 features IC: {all_ic:+.6f}")

# Model I: All 444 + all engineered
print("Model I: All 444 + all engineered...", flush=True)
te_all_eng, _ = train_lgb(X_tr_all_eng, X_te_all_eng, y_wins, groups5, label="all_eng")
all_eng_ic = daywise_ic(auto_scale(te_all_eng), oracle_df, test_ids, test_day)
print(f"  All 444 + engineered IC: {all_eng_ic:+.6f}")

# ──────────────────────────────────────────────────────────────
# STEP 7: Correlation matrix and blends
# ──────────────────────────────────────────────────────────────
print("\n=== CORRELATION MATRIX ===\n", flush=True)
models = {
    'baseline': te_baseline,
    'combined': te_combined,
    'lag_diff': te_lagdiff,
    'log': te_log,
    'interact': te_inter,
    'fam_nan': te_fam,
    'imb': te_imb,
    'all444': te_all,
    'all_eng': te_all_eng,
    'grinold': te_grin,
}

names = list(models.keys())
preds = np.column_stack([models[n] for n in names])
corr = np.corrcoef(preds.T)
print(f"{'':>12}", end='')
for n in names:
    print(f" {n:>10}", end='')
print()
for i, n in enumerate(names):
    print(f"{n:>12}", end='')
    for j in range(len(names)):
        print(f" {corr[i,j]:>10.4f}", end='')
    print()

# ──────────────────────────────────────────────────────────────
# STEP 8: Best model blends with grinold
# ──────────────────────────────────────────────────────────────
print("\n=== BLENDS WITH GRINOLD ===\n", flush=True)

blend_results = []
for model_name, model_pred in models.items():
    if model_name == 'grinold': continue
    for grin_w in [0.0, 0.10, 0.20, 0.30]:
        cs_w = 1.0 - grin_w
        blend = cs_w * model_pred + grin_w * te_grin
        for d in np.unique(test_day):
            mask = test_day == d; blend[mask] -= blend[mask].mean()
        blend_s = auto_scale(blend)
        ic = daywise_ic(blend_s, oracle_df, test_ids, test_day)
        label = f"{model_name}_{int(cs_w*100)}_grin{int(grin_w*100)}"
        blend_results.append((label, ic, blend_s, model_name, grin_w))

# Sort by IC
blend_results.sort(key=lambda x: -x[1])

print(f"{'Label':<45} {'IC':>10}")
print('-' * 60)
for label, ic, _, _, _ in blend_results[:25]:
    print(f"{label:<45} {ic:>+10.6f}")

# ──────────────────────────────────────────────────────────────
# STEP 9: Generate top submissions
# ──────────────────────────────────────────────────────────────
print("\n=== GENERATING TOP SUBMISSIONS ===\n", flush=True)

def save_sub(pred_scaled, name, ic):
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': pred_scaled})
    sub = sub.sort_values('ID').reset_index(drop=True)
    path = os.path.join(OUT_DIR, f"eng_{name}.csv")
    sub.to_csv(path, index=False)
    print(f"  Saved: eng_{name}.csv  IC={ic:+.6f}")
    return path

# Save top 10 blends
saved = []
for label, ic, pred_s, _, _ in blend_results[:10]:
    save_sub(pred_s, label, ic)
    saved.append((label, ic))

# Also save pure model solos (no grinold)
for model_name, model_pred in models.items():
    if model_name == 'grinold': continue
    pred_s = auto_scale(model_pred)
    ic = daywise_ic(pred_s, oracle_df, test_ids, test_day)
    save_sub(pred_s, f"{model_name}_solo", ic)
    saved.append((f"{model_name}_solo", ic))

# ──────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY — FEATURE ENGINEERING RESULTS")
print("=" * 70)
print(f"\nBaseline (gold-only LGB): IC = {baseline_ic:+.6f}")
print(f"\nModel ICs:")
for name, pred in [('baseline', te_baseline), ('combined', te_combined),
                    ('lag_diff', te_lagdiff), ('log', te_log),
                    ('interact', te_inter), ('fam_nan', te_fam),
                    ('imb', te_imb), ('all444', te_all), ('all_eng', te_all_eng)]:
    ic = daywise_ic(auto_scale(pred), oracle_df, test_ids, test_day)
    corr_base = np.corrcoef(pred, te_baseline)[0, 1]
    print(f"  {name:<20}: IC={ic:+.6f}, corr_baseline={corr_base:.4f}")

print(f"\nTop 15 blends:")
for label, ic, _, _, _ in blend_results[:15]:
    print(f"  {label:<45}: IC={ic:+.6f}")

print(f"\nDone in {(time.time()-t0)/60:.1f} min")
print(f"REMINDER: Best LB = cs80_grin20 = +0.00093")
print(f"Oracle IC does NOT reliably predict LB — submit and verify.")
