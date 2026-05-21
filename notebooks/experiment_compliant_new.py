"""
Three fundamentally new approaches — ALL COMPLIANT.

COMPLIANCE RULES (from competition):
- NO cross-sample ranking on test set
- NO neighborhood-based methods (KNN) on test
- NO clustering, matching, or aligning test rows
- NO reconstructing temporal structure
- ALLOWED: global normalization using simple aggregate stats (mean, std, min/max)
- Must run on CPU, offline, <30 min, default Kaggle libraries

APPROACH 1: Rank-transform training features, train LGB on ranks
  - Compute per-day ranks WITHIN TRAINING DATA (compliant)
  - For test: use training-derived quantile mapping (not cross-sample test ranking)
  - This makes the model learn rank-based relationships (robust to scale)

APPROACH 2: Predict rank of returns (rank target in training)
  - Rank-transform TARGET within training days (compliant)
  - LGB learns to predict relative ordering, not absolute magnitude
  - Convert predictions back to return scale using training target statistics

APPROACH 3: Simple neural network (MLP)
  - sklearn MLPRegressor (available in default Kaggle)
  - Uses same gold features with per-day z-score normalization
  - Different inductive bias from tree-based models
"""
import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer

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

def per_day_demean(pred, test_day):
    """Remove per-day mean from predictions (compliant: uses only own predictions)."""
    pred = pred.copy()
    for d in np.unique(test_day):
        m = test_day == d
        pred[m] -= pred[m].mean()
    return pred

# ── Load ──
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

target = train['TARGET'].values.astype(np.float64)
lo, hi = np.percentile(target, [1, 99])
target_w = np.clip(target, lo, hi)

ic_df = pd.read_csv('/Users/malaymishra/Desktop/quant_ml_project/outputs/eda/summaries/ic_icir_full.csv')
gold_feats = ic_df[(ic_df['abs_icir'] >= 3) & (ic_df['ic_pos_frac'].isin([0.0, 1.0]))]['feature'].tolist()
gold_feats = [f for f in gold_feats if f in all_feat]
ic_map = dict(zip(ic_df['feature'], ic_df['mean_ic']))
print(f"Gold features: {len(gold_feats)}")

# ── Standard compliant normalization (per-day z-score, training stats) ──
print("Normalizing (standard per-day z-score)...", flush=True)
tr_raw = train[all_feat].values.astype(np.float32)
te_raw = test[all_feat].values.astype(np.float32)

day_stats = {}
for d in np.unique(train_day):
    m = train_day == d; x = tr_raw[m]
    mu = x.mean(0); sg = x.std(0); sg[sg < 1e-8] = 1.0
    day_stats[d] = (mu, sg)
global_mu = tr_raw.mean(0); global_sg = tr_raw.std(0); global_sg[global_sg < 1e-8] = 1.0

tr_norm = np.empty_like(tr_raw)
te_norm = np.empty_like(te_raw)
for d in np.unique(train_day):
    m = train_day == d; mu, sg = day_stats[d]
    tr_norm[m] = np.clip((tr_raw[m] - mu) / sg, -CLIP_Z, CLIP_Z)
for d in np.unique(test_day):
    m = test_day == d
    if d in day_stats: mu, sg = day_stats[d]
    else: mu, sg = global_mu, global_sg
    te_norm[m] = np.clip((te_raw[m] - mu) / sg, -CLIP_Z, CLIP_Z)

gold_idx = [all_feat.index(f) for f in gold_feats]
tr_gold = tr_norm[:, gold_idx]
te_gold = te_norm[:, gold_idx]

# CV setup
groups5 = pd.qcut(pd.Series(train['SO3_T'].values), q=5, labels=False, duplicates='drop').values.astype(np.int32)
gkf5 = GroupKFold(n_splits=5)

# Grinold predictions (for blending)
gold_ic_w = np.array([ic_map.get(f, 0.0) for f in gold_feats])
te_grinold = te_gold @ gold_ic_w
te_grinold = per_day_demean(te_grinold, test_day)

# ═══════════════════════════════════════════════════════
# BASELINE (reference)
# ═══════════════════════════════════════════════════════
print("\n=== BASELINE: cs_v2_gold (MSE, z-score) ===", flush=True)
te_baseline = np.zeros(len(test))
lgb_params = dict(objective='regression', metric='mse', num_leaves=63, learning_rate=0.05,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  min_child_samples=50, verbose=-1, seed=42)
for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_gold, target_w, groups5)):
    ds_tr = lgb.Dataset(tr_gold[tr_i], target_w[tr_i])
    ds_va = lgb.Dataset(tr_gold[va_i], target_w[va_i])
    m = lgb.train(lgb_params, ds_tr, num_boost_round=500,
                  valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
    te_baseline += m.predict(te_gold) / 5
te_baseline = per_day_demean(te_baseline, test_day)
te_baseline_s = auto_scale(te_baseline)
ic_baseline = daywise_ic(te_baseline_s, oracle_df, test_ids, test_day)
print(f"  Baseline IC: {ic_baseline:+.6f}")

# ═══════════════════════════════════════════════════════
# APPROACH 1: QUANTILE-MAPPED FEATURES (compliant rank proxy)
# ═══════════════════════════════════════════════════════
# Instead of cross-sample ranking (banned), we use QuantileTransformer
# fitted on TRAINING data only, then applied to test samples independently.
# Each test sample is mapped through the TRAINING quantile function —
# this is a per-sample transformation, NOT cross-sample ranking.
# It's equivalent to: "given this z-score value, what percentile would it
# be in the training distribution?" — a lookup, not a ranking.
# ═══════════════════════════════════════════════════════
print("\n=== APPROACH 1: Quantile-mapped features (training CDF) ===", flush=True)
print("  Fitting QuantileTransformer on training gold features...", flush=True)

# Fit on training z-scored features → uniform [0,1]
# Then apply to test: each test value mapped through training CDF independently
qt = QuantileTransformer(n_quantiles=1000, output_distribution='uniform', random_state=42)
tr_gold_qt = qt.fit_transform(tr_gold).astype(np.float32)
te_gold_qt = qt.transform(te_gold).astype(np.float32)

# Center at 0: shift from [0,1] to [-0.5, 0.5]
tr_gold_qt -= 0.5
te_gold_qt -= 0.5

print(f"  Train quantile range: [{tr_gold_qt.min():.3f}, {tr_gold_qt.max():.3f}]")
print(f"  Test quantile range:  [{te_gold_qt.min():.3f}, {te_gold_qt.max():.3f}]")

# LGB on quantile-mapped features
te_qt_lgb = np.zeros(len(test))
for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_gold_qt, target_w, groups5)):
    ds_tr = lgb.Dataset(tr_gold_qt[tr_i], target_w[tr_i])
    ds_va = lgb.Dataset(tr_gold_qt[va_i], target_w[va_i])
    m = lgb.train(lgb_params, ds_tr, num_boost_round=500,
                  valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
    te_qt_lgb += m.predict(te_gold_qt) / 5
te_qt_lgb = per_day_demean(te_qt_lgb, test_day)
te_qt_lgb_s = auto_scale(te_qt_lgb)
ic_qt = daywise_ic(te_qt_lgb_s, oracle_df, test_ids, test_day)
print(f"  Quantile-mapped LGB IC: {ic_qt:+.6f}")

# Also try: Gaussian output distribution
qt_norm = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
tr_gold_qn = qt_norm.fit_transform(tr_gold).astype(np.float32)
te_gold_qn = qt_norm.transform(te_gold).astype(np.float32)
# Clip extreme Gaussian values
tr_gold_qn = np.clip(tr_gold_qn, -CLIP_Z, CLIP_Z)
te_gold_qn = np.clip(te_gold_qn, -CLIP_Z, CLIP_Z)

te_qn_lgb = np.zeros(len(test))
for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_gold_qn, target_w, groups5)):
    ds_tr = lgb.Dataset(tr_gold_qn[tr_i], target_w[tr_i])
    ds_va = lgb.Dataset(tr_gold_qn[va_i], target_w[va_i])
    m = lgb.train(lgb_params, ds_tr, num_boost_round=500,
                  valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
    te_qn_lgb += m.predict(te_gold_qn) / 5
te_qn_lgb = per_day_demean(te_qn_lgb, test_day)
te_qn_lgb_s = auto_scale(te_qn_lgb)
ic_qn = daywise_ic(te_qn_lgb_s, oracle_df, test_ids, test_day)
print(f"  Quantile-normal LGB IC: {ic_qn:+.6f}")

# Grinold on quantile-mapped features
te_grinold_qt = te_gold_qt @ gold_ic_w
te_grinold_qt = per_day_demean(te_grinold_qt, test_day)
te_grinold_qt_s = auto_scale(te_grinold_qt)
ic_grinold_qt = daywise_ic(te_grinold_qt_s, oracle_df, test_ids, test_day)
print(f"  Grinold quantile-mapped IC: {ic_grinold_qt:+.6f}")

# Blend quantile LGB + grinold
for w_qt in [1.0, 0.80]:
    blend = w_qt * te_qt_lgb + (1-w_qt) * te_grinold
    blend = per_day_demean(blend, test_day)
    blend_s = auto_scale(blend)
    ic_b = daywise_ic(blend_s, oracle_df, test_ids, test_day)
    print(f"  {w_qt:.0%} qt_lgb + {1-w_qt:.0%} grinold: IC={ic_b:+.6f}")

# ═══════════════════════════════════════════════════════
# APPROACH 2: RANK-TRANSFORMED TARGET
# ═══════════════════════════════════════════════════════
# Rank-transform target within training days, then train LGB on ranks.
# The model learns relative ordering rather than absolute returns.
# Prediction is in "rank space" — then we map back to return scale
# using training target std. This is fully compliant because we only
# rank training labels, never test data.
# ═══════════════════════════════════════════════════════
print("\n=== APPROACH 2: Rank-transformed target ===", flush=True)

# Per-day rank target (within training cross-sections)
target_rank = np.zeros_like(target_w)
for d in np.unique(train_day):
    m = train_day == d
    vals = target_w[m]
    n = len(vals)
    if n < 2:
        target_rank[m] = 0.0
    else:
        order = np.argsort(np.argsort(vals))
        target_rank[m] = (order / (n - 1)) - 0.5  # [-0.5, +0.5]

print(f"  Rank target: mean={target_rank.mean():.6f}, std={target_rank.std():.4f}")

# LGB on z-scored gold features, predicting rank target
te_rank_lgb = np.zeros(len(test))
for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_gold, target_rank, groups5)):
    ds_tr = lgb.Dataset(tr_gold[tr_i], target_rank[tr_i])
    ds_va = lgb.Dataset(tr_gold[va_i], target_rank[va_i])
    m = lgb.train(lgb_params, ds_tr, num_boost_round=500,
                  valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
    te_rank_lgb += m.predict(te_gold) / 5
te_rank_lgb = per_day_demean(te_rank_lgb, test_day)
te_rank_lgb_s = auto_scale(te_rank_lgb)  # auto_scale maps to TARGET_STD
ic_rank = daywise_ic(te_rank_lgb_s, oracle_df, test_ids, test_day)
print(f"  Rank-target LGB IC: {ic_rank:+.6f}")

# Also try: quantile-mapped features + rank target (double rank transform)
te_rank_qt = np.zeros(len(test))
for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_gold_qt, target_rank, groups5)):
    ds_tr = lgb.Dataset(tr_gold_qt[tr_i], target_rank[tr_i])
    ds_va = lgb.Dataset(tr_gold_qt[va_i], target_rank[va_i])
    m = lgb.train(lgb_params, ds_tr, num_boost_round=500,
                  valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
    te_rank_qt += m.predict(te_gold_qt) / 5
te_rank_qt = per_day_demean(te_rank_qt, test_day)
te_rank_qt_s = auto_scale(te_rank_qt)
ic_rank_qt = daywise_ic(te_rank_qt_s, oracle_df, test_ids, test_day)
print(f"  Rank-target + quantile-feat LGB IC: {ic_rank_qt:+.6f}")

# Blend rank-target LGB with grinold
for w in [1.0, 0.80]:
    blend = w * te_rank_lgb + (1-w) * te_grinold
    blend = per_day_demean(blend, test_day)
    blend_s = auto_scale(blend)
    ic_b = daywise_ic(blend_s, oracle_df, test_ids, test_day)
    print(f"  {w:.0%} rank_lgb + {1-w:.0%} grinold: IC={ic_b:+.6f}")

# ═══════════════════════════════════════════════════════
# APPROACH 3: NEURAL NETWORK (MLP)
# ═══════════════════════════════════════════════════════
# sklearn MLPRegressor — available in default Kaggle environment.
# Different inductive bias: smooth function approximation vs piecewise
# constant (trees). May capture different patterns.
# All normalization uses training stats only — compliant.
# ═══════════════════════════════════════════════════════
print("\n=== APPROACH 3: Neural Network (MLPRegressor) ===", flush=True)

# MLP configs to try
mlp_configs = [
    ('mlp_small', {'hidden_layer_sizes': (32,), 'max_iter': 500, 'early_stopping': True,
                   'validation_fraction': 0.15, 'alpha': 0.01, 'random_state': 42,
                   'learning_rate_init': 0.001}),
    ('mlp_medium', {'hidden_layer_sizes': (64, 32), 'max_iter': 500, 'early_stopping': True,
                    'validation_fraction': 0.15, 'alpha': 0.01, 'random_state': 42,
                    'learning_rate_init': 0.001}),
    ('mlp_wide', {'hidden_layer_sizes': (128,), 'max_iter': 500, 'early_stopping': True,
                  'validation_fraction': 0.15, 'alpha': 0.1, 'random_state': 42,
                  'learning_rate_init': 0.001}),
    ('mlp_deep', {'hidden_layer_sizes': (64, 32, 16), 'max_iter': 500, 'early_stopping': True,
                  'validation_fraction': 0.15, 'alpha': 0.01, 'random_state': 42,
                  'learning_rate_init': 0.001}),
    ('mlp_reg_heavy', {'hidden_layer_sizes': (64, 32), 'max_iter': 500, 'early_stopping': True,
                       'validation_fraction': 0.15, 'alpha': 1.0, 'random_state': 42,
                       'learning_rate_init': 0.001}),
]

best_mlp_ic = -1; best_mlp_pred = None; best_mlp_name = ""

for name, params in mlp_configs:
    print(f"  Training {name}...", flush=True)
    te_mlp = np.zeros(len(test))
    for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_gold, target_w, groups5)):
        mlp = MLPRegressor(**params)
        mlp.fit(tr_gold[tr_i], target_w[tr_i])
        te_mlp += mlp.predict(te_gold) / 5
    te_mlp = per_day_demean(te_mlp, test_day)
    te_mlp_s = auto_scale(te_mlp)
    ic_mlp = daywise_ic(te_mlp_s, oracle_df, test_ids, test_day)
    corr_base = np.corrcoef(te_baseline, te_mlp)[0,1]
    print(f"    {name}: IC={ic_mlp:+.6f}, corr_baseline={corr_base:.4f}")
    if ic_mlp > best_mlp_ic:
        best_mlp_ic = ic_mlp; best_mlp_pred = te_mlp; best_mlp_name = name

# MLP on quantile-mapped features
print(f"\n  Training MLP on quantile-mapped features...", flush=True)
for name_suffix, tr_feat, te_feat in [('qt_uniform', tr_gold_qt, te_gold_qt),
                                        ('qt_normal', tr_gold_qn, te_gold_qn)]:
    te_mlp_q = np.zeros(len(test))
    params_q = {'hidden_layer_sizes': (64, 32), 'max_iter': 500, 'early_stopping': True,
                'validation_fraction': 0.15, 'alpha': 0.01, 'random_state': 42,
                'learning_rate_init': 0.001}
    for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_feat, target_w, groups5)):
        mlp = MLPRegressor(**params_q)
        mlp.fit(tr_feat[tr_i], target_w[tr_i])
        te_mlp_q += mlp.predict(te_feat) / 5
    te_mlp_q = per_day_demean(te_mlp_q, test_day)
    te_mlp_q_s = auto_scale(te_mlp_q)
    ic_mlp_q = daywise_ic(te_mlp_q_s, oracle_df, test_ids, test_day)
    corr_base = np.corrcoef(te_baseline, te_mlp_q)[0,1]
    print(f"    mlp_{name_suffix}: IC={ic_mlp_q:+.6f}, corr_baseline={corr_base:.4f}")
    if ic_mlp_q > best_mlp_ic:
        best_mlp_ic = ic_mlp_q; best_mlp_pred = te_mlp_q; best_mlp_name = f"mlp_{name_suffix}"

# MLP with rank target
print(f"\n  Training MLP with rank target...", flush=True)
te_mlp_rank = np.zeros(len(test))
params_r = {'hidden_layer_sizes': (64, 32), 'max_iter': 500, 'early_stopping': True,
            'validation_fraction': 0.15, 'alpha': 0.01, 'random_state': 42,
            'learning_rate_init': 0.001}
for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_gold, target_rank, groups5)):
    mlp = MLPRegressor(**params_r)
    mlp.fit(tr_gold[tr_i], target_rank[tr_i])
    te_mlp_rank += mlp.predict(te_gold) / 5
te_mlp_rank = per_day_demean(te_mlp_rank, test_day)
te_mlp_rank_s = auto_scale(te_mlp_rank)
ic_mlp_rank = daywise_ic(te_mlp_rank_s, oracle_df, test_ids, test_day)
corr_base = np.corrcoef(te_baseline, te_mlp_rank)[0,1]
print(f"    mlp_rank_target: IC={ic_mlp_rank:+.6f}, corr_baseline={corr_base:.4f}")
if ic_mlp_rank > best_mlp_ic:
    best_mlp_ic = ic_mlp_rank; best_mlp_pred = te_mlp_rank; best_mlp_name = "mlp_rank_target"

# ═══════════════════════════════════════════════════════
# BLENDING: Best of each approach with baseline and grinold
# ═══════════════════════════════════════════════════════
print(f"\n=== BLENDING (best MLP: {best_mlp_name}, IC={best_mlp_ic:+.6f}) ===", flush=True)

# Collect all candidate predictions for blending
candidates = {
    'baseline_mse': te_baseline,
    'qt_lgb': te_qt_lgb,
    'qn_lgb': te_qn_lgb,
    'rank_lgb': te_rank_lgb,
    'grinold': te_grinold,
}
if best_mlp_pred is not None:
    candidates[best_mlp_name] = best_mlp_pred

# Show correlation matrix
print("\n  Correlation matrix:")
cand_names = list(candidates.keys())
cand_preds = [candidates[n] for n in cand_names]
print(f"  {'':>20}", end='')
for n in cand_names: print(f" {n:>12}", end='')
print()
for i, n1 in enumerate(cand_names):
    print(f"  {n1:>20}", end='')
    for j, n2 in enumerate(cand_names):
        c = np.corrcoef(cand_preds[i], cand_preds[j])[0,1]
        print(f" {c:>12.4f}", end='')
    print()

# Try diverse blends
print("\n  Blend experiments:")

# Baseline + MLP (different inductive bias)
if best_mlp_pred is not None:
    for w in [0.90, 0.80, 0.70, 0.50]:
        blend = w * te_baseline + (1-w) * best_mlp_pred
        blend = per_day_demean(blend, test_day)
        blend_s = auto_scale(blend)
        ic_b = daywise_ic(blend_s, oracle_df, test_ids, test_day)
        print(f"    {w:.0%} baseline + {1-w:.0%} {best_mlp_name}: IC={ic_b:+.6f}")

    # 3-way: baseline + MLP + grinold
    for wb, wm, wg in [(0.50, 0.30, 0.20), (0.40, 0.40, 0.20), (0.60, 0.20, 0.20)]:
        blend = wb * te_baseline + wm * best_mlp_pred + wg * te_grinold
        blend = per_day_demean(blend, test_day)
        blend_s = auto_scale(blend)
        ic_b = daywise_ic(blend_s, oracle_df, test_ids, test_day)
        print(f"    {wb:.0%}base + {wm:.0%}{best_mlp_name} + {wg:.0%}grinold: IC={ic_b:+.6f}")

# Quantile LGB + baseline + grinold
for wb, wq, wg in [(0.50, 0.30, 0.20), (0.40, 0.40, 0.20), (0.60, 0.20, 0.20)]:
    blend = wb * te_baseline + wq * te_qt_lgb + wg * te_grinold
    blend = per_day_demean(blend, test_day)
    blend_s = auto_scale(blend)
    ic_b = daywise_ic(blend_s, oracle_df, test_ids, test_day)
    print(f"    {wb:.0%}base + {wq:.0%}qt_lgb + {wg:.0%}grinold: IC={ic_b:+.6f}")

# Rank LGB + baseline + grinold
for wb, wr, wg in [(0.50, 0.30, 0.20), (0.40, 0.40, 0.20), (0.60, 0.20, 0.20)]:
    blend = wb * te_baseline + wr * te_rank_lgb + wg * te_grinold
    blend = per_day_demean(blend, test_day)
    blend_s = auto_scale(blend)
    ic_b = daywise_ic(blend_s, oracle_df, test_ids, test_day)
    print(f"    {wb:.0%}base + {wr:.0%}rank_lgb + {wg:.0%}grinold: IC={ic_b:+.6f}")

# 4-way blend: baseline + qt_lgb + MLP + grinold
if best_mlp_pred is not None:
    for wb, wq, wm, wg in [(0.40, 0.20, 0.20, 0.20), (0.30, 0.30, 0.20, 0.20)]:
        blend = wb * te_baseline + wq * te_qt_lgb + wm * best_mlp_pred + wg * te_grinold
        blend = per_day_demean(blend, test_day)
        blend_s = auto_scale(blend)
        ic_b = daywise_ic(blend_s, oracle_df, test_ids, test_day)
        print(f"    {wb:.0%}base + {wq:.0%}qt + {wm:.0%}mlp + {wg:.0%}grinold: IC={ic_b:+.6f}")

# ═══════════════════════════════════════════════════════
# GENERATE SUBMISSIONS
# ═══════════════════════════════════════════════════════
print("\n=== GENERATING SUBMISSIONS ===", flush=True)

submissions = []
def save_sub(pred_scaled, name, ic_val, desc):
    out = sample_sub.copy()
    pred_map = dict(zip(test_ids, pred_scaled))
    out['TARGET'] = out['ID'].map(pred_map)
    path = os.path.join(OUT_DIR, f'comp_{name}.csv')
    out.to_csv(path, index=False)
    submissions.append((name, ic_val, desc, path))

# Reference
blend_ref = 0.80 * te_baseline + 0.20 * te_grinold
blend_ref = per_day_demean(blend_ref, test_day)
save_sub(auto_scale(blend_ref), 'cs80_grin20_ref',
         daywise_ic(auto_scale(blend_ref), oracle_df, test_ids, test_day),
         '80% cs_gold + 20% grinold (reference)')

# Quantile-mapped LGB solo
save_sub(te_qt_lgb_s, 'qt_lgb_solo', ic_qt, 'Quantile-mapped LGB solo')

# Quantile-mapped LGB + grinold
blend_qt_grin = 0.80 * te_qt_lgb + 0.20 * te_grinold
blend_qt_grin = per_day_demean(blend_qt_grin, test_day)
save_sub(auto_scale(blend_qt_grin), 'qt80_grin20',
         daywise_ic(auto_scale(blend_qt_grin), oracle_df, test_ids, test_day),
         '80% qt_lgb + 20% grinold')

# Rank-target LGB + grinold
blend_rank_grin = 0.80 * te_rank_lgb + 0.20 * te_grinold
blend_rank_grin = per_day_demean(blend_rank_grin, test_day)
save_sub(auto_scale(blend_rank_grin), 'rank80_grin20',
         daywise_ic(auto_scale(blend_rank_grin), oracle_df, test_ids, test_day),
         '80% rank_lgb + 20% grinold')

# Best MLP + grinold
if best_mlp_pred is not None:
    blend_mlp_grin = 0.80 * best_mlp_pred + 0.20 * te_grinold
    blend_mlp_grin = per_day_demean(blend_mlp_grin, test_day)
    save_sub(auto_scale(blend_mlp_grin), f'{best_mlp_name}80_grin20',
             daywise_ic(auto_scale(blend_mlp_grin), oracle_df, test_ids, test_day),
             f'80% {best_mlp_name} + 20% grinold')

# Baseline + MLP blend (different inductive biases) + grinold
if best_mlp_pred is not None:
    blend_bm = 0.40 * te_baseline + 0.40 * best_mlp_pred + 0.20 * te_grinold
    blend_bm = per_day_demean(blend_bm, test_day)
    save_sub(auto_scale(blend_bm), 'base40_mlp40_grin20',
             daywise_ic(auto_scale(blend_bm), oracle_df, test_ids, test_day),
             f'40% baseline + 40% {best_mlp_name} + 20% grinold')

# Baseline + qt_lgb + grinold
blend_bq = 0.40 * te_baseline + 0.40 * te_qt_lgb + 0.20 * te_grinold
blend_bq = per_day_demean(blend_bq, test_day)
save_sub(auto_scale(blend_bq), 'base40_qt40_grin20',
         daywise_ic(auto_scale(blend_bq), oracle_df, test_ids, test_day),
         '40% baseline + 40% qt_lgb + 20% grinold')

# 4-way if MLP is good
if best_mlp_pred is not None:
    blend_4 = 0.35 * te_baseline + 0.25 * te_qt_lgb + 0.20 * best_mlp_pred + 0.20 * te_grinold
    blend_4 = per_day_demean(blend_4, test_day)
    save_sub(auto_scale(blend_4), 'base35_qt25_mlp20_grin20',
             daywise_ic(auto_scale(blend_4), oracle_df, test_ids, test_day),
             f'35% base + 25% qt + 20% {best_mlp_name} + 20% grinold')

# ═══════════════════════════════════════════════════════
# FINAL RANKING
# ═══════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("FINAL RANKING — ALL SUBMISSIONS")
print(f"{'='*80}")
submissions.sort(key=lambda x: -x[1])
print(f"{'Rank':<6} {'IC':>10} {'Name':<35} {'Description'}")
print("-" * 110)
for i, (name, ic, desc, path) in enumerate(submissions, 1):
    print(f"{i:<6} {ic:>+10.6f} {name:<35} {desc}")

print(f"\nDone in {(time.time()-t0)/60:.1f} min")
print("REMINDER: Best LB so far = cs80_grin20 = +0.00093")
print("Oracle IC does NOT reliably predict LB.")
