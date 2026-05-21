# ================================================================
# ICIR-STABLE + RANK NORMALIZATION STRATEGIES
# ================================================================
# Key finding from ICIR analysis:
#   - Top features are almost entirely LagT1/T2/T3 (mean-reversion signal)
#   - Top feature ICIR = 6.37 with mean_ic = -0.032 (always negative)
#   - 51 "gold" features: abs_icir >= 3 AND never flip sign across 428 days
#   - Previous transductive failure: selected top-50 features by THAT DAY's
#     IC → noisy selection, different features every day
#
# Fix: pre-select features by GLOBAL ICIR stability → same features every day
#
# Two normalization strategies compared:
#   A. Z-score (subtract daily mean, divide by daily std, clip ±5)
#   B. Rank   (convert to daily percentile [0,1])  ← robust to outliers
#
# Two model scopes compared:
#   1. Transductive  (per-day: use same-day labeled assets)
#   2. Global        (train on all days, GroupKFold on SO3_T quintiles)
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import rankdata

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR    = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH  = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH   = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH   = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'outputs/submissions')
t0 = time.time()

# ── Feature selection ────────────────────────────────────────────
icir_df   = pd.read_csv(ICIR_PATH)

# Gold: abs_icir >= 3 AND sign never flips (ic_pos_frac = 0 or 1)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_feats = icir_df[gold_mask].sort_values('abs_icir', ascending=False)['feature'].tolist()

# Silver: abs_icir >= 2 (includes less stable but still predictive)
silver_feats = icir_df[icir_df['abs_icir'] >= 2].sort_values(
    'abs_icir', ascending=False)['feature'].tolist()

print(f"Gold features (ICIR>=3, never flips sign): {len(gold_feats)}")
print(f"Silver features (ICIR>=2):                 {len(silver_feats)}")
print(f"Top 5 gold: {gold_feats[:5]}")

# ── Load data ────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)

# Keep only features that exist in the dataset
all_cols   = set(train.columns) - {'ID', 'TARGET'}
gold_feats = [f for f in gold_feats   if f in all_cols]
silver_feats = [f for f in silver_feats if f in all_cols]
print(f"Gold feats in dataset:   {len(gold_feats)}")
print(f"Silver feats in dataset: {len(silver_feats)}")

train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days = set(train['day_id'].unique())
overlap    = train_days & set(test['day_id'].unique())
new_days   = set(test['day_id'].unique()) - train_days

y_train  = train['TARGET'].values.astype(np.float64)
test_ids = test['ID'].values
rng      = np.random.default_rng(42)

print(f"Overlap days: {len(overlap)}  |  New days: {len(new_days)}")


# ── Normalization helpers ─────────────────────────────────────────
def zscore_day(X_tr, X_te=None, clip=5.0):
    """Cross-sectional z-score using training day stats."""
    m   = X_tr.mean(axis=0)
    s   = X_tr.std(axis=0)
    s   = np.where(s < 1e-8, 1.0, s)
    Xtz = np.clip((X_tr - m) / s, -clip, clip)
    Xez = None
    if X_te is not None:
        Xez = np.clip((X_te - m) / s, -clip, clip)
    return Xtz, Xez

def rank_day(X_tr, X_te=None):
    """
    Within-day percentile rank [0,1].
    Test rows are ranked using the combined distribution of train+test
    for that day (the only unbiased approach without future leakage).
    """
    if X_te is None:
        n = X_tr.shape[0]
        Xtr_r = np.apply_along_axis(
            lambda col: rankdata(col, method='average') / n, 0, X_tr)
        return Xtr_r, None

    # Combine, rank together, split back
    n_tr = X_tr.shape[0]
    X_all  = np.vstack([X_tr, X_te])
    n_all  = X_all.shape[0]
    X_rank = np.apply_along_axis(
        lambda col: rankdata(col, method='average') / n_all, 0, X_all)
    return X_rank[:n_tr], X_rank[n_tr:]

def winsorise(y, pct=5):
    lo, hi = np.percentile(y, pct), np.percentile(y, 100 - pct)
    return np.clip(y, lo, hi)

def fit_predict_ridge(X_tr, y_tr, X_te, alpha=100):
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_tr, winsorise(y_tr))
    return model.predict(X_te)


# ── Global fallback (new days) ────────────────────────────────────
print("\nFitting global fallbacks (for 84 new days)...")

def fit_global_fallback(feat_cols, norm_fn):
    """Train global Ridge on all training data with given normalization."""
    X_all = train[feat_cols].fillna(0).values.astype(np.float64)
    # Global normalization (across all rows — approximate since we can't
    # do per-day for a global model used as fallback)
    m, s = X_all.mean(0), X_all.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    X_z = np.clip((X_all - m) / s, -5, 5)
    model = Ridge(alpha=1000, fit_intercept=True)
    model.fit(X_z, y_train)
    return model, m, s

fb_gold_z,   mean_gz, std_gz   = fit_global_fallback(gold_feats,   zscore_day)
fb_silver_z, mean_sz, std_sz   = fit_global_fallback(silver_feats, zscore_day)
print("  Fallbacks fitted.")


# ================================================================
# EXPERIMENT MATRIX
# We run 4 experiments in one loop:
#   gold_z    : gold features + z-score
#   gold_r    : gold features + rank
#   silver_z  : silver features + z-score
#   silver_r  : silver features + rank
# ================================================================

experiments = {
    'gold_z':    (gold_feats,   'zscore', fb_gold_z,   mean_gz, std_gz),
    'gold_r':    (gold_feats,   'rank',   fb_gold_z,   mean_gz, std_gz),
    'silver_z':  (silver_feats, 'zscore', fb_silver_z, mean_sz, std_sz),
    'silver_r':  (silver_feats, 'rank',   fb_silver_z, mean_sz, std_sz),
}

oof_results   = {}
test_results  = {}
day_r2_results = {}

print("\n" + "="*60)
print("TRANSDUCTIVE EXPERIMENTS (per-day, within-day 80/20 OOF)")
print("="*60)

for exp_name, (feat_cols, norm_type, fb_model, fb_mean, fb_std) in experiments.items():
    t_exp = time.time()
    oof       = np.zeros(len(train))
    te_preds  = np.zeros(len(test))
    day_r2s   = {}

    for day, grp in train.groupby('day_id'):
        n = len(grp)
        if n < 15:
            oof[grp.index] = y_train[grp.index].mean()
            continue

        perm   = rng.permutation(n)
        n_tr   = int(n * 0.8)
        tr_idx = grp.index[perm[:n_tr]]
        va_idx = grp.index[perm[n_tr:]]

        X_tr_raw = grp.loc[tr_idx, feat_cols].fillna(0).values.astype(np.float64)
        X_va_raw = grp.loc[va_idx, feat_cols].fillna(0).values.astype(np.float64)
        y_tr     = y_train[tr_idx]
        y_va     = y_train[va_idx]

        if norm_type == 'zscore':
            X_tr_n, X_va_n = zscore_day(X_tr_raw, X_va_raw)
        else:
            X_tr_n, X_va_n = rank_day(X_tr_raw, X_va_raw)

        oof[va_idx] = fit_predict_ridge(X_tr_n, y_tr, X_va_n)

        if len(y_va) > 5:
            day_r2s[day] = r2_score(y_va, oof[va_idx])

    # Build test predictions
    for day, grp_te in test.groupby('day_id'):
        te_idx   = grp_te.index
        X_te_raw = grp_te[feat_cols].fillna(0).values.astype(np.float64)

        if day in train_days:
            grp_tr   = train[train['day_id'] == day]
            n_tr_day = len(grp_tr)

            if n_tr_day < 15:
                # Sparse day: global fallback
                X_z = np.clip((X_te_raw - fb_mean) / fb_std, -5, 5)
                te_preds[te_idx] = fb_model.predict(X_z)
            else:
                X_tr_raw = grp_tr[feat_cols].fillna(0).values.astype(np.float64)
                y_tr     = y_train[grp_tr.index]

                if norm_type == 'zscore':
                    X_tr_n, X_te_n = zscore_day(X_tr_raw, X_te_raw)
                else:
                    X_tr_n, X_te_n = rank_day(X_tr_raw, X_te_raw)

                te_preds[te_idx] = fit_predict_ridge(X_tr_n, y_tr, X_te_n)
        else:
            X_z = np.clip((X_te_raw - fb_mean) / fb_std, -5, 5)
            te_preds[te_idx] = fb_model.predict(X_z)

    # Clip to ±3σ
    clip_bound = 3.0 * y_train.std()
    te_preds   = np.clip(te_preds, -clip_bound, clip_bound)

    oof_r2     = r2_score(y_train, oof)
    med_day_r2 = np.median(list(day_r2s.values()))
    oof_results[exp_name]   = oof_r2
    test_results[exp_name]  = te_preds.copy()
    day_r2_results[exp_name] = day_r2s

    print(f"\n  [{exp_name}]  {len(feat_cols)} feats  norm={norm_type}")
    print(f"    OOF R²         : {oof_r2:+.6f}")
    print(f"    Median day R²  : {med_day_r2:+.6f}")
    print(f"    Test pred std  : {te_preds.std():.6f}")
    print(f"    Test mean      : {te_preds.mean():+.8f}")
    print(f"    Test pct_pos   : {(te_preds > 0).mean()*100:.1f}%")
    print(f"    Test skew      : {pd.Series(te_preds).skew():+.3f}")
    print(f"    Time           : {time.time()-t_exp:.0f}s")


# ── Summary comparison ───────────────────────────────────────────
print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
print(f"  {'Experiment':<12} {'OOF R²':>10} {'Med day R²':>12} {'Test std':>10}")
print(f"  {'-'*46}")
for name, r2 in sorted(oof_results.items(), key=lambda x: x[1]):
    te = test_results[name]
    med = np.median(list(day_r2_results[name].values()))
    print(f"  {name:<12} {r2:+10.6f} {med:+12.6f} {te.std():10.6f}")

print(f"\n  Reference — transductive_v4_005 (top-50 per-day IC):")
print(f"  {'old_trans':12} {'?':>10} {'?':>12} {'0.008653':>10}  LB=+0.00003")
print(f"  Reference — fold_safe_v1:")
print(f"  {'fold_v1':12} {'+0.000544':>10} {'?':>12} {'0.000624':>10}  LB=+0.00005")


# ── Save submissions with scaling variants ───────────────────────
sample_sub = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/sample_submission.csv'))[['ID']]

def save_sub(preds, name):
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': preds})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    path = os.path.join(OUT_DIR, f'{name}.csv')
    sub.to_csv(path, index=False)
    t = sub['TARGET']
    return t.std(), t.mean(), (t > 0).mean(), t.skew()

print("\n" + "="*60)
print("SAVED SUBMISSIONS")
print("="*60)

# We care most about std being near fold_safe_v1's 0.000624
# Raw test std is ~0.008-0.012 for transductive models
# Target scale: ~0.05 of raw = std ≈ 0.0004-0.0006

for exp_name in experiments:
    raw = test_results[exp_name]
    raw_std = raw.std()
    print(f"\n  {exp_name} (raw std={raw_std:.6f}):")
    for alpha, label in [(0.05, '5pct'), (0.03, '3pct'), (0.07, '7pct')]:
        std, mean, ppos, skew = save_sub(raw * alpha, f'icir_{exp_name}_{label}')
        print(f"    icir_{exp_name}_{label}: "
              f"std={std:.7f}  mean={mean:+.8f}  "
              f"pct_pos={ppos*100:.1f}%  skew={skew:+.3f}")

print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")

# ── Recommendation ───────────────────────────────────────────────
best_exp = max(oof_results, key=oof_results.get)
print(f"""
WHICH TO SUBMIT:
────────────────
Key principle from competition history:
  OOF R² is unreliable — lower OOF sometimes = better LB.
  We trust LB pattern over OOF.

What's genuinely different here vs previous failures:
  ✓ Pre-selected gold features by GLOBAL ICIR (not per-day IC mining)
  ✓ Features never flip sign across 428 days — they ARE the signal
  ✓ Rank normalization is robust to the universe-size fluctuation

What to look for in std values above:
  • Closest to fold_safe_v1 std (0.000624) = safest starting point
  • Rank model std slightly different from z-score (different scale)
  • Pick the variant whose std is between 0.0004 and 0.0007

Gold features encode mean-reversion: LagT1 features with mean_ic=-0.032
ALWAYS predict lower return for assets with high recent values.
This structural signal should generalize better than any model
that data-mines per-day relationships.
""")
