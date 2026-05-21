# ================================================================
# TRANSDUCTIVE DAILY PREDICTION
# ================================================================
# KEY FINDING: 83.6% of test days exist in training (same SO3_T
# rounded to 5dp). This is NOT a future-prediction problem.
# It is: "predict unlabeled assets on the SAME day as labeled ones."
#
# Algorithm:
#   For each test row on overlap day D:
#     1. Get all ~1546 training rows from day D
#     2. Compute per-day IC (feature-TARGET correlation)
#     3. IC-weighted prediction using top-50 features
#
#   For test rows on 84 new days (no training data):
#     → Fall back to global IC-weighted prediction
#
# Validation:
#   Split each training day 80/20 → estimate true within-day R²
#   Use increasing Ridge alpha to find optimal regularisation
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
t0 = time.time()

print("Loading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET'}]

train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)

train_days = set(train['day_id'].unique())
overlap    = train_days & set(test['day_id'].unique())
new_days   = set(test['day_id'].unique()) - train_days

print(f"  Overlap days: {len(overlap)}")
print(f"  New-only days: {len(new_days)}")
print(f"  Test rows on overlap days: {test['day_id'].isin(overlap).sum():,} ({test['day_id'].isin(overlap).mean()*100:.1f}%)")

y_train = train['TARGET'].values.astype(np.float64)
rng = np.random.default_rng(42)

# ── Pre-compute global fallback model (for new days) ─────────────
print("\nFitting global fallback Ridge (new days)...")
X_all  = train[feat_cols].fillna(0).values.astype(np.float64)
mean_g = X_all.mean(axis=0)
std_g  = X_all.std(axis=0)
std_g  = np.where(std_g < 1e-8, 1.0, std_g)
X_all_z = np.clip((X_all - mean_g) / std_g, -5.0, 5.0)

# Global ICs on z-scored data
global_ics = np.array([np.corrcoef(X_all_z[:, k], y_train)[0, 1]
                        for k in range(len(feat_cols))], dtype=np.float64)
global_ics = np.nan_to_num(global_ics)
global_top50_idx = np.argsort(np.abs(global_ics))[-50:]

# Global Ridge on top-50 z-scored features
from sklearn.linear_model import Ridge as _Ridge
global_model = _Ridge(alpha=1000, fit_intercept=True)
global_model.fit(X_all_z[:, global_top50_idx], y_train)
print(f"  Global top IC max: {np.abs(global_ics).max():.5f}")

del X_all, X_all_z
gc.collect()


# ── OOF VALIDATION: within-day 80/20 split ────────────────────────
print("\nWithin-day OOF validation (80/20 split)...")

def predict_per_day(X_test_rows, X_train_rows, y_train_rows, top_k=50, ridge_alpha=1000):
    """
    Per-day transductive prediction.
    Steps:
      1. Cross-sectionally z-score all features using training-day mean/std
      2. Select top-50 features by |IC| on z-scored data
      3. Fit Ridge(alpha) on z-scored top features
      4. Predict test rows; predictions already in TARGET-scale
    """
    # 1. Cross-sectional z-score (within-day, using training stats)
    mean_tr = X_train_rows.mean(axis=0)
    std_tr  = X_train_rows.std(axis=0)
    std_tr  = np.where(std_tr < 1e-8, 1.0, std_tr)

    # Clip to [-5, 5] after z-scoring to prevent extreme outlier influence
    CLIP = 5.0
    X_tr_z = np.clip((X_train_rows - mean_tr) / std_tr, -CLIP, CLIP)
    X_te_z = np.clip((X_test_rows  - mean_tr) / std_tr, -CLIP, CLIP)

    # 2. Winsorise TARGET
    lo, hi    = np.percentile(y_train_rows, 5), np.percentile(y_train_rows, 95)
    y_tr_clip = np.clip(y_train_rows, lo, hi)

    # 3. Per-day IC on z-scored features → select top-k
    ics = np.zeros(X_train_rows.shape[1])
    for k in range(X_train_rows.shape[1]):
        c = np.corrcoef(X_tr_z[:, k], y_tr_clip)[0, 1]
        if not np.isnan(c):
            ics[k] = c

    top_idx = np.argsort(np.abs(ics))[-top_k:]

    # 4. Ridge on z-scored top-k features
    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(X_tr_z[:, top_idx], y_tr_clip)
    return model.predict(X_te_z[:, top_idx])


oof = np.zeros(len(train))
day_r2s = {}

for day, grp in train.groupby('day_id'):
    n = len(grp)
    if n < 10:
        oof[grp.index] = y_train[grp.index].mean()
        continue

    perm   = rng.permutation(n)
    n_tr   = int(n * 0.8)
    tr_idx = grp.index[perm[:n_tr]]
    va_idx = grp.index[perm[n_tr:]]

    X_tr = grp.loc[tr_idx, feat_cols].fillna(0).values.astype(np.float64)
    y_tr = y_train[tr_idx]
    X_va = grp.loc[va_idx, feat_cols].fillna(0).values.astype(np.float64)
    y_va = y_train[va_idx]

    preds_va = predict_per_day(X_va, X_tr, y_tr, top_k=50)
    oof[va_idx] = preds_va

    if len(y_va) > 5:
        day_r2s[day] = r2_score(y_va, preds_va)

oof_r2 = r2_score(y_train, oof)
daily_r2s = list(day_r2s.values())
print(f"  OOF R² (IC-weighted, top-50 per day): {oof_r2:+.6f}")
print(f"  Median per-day R²: {np.median(daily_r2s):+.6f}")
print(f"  Mean per-day R²:   {np.mean(daily_r2s):+.6f}")
print(f"  Pred std:          {np.std(oof):.6f}")
print(f"  Pct positive:      {(oof > 0).mean()*100:.1f}%")
print(f"\n  Comparison: fold_safe_v1 OOF=+0.000544  LB=+0.00005")

# ── BUILD ACTUAL TEST PREDICTIONS ────────────────────────────────
print("\nBuilding test predictions...")
test_preds = np.zeros(len(test))
test_ids   = test['ID'].values

for day, grp_te in test.groupby('day_id'):
    te_idx    = grp_te.index
    X_te      = grp_te[feat_cols].fillna(0).values.astype(np.float64)

    if day in train_days:
        # Use same-day training data
        grp_tr = train[train['day_id'] == day]
        X_tr   = grp_tr[feat_cols].fillna(0).values.astype(np.float64)
        y_tr   = y_train[grp_tr.index]
        preds  = predict_per_day(X_te, X_tr, y_tr, top_k=50)
    else:
        # Fallback: global Ridge on z-scored features
        X_te_z = np.clip((X_te - mean_g) / std_g, -5.0, 5.0)
        preds  = global_model.predict(X_te_z[:, global_top50_idx])

    test_preds[te_idx] = preds

# Clip to ±3σ of train TARGET to prevent extreme submissions
target_std  = y_train.std()
clip_bound  = 3.0 * target_std
test_preds  = np.clip(test_preds, -clip_bound, clip_bound)
print(f"  Test pred std:    {test_preds.std():.6f}")
print(f"  Test pred mean:   {test_preds.mean():+.8f}")
print(f"  Test pct_pos:     {(test_preds > 0).mean()*100:.1f}%")
print(f"  Test skew:        {pd.Series(test_preds).skew():+.3f}")

# ── SHRINKAGE VARIANTS ────────────────────────────────────────────
# Inverse OOF-LB pattern: scale std to match fold_safe_v1's std
# OOF std=0.000624 → LB=+0.00005
# Optimal std ≈ 0.000370 from shrinkage math
sample_sub = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/sample_submission.csv'))[['ID']]

print("\nSaving submission variants...")
for alpha, label in [(1.0, 'v1_raw'),
                     (0.5, 'v2_half'),
                     (0.1, 'v3_tenth'),
                     (0.05, 'v4_005')]:
    scaled = test_preds * alpha
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': scaled})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    path = os.path.join(OUT_DIR, f'transductive_{label}.csv')
    sub.to_csv(path, index=False)
    t = sub['TARGET']
    print(f"  transductive_{label}: std={t.std():.7f}  mean={t.mean():+.8f}  "
          f"pct_pos={(t>0).mean()*100:.1f}%  skew={t.skew():+.3f}")

print(f"\nElapsed: {(time.time()-t0)/60:.1f} min")
