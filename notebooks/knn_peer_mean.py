# ================================================================
# KNN PEER MEAN — TRANSDUCTIVE SAME-DAY NEAREST NEIGHBORS
# ================================================================
# Core insight: 83.6% of test days exist in training (same day_id).
# For each test row on an overlap day, the K most similar labeled
# training assets from THAT SAME DAY are its best predictors.
#
# Algorithm:
#   For each test row on overlap day D:
#     1. Cross-sectionally z-score features using D's training stats
#     2. Select top-50 features by |per-day IC| to avoid noise dims
#     3. Find K nearest training neighbors in z-scored IC space
#     4. Predict = distance-weighted avg TARGET of K neighbors
#
#   For test rows on 84 new days:
#     → Fall back to global Ridge (same as transductive_daily.py)
#
# Key fix over user's Kaggle notebook:
#   The original used raw Euclidean distance on unnormalized features.
#   S01_F03_U01_LagT1 has std=26,560 — it would dominate every distance.
#   This script z-scores within each day BEFORE computing KNN.
#
# Validation:
#   Within-day 80/20 OOF split (same structure as transductive_daily.py)
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
t0 = time.time()

# ── Load ─────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET'}]

train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)

train_days = set(train['day_id'].unique())
overlap    = train_days & set(test['day_id'].unique())
new_days   = set(test['day_id'].unique()) - train_days

print(f"  Overlap days: {len(overlap)}  |  New-only days: {len(new_days)}")
print(f"  Test rows on overlap days: {test['day_id'].isin(overlap).sum():,} "
      f"({test['day_id'].isin(overlap).mean()*100:.1f}%)")

y_train = train['TARGET'].values.astype(np.float64)
rng = np.random.default_rng(42)

# ── Pre-compute global fallback (for new days) ───────────────────
print("\nFitting global fallback Ridge (new days)...")
X_all   = train[feat_cols].fillna(0).values.astype(np.float64)
mean_g  = X_all.mean(axis=0)
std_g   = X_all.std(axis=0)
std_g   = np.where(std_g < 1e-8, 1.0, std_g)
X_all_z = np.clip((X_all - mean_g) / std_g, -5.0, 5.0)

global_ics = np.array([np.corrcoef(X_all_z[:, k], y_train)[0, 1]
                        for k in range(len(feat_cols))], dtype=np.float64)
global_ics = np.nan_to_num(global_ics)
global_top50_idx = np.argsort(np.abs(global_ics))[-50:]

global_model = Ridge(alpha=1000, fit_intercept=True)
global_model.fit(X_all_z[:, global_top50_idx], y_train)
print(f"  Global top IC max: {np.abs(global_ics).max():.5f}")

del X_all, X_all_z
gc.collect()


# ── Core per-day z-scoring + IC selection helper ─────────────────
def zscore_and_select_ics(X_train_rows, y_train_rows, X_test_rows=None,
                           top_k=50, clip=5.0, winsor_pct=5):
    """
    Returns (X_tr_z_topk, X_te_z_topk, top_idx, y_tr_clip)
    using training-day stats for z-scoring.
    """
    mean_tr = X_train_rows.mean(axis=0)
    std_tr  = X_train_rows.std(axis=0)
    std_tr  = np.where(std_tr < 1e-8, 1.0, std_tr)

    X_tr_z = np.clip((X_train_rows - mean_tr) / std_tr, -clip, clip)

    # Winsorise TARGET
    lo, hi    = np.percentile(y_train_rows, winsor_pct), np.percentile(y_train_rows, 100 - winsor_pct)
    y_tr_clip = np.clip(y_train_rows, lo, hi)

    # Per-day IC → top-k features
    ics = np.zeros(X_train_rows.shape[1])
    for k in range(X_train_rows.shape[1]):
        c = np.corrcoef(X_tr_z[:, k], y_tr_clip)[0, 1]
        if not np.isnan(c):
            ics[k] = c
    top_idx = np.argsort(np.abs(ics))[-top_k:]

    X_te_z_topk = None
    if X_test_rows is not None:
        X_te_z = np.clip((X_test_rows - mean_tr) / std_tr, -clip, clip)
        X_te_z_topk = X_te_z[:, top_idx]

    return X_tr_z[:, top_idx], X_te_z_topk, top_idx, y_tr_clip


def predict_knn(X_te_z, X_tr_z, y_tr, K=5, use_distance_weights=True):
    """
    K-NN prediction: for each test point, find K nearest training points
    and return (distance-)weighted average of their TARGET values.
    """
    n_te = X_te_z.shape[0]
    n_tr = X_tr_z.shape[0]
    K_eff = min(K, n_tr)

    nbrs = NearestNeighbors(n_neighbors=K_eff, algorithm='auto',
                            metric='euclidean', n_jobs=-1)
    nbrs.fit(X_tr_z)
    dists, idxs = nbrs.kneighbors(X_te_z)  # (n_te, K_eff)

    if use_distance_weights:
        # Inverse distance weighting; add small eps to avoid div/0
        eps = 1e-8
        w   = 1.0 / (dists + eps)           # (n_te, K_eff)
        w  /= w.sum(axis=1, keepdims=True)  # normalise rows to sum=1
    else:
        w = np.ones_like(dists) / K_eff

    preds = (w * y_tr[idxs]).sum(axis=1)   # (n_te,)
    return preds


def predict_ridge(X_te_z, X_tr_z, y_tr, alpha=1000):
    """Standard Ridge regression on z-scored top-k features."""
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_tr_z, y_tr)
    return model.predict(X_te_z)


# ── OOF VALIDATION: within-day 80/20 split ───────────────────────
print("\nWithin-day OOF validation (80/20 split)...")

K_VALS   = [3, 5, 10]
oof_knn  = {K: np.zeros(len(train)) for K in K_VALS}
oof_ridge = np.zeros(len(train))
oof_blend = {K: np.zeros(len(train)) for K in K_VALS}   # 50% KNN + 50% Ridge
day_r2s_knn  = {K: {} for K in K_VALS}
day_r2s_ridge = {}

for day, grp in train.groupby('day_id'):
    n = len(grp)
    if n < 10:
        for K in K_VALS:
            oof_knn[K][grp.index] = y_train[grp.index].mean()
        oof_ridge[grp.index] = y_train[grp.index].mean()
        continue

    perm   = rng.permutation(n)
    n_tr   = int(n * 0.8)
    tr_idx = grp.index[perm[:n_tr]]
    va_idx = grp.index[perm[n_tr:]]

    X_tr_raw = grp.loc[tr_idx, feat_cols].fillna(0).values.astype(np.float64)
    y_tr     = y_train[tr_idx]
    X_va_raw = grp.loc[va_idx, feat_cols].fillna(0).values.astype(np.float64)
    y_va     = y_train[va_idx]

    X_tr_z, X_va_z, _, y_tr_clip = zscore_and_select_ics(
        X_tr_raw, y_tr, X_va_raw, top_k=50)

    # Ridge baseline
    p_ridge = predict_ridge(X_va_z, X_tr_z, y_tr_clip)
    oof_ridge[va_idx] = p_ridge
    if len(y_va) > 5:
        day_r2s_ridge[day] = r2_score(y_va, p_ridge)

    # KNN variants
    for K in K_VALS:
        p_knn = predict_knn(X_va_z, X_tr_z, y_tr_clip, K=K)
        oof_knn[K][va_idx] = p_knn
        if len(y_va) > 5:
            day_r2s_knn[K][day] = r2_score(y_va, p_knn)
        oof_blend[K][va_idx] = 0.5 * p_knn + 0.5 * p_ridge

print(f"\n  {'Method':<20} {'OOF R²':>12} {'Med/day R²':>12} {'Pred std':>10}")
print(f"  {'-'*56}")

oof_r2_ridge = r2_score(y_train, oof_ridge)
print(f"  {'Ridge(top50)':20} {oof_r2_ridge:+12.6f} "
      f"{np.median(list(day_r2s_ridge.values())):+12.6f} "
      f"{np.std(oof_ridge):10.6f}")

for K in K_VALS:
    oof_r2_knn = r2_score(y_train, oof_knn[K])
    oof_r2_blend = r2_score(y_train, oof_blend[K])
    med_r2 = np.median(list(day_r2s_knn[K].values()))
    print(f"  {'KNN K='+str(K)+'(top50)':20} {oof_r2_knn:+12.6f} "
          f"{med_r2:+12.6f} "
          f"{np.std(oof_knn[K]):10.6f}")
    print(f"  {'Blend K='+str(K)+' (50/50)':20} {oof_r2_blend:+12.6f} "
          f"{'—':>12} "
          f"{np.std(oof_blend[K]):10.6f}")

print(f"\n  Reference — transductive_v4_005: LB=+0.00003  (scaled to std≈0.0004)")
print(f"  Reference — fold_safe_v1:        LB=+0.00005  std=0.000624")

# Choose best K based on OOF
best_K = max(K_VALS, key=lambda K: r2_score(y_train, oof_knn[K]))
print(f"\n  Best K by OOF R²: K={best_K}")


# ── BUILD TEST PREDICTIONS ────────────────────────────────────────
print("\nBuilding test predictions...")
test_preds_knn   = {K: np.zeros(len(test)) for K in K_VALS}
test_preds_ridge = np.zeros(len(test))
test_ids         = test['ID'].values

for day, grp_te in test.groupby('day_id'):
    te_idx   = grp_te.index
    X_te_raw = grp_te[feat_cols].fillna(0).values.astype(np.float64)

    if day in train_days:
        grp_tr   = train[train['day_id'] == day]
        X_tr_raw = grp_tr[feat_cols].fillna(0).values.astype(np.float64)
        y_tr     = y_train[grp_tr.index]

        X_tr_z, X_te_z, _, y_tr_clip = zscore_and_select_ics(
            X_tr_raw, y_tr, X_te_raw, top_k=50)

        # Ridge
        test_preds_ridge[te_idx] = predict_ridge(X_te_z, X_tr_z, y_tr_clip)

        # KNN variants
        for K in K_VALS:
            test_preds_knn[K][te_idx] = predict_knn(X_te_z, X_tr_z, y_tr_clip, K=K)

    else:
        # Global fallback
        X_te_z = np.clip((X_te_raw - mean_g) / std_g, -5.0, 5.0)
        fb = global_model.predict(X_te_z[:, global_top50_idx])
        test_preds_ridge[te_idx] = fb
        for K in K_VALS:
            test_preds_knn[K][te_idx] = fb

# Clip to ±3σ of train TARGET
clip_bound = 3.0 * y_train.std()
test_preds_ridge = np.clip(test_preds_ridge, -clip_bound, clip_bound)
for K in K_VALS:
    test_preds_knn[K] = np.clip(test_preds_knn[K], -clip_bound, clip_bound)

print(f"  Ridge test std: {test_preds_ridge.std():.6f}")
for K in K_VALS:
    print(f"  KNN K={K} test std: {test_preds_knn[K].std():.6f}")


# ── SAVE SUBMISSIONS ─────────────────────────────────────────────
sample_sub = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/sample_submission.csv'))[['ID']]

def save_sub(preds, name):
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': preds})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    path = os.path.join(OUT_DIR, f'{name}.csv')
    sub.to_csv(path, index=False)
    t = sub['TARGET']
    print(f"  {name}: std={t.std():.7f}  mean={t.mean():+.8f}  "
          f"pct_pos={(t>0).mean()*100:.1f}%  skew={t.skew():+.3f}")

print("\nSaving submission variants...")

# Scaling analysis:
# Raw KNN std ≈ 0.008-0.015 (averaging neighbour TARGETs)
# fold_safe_v1 LB-best std ≈ 0.000624
# transductive_v4_005 LB=+0.00003 std ≈ 0.0004
# Target range for positive LB: std ≈ 0.0003 – 0.0007

for alpha, label in [(1.0, 'raw'), (0.1, '10pct'), (0.05, '5pct'),
                     (0.03, '3pct'), (0.01, '1pct')]:
    save_sub(test_preds_knn[best_K] * alpha,
             f'knn_K{best_K}_{label}')

# Blended (best_K): 50% KNN + 50% Ridge, then scale
blend_raw = 0.5 * test_preds_knn[best_K] + 0.5 * test_preds_ridge
for alpha, label in [(0.1, '10pct'), (0.05, '5pct'), (0.03, '3pct')]:
    save_sub(blend_raw * alpha, f'knn_ridge_blend_K{best_K}_{label}')

# Also save all K=5 variants explicitly (most standard choice)
for alpha, label in [(0.1, '10pct'), (0.05, '5pct'), (0.03, '3pct')]:
    save_sub(test_preds_knn[5] * alpha, f'knn_K5_{label}')

print(f"\nElapsed: {(time.time()-t0)/60:.1f} min")

# ── SUBMISSION GUIDE ─────────────────────────────────────────────
print(f"""
SUBMISSION STRATEGY (2 submissions remaining):
─────────────────────────────────────────────
Current best LB: fold_safe_v1 = +0.00005  (std=0.000624)

KNN key insight:
  KNN directly uses ACTUAL TARGET values of nearest labeled peers
  on the SAME day — this is the most direct exploitation of the
  83.6% overlap finding.

SUBMISSION 1 (best expected): knn_K{best_K}_5pct or knn_K5_5pct
  → Pure KNN with 5% scaling (std ≈ 0.0004-0.0006)
  → If KNN signal > Ridge signal → beats +0.00005

SUBMISSION 2 (fallback): knn_ridge_blend_K{best_K}_5pct
  → 50% KNN + 50% Ridge: averages two signal sources
  → More conservative, slightly smoother predictions

IMPORTANT: Look at the std values above:
  • Target std range: 0.0003 – 0.0007 (near fold_safe std of 0.000624)
  • If raw KNN std >> 0.010 → use 3pct or 5pct scaling
  • If raw KNN std ≈ 0.008 → use 5pct scaling → final std ≈ 0.0004

Do NOT use raw (1.0) — raw predictions are too large.
Do NOT use 1pct — std ≈ 0.00008, too small (signal gets killed by noise).
""")
