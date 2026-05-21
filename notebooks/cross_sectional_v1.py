# ================================================================
# CROSS-SECTIONAL NORMALIZED MODEL
# ================================================================
# Core insight: OOF and LB are INVERSELY correlated. Every model
# that learns more temporal signal gets worse LB. The test period
# has different temporal dynamics than training.
#
# Solution: remove all temporal information from features by
# z-scoring each feature within its trading day (time_ID group).
# After this transform:
#   - Features encode only "is this asset above/below average TODAY"
#   - The model learns purely cross-sectional relationships
#   - These generalise to test because the RANK structure of assets
#     is more stable than the absolute level of features over time
#
# Architecture:
#   - time_ID = SO3_T.round(5).astype(str)  [428 trading days]
#   - Cross-sectional z-score each feature within time_ID
#   - GroupKFold(5) on SO3_T quintiles — same as fold_safe_v1 baseline
#   - LightGBM, all 445 features, winsorize TARGET only
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
t0 = time.time()

N_FOLDS    = 5
NUM_LEAVES = 63
N_EST      = 2000
LR         = 0.05
ES_ROUNDS  = 50

# ── Load ─────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET'}]
print(f"  Train: {len(train):,}  Test: {len(test):,}  Features: {len(feat_cols)}")

# ── Reconstruct trading day IDs ───────────────────────────────────
train_time_ids = train['SO3_T'].round(5).astype(str).values
test_time_ids  = test['SO3_T'].round(5).astype(str).values
print(f"  Unique trading days — train: {len(np.unique(train_time_ids))}  "
      f"test: {len(np.unique(test_time_ids))}")

y_train  = train['TARGET'].values.astype(np.float32)
test_ids = test['ID'].values

# ── Cross-sectional z-score normalization ─────────────────────────
# For each feature and each trading day: x_norm = (x - daily_mean) / daily_std
# This strips temporal drift, leaving only cross-sectional variation
print("\nApplying cross-sectional z-score normalization...")
t1 = time.time()

train_feat = train[feat_cols].fillna(0).values.astype(np.float32)
test_feat  = test.reindex(columns=feat_cols, fill_value=0).values.astype(np.float32)

# Train normalization
train_norm = np.zeros_like(train_feat)
for tid in np.unique(train_time_ids):
    mask = train_time_ids == tid
    x    = train_feat[mask]
    m    = x.mean(axis=0)
    s    = x.std(axis=0)
    s    = np.where(s < 1e-8, 1.0, s)   # avoid divide-by-zero for constant features
    train_norm[mask] = (x - m) / s

# Test normalization
test_norm = np.zeros_like(test_feat)
for tid in np.unique(test_time_ids):
    mask = test_time_ids == tid
    x    = test_feat[mask]
    m    = x.mean(axis=0)
    s    = x.std(axis=0)
    s    = np.where(s < 1e-8, 1.0, s)
    test_norm[mask] = (x - m) / s

print(f"  Done in {time.time()-t1:.1f}s")
print(f"  Sample stats after norm — mean: {train_norm.mean():.4f}  std: {train_norm.std():.4f}")

del train_feat, test_feat, train, test
gc.collect()

X_train = train_norm.astype(np.float32)
X_test  = test_norm.astype(np.float32)
del train_norm, test_norm
gc.collect()

# ── GroupKFold on SO3_T quintiles (same as fold_safe_v1) ─────────
# We still use SO3_T quintile grouping (NOT time_ID) — this is what
# accidentally works. It forces cross-sectional generalisation.
so3t_idx  = feat_cols.index('SO3_T')
so3t_vals = X_train[:, so3t_idx]
groups    = pd.qcut(pd.Series(so3t_vals), q=N_FOLDS,
                    labels=False, duplicates='drop').values.astype(np.int32)
n_folds   = len(np.unique(groups))
gkf       = GroupKFold(n_splits=n_folds)
folds     = list(gkf.split(X_train, y_train, groups=groups))

print(f"\nGroupKFold: {n_folds} folds on SO3_T quintiles")
for i, (tr, va) in enumerate(folds):
    print(f"  Fold {i+1}: train={len(tr):,}  val={len(va):,}")

# ── LightGBM params (identical to fold_safe_v1) ───────────────────
lgb_params = dict(
    objective         = 'regression',
    metric            = 'rmse',
    num_leaves        = NUM_LEAVES,
    learning_rate     = LR,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 1,
    min_child_samples = 50,
    lambda_l1         = 0.1,
    lambda_l2         = 1.0,
    n_jobs            = -1,
    verbose           = -1,
    seed              = 42,
)

# ── Training loop ─────────────────────────────────────────────────
oof_preds  = np.zeros(len(y_train), dtype=np.float64)
test_preds = np.zeros(len(X_test),  dtype=np.float64)
fold_r2s   = []
best_iters = []

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    tf = time.time()
    print(f"\n{'─'*55}")
    print(f"FOLD {fold_idx+1}/{n_folds}")

    X_tr = X_train[tr_idx].copy()
    y_tr = y_train[tr_idx].astype(np.float64)
    y_va = y_train[va_idx].astype(np.float64)

    # Winsorise TARGET only (features are already z-scored cross-sectionally)
    lo, hi = np.percentile(y_tr, 1), np.percentile(y_tr, 99)
    y_tr   = np.clip(y_tr, lo, hi)

    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
    dval   = lgb.Dataset(X_train[va_idx], label=y_va, reference=dtrain, free_raw_data=True)

    del X_tr
    gc.collect()

    model = lgb.train(
        lgb_params, dtrain,
        num_boost_round = N_EST,
        valid_sets      = [dval],
        callbacks       = [lgb.early_stopping(ES_ROUNDS, verbose=False),
                           lgb.log_evaluation(200)],
    )

    best_iter = model.best_iteration
    best_iters.append(best_iter)

    oof_preds[va_idx] = model.predict(X_train[va_idx], num_iteration=best_iter)
    test_preds       += model.predict(X_test, num_iteration=best_iter) / n_folds

    fold_r2 = r2_score(y_va, oof_preds[va_idx])
    fold_r2s.append(fold_r2)
    print(f"  best_iter={best_iter}  fold_R²={fold_r2:+.6f}  ({time.time()-tf:.0f}s)")

    del dtrain, dval, model
    gc.collect()

# ── Results ───────────────────────────────────────────────────────
oof_r2   = r2_score(y_train, oof_preds)
pred_std = oof_preds.std()
pct_pos  = (test_preds > 0).mean()

print(f"\n{'='*55}")
print(f"cross_sectional_v1 RESULTS")
print(f"{'='*55}")
print(f"  OOF R²        : {oof_r2:+.6f}")
print(f"  Per-fold R²   : {[f'{r:+.6f}' for r in fold_r2s]}")
print(f"  Best iters    : {best_iters}")
print(f"  OOF pred std  : {pred_std:.6f}")
print(f"  Test pred std : {test_preds.std():.6f}")
print(f"  Test pct_pos  : {pct_pos*100:.1f}%")
print(f"  Test skew     : {pd.Series(test_preds).skew():+.3f}")
print(f"\n  Reference: fold_safe_v1  OOF=+0.000544  LB=+0.00005  std=0.000624")
print(f"  OOF Delta vs v1: {oof_r2 - 0.000544:+.6f}")
print(f"\n  *** OOF higher ≠ LB higher. Target is cross-sectional correlation. ***")
print(f"  *** If test pred std ≈ 0.0003-0.0006 and skew near 0 → good candidate ***")

# ── Save ──────────────────────────────────────────────────────────
sample_sub = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/sample_submission.csv'))[['ID']]
sub = pd.DataFrame({'ID': test_ids, 'TARGET': test_preds})
sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
out_path = os.path.join(OUT_DIR, 'cross_sectional_v1.csv')
sub.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")
print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")
