"""
04_fold_safe_cross_sectional.py — Memory-Safe In-Fold Cross-Sectional Pipeline
================================================================================
Cross-sectional alpha (Z-scores) computed INSIDE the CV loop to prevent leakage,
with in-place array mutation to stay within 8 GB RAM.

Anti-leakage guarantees
------------------------
1. StandardScaler fitted ONLY on X_tr (training fold). Never on val or test.
2. X_test scaled in-place per fold → predictions collected → inverse_transform
   restores X_test to raw state before next fold. No fold-to-fold contamination.
3. Winsorization bounds computed from y_tr percentiles only. Applied to y_va
   using those same bounds (never re-computed on val).
4. GroupKFold on SO3_T quantile buckets. No TARGET information in the groups.

Memory strategy
---------------
- All feature arrays kept as float32 (halves memory vs float64).
- StandardScaler(copy=False) avoids creating a second copy where possible.
- X_tr[:] = ... and X_va[:] = ... write results back in-place; temporaries
  are freed immediately by Python's reference counting.
- X_test is mutated per fold and restored immediately after predictions.
- del + gc.collect() at end of every fold releases LGB Dataset/model memory.

Peak RAM estimate
------------------
  X_train (float32): 661k × 445 × 4B ≈ 1.18 GB (held full-time)
  X_test  (float32): 410k × 445 × 4B ≈ 0.73 GB (held full-time)
  X_tr slice (float32): ~0.94 GB  (freed end of fold)
  X_va slice (float32): ~0.24 GB  (freed end of fold)
  Transform temporary (float64): ~0.94 GB peak, freed immediately
  LightGBM Dataset + model:      ~0.50 GB
  ─────────────────────────────────────────
  Total peak: ~4.5 GB  (comfortably within 8 GB)
"""

import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = '/Users/malaymishra/Desktop/quant_ml_project'
os.chdir(BASE_DIR)

# Handle both possible raw train filenames
_train_path_a = os.path.join(BASE_DIR, 'data/raw/train.parquet')
_train_path_b = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TRAIN_RAW  = _train_path_a if os.path.exists(_train_path_a) else _train_path_b
TEST_RAW   = os.path.join(BASE_DIR, 'data/raw/test.parquet')
SAMPLE_SUB = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')

os.makedirs('outputs/oof_predictions', exist_ok=True)
os.makedirs('outputs/submissions',     exist_ok=True)
os.makedirs('models/checkpoints',      exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
N_FOLDS      = 5
N_ESTIMATORS = 3000
EARLY_STOP   = 150
FAIR_C       = 1.0
np.random.seed(RANDOM_SEED)

# ── LightGBM params ─────────────────────────────────────────────────────────
LGB_PARAMS = {
    'num_leaves'       : 63,
    'learning_rate'    : 0.02,
    'feature_fraction' : 0.4,
    'bagging_fraction' : 0.7,
    'bagging_freq'     : 1,
    'min_child_samples': 250,
    'reg_alpha'        : 0.5,
    'reg_lambda'       : 10.0,
    'random_state'     : RANDOM_SEED,
    'n_jobs'           : -1,
    'verbose'          : -1,
}

# ── Fair loss objective ─────────────────────────────────────────────────────
# L(r) = c² * (|r|/c - log(1 + |r|/c))
# More robust to fat-tailed financial returns than MSE.
# Gradient and Hessian are derived analytically.
def fair_obj_lgb(y_pred, dataset):
    y_true = dataset.get_label()
    r      = y_pred - y_true
    grad   = r / (1.0 + np.abs(r) / FAIR_C)
    hess   = FAIR_C ** 2 / (FAIR_C + np.abs(r)) ** 2
    return grad, hess

# ── R² evaluation metric ────────────────────────────────────────────────────
def r2_eval_lgb(y_pred, dataset):
    y_true  = dataset.get_label()
    ss_res  = np.sum((y_true - y_pred) ** 2)
    ss_tot  = np.sum((y_true - y_true.mean()) ** 2)
    r2      = 1.0 - ss_res / (ss_tot + 1e-15)
    return 'r2', r2, True   # True = higher is better

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & downcast
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1 — Loading raw data")
print("=" * 70)
print(f"  Train: {TRAIN_RAW}")
t0 = time.time()

train_raw = pd.read_parquet(TRAIN_RAW)
test_raw  = pd.read_parquet(TEST_RAW)
print(f"  Train shape: {train_raw.shape}")
print(f"  Test  shape: {test_raw.shape}")

NON_FEAT  = {'ID', 'TARGET'}
feat_cols = [c for c in train_raw.columns if c not in NON_FEAT]
test_feat_cols = [c for c in feat_cols if c in test_raw.columns]

# Downcast all feature columns to float32
# This halves the memory footprint vs the default float64
print("  Downcasting features to float32 ...")
for col in feat_cols:
    train_raw[col] = train_raw[col].astype(np.float32)
for col in test_feat_cols:
    test_raw[col] = test_raw[col].astype(np.float32)

y_train  = train_raw['TARGET'].values.astype(np.float32)
test_ids = test_raw['ID'].values

print(f"  TARGET — mean:{y_train.mean():.6f}  std:{y_train.std():.6f}  "
      f"min:{y_train.min():.4f}  max:{y_train.max():.4f}")
print(f"  Load + downcast elapsed: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — NaN imputation (train medians only, applied to both sets)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2 — NaN imputation")
print("=" * 70)

train_nans = train_raw[feat_cols].isna().sum().sum()
test_nans  = test_raw[test_feat_cols].isna().sum().sum()
print(f"  Train NaNs: {train_nans:,}    Test NaNs: {test_nans:,}")

if train_nans > 0 or test_nans > 0:
    # Medians computed strictly on train set
    medians = train_raw[feat_cols].median()
    train_raw[feat_cols] = train_raw[feat_cols].fillna(medians)
    test_raw[test_feat_cols] = test_raw[test_feat_cols].fillna(medians[test_feat_cols])
    print("  Imputation applied using train medians.")
else:
    print("  No NaNs found — skipping imputation.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Build feature matrices (keep as float32 C-contiguous arrays)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3 — Building feature matrices")
print("=" * 70)

# np.ascontiguousarray ensures the array is C-contiguous.
# This is required for StandardScaler(copy=False) to operate in-place.
X_train = np.ascontiguousarray(train_raw[feat_cols].values, dtype=np.float32)
X_test  = np.ascontiguousarray(test_raw[test_feat_cols].values,  dtype=np.float32)

print(f"  X_train: {X_train.shape}  ({X_train.nbytes/1e6:.0f} MB, float32)")
print(f"  X_test : {X_test.shape}  ({X_test.nbytes/1e6:.0f} MB, float32)")

del train_raw, test_raw
gc.collect()
print("  Raw DataFrames freed.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — GroupKFold on SO3_T quantile buckets
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4 — GroupKFold on SO3_T quantile buckets")
print("=" * 70)

so3t_idx = feat_cols.index('SO3_T')
so3t_vals = X_train[:, so3t_idx]

groups = pd.qcut(
    pd.Series(so3t_vals),
    q=N_FOLDS,
    labels=False,
    duplicates='drop'
).values.astype(np.int32)

n_actual_groups = len(np.unique(groups))
if n_actual_groups < N_FOLDS:
    print(f"  WARNING: SO3_T qcut yielded {n_actual_groups} groups "
          f"(boundary ties). Using {n_actual_groups} folds.")
    actual_folds = n_actual_groups
else:
    actual_folds = N_FOLDS

print(f"  Groups:")
for g, cnt in zip(*np.unique(groups, return_counts=True)):
    mask = groups == g
    print(f"    Group {g}: {cnt:,} rows  "
          f"SO3_T ∈ [{so3t_vals[mask].min():.6f}, {so3t_vals[mask].max():.6f}]")

gkf   = GroupKFold(n_splits=actual_folds)
folds = list(gkf.split(X_train, y_train, groups=groups))

print(f"\n  Fold summary:")
for i, (tr, va) in enumerate(folds):
    print(f"    Fold {i+1}: train={len(tr):,}  val={len(va):,}  "
          f"val_group={sorted(np.unique(groups[va]).tolist())}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Memory-safe in-fold CV loop
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5 — In-fold cross-sectional scaling + training")
print("=" * 70)
print("  Memory strategy: in-place mutation via [:] assignment.")
print("  X_test is scaled → predicted → inverse_transformed each fold.\n")

oof        = np.zeros(len(y_train), dtype=np.float32)
test_preds = np.zeros(len(X_test),  dtype=np.float64)
fold_r2s   = []
total_start = time.time()

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    fold_start = time.time()
    print(f"{'─'*60}")
    print(f"  FOLD {fold_idx+1}/{actual_folds}  "
          f"(train={len(tr_idx):,}  val={len(va_idx):,})")
    print(f"{'─'*60}")

    # ── Slice — creates independent copies (fancy indexing) ─────────────
    # X_tr and X_va are contiguous float32 arrays, independent of X_train.
    # Mutations to X_tr/X_va do NOT affect X_train.
    X_tr = np.ascontiguousarray(X_train[tr_idx], dtype=np.float32)
    X_va = np.ascontiguousarray(X_train[va_idx], dtype=np.float32)
    y_tr = y_train[tr_idx].copy()
    y_va = y_train[va_idx].copy()

    # ── Winsorization (fold-safe) ────────────────────────────────────────
    # Clip bounds derived ONLY from y_tr. Applied to y_va using those same
    # bounds. y_va's own distribution never informs the clip limits.
    clip_low  = np.percentile(y_tr, 1)
    clip_high = np.percentile(y_tr, 99)
    y_tr = np.clip(y_tr, clip_low, clip_high).astype(np.float32)
    y_va = np.clip(y_va, clip_low, clip_high).astype(np.float32)
    print(f"  Winsorization: [{clip_low:.5f}, {clip_high:.5f}]  "
          f"(y_tr 1st–99th pct)")
    print(f"  y_tr after clip — mean:{y_tr.mean():.6f}  std:{y_tr.std():.6f}")
    print(f"  y_va after clip — mean:{y_va.mean():.6f}  std:{y_va.std():.6f}")

    # ── In-place feature scaling (fold-safe) ────────────────────────────
    # StandardScaler(copy=False): avoids internal copy where dtype allows.
    # The [:] assignment pattern guarantees in-place write regardless of
    # whether sklearn internally copies (dtype mismatch = float32 → float64).
    # The temporary float64 array from .transform() is freed immediately
    # after the assignment by Python's reference counting.
    scaler = StandardScaler(copy=False)
    scaler.fit(X_tr)                             # fit on TRAIN FOLD ONLY

    X_tr[:]    = scaler.transform(X_tr)          # in-place: train fold
    X_va[:]    = scaler.transform(X_va)          # in-place: val fold (train stats)
    X_test[:]  = scaler.transform(X_test)        # in-place: test (train stats)
    # ↑ X_test is now in the SCALED space of this fold's training distribution.
    #   It will be inverse-transformed back after predictions.

    print(f"  Scaling applied. X_tr mean≈{X_tr.mean():.4f} std≈{X_tr.std():.4f}")

    # ── LightGBM training ────────────────────────────────────────────────
    dtrain = lgb.Dataset(X_tr, label=y_tr,
                         feature_name=feat_cols, free_raw_data=True)
    dvalid = lgb.Dataset(X_va, label=y_va,
                         reference=dtrain, free_raw_data=True)

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
        lgb.log_evaluation(period=300),
    ]

    model = lgb.train(
        {**LGB_PARAMS, 'objective': fair_obj_lgb},
        dtrain,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[dvalid],
        feval=r2_eval_lgb,
        callbacks=callbacks,
    )

    # ── Predictions ─────────────────────────────────────────────────────
    # Both X_va and X_test are currently in this fold's scaled space.
    oof[va_idx]  = model.predict(X_va).astype(np.float32)
    test_preds  += model.predict(X_test) / actual_folds

    fold_r2 = r2_score(y_va, oof[va_idx])
    fold_r2s.append(fold_r2)

    # ── CRITICAL: Restore X_test to raw state ───────────────────────────
    # Inverse-transform reverts the in-place scaling using this fold's
    # scaler statistics. X_test returns to float32 raw feature space,
    # ready to be scaled by the NEXT fold's scaler.
    X_test[:] = scaler.inverse_transform(X_test)

    elapsed = time.time() - fold_start
    print(f"\n  ── Fold {fold_idx+1} results ──")
    print(f"     Best iteration : {model.best_iteration}")
    print(f"     Fold R²        : {fold_r2:.6f}")
    print(f"     OOF pred mean  : {oof[va_idx].mean():.6f}  "
          f"std: {oof[va_idx].std():.6f}")
    print(f"     Elapsed        : {elapsed:.1f}s")

    model.save_model(f'models/checkpoints/fold_safe_lgbm_fold{fold_idx+1}.txt')

    # ── Free fold-level memory ───────────────────────────────────────────
    del X_tr, X_va, y_tr, y_va, dtrain, dvalid, model, scaler
    gc.collect()
    print(f"     Fold memory freed.")

total_elapsed = time.time() - total_start

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — Final OOF evaluation & submission
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6 — Final evaluation")
print("=" * 70)

oof_r2 = r2_score(y_train, oof)

print(f"\n  Per-fold R² scores:")
for i, r2 in enumerate(fold_r2s):
    bar = '█' * max(0, int(r2 * 10000))
    print(f"    Fold {i+1}: {r2:+.6f}  {bar}")

print(f"\n  OOF R² (combined) : {oof_r2:+.6f}")
print(f"  OOF std dev       : {oof.std():.6f}")
print(f"  OOF range         : [{oof.min():.6f}, {oof.max():.6f}]")
print(f"  Total time        : {total_elapsed/60:.1f} min")

# Save OOF and test predictions
np.save('outputs/oof_predictions/fold_safe_lgbm_oof.npy',        oof)
np.save('outputs/oof_predictions/fold_safe_lgbm_test_preds.npy', test_preds)
print("\n  SAVED: outputs/oof_predictions/fold_safe_lgbm_oof.npy")
print("  SAVED: outputs/oof_predictions/fold_safe_lgbm_test_preds.npy")

# Submission CSV aligned to sample_submission ordering
sample_sub = pd.read_csv(SAMPLE_SUB)
sub = pd.DataFrame({'ID': test_ids, 'TARGET': test_preds})
sub = sample_sub[['ID']].merge(sub, on='ID', how='left')
null_count = sub['TARGET'].isnull().sum()
if null_count:
    print(f"  WARNING: {null_count} IDs missing from predictions — filling 0.0")
    sub['TARGET'] = sub['TARGET'].fillna(0.0)

out_path = 'outputs/submissions/fold_safe_v1.csv'
sub.to_csv(out_path, index=False)
print(f"\n  SAVED: {out_path}  ({len(sub):,} rows)")
print(f"  Test pred stats — "
      f"mean:{test_preds.mean():.6f}  std:{test_preds.std():.6f}  "
      f"min:{test_preds.min():.6f}  max:{test_preds.max():.6f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Pipeline        : Fold-safe Z-score cross-sectional scaling")
print(f"  Features        : {X_train.shape[1]} raw (445 float32)")
print(f"  CV strategy     : {actual_folds}-fold GroupKFold on SO3_T quantiles")
print(f"  Objective       : Fair loss (c={FAIR_C})")
print(f"  OOF R²          : {oof_r2:+.6f}")
print(f"  Expected LB R²  : close to OOF (no leakage by design)")
print(f"  Submission      : {out_path}")
print("=" * 70)
