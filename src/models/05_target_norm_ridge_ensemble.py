"""
05_target_norm_ridge_ensemble.py
=================================
Upgraded fold_safe pipeline with two additions on top of fold_safe_v1:

1. TARGET Z-SCORE STANDARDISATION (the "Fold 4 fix")
   - Inside each fold, y_tr is Winsorized then Z-scored using its own mean/std.
   - The exact same (mean, std) from y_tr is applied to y_va and used to
     inverse-transform predictions back to original percentage scale.
   - Forces the model to learn pure directional patterns, immune to
     cross-regime volatility differences.

2. RIDGE REGRESSION (the "diffuse signal aggregator")
   - Feature importance showed signal spread nearly uniformly across all 445
     features (top-15 captured only 11.7% of gain). Ridge's L2 penalty
     distributes weight smoothly across all features simultaneously —
     exactly the right inductive bias for a flat, multi-collinear signal.
   - Trained on the same Z-scored X_tr and y_tr_norm as LightGBM.
   - Adds essentially zero RAM (Ridge just solves X^T X, no tree structures).

3. SCIPY-OPTIMISED ENSEMBLE
   - scipy.optimize.minimize_scalar finds the LGB weight w* that maximises
     OOF R² over the full training set in original scale.
     Ensemble = w* * lgb_oof + (1 - w*) * ridge_oof

Anti-leakage guarantees (unchanged from fold_safe_v1)
------------------------------------------------------
1. StandardScaler fitted ONLY on X_tr.  Never on val or test.
2. X_test scaled in-place per fold → predictions → inverse_transform.
3. Winsorization bounds and target (mean, std) from y_tr ONLY.
4. GroupKFold on SO3_T quantile buckets.

Memory strategy (unchanged from fold_safe_v1)
---------------------------------------------
- float32 throughout; StandardScaler(copy=False).
- X_tr[:] / X_va[:] / X_test[:] in-place writes.
- del + gc.collect() at end of every fold.
- Ridge uses Cholesky on the 445×445 gram matrix — no large float64 copy.

Peak RAM estimate
-----------------
  X_train (float32) : 661k × 445 × 4B  ≈ 1.18 GB
  X_test  (float32) : 410k × 445 × 4B  ≈ 0.73 GB
  X_tr slice        :                   ≈ 0.94 GB
  X_va slice        :                   ≈ 0.24 GB
  LGB Dataset+model :                   ≈ 0.50 GB
  Ridge gram matrix :                   ≈ 0.002 GB (trivial)
  ──────────────────────────────────────────────────
  Total peak        :                   ≈ 3.6 GB  (well within 8 GB)
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = '/Users/malaymishra/Desktop/quant_ml_project'
os.chdir(BASE_DIR)

_train_path_a = os.path.join(BASE_DIR, 'data/raw/train.parquet')
_train_path_b = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TRAIN_RAW  = _train_path_a if os.path.exists(_train_path_a) else _train_path_b
TEST_RAW   = os.path.join(BASE_DIR, 'data/raw/test.parquet')
SAMPLE_SUB = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')

os.makedirs('outputs/oof_predictions', exist_ok=True)
os.makedirs('outputs/submissions',     exist_ok=True)
os.makedirs('models/checkpoints',      exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
N_FOLDS      = 5
N_ESTIMATORS = 3000
EARLY_STOP   = 150
FAIR_C       = 1.0
RIDGE_ALPHA  = 10_000.0   # strong regularisation for diffuse low-SNR signal
np.random.seed(RANDOM_SEED)

# ── LightGBM params ─────────────────────────────────────────────────────────
LGB_PARAMS = {
    'num_leaves'        : 63,
    'learning_rate'     : 0.02,
    'feature_fraction'  : 0.4,
    'bagging_fraction'  : 0.7,
    'bagging_freq'      : 1,
    'min_child_samples' : 250,
    'reg_alpha'         : 0.5,
    'reg_lambda'        : 10.0,
    'random_state'      : RANDOM_SEED,
    'n_jobs'            : -1,
    'verbose'           : -1,
}

# ── Fair loss (robust to fat financial tails) ───────────────────────────────
def fair_obj_lgb(y_pred, dataset):
    y_true = dataset.get_label()
    r      = y_pred - y_true
    grad   = r / (1.0 + np.abs(r) / FAIR_C)
    hess   = FAIR_C ** 2 / (FAIR_C + np.abs(r)) ** 2
    return grad, hess

# ── R² evaluation metric ────────────────────────────────────────────────────
def r2_eval_lgb(y_pred, dataset):
    y_true = dataset.get_label()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2     = 1.0 - ss_res / (ss_tot + 1e-15)
    return 'r2', r2, True   # True = higher is better


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & downcast
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1 — Loading raw data")
print("=" * 70)
t0 = time.time()

train_raw = pd.read_parquet(TRAIN_RAW)
test_raw  = pd.read_parquet(TEST_RAW)
print(f"  Train shape : {train_raw.shape}")
print(f"  Test  shape : {test_raw.shape}")

NON_FEAT       = {'ID', 'TARGET'}
feat_cols      = [c for c in train_raw.columns if c not in NON_FEAT]
test_feat_cols = [c for c in feat_cols if c in test_raw.columns]

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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — NaN imputation (train medians, applied to both)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2 — NaN imputation")
print("=" * 70)

train_nans = train_raw[feat_cols].isna().sum().sum()
test_nans  = test_raw[test_feat_cols].isna().sum().sum()
print(f"  Train NaNs: {train_nans:,}    Test NaNs: {test_nans:,}")

if train_nans > 0 or test_nans > 0:
    medians = train_raw[feat_cols].median()
    train_raw[feat_cols]       = train_raw[feat_cols].fillna(medians)
    test_raw[test_feat_cols]   = test_raw[test_feat_cols].fillna(medians[test_feat_cols])
    print("  Imputation applied using train medians.")
else:
    print("  No NaNs found — skipping imputation.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Build feature matrices
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3 — Building feature matrices")
print("=" * 70)

X_train = np.ascontiguousarray(train_raw[feat_cols].values,      dtype=np.float32)
X_test  = np.ascontiguousarray(test_raw[test_feat_cols].values,  dtype=np.float32)

print(f"  X_train : {X_train.shape}  ({X_train.nbytes/1e6:.0f} MB, float32)")
print(f"  X_test  : {X_test.shape}  ({X_test.nbytes/1e6:.0f} MB, float32)")

del train_raw, test_raw
gc.collect()
print("  Raw DataFrames freed.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — GroupKFold on SO3_T quantile buckets
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4 — GroupKFold on SO3_T quantile buckets")
print("=" * 70)

so3t_idx  = feat_cols.index('SO3_T')
so3t_vals = X_train[:, so3t_idx]

groups = pd.qcut(
    pd.Series(so3t_vals), q=N_FOLDS, labels=False, duplicates='drop'
).values.astype(np.int32)

n_actual_groups = len(np.unique(groups))
actual_folds    = n_actual_groups if n_actual_groups < N_FOLDS else N_FOLDS

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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Memory-safe in-fold CV loop
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5 — In-fold training: Target Norm + LGB + Ridge")
print("=" * 70)

# Separate OOF arrays for LGB and Ridge (original scale)
lgb_oof        = np.zeros(len(y_train), dtype=np.float32)
ridge_oof      = np.zeros(len(y_train), dtype=np.float32)

# Test prediction accumulators (original scale, averaged across folds)
lgb_test_preds   = np.zeros(len(X_test), dtype=np.float64)
ridge_test_preds = np.zeros(len(X_test), dtype=np.float64)

lgb_fold_r2s   = []
ridge_fold_r2s = []
total_start    = time.time()

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    fold_start = time.time()
    print(f"\n{'─'*60}")
    print(f"  FOLD {fold_idx+1}/{actual_folds}  "
          f"(train={len(tr_idx):,}  val={len(va_idx):,})")
    print(f"{'─'*60}")

    # ── 1. Slice (fancy indexing → independent copies) ───────────────────
    X_tr = np.ascontiguousarray(X_train[tr_idx], dtype=np.float32)
    X_va = np.ascontiguousarray(X_train[va_idx], dtype=np.float32)
    y_tr = y_train[tr_idx].copy()
    y_va = y_train[va_idx].copy()

    # ── 2. Winsorize (bounds from y_tr only) ─────────────────────────────
    clip_low  = np.percentile(y_tr, 1)
    clip_high = np.percentile(y_tr, 99)
    y_tr      = np.clip(y_tr, clip_low, clip_high).astype(np.float32)
    y_va      = np.clip(y_va, clip_low, clip_high).astype(np.float32)
    print(f"  Winsorize: [{clip_low:.5f}, {clip_high:.5f}]")

    # ── 3. TARGET Z-SCORE STANDARDISATION (NEW) ──────────────────────────
    # Compute mean and std strictly from the Winsorized training target.
    # Same transform applied to y_va — no information leaks from val.
    # std clamped to 1e-8 to guard against degenerate folds (e.g. Fold 4).
    mean_y = float(y_tr.mean())
    std_y  = float(y_tr.std())
    std_y  = max(std_y, 1e-8)

    y_tr_norm = ((y_tr - mean_y) / std_y).astype(np.float32)
    y_va_norm = ((y_va - mean_y) / std_y).astype(np.float32)
    print(f"  Target norm — mean_y={mean_y:.6f}  std_y={std_y:.6f}")
    print(f"  y_tr_norm  — mean:{y_tr_norm.mean():.4f}  std:{y_tr_norm.std():.4f}")
    print(f"  y_va_norm  — mean:{y_va_norm.mean():.4f}  std:{y_va_norm.std():.4f}")

    # ── 4. Feature scaling (fold-safe, in-place) ──────────────────────────
    scaler = StandardScaler(copy=False)
    scaler.fit(X_tr)
    X_tr[:]   = scaler.transform(X_tr)
    X_va[:]   = scaler.transform(X_va)
    X_test[:] = scaler.transform(X_test)
    print(f"  Feature scaling done. X_tr mean≈{X_tr.mean():.4f} std≈{X_tr.std():.4f}")

    # ── 5. LightGBM — trained on normalised target ────────────────────────
    print(f"  Training LightGBM ...")
    dtrain = lgb.Dataset(X_tr, label=y_tr_norm,
                         feature_name=feat_cols, free_raw_data=True)
    dvalid = lgb.Dataset(X_va, label=y_va_norm,
                         reference=dtrain, free_raw_data=True)

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
        lgb.log_evaluation(period=300),
    ]

    lgb_model = lgb.train(
        {**LGB_PARAMS, 'objective': fair_obj_lgb},
        dtrain,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[dvalid],
        feval=r2_eval_lgb,
        callbacks=callbacks,
    )

    # Predictions in normalised space → inverse-transform to original scale
    lgb_oof_norm  = lgb_model.predict(X_va).astype(np.float32)
    lgb_test_norm = lgb_model.predict(X_test)

    lgb_oof[va_idx]    = (lgb_oof_norm  * std_y + mean_y).astype(np.float32)
    lgb_test_preds    += (lgb_test_norm * std_y + mean_y) / actual_folds

    lgb_fold_r2 = r2_score(y_va, lgb_oof[va_idx])
    lgb_fold_r2s.append(lgb_fold_r2)
    print(f"  LGB  best_iter={lgb_model.best_iteration:>4}  "
          f"fold_R²={lgb_fold_r2:+.6f}")

    del dtrain, dvalid, lgb_model, lgb_oof_norm, lgb_test_norm
    gc.collect()

    # ── 6. Ridge — trained on the same normalised target ─────────────────
    print(f"  Training Ridge (alpha={RIDGE_ALPHA:.0f}) ...")
    ridge_model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge_model.fit(X_tr, y_tr_norm)

    ridge_oof_norm  = ridge_model.predict(X_va).astype(np.float32)
    ridge_test_norm = ridge_model.predict(X_test)

    ridge_oof[va_idx]    = (ridge_oof_norm  * std_y + mean_y).astype(np.float32)
    ridge_test_preds    += (ridge_test_norm * std_y + mean_y) / actual_folds

    ridge_fold_r2 = r2_score(y_va, ridge_oof[va_idx])
    ridge_fold_r2s.append(ridge_fold_r2)
    print(f"  Ridge fold_R²={ridge_fold_r2:+.6f}")

    del ridge_model, ridge_oof_norm, ridge_test_norm
    gc.collect()

    # ── 7. Restore X_test to raw space ────────────────────────────────────
    X_test[:] = scaler.inverse_transform(X_test)

    elapsed = time.time() - fold_start
    print(f"\n  ── Fold {fold_idx+1} summary ──")
    print(f"     LGB   R² : {lgb_fold_r2:+.6f}")
    print(f"     Ridge R² : {ridge_fold_r2:+.6f}")
    print(f"     Elapsed  : {elapsed:.1f}s")

    del X_tr, X_va, y_tr, y_va, y_tr_norm, y_va_norm, scaler
    gc.collect()
    print(f"     Fold memory freed.")

total_elapsed = time.time() - total_start


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — OOF evaluation per model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6 — OOF evaluation")
print("=" * 70)

lgb_oof_r2   = r2_score(y_train, lgb_oof)
ridge_oof_r2 = r2_score(y_train, ridge_oof)

print(f"\n  Per-fold R² (LGB vs Ridge):")
for i, (lr, rr) in enumerate(zip(lgb_fold_r2s, ridge_fold_r2s), 1):
    winner = "LGB  >" if lr > rr else "Ridge>"
    print(f"    Fold {i}:  LGB={lr:+.6f}   Ridge={rr:+.6f}   {winner}")

print(f"\n  LGB   OOF R² : {lgb_oof_r2:+.6f}")
print(f"  Ridge OOF R² : {ridge_oof_r2:+.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Scipy-optimised ensemble blend
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7 — Optimising ensemble blend weight")
print("=" * 70)

def neg_r2_blend(w):
    """w = LGB weight; (1-w) = Ridge weight."""
    blend = w * lgb_oof + (1.0 - w) * ridge_oof
    return -r2_score(y_train, blend)

result  = minimize_scalar(neg_r2_blend, bounds=(0.0, 1.0), method='bounded')
w_lgb   = float(result.x)
w_ridge = 1.0 - w_lgb

ensemble_oof      = w_lgb * lgb_oof   + w_ridge * ridge_oof
ensemble_test     = w_lgb * lgb_test_preds + w_ridge * ridge_test_preds
ensemble_oof_r2   = r2_score(y_train, ensemble_oof)

print(f"  Optimal LGB weight   : {w_lgb:.4f}  ({w_lgb*100:.1f}%)")
print(f"  Optimal Ridge weight : {w_ridge:.4f}  ({w_ridge*100:.1f}%)")
print(f"  Ensemble OOF R²      : {ensemble_oof_r2:+.6f}")
print(f"  vs LGB alone         : {lgb_oof_r2:+.6f}  "
      f"({'improvement' if ensemble_oof_r2 > lgb_oof_r2 else 'no improvement'})")
print(f"  vs Ridge alone       : {ridge_oof_r2:+.6f}  "
      f"({'improvement' if ensemble_oof_r2 > ridge_oof_r2 else 'no improvement'})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Save OOF arrays and submission CSV
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8 — Saving outputs")
print("=" * 70)

np.save('outputs/oof_predictions/tgt_norm_lgb_oof.npy',        lgb_oof)
np.save('outputs/oof_predictions/tgt_norm_ridge_oof.npy',      ridge_oof)
np.save('outputs/oof_predictions/tgt_norm_ensemble_oof.npy',   ensemble_oof)
np.save('outputs/oof_predictions/tgt_norm_lgb_test.npy',       lgb_test_preds)
np.save('outputs/oof_predictions/tgt_norm_ridge_test.npy',     ridge_test_preds)

sample_sub = pd.read_csv(SAMPLE_SUB)
sub        = pd.DataFrame({'ID': test_ids, 'TARGET': ensemble_test})
sub        = sample_sub[['ID']].merge(sub, on='ID', how='left')

null_count = sub['TARGET'].isnull().sum()
if null_count:
    print(f"  WARNING: {null_count} IDs missing — filling 0.0")
    sub['TARGET'] = sub['TARGET'].fillna(0.0)

out_path = 'outputs/submissions/target_norm_ridge_v1.csv'
sub.to_csv(out_path, index=False)
print(f"  SAVED: {out_path}  ({len(sub):,} rows)")

print(f"\n  Test ensemble stats:")
print(f"    mean  : {ensemble_test.mean():.7f}")
print(f"    std   : {ensemble_test.std():.7f}")
print(f"    min   : {ensemble_test.min():.7f}")
print(f"    max   : {ensemble_test.max():.7f}")
print(f"    +ve % : {(ensemble_test > 0).mean()*100:.1f}%")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Pipeline         : Target-Norm + fold-safe Z-score + LGB + Ridge")
print(f"  Features         : {X_train.shape[1]} raw (float32)")
print(f"  CV strategy      : {actual_folds}-fold GroupKFold on SO3_T quantiles")
print(f"  LGB objective    : Fair loss (c={FAIR_C})")
print(f"  Ridge alpha      : {RIDGE_ALPHA}")
print(f"  LGB   OOF R²     : {lgb_oof_r2:+.6f}")
print(f"  Ridge OOF R²     : {ridge_oof_r2:+.6f}")
print(f"  Ensemble OOF R²  : {ensemble_oof_r2:+.6f}")
print(f"  LGB / Ridge wts  : {w_lgb*100:.1f}% / {w_ridge*100:.1f}%")
print(f"  Total time       : {total_elapsed/60:.1f} min")
print(f"  Submission       : {out_path}")
print("=" * 70)
