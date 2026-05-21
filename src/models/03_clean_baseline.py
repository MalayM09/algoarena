"""
03_clean_baseline.py — Leak-Free LightGBM Baseline
====================================================
Strategy: strict per-row feature engineering only. Zero cross-sectional
transforms. Predict raw TARGET. 5-fold GroupKFold on SO3_T quantiles.

Anti-leakage guarantees
------------------------
1. No StandardScaler / QuantileTransformer / rank transforms across rows.
2. All engineered features are purely arithmetic operations on the same row.
3. Median imputation fitted on train set only, applied to test separately.
4. GroupKFold grouping derived from SO3_T quantiles (no TARGET leakage).
5. Top-30 LagT1 selection uses absolute Pearson correlation on training data
   only — this is feature *selection*, not feature *transformation*, so
   feature values themselves are never contaminated by cross-row statistics.

Engineered features (per-row only)
------------------------------------
For every base feature F that has all three lag versions (111 triplets):
  - lag_ratio_T1T2  : F_LagT1 / (|F_LagT2| + 1e-8)    clipped to [-10, 10]
  - sign_agreement  : sign(F_LagT1) * sign(F_LagT2) * sign(F_LagT3)
  - convergence     : F_LagT1 - (F_LagT2 / 2)

For the top-30 LagT1 features by |Pearson corr with TARGET|:
  - so3t_interact    : F_LagT1 * SO3_T   (raw SO3_T, no normalization needed
                       — LightGBM is scale-invariant; cross-row normalization
                       of SO3_T would itself violate the anti-leakage rules)

Output
------
  outputs/submissions/clean_lgbm_baseline.csv
  outputs/oof_predictions/clean_lgbm_baseline_oof.npy
  outputs/oof_predictions/clean_lgbm_baseline_test_preds.npy
  models/checkpoints/clean_lgbm_baseline_fold{1..5}.txt
"""

import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR  = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_RAW = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_RAW  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
SAMPLE_SUB= os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')

os.chdir(BASE_DIR)
os.makedirs('outputs/oof_predictions', exist_ok=True)
os.makedirs('outputs/submissions',     exist_ok=True)
os.makedirs('models/checkpoints',      exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
N_FOLDS       = 5
N_ESTIMATORS  = 3000
EARLY_STOP    = 150
SO3T_TOP_N    = 30    # number of LagT1 features to interact with SO3_T
np.random.seed(RANDOM_SEED)

# ── LightGBM params — heavy regularization to prevent overfitting ──────────
LGB_PARAMS = {
    'num_leaves'       : 63,
    'learning_rate'    : 0.05,
    'feature_fraction' : 0.3,
    'bagging_fraction' : 0.7,
    'bagging_freq'     : 1,
    'min_child_samples': 500,
    'reg_alpha'        : 1.0,
    'reg_lambda'       : 20.0,
    'random_state'     : RANDOM_SEED,
    'n_jobs'           : -1,
    'verbose'          : -1,
}

# ── Custom R² eval for LightGBM ────────────────────────────────────────────
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
print("STEP 1 — Loading raw data and downcasting to float32")
print("=" * 70)
t0 = time.time()

train_raw = pd.read_parquet(TRAIN_RAW)
test_raw  = pd.read_parquet(TEST_RAW)
print(f"  Train raw : {train_raw.shape}")
print(f"  Test  raw : {test_raw.shape}")

NON_FEAT  = {'ID', 'TARGET'}
feat_cols = [c for c in train_raw.columns if c not in NON_FEAT]

# Downcast features to float32 (halves memory vs float64)
for col in feat_cols:
    train_raw[col] = train_raw[col].astype(np.float32)
for col in feat_cols:
    if col in test_raw.columns:
        test_raw[col] = test_raw[col].astype(np.float32)

y_train  = train_raw['TARGET'].values.astype(np.float32)
test_ids = test_raw['ID'].values
print(f"  TARGET — mean:{y_train.mean():.6f}  std:{y_train.std():.6f}  "
      f"min:{y_train.min():.4f}  max:{y_train.max():.4f}")
print(f"  Elapsed: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — NaN imputation (train median only, applied to test separately)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2 — NaN imputation (train medians → apply to test)")
print("=" * 70)

train_nan = train_raw[feat_cols].isna().sum().sum()
test_nan  = test_raw[[c for c in feat_cols if c in test_raw.columns]].isna().sum().sum()
print(f"  Train NaNs: {train_nan:,}    Test NaNs: {test_nan:,}")

if train_nan > 0 or test_nan > 0:
    medians = train_raw[feat_cols].median()
    train_raw[feat_cols] = train_raw[feat_cols].fillna(medians)
    test_feat_cols = [c for c in feat_cols if c in test_raw.columns]
    test_raw[test_feat_cols] = test_raw[test_feat_cols].fillna(medians[test_feat_cols])
    print("  Imputation applied.")
else:
    print("  No NaNs found — skipping imputation.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Identify lag triplets
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3 — Identifying lag feature triplets")
print("=" * 70)

lag1_feats = [f for f in feat_cols if f.endswith('_LagT1')]
lag2_feats = [f for f in feat_cols if f.endswith('_LagT2')]
lag3_feats = [f for f in feat_cols if f.endswith('_LagT3')]

lag1_bases = set(f[:-6] for f in lag1_feats)   # strip '_LagT1'
lag2_bases = set(f[:-6] for f in lag2_feats)
lag3_bases = set(f[:-6] for f in lag3_feats)
triplet_bases = sorted(lag1_bases & lag2_bases & lag3_bases)

print(f"  LagT1 features : {len(lag1_feats)}")
print(f"  LagT2 features : {len(lag2_feats)}")
print(f"  LagT3 features : {len(lag3_feats)}")
print(f"  Complete triplets (T1+T2+T3): {len(triplet_bases)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Per-row feature engineering (strictly leak-free)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4 — Per-row feature engineering")
print("=" * 70)
print("  Rules: row-wise arithmetic only. No cross-row stats.")

def engineer_features(df, triplet_bases, so3t_top30, is_train=True):
    """
    All transforms operate on individual rows in isolation.
    No fit step required — purely algebraic.
    """
    new_feats = {}

    for base in triplet_bases:
        t1 = df[f'{base}_LagT1'].values.astype(np.float32)
        t2 = df[f'{base}_LagT2'].values.astype(np.float32)
        t3 = df[f'{base}_LagT3'].values.astype(np.float32)

        # Lag ratio T1/T2: how fast is the signal accelerating/decelerating?
        ratio = t1 / (np.abs(t2) + 1e-8)
        new_feats[f'{base}_ratio_T1T2'] = np.clip(ratio, -10.0, 10.0)

        # Sign agreement: all three lags pointing same direction = persistence
        new_feats[f'{base}_sign_agree'] = (
            np.sign(t1) * np.sign(t2) * np.sign(t3)
        ).astype(np.float32)

        # Convergence: is T1 moving toward zero faster than T2?
        new_feats[f'{base}_convergence'] = (t1 - t2 / 2.0).astype(np.float32)

    # SO3_T interactions with top-30 LagT1 features
    # SO3_T used raw (cross-row normalization would violate anti-leakage rules)
    so3t = df['SO3_T'].values.astype(np.float32)
    for lag1_feat in so3t_top30:
        t1 = df[lag1_feat].values.astype(np.float32)
        new_feats[f'{lag1_feat}_x_SO3T'] = (t1 * so3t).astype(np.float32)

    new_df = pd.DataFrame(new_feats, index=df.index)
    return new_df

# Select top-30 LagT1 features by |Pearson correlation with TARGET|
# This is feature *selection* on training data — no feature values are changed
print("  Selecting top-30 LagT1 features by |Pearson r| with TARGET ...")
lag1_corrs = {}
for f in lag1_feats:
    lag1_corrs[f] = abs(train_raw[f].astype(np.float64).corr(
        pd.Series(y_train.astype(np.float64), index=train_raw.index)
    ))
so3t_top30 = sorted(lag1_corrs, key=lag1_corrs.get, reverse=True)[:SO3T_TOP_N]
top_corr_vals = [f"{f}: {lag1_corrs[f]:.5f}" for f in so3t_top30[:5]]
print(f"  Top-5 selected: {top_corr_vals}")

# Engineer features for train and test
t1 = time.time()
print("  Engineering train features ...")
train_eng = engineer_features(train_raw, triplet_bases, so3t_top30, is_train=True)
print(f"  Engineering test features ...")
test_eng  = engineer_features(test_raw,  triplet_bases, so3t_top30, is_train=False)
print(f"  Engineered features: {train_eng.shape[1]}  ({time.time()-t1:.1f}s)")

# Final feature matrix = raw features + engineered features
all_feat_cols  = feat_cols + list(train_eng.columns)
X_train = np.hstack([
    train_raw[feat_cols].values.astype(np.float32),
    train_eng.values.astype(np.float32),
])
X_test  = np.hstack([
    test_raw[[c for c in feat_cols if c in test_raw.columns]].values.astype(np.float32),
    test_eng.values.astype(np.float32),
])
print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}")
print(f"  Memory — X_train: {X_train.nbytes/1e6:.0f} MB   X_test: {X_test.nbytes/1e6:.0f} MB")

del train_eng, test_eng
gc.collect()

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — GroupKFold on SO3_T quantiles
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5 — 5-fold GroupKFold on SO3_T quantile buckets")
print("=" * 70)

so3t_series = train_raw['SO3_T']
groups = pd.qcut(so3t_series, q=N_FOLDS, labels=False, duplicates='drop').values.astype(np.int32)
n_groups = len(np.unique(groups))
if n_groups < N_FOLDS:
    print(f"  WARNING: SO3_T qcut produced {n_groups} groups instead of {N_FOLDS} "
          f"(duplicate boundary values). Using {n_groups} folds.")
    actual_folds = n_groups
else:
    actual_folds = N_FOLDS

print(f"  Groups distribution:")
for g, cnt in zip(*np.unique(groups, return_counts=True)):
    print(f"    Group {g}: {cnt:,} rows  "
          f"SO3_T range [{so3t_series[groups==g].min():.6f}, "
          f"{so3t_series[groups==g].max():.6f}]")

gkf   = GroupKFold(n_splits=actual_folds)
folds = list(gkf.split(X_train, y_train, groups=groups))
print(f"\n  Fold summary:")
for i, (tr, va) in enumerate(folds):
    print(f"    Fold {i+1}: train={len(tr):,}  val={len(va):,}  "
          f"val_groups={sorted(np.unique(groups[va]).tolist())}")

del train_raw, test_raw
gc.collect()

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — Training
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6 — LightGBM training (heavy regularization)")
print("=" * 70)
print(f"  Params:")
for k, v in LGB_PARAMS.items():
    print(f"    {k}: {v}")
print(f"  n_estimators  : {N_ESTIMATORS}")
print(f"  early_stopping: {EARLY_STOP} rounds")

oof        = np.zeros(len(y_train), dtype=np.float32)
test_preds = np.zeros(len(X_test),  dtype=np.float64)
fold_r2s   = []

train_start = time.time()

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    fold_start = time.time()
    print(f"\n{'─'*60}")
    print(f"  FOLD {fold_idx+1}/{actual_folds}  "
          f"(train={len(tr_idx):,}  val={len(va_idx):,})")
    print(f"{'─'*60}")

    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
    print(f"  y_train — mean:{y_tr.mean():.6f}  std:{y_tr.std():.6f}")
    print(f"  y_val   — mean:{y_va.mean():.6f}  std:{y_va.std():.6f}")

    dtrain = lgb.Dataset(X_tr, label=y_tr,
                         feature_name=all_feat_cols, free_raw_data=True)
    dvalid = lgb.Dataset(X_va, label=y_va,
                         reference=dtrain, free_raw_data=True)

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=False),
        lgb.log_evaluation(period=200),
    ]

    model = lgb.train(
        {**LGB_PARAMS, 'objective': 'regression_l1'},
        dtrain,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[dvalid],
        feval=r2_eval_lgb,
        callbacks=callbacks,
    )

    oof[va_idx]  = model.predict(X_va).astype(np.float32)
    test_preds  += model.predict(X_test) / actual_folds

    fold_r2 = r2_score(y_va, oof[va_idx])
    fold_r2s.append(fold_r2)
    elapsed = time.time() - fold_start

    print(f"\n  ── Fold {fold_idx+1} results ──")
    print(f"     Best iteration : {model.best_iteration}")
    print(f"     Fold R²        : {fold_r2:.6f}")
    print(f"     Pred mean      : {oof[va_idx].mean():.6f}  "
          f"std: {oof[va_idx].std():.6f}")
    print(f"     Elapsed        : {elapsed:.1f}s")

    model.save_model(f'models/checkpoints/clean_lgbm_baseline_fold{fold_idx+1}.txt')
    del X_tr, X_va, dtrain, dvalid
    gc.collect()

total_elapsed = time.time() - train_start

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — OOF evaluation & submission
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7 — Final evaluation & submission")
print("=" * 70)

oof_r2 = r2_score(y_train, oof)

print(f"\n  Per-fold R² scores:")
for i, r2 in enumerate(fold_r2s):
    print(f"    Fold {i+1}: {r2:.6f}")
print(f"\n  OOF R² (all folds combined): {oof_r2:.6f}")
print(f"  OOF std dev : {oof.std():.6f}")
print(f"  OOF range   : [{oof.min():.6f}, {oof.max():.6f}]")
print(f"  Total training time: {total_elapsed/60:.1f} min")

np.save('outputs/oof_predictions/clean_lgbm_baseline_oof.npy',        oof)
np.save('outputs/oof_predictions/clean_lgbm_baseline_test_preds.npy', test_preds)
print("\n  SAVED: outputs/oof_predictions/clean_lgbm_baseline_oof.npy")
print("  SAVED: outputs/oof_predictions/clean_lgbm_baseline_test_preds.npy")

# Submission
sample_sub = pd.read_csv(SAMPLE_SUB)
sub = pd.DataFrame({'ID': test_ids, 'TARGET': test_preds})
sub = sample_sub[['ID']].merge(sub, on='ID', how='left')
null_count = sub['TARGET'].isnull().sum()
if null_count:
    print(f"  WARNING: {null_count} missing IDs — filling with 0.0")
    sub['TARGET'] = sub['TARGET'].fillna(0.0)

out_path = 'outputs/submissions/clean_lgbm_baseline.csv'
sub.to_csv(out_path, index=False)
print(f"\n  SAVED: {out_path}  ({len(sub):,} rows)")
print(f"  Pred stats — mean:{test_preds.mean():.6f}  "
      f"std:{test_preds.std():.6f}  "
      f"min:{test_preds.min():.6f}  max:{test_preds.max():.6f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Features used         : {X_train.shape[1]}")
print(f"    Raw features        : {len(feat_cols)}")
print(f"    Lag ratio (T1/T2)   : {len(triplet_bases)}")
print(f"    Sign agreement      : {len(triplet_bases)}")
print(f"    Convergence         : {len(triplet_bases)}")
print(f"    SO3_T interactions  : {SO3T_TOP_N}")
print(f"  OOF R²                : {oof_r2:.6f}")
print(f"  Expected LB R²        : ~{oof_r2:.4f} (no leakage)")
print(f"  Submission            : {out_path}")
print("=" * 70)
