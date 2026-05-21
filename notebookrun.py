# ┌─────────────────────────────────────────────────────────────────────────┐
# │  KAGGLE NOTEBOOK: Rank + Regime-Neutral LightGBM & CatBoost Ensemble   │
# │                                                                         │
# │  Builds on the validated fold_safe_v1 pipeline (LB R² = +0.00005):     │
# │    fold_safe_v1 : Z-score fitted on training fold  → LB +0.00005       │
# │    this notebook: Rank fitted on training fold CDF → LB expected higher │
# │                   + Regime Neutralization (SO3_T orthogonalization)     │
# │                   + CatBoost added for ensemble diversity               │
# │                                                                         │
# │  Anti-leakage guarantees:                                               │
# │    1. Rankings computed via training fold CDF for val AND test          │
# │    2. Neutralization betas computed on training fold only               │
# │    3. SO3_T mean for centering saved from training fold, reused on val/test│
# │    4. Winsorization bounds from y_tr percentiles only                   │
# └─────────────────────────────────────────────────────────────────────────┘


# ═══════════════════════════════════════════════════════════════════════════
# CELL 1: IMPORTS & SETUP
# ═══════════════════════════════════════════════════════════════════════════

import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from scipy.stats import rankdata
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

# Kaggle Paths
INPUT_DIR  = '/kaggle/input/<YOUR_COMPETITION_DATASET_NAME>/'
OUTPUT_DIR = '/kaggle/working/'

os.makedirs(OUTPUT_DIR + 'oof_predictions', exist_ok=True)
os.makedirs(OUTPUT_DIR + 'submissions',     exist_ok=True)
os.makedirs(OUTPUT_DIR + 'checkpoints',     exist_ok=True)

RANDOM_SEED = 42
N_FOLDS     = 5
np.random.seed(RANDOM_SEED)

print("Environment Ready.")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 2: LOAD DATA & PER-ROW FEATURES
# ═══════════════════════════════════════════════════════════════════════════

print("Loading raw data...")
train = pd.read_parquet(INPUT_DIR + 'train.parquet')
test  = pd.read_parquet(INPUT_DIR + 'test.parquet')

# SO3_T is used only for grouping and neutralization — kept separate from features
feat_cols = [c for c in train.columns if c not in ['ID', 'TARGET', 'SO3_T']]

# 1. Downcast to float32
for col in feat_cols + ['TARGET', 'SO3_T']:
    if col in train.columns: train[col] = train[col].astype(np.float32)
    if col in test.columns:  test[col]  = test[col].astype(np.float32)

# 2. NaN imputation — medians computed on TRAIN ONLY, applied to both
medians = train[feat_cols + ['SO3_T']].median()
train.fillna(medians, inplace=True)
test.fillna(medians,  inplace=True)

# 3. Per-Row Features (lag ratios & convergence — purely row-wise arithmetic)
# Safe by construction: each row computed independently, no cross-row stats.
print("Building per-row features...")
base_cols = [c for c in feat_cols if '_LagT' not in c]
new_train_feats, new_test_feats = {}, {}

for bc in base_cols:
    l1, l2 = f"{bc}_LagT1", f"{bc}_LagT2"
    if l1 in train.columns and l2 in train.columns:
        new_train_feats[f'{bc}_lagrat'] = (train[l1] / (train[l2].abs() + 1e-8)).clip(-10, 10)
        new_test_feats[f'{bc}_lagrat']  = (test[l1]  / (test[l2].abs()  + 1e-8)).clip(-10, 10)
        new_train_feats[f'{bc}_conv']   = train[l1] - (train[l2] / 2)
        new_test_feats[f'{bc}_conv']    = test[l1]  - (test[l2]  / 2)

train = pd.concat([train, pd.DataFrame(new_train_feats, index=train.index)], axis=1)
test  = pd.concat([test,  pd.DataFrame(new_test_feats,  index=test.index)],  axis=1)

# Extract final raw matrices (rankings computed INSIDE fold loop against training CDF)
all_feats   = [c for c in train.columns if c not in ['ID', 'TARGET', 'SO3_T']]
X_train_raw = train[all_feats].values.astype(np.float32)
X_test_raw  = test[all_feats].values.astype(np.float32)
y_train_raw = train['TARGET'].values.astype(np.float32)
so3_train   = train['SO3_T'].values.astype(np.float32)
so3_test    = test['SO3_T'].values.astype(np.float32)
test_ids    = test['ID'].values

del train, test, new_train_feats, new_test_feats
gc.collect()
print(f"Raw feature matrices built: {X_train_raw.shape[1]} features.")
print(f"Rankings will be computed per-fold against the training fold CDF.")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 3: CUSTOM FUNCTIONS (LOSS, METRIC, RANKING, NEUTRALIZATION)
# ═══════════════════════════════════════════════════════════════════════════

# ── Fair Loss (LightGBM objective) ─────────────────────────────────────────
# Robust to fat-tailed financial return distributions vs MSE.
FAIR_C = 1.0

def fair_obj_lgb(y_pred, dataset):
    y_true = dataset.get_label()
    r      = y_pred - y_true
    grad   = r / (1.0 + np.abs(r) / FAIR_C)
    hess   = FAIR_C ** 2 / (FAIR_C + np.abs(r)) ** 2
    return grad, hess

# ── R² evaluation metric ───────────────────────────────────────────────────
def r2_eval_lgb(y_pred, dataset):
    y_true  = dataset.get_label()
    ss_res  = np.sum((y_true - y_pred) ** 2)
    ss_tot  = np.sum((y_true - y_true.mean()) ** 2)
    r2      = 1.0 - ss_res / (ss_tot + 1e-15)
    return 'r2', r2, True   # True = higher is better

# ── In-fold ranking against training CDF (leak-free) ─────────────────────
# CRITICAL: X_apply (val or test) is ranked by its percentile position
# within the TRAINING FOLD's empirical distribution — not its own.
# This ensures val and test features are on the same scale as training features.
# Contrast with fast_rank2d(X_va) which would rank va against itself — wrong.
def rank_against_reference(X_ref, X_apply):
    """
    For each feature column:
      - Sort the training fold values (X_ref)
      - Find where each X_apply value sits within that sorted array
      - Normalize to [0, 1] by dividing by len(X_ref)

    This maps X_apply values to their percentile rank in the
    training fold's distribution. Same raw value → same rank
    regardless of which set (val/test) it comes from.
    """
    n_ref = X_ref.shape[0]
    ranks = np.empty((X_apply.shape[0], X_apply.shape[1]), dtype=np.float32)
    for i in range(X_ref.shape[1]):
        sorted_col    = np.sort(X_ref[:, i])
        ranks[:, i]   = np.searchsorted(sorted_col, X_apply[:, i]) / n_ref
    return ranks

def rank_train_fold(X_tr):
    """Rank the training fold against itself (standard within-fold ranking)."""
    n = X_tr.shape[0]
    ranks = np.empty_like(X_tr, dtype=np.float32)
    for i in range(X_tr.shape[1]):
        ranks[:, i] = rankdata(X_tr[:, i]) / n
    return ranks

# ── Regime Neutralization (SO3_T orthogonalization) ───────────────────────
# Removes the linear component of SO3_T from all ranked features.
# This makes the signal regime-invariant — the model predicts alpha that
# is orthogonal to the market state (SO3_T), which should generalize better
# across different regimes in the test set.
#
# Fix applied: so3_mean is computed from the TRAINING FOLD and passed
# explicitly when applying to val/test. Without this, the centering
# s = subset_so3 - mean(subset_so3) would use val/test's own mean,
# creating a subtle inconsistency in the orthogonalization.
def neutralize_features(X, subset_so3, betas=None, so3_mean=None):
    """
    Args:
        X          : ranked feature matrix
        subset_so3 : SO3_T ranks for this subset
        betas      : pre-computed regression coefficients (None = compute from X)
        so3_mean   : mean SO3_T rank from the TRAINING FOLD (None = compute here)

    Returns:
        X_neut, betas, so3_mean
    """
    # Use training fold's SO3_T mean if provided to ensure consistent centering
    if so3_mean is None:
        so3_mean = np.mean(subset_so3)

    s = subset_so3 - so3_mean   # centered SO3_T ranks

    if betas is None:
        # Compute regression coefficient: beta = Cov(X_col, s) / Var(s)
        # Fitting on training fold only
        var_s   = np.sum(s ** 2) + 1e-8
        cov_xs  = np.sum(X * s[:, None], axis=0)
        betas   = cov_xs / var_s

    X_neut = X - betas * s[:, None]
    return X_neut.astype(np.float32), betas, so3_mean

print("Custom functions defined.")
print("  - fair_obj_lgb      : Fair loss objective")
print("  - r2_eval_lgb       : R² custom metric")
print("  - rank_against_reference : Fold-safe ranking (training CDF)")
print("  - rank_train_fold   : Training fold self-ranking")
print("  - neutralize_features : SO3_T orthogonalization")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 4: CV SPLIT SETUP
# ═══════════════════════════════════════════════════════════════════════════

# GroupKFold on SO3_T quantile buckets
# Each fold holds out a different region of the SO3_T distribution,
# testing whether the model generalizes across market regimes.
groups = pd.qcut(
    pd.Series(so3_train), q=N_FOLDS,
    labels=False, duplicates='drop'
).values.astype(np.int32)

gkf   = GroupKFold(n_splits=N_FOLDS)
folds = list(gkf.split(X_train_raw, y_train_raw, groups=groups))

print(f"GroupKFold: {N_FOLDS} folds on SO3_T quantile buckets")
for i, (tr, va) in enumerate(folds):
    print(f"  Fold {i+1}: train={len(tr):,}  val={len(va):,}  "
          f"val_group={sorted(np.unique(groups[va]).tolist())}")

# NOTE: X_test_rank is NO LONGER pre-computed here.
# In the previous (buggy) version, X_test_rank = fast_rank2d(X_test_raw)
# was computed outside the loop using the TEST SET's own distribution.
# This caused a distribution mismatch: training ranks were relative to ~528k
# training rows, but test ranks were relative to 410k test rows.
# FIX: test and val are now ranked against the training fold CDF inside the loop.
print("\nTest set ranking deferred to inside fold loop (training CDF reference).")
print("CV setup complete.")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 5: MASTER FOLD LOOP (LightGBM + CatBoost, GPU Accelerated)
# ═══════════════════════════════════════════════════════════════════════════

lgb_oof = np.zeros(len(y_train_raw), dtype=np.float32)
cb_oof  = np.zeros(len(y_train_raw), dtype=np.float32)

lgb_test_preds = np.zeros(len(X_test_raw), dtype=np.float64)
cb_test_preds  = np.zeros(len(X_test_raw), dtype=np.float64)

# ── LightGBM params ────────────────────────────────────────────────────────
# FIX 1: 'objective' removed from params dict — custom Python objectives
#         must be passed as obj= in lgb.train(), not in the params dict.
# NOTE: num_leaves=127 with max_depth=6 → effective leaves = min(127, 2^6=64).
#       max_depth is the binding constraint. This is conservative by design.
lgb_params = {
    'num_leaves'       : 127,
    'max_depth'        : 6,
    'min_child_samples': 500,
    'learning_rate'    : 0.01,
    'feature_fraction' : 0.3,
    'bagging_fraction' : 0.7,
    'bagging_freq'     : 1,
    'reg_lambda'       : 20.0,
    'max_bin'          : 127,    # moved from Dataset params to main params
    'device'           : 'gpu',
    'n_jobs'           : 1,
    'verbose'          : -1,
    'random_state'     : RANDOM_SEED,
}
LGB_ROUNDS = 5000   # FIX 2: was missing → lgb.train() defaulted to 100 rounds

# ── CatBoost params ────────────────────────────────────────────────────────
cb_params = {
    'iterations'         : 5000,
    'learning_rate'      : 0.02,
    'depth'              : 6,
    'l2_leaf_reg'        : 30,
    'loss_function'      : 'RMSE',
    'od_type'            : 'Iter',   # added: reliable iteration-based early stopping
    'od_wait'            : 150,
    'task_type'          : 'GPU',
    'verbose'            : False,
    'random_seed'        : RANDOM_SEED,
}

print("=" * 60)
print("MASTER FOLD LOOP")
print("=" * 60)
print(f"LightGBM: max {LGB_ROUNDS} rounds, lr=0.01, early_stop=150")
print(f"CatBoost: {cb_params['iterations']} iterations, lr=0.02, early_stop=150")
print()

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    print(f"\n{'='*50}")
    print(f"  FOLD {fold_idx+1}/{N_FOLDS}  "
          f"(train={len(tr_idx):,}  val={len(va_idx):,})")
    print(f"{'='*50}")

    # ── Step 1: Slice raw data ───────────────────────────────────────────
    X_tr = X_train_raw[tr_idx].copy()
    X_va = X_train_raw[va_idx].copy()
    y_tr = y_train_raw[tr_idx].copy()
    y_va = y_train_raw[va_idx].copy()
    s_tr = so3_train[tr_idx].copy()
    s_va = so3_train[va_idx].copy()

    # ── Step 2: Target winsorization (fold-safe) ─────────────────────────
    # Bounds from y_tr only. Applied to y_va using SAME bounds (not y_va's own).
    clip_low  = np.percentile(y_tr, 1)
    clip_high = np.percentile(y_tr, 99)
    y_tr      = np.clip(y_tr, clip_low, clip_high).astype(np.float32)
    y_va      = np.clip(y_va, clip_low, clip_high).astype(np.float32)
    print(f"  Winsorization: [{clip_low:.5f}, {clip_high:.5f}]")

    # ── Step 3: Fold-safe cross-sectional ranking ─────────────────────────
    # FIX 3 (Bug 3 + Bug 7): Previously:
    #   - X_test_rank was pre-computed OUTSIDE the loop on the test set's own
    #     distribution (410k rows) — different reference population than training
    #   - X_va_rank = fast_rank2d(X_va) ranked val against itself — inconsistent
    #     with training fold ranks
    #
    # Now: ALL ranking uses the TRAINING FOLD as the reference distribution.
    #   - X_tr_rank: training fold ranked against itself (standard)
    #   - X_va_rank: val values positioned in training fold's CDF
    #   - X_test_rank_fold: test values positioned in training fold's CDF
    # → Same raw value → same rank regardless of which set it came from.
    print("  -> Computing fold-safe ranks (training CDF reference)...")
    X_tr_rank        = rank_train_fold(X_tr)
    X_va_rank        = rank_against_reference(X_tr, X_va)
    X_test_rank_fold = rank_against_reference(X_tr, X_test_raw)

    # SO3_T also ranked against training fold CDF
    s_tr_rank  = rankdata(s_tr) / len(s_tr)
    s_va_rank  = np.searchsorted(np.sort(s_tr), s_va) / len(s_tr)
    so3_test_rank_fold = np.searchsorted(np.sort(s_tr), so3_test) / len(s_tr)
    s_tr_rank  = s_tr_rank.astype(np.float32)
    s_va_rank  = s_va_rank.astype(np.float32)
    so3_test_rank_fold = so3_test_rank_fold.astype(np.float32)

    # ── Step 4: Regime neutralization ────────────────────────────────────
    # FIX 4 (Bug 4): so3_mean computed from training fold and saved.
    # Applied consistently to val and test centering.
    # Previously: np.mean(subset_so3) would use val/test's own mean — wrong.
    print("  -> Neutralizing SO3_T regime component...")
    X_tr_neut, fold_betas, fold_so3_mean = neutralize_features(
        X_tr_rank, s_tr_rank, betas=None, so3_mean=None
    )
    X_va_neut, _, _ = neutralize_features(
        X_va_rank, s_va_rank,
        betas=fold_betas, so3_mean=fold_so3_mean   # training fold stats
    )
    X_test_neut, _, _ = neutralize_features(
        X_test_rank_fold, so3_test_rank_fold,
        betas=fold_betas, so3_mean=fold_so3_mean   # training fold stats
    )

    # ── Step 5: Train LightGBM ────────────────────────────────────────────
    # FIX 1: obj=fair_obj_lgb passed directly (not in params dict)
    # FIX 2: num_boost_round=LGB_ROUNDS (was missing, defaulted to 100)
    print("  -> Training LightGBM...")
    dtrain = lgb.Dataset(X_tr_neut, label=y_tr, free_raw_data=True)
    dvalid = lgb.Dataset(X_va_neut, label=y_va, reference=dtrain, free_raw_data=True)

    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=LGB_ROUNDS,
        fobj=fair_obj_lgb,   # Kaggle LightGBM uses fobj=, not obj=
        valid_sets=[dvalid],
        feval=r2_eval_lgb,
        callbacks=[lgb.early_stopping(150, verbose=False),
                   lgb.log_evaluation(500)],
    )

    lgb_oof[va_idx]  = model_lgb.predict(X_va_neut).astype(np.float32)
    lgb_test_preds  += model_lgb.predict(X_test_neut) / N_FOLDS

    fold_r2_lgb = r2_score(y_va, lgb_oof[va_idx])
    print(f"  LightGBM — best_iter: {model_lgb.best_iteration}  "
          f"fold R²: {fold_r2_lgb:.6f}")
    model_lgb.save_model(f"{OUTPUT_DIR}checkpoints/lgbm_fold{fold_idx+1}.txt")

    # ── Step 6: Train CatBoost ────────────────────────────────────────────
    print("  -> Training CatBoost...")
    train_pool = Pool(X_tr_neut, y_tr)
    valid_pool = Pool(X_va_neut, y_va)

    model_cb = CatBoostRegressor(**cb_params)
    model_cb.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=150)

    cb_oof[va_idx]  = model_cb.predict(X_va_neut).astype(np.float32)
    cb_test_preds  += model_cb.predict(X_test_neut) / N_FOLDS

    fold_r2_cb = r2_score(y_va, cb_oof[va_idx])
    print(f"  CatBoost  — best_iter: {model_cb.best_iteration_}  "
          f"fold R²: {fold_r2_cb:.6f}")
    model_cb.save_model(f"{OUTPUT_DIR}checkpoints/catboost_fold{fold_idx+1}.cbm")

    # ── Step 7: Aggressive memory cleanup ────────────────────────────────
    del (X_tr, X_va, y_tr, y_va, s_tr, s_va,
         X_tr_rank, X_va_rank, X_test_rank_fold,
         s_tr_rank, s_va_rank, so3_test_rank_fold,
         X_tr_neut, X_va_neut, X_test_neut,
         dtrain, dvalid, model_lgb,
         train_pool, valid_pool, model_cb)
    gc.collect()
    print(f"  Fold {fold_idx+1} memory freed.")

print(f"\n{'='*50}")
print("All Folds Completed!")
print(f"{'='*50}")
print(f"  LightGBM OOF R²  : {r2_score(y_train_raw, lgb_oof):.6f}")
print(f"  CatBoost  OOF R²  : {r2_score(y_train_raw, cb_oof):.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 6: OPTIMAL ENSEMBLE WEIGHTS + SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════

lgb_oof_r2 = r2_score(y_train_raw, lgb_oof)
cb_oof_r2  = r2_score(y_train_raw, cb_oof)

print("\nOptimizing ensemble blend weights on OOF R²...")

# FIX 5: Optimal weights via Nelder-Mead (was hardcoded 0.5/0.5)
# The best blend weight w* minimises -R²(y, w*lgb + (1-w)*cb)
res = minimize_scalar(
    lambda w: -r2_score(y_train_raw, w * lgb_oof + (1 - w) * cb_oof),
    bounds=(0.0, 1.0),
    method='bounded',
)
opt_lgb_w = res.x
opt_cb_w  = 1.0 - opt_lgb_w

print(f"  LightGBM OOF R²   : {lgb_oof_r2:.6f}")
print(f"  CatBoost  OOF R²  : {cb_oof_r2:.6f}")
print(f"  Optimal LGB weight: {opt_lgb_w:.4f}")
print(f"  Optimal  CB weight: {opt_cb_w:.4f}")

ensemble_oof        = opt_lgb_w * lgb_oof        + opt_cb_w * cb_oof
ensemble_test_preds = opt_lgb_w * lgb_test_preds + opt_cb_w * cb_test_preds

ensemble_r2 = r2_score(y_train_raw, ensemble_oof)
print(f"  Ensemble OOF R²   : {ensemble_r2:.6f}")

# Save individual OOF arrays for future offline blending experiments
np.save(f"{OUTPUT_DIR}oof_predictions/lgbm_rank_oof.npy",   lgb_oof)
np.save(f"{OUTPUT_DIR}oof_predictions/cb_rank_oof.npy",     cb_oof)
np.save(f"{OUTPUT_DIR}oof_predictions/lgbm_rank_test.npy",  lgb_test_preds)
np.save(f"{OUTPUT_DIR}oof_predictions/cb_rank_test.npy",    cb_test_preds)

# Build submission aligned to sample_submission ordering
sample_sub = pd.read_csv(INPUT_DIR + 'sample_submission.csv')
sub = pd.DataFrame({'ID': test_ids, 'TARGET': ensemble_test_preds})
sub = sample_sub[['ID']].merge(sub, on='ID', how='left')
sub.to_csv(f"{OUTPUT_DIR}submissions/ensemble_rank_neutral_v1.csv", index=False)

print(f"\nSAVED: {OUTPUT_DIR}submissions/ensemble_rank_neutral_v1.csv "
      f"({len(sub):,} rows)")
print(f"  Pred mean : {ensemble_test_preds.mean():.6f}")
print(f"  Pred std  : {ensemble_test_preds.std():.6f}")
print(f"  Pred range: [{ensemble_test_preds.min():.6f}, "
      f"{ensemble_test_preds.max():.6f}]")
print("\nReady for Kaggle Leaderboard submission.")
print("\nWhen sharing results, please provide:")
print("  - Per-fold R² for LightGBM and CatBoost separately")
print("  - Optimal blend weight (printed above)")
print("  - Ensemble OOF R² vs Leaderboard R² (to track efficiency ratio)")
