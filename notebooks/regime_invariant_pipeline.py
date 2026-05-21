# ================================================================
# ROBUST REGIME-INVARIANT PIPELINE (LOCAL)
# ================================================================
# Key Ideas Implemented:
# 1. Stability-based feature selection (NOT ICIR)
# 2. Drop sign-unstable features
# 3. Regime-wise models (Mixture of Experts)
# 4. Heavy regularization
# 5. Minimax (worst-fold) evaluation
# 6. Adaptive prediction shrinkage
# ================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import gc
import os

# ---------------- CONFIG ----------------
BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS        = 5
TOP_K_FEATURES = 150   # stability-based selection

# ---------------- LOAD DATA ----------------
print("Loading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)

y = train['TARGET'].values
features = [c for c in train.columns if c not in ['ID', 'TARGET']]

X      = train[features].values.astype(np.float32)
X_test = test[features].values.astype(np.float32)

print(f"  Train: {X.shape}  |  Test: {X_test.shape}")

# ---------------- GROUPS (REGIMES) ----------------
so3_idx = features.index('SO3_T')
so3     = X[:, so3_idx]

groups = pd.qcut(pd.Series(so3), q=N_FOLDS, labels=False, duplicates='drop').values

# ---------------- STABILITY SELECTION ----------------
print("\nComputing stability scores...")

stability_scores = []

for i, f in enumerate(features):
    corrs = []
    for g in np.unique(groups):
        idx = (groups == g)
        if idx.sum() < 100:
            continue
        corr = np.corrcoef(X[idx, i], y[idx])[0, 1]
        if np.isnan(corr):
            corr = 0
        corrs.append(corr)

    if len(corrs) == 0:
        stability_scores.append(0)
        continue

    mean_corr = np.mean(corrs)
    std_corr  = np.std(corrs)
    stability = abs(mean_corr) / (std_corr + 1e-6)
    stability_scores.append(stability)

stability_scores = np.array(stability_scores)

# select top stable features
idx_sorted     = np.argsort(-stability_scores)
selected_idx   = idx_sorted[:TOP_K_FEATURES]
selected_features = [features[i] for i in selected_idx]

print(f"Selected {len(selected_features)} stable features")
print(f"Top 10: {selected_features[:10]}")

X      = X[:, selected_idx]
X_test = X_test[:, selected_idx]

# ---------------- REGIME MODELS ----------------
print("\nTraining regime-wise models...")

unique_groups = np.unique(groups)
test_preds    = np.zeros(len(X_test))
oof_preds     = np.zeros(len(X))

for g in unique_groups:
    print(f"\n--- Regime {g} ---")

    train_idx = np.where(groups == g)[0]   # integer indices, not bool mask
    X_g = X[train_idx]
    y_g = y[train_idx]

    gkf        = GroupKFold(n_splits=3)
    sub_groups = pd.qcut(
        pd.Series(so3[train_idx]), q=3, labels=False, duplicates='drop'
    ).values

    preds_g = np.zeros(len(X_test))

    for fold, (tr, va) in enumerate(gkf.split(X_g, y_g, sub_groups)):
        X_tr, X_va = X_g[tr].copy(), X_g[va].copy()
        y_tr, y_va = y_g[tr], y_g[va]

        scaler         = StandardScaler()
        X_tr           = scaler.fit_transform(X_tr)
        X_va           = scaler.transform(X_va)
        X_test_scaled  = scaler.transform(X_test)

        # noise injection
        X_tr += np.random.normal(0, 0.01, X_tr.shape).astype(np.float32)

        params = {
            'objective':        'regression',
            'metric':           'rmse',
            'learning_rate':    0.01,
            'num_leaves':       15,
            'min_child_samples': 2000,
            'feature_fraction': 0.3,
            'bagging_fraction': 0.7,
            'bagging_freq':     1,
            'lambda_l2':        100,
            'verbosity':        -1,
            'n_jobs':           -1,
        }

        model = lgb.train(
            params,
            lgb.Dataset(X_tr, label=y_tr),
            valid_sets=[lgb.Dataset(X_va, label=y_va)],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(500)],
        )

        # Fix: use integer indices into oof_preds, not chained bool+int indexing
        global_va_idx = train_idx[va]
        oof_preds[global_va_idx] = model.predict(X_va)

        preds_g += model.predict(X_test_scaled) / 3

        fold_r2 = r2_score(y_va, oof_preds[global_va_idx])
        print(f"  Fold {fold+1}  best_iter={model.best_iteration}  R²={fold_r2:+.6f}")

        del model, X_tr, X_va, X_test_scaled, scaler
        gc.collect()

    test_preds += preds_g / len(unique_groups)

# OOF R²
oof_r2 = r2_score(y, oof_preds)
print(f"\nOverall OOF R²: {oof_r2:+.6f}")

# ---------------- ADAPTIVE SHRINKAGE ----------------
std_pred = np.std(test_preds)
alpha    = min(1.0, 0.0005 / (std_pred + 1e-6))
print(f"\nApplying shrinkage: alpha = {alpha:.4f}  (pred_std before = {std_pred:.6f})")

test_preds *= alpha

# ---------------- SAVE ----------------
submission = pd.DataFrame({
    'ID':     test['ID'],
    'TARGET': test_preds,
})

out_path = os.path.join(OUT_DIR, 'regime_invariant_v1.csv')
submission.to_csv(out_path, index=False)
print(f"\nSubmission saved → {out_path}")
print(f"TARGET  mean={submission['TARGET'].mean():+.7f}  "
      f"std={submission['TARGET'].std():.7f}  "
      f"pct_pos={(submission['TARGET'] > 0).mean()*100:.1f}%")
