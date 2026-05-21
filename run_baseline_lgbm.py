"""
Baseline LightGBM — raw 445 features only, zero feature engineering.
Purpose: clean leakage-free benchmark to compare against engineered model.
Target   : TARGET (raw, no winsorization)
Submission: outputs/submissions/lgbm_baseline_v1.csv
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import pickle
import os
import gc
import warnings
warnings.filterwarnings('ignore')

os.chdir('/Users/malaymishra/Desktop/quant_ml_project')
os.makedirs('outputs/oof_predictions', exist_ok=True)
os.makedirs('models/checkpoints', exist_ok=True)
os.makedirs('outputs/submissions', exist_ok=True)

RANDOM_SEED = 42
N_FOLDS     = 3
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("BASELINE LightGBM — raw 445 features, no engineering")
print("=" * 70)

# ── Load raw data ───────────────────────────────────────────────────────────
print("\n--- Loading raw data ---")
train = pd.read_parquet('data/raw/train-001.parquet')
test  = pd.read_parquet('data/raw/test.parquet')
print(f"Train: {train.shape},  Test: {test.shape}")

# ── Features & target ───────────────────────────────────────────────────────
NON_FEAT = ['ID', 'TARGET']
feat_cols = [c for c in train.columns if c not in NON_FEAT]
print(f"Features: {len(feat_cols)}")

y_train  = train['TARGET'].values.astype(np.float32)
test_ids = test['ID'].values

X_train = train[feat_cols].values.astype(np.float32)
X_test  = test[feat_cols].values.astype(np.float32)
print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
print(f"TARGET — min:{y_train.min():.4f}  max:{y_train.max():.4f}  "
      f"mean:{y_train.mean():.4f}  std:{y_train.std():.4f}")

# ── Regime groups (bin SO3_T into 3 quantile groups for GroupKFold) ─────────
groups = pd.qcut(train['SO3_T'], q=N_FOLDS, labels=False).values.astype(np.int32)
print(f"Regime groups: {dict(zip(*np.unique(groups, return_counts=True)))}")

del train, test
gc.collect()

# ── CV setup ────────────────────────────────────────────────────────────────
gkf   = GroupKFold(n_splits=N_FOLDS)
folds = list(gkf.split(X_train, y_train, groups=groups))
print(f"\nGroupKFold: {N_FOLDS} folds")
for i, (tr, va) in enumerate(folds):
    print(f"  Fold {i+1}: train={len(tr):,}  val={len(va):,}  "
          f"groups_val={np.unique(groups[va]).tolist()}")

# ── Fair loss ───────────────────────────────────────────────────────────────
FAIR_C = 1.0

def fair_obj(y_pred, dataset):
    y_true = dataset.get_label()
    r    = y_pred - y_true
    grad = r / (1.0 + np.abs(r) / FAIR_C)
    hess = FAIR_C ** 2 / (FAIR_C + np.abs(r)) ** 2
    return grad, hess

def r2_eval(y_pred, dataset):
    y_true = dataset.get_label()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-15)
    return 'r2', r2, True   # higher is better

# ── LightGBM params ─────────────────────────────────────────────────────────
lgb_params = {
    'num_leaves'       : 127,
    'learning_rate'    : 0.05,
    'n_estimators'     : 2000,
    'feature_fraction' : 0.6,
    'bagging_fraction' : 0.8,
    'bagging_freq'     : 1,
    'min_child_samples': 100,
    'reg_alpha'        : 0.1,
    'reg_lambda'       : 1.0,
    'random_state'     : RANDOM_SEED,
    'n_jobs'           : -1,
    'verbose'          : -1,
}

# ── Training ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Training LightGBM baseline (Fair loss, GroupKFold 3-fold)")
print("=" * 70)

oof        = np.zeros(len(y_train), dtype=np.float32)
test_preds = np.zeros(len(X_test),  dtype=np.float64)

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_cols, free_raw_data=True)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, free_raw_data=True)

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=200),
    ]

    model = lgb.train(
        {**lgb_params, 'objective': fair_obj},
        dtrain,
        valid_sets=[dvalid],
        feval=r2_eval,
        callbacks=callbacks,
    )

    oof[va_idx]  = model.predict(X_va).astype(np.float32)
    test_preds  += model.predict(X_test) / N_FOLDS

    fold_r2 = r2_score(y_va, oof[va_idx])
    print(f"    Best iter: {model.best_iteration}  |  Fold R²: {fold_r2:.6f}")

    model.save_model(f'models/checkpoints/lgbm_baseline_fold{fold_idx+1}.txt')
    del X_tr, X_va, dtrain, dvalid
    gc.collect()

oof_r2 = r2_score(y_train, oof)
print(f"\nBaseline LightGBM OOF R²: {oof_r2:.6f}")

np.save('outputs/oof_predictions/lgbm_baseline_oof.npy',        oof)
np.save('outputs/oof_predictions/lgbm_baseline_test_preds.npy', test_preds)

# ── Submission ───────────────────────────────────────────────────────────────
sample_sub = pd.read_csv('data/raw/sample_submission.csv')
sub = pd.DataFrame({'ID': test_ids, 'TARGET': test_preds})
sub = sample_sub[['ID']].merge(sub, on='ID', how='left')
sub.to_csv('outputs/submissions/lgbm_baseline_v1.csv', index=False)

print(f"\nSAVED: outputs/submissions/lgbm_baseline_v1.csv  ({len(sub):,} rows)")
print(f"Predictions — min:{test_preds.min():.6f}  max:{test_preds.max():.6f}  "
      f"mean:{test_preds.mean():.6f}")
print()
print("=" * 70)
print(f"  Baseline OOF R²: {oof_r2:.6f}")
print("  Compare to engineered LightGBM OOF R²: 0.101804")
print("  If baseline LB > -0.04 → engineered features had leakage")
print("=" * 70)
