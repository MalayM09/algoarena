"""
Modelling v2 — XGBoost, CatBoost, Ridge
LightGBM already complete; loads saved OOF from v1 run.
Fix applied: XGBoost early stopping now uses neg_r2 (lower=better) +
             EarlyStopping callback with maximize=False.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge
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
print("MODELLING v2 — XGBoost / CatBoost / Ridge")
print("=" * 70)

# ── Load Data ──────────────────────────────────────────────────────────────
print("\n--- Loading engineered datasets ---")
train_eng = pd.read_parquet('data/processed/train_engineered.parquet')
test_eng  = pd.read_parquet('data/processed/test_engineered.parquet')
print(f"Train: {train_eng.shape}, Test: {test_eng.shape}")

with open('outputs/feature_engineering/feature_sets.pkl', 'rb') as f:
    feature_sets = pickle.load(f)

tree_features   = feature_sets['tree_features']
linear_features = feature_sets['linear_features']

NON_FEATURE_COLS = ['TARGET', 'TARGET_raw', 'TARGET_wins', 'ID', 'regime']

y_train  = train_eng['TARGET_wins'].values.astype(np.float32)
groups   = train_eng['regime'].values.astype(np.int32)
test_ids = test_eng['ID'].values

tree_feats   = [f for f in tree_features   if f in train_eng.columns
                                            and f not in NON_FEATURE_COLS]
linear_feats = [f for f in linear_features if f in train_eng.columns
                                            and f not in NON_FEATURE_COLS]

print(f"Tree features   : {len(tree_feats)}")
print(f"Linear features : {len(linear_feats)}")

X_train_tree   = train_eng[tree_feats].values.astype(np.float32)
X_test_tree    = test_eng[tree_feats].values.astype(np.float32)
X_train_linear = train_eng[linear_feats].values.astype(np.float64)
X_test_linear  = test_eng[linear_feats].values.astype(np.float64)

del train_eng, test_eng
gc.collect()
print("Feature matrices built.")

# ── CV Setup ───────────────────────────────────────────────────────────────
gkf   = GroupKFold(n_splits=N_FOLDS)
folds = list(gkf.split(X_train_tree, y_train, groups=groups))
print(f"\nGroupKFold: {N_FOLDS} folds on regime labels")
for i, (tr, va) in enumerate(folds):
    print(f"  Fold {i+1}: train={len(tr):,}  val={len(va):,}  regimes_val={np.unique(groups[va]).tolist()}")

# ── Fair Loss ──────────────────────────────────────────────────────────────
FAIR_C = 1.0

def fair_obj_xgb(y_pred, dtrain):
    y_true = dtrain.get_label()
    r = y_pred - y_true
    grad = r / (1.0 + np.abs(r) / FAIR_C)
    hess = FAIR_C ** 2 / (FAIR_C + np.abs(r)) ** 2
    return grad, hess

# neg_r2: lower = better, so XGBoost early stopping (minimize) = maximize R²
def neg_r2_eval_xgb(y_pred, dtrain):
    y_true = dtrain.get_label()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-15)
    return 'neg_r2', -r2   # minimize neg_r2 = maximize r2

# ── Load LightGBM results (already computed) ───────────────────────────────
print("\n--- Loading pre-computed LightGBM OOF ---")
lgb_oof        = np.load('outputs/oof_predictions/lgbm_oof.npy')
lgb_test_preds = np.load('outputs/oof_predictions/lgbm_test_preds.npy')
lgb_oof_r2     = r2_score(y_train, lgb_oof)
print(f"LightGBM OOF R² (loaded): {lgb_oof_r2:.6f}")

# ── 2. XGBoost ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL 2: XGBoost (Fair loss + neg_r2 early stop, GroupKFold 3-fold)")
print("=" * 70)

xgb_base_params = {
    'max_depth'        : 6,
    'learning_rate'    : 0.03,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.6,
    'min_child_weight' : 100,
    'reg_alpha'        : 0.1,
    'reg_lambda'       : 1.0,
    'tree_method'      : 'hist',
    'seed'             : RANDOM_SEED,
    'nthread'          : -1,
    'verbosity'        : 0,
    'disable_default_eval_metric': 1,
}
XGB_ROUNDS = 3000

xgb_oof        = np.zeros(len(y_train), dtype=np.float32)
xgb_test_preds = np.zeros(len(X_test_tree), dtype=np.float64)

# Build test DMatrix once (shared across folds)
print("  Building test DMatrix...")
dtest = xgb.DMatrix(X_test_tree, feature_names=tree_feats)
print("  Test DMatrix ready.")

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")
    X_tr, X_va = X_train_tree[tr_idx], X_train_tree[va_idx]
    y_tr, y_va = y_train[tr_idx],      y_train[va_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=tree_feats)
    dvalid = xgb.DMatrix(X_va, label=y_va, feature_names=tree_feats)

    # EarlyStopping callback: minimizes neg_r2 → effectively maximises R²
    early_stop = xgb.callback.EarlyStopping(
        rounds=100,
        metric_name='neg_r2',
        data_name='val',
        maximize=False,   # minimize neg_r2
        save_best=True,
    )

    model = xgb.train(
        xgb_base_params,
        dtrain,
        num_boost_round=XGB_ROUNDS,
        obj=fair_obj_xgb,
        custom_metric=neg_r2_eval_xgb,
        evals=[(dvalid, 'val')],
        callbacks=[early_stop],
        verbose_eval=200,
    )

    best_iter = model.best_iteration
    xgb_oof[va_idx] = model.predict(dvalid, iteration_range=(0, best_iter + 1)).astype(np.float32)
    xgb_test_preds += model.predict(dtest, iteration_range=(0, best_iter + 1)) / N_FOLDS

    fold_r2 = r2_score(y_va, xgb_oof[va_idx])
    print(f"    Best iter: {best_iter}  |  Fold R²: {fold_r2:.6f}")

    model.save_model(f'models/checkpoints/xgb_fold{fold_idx+1}.json')
    del X_tr, X_va, dtrain, dvalid, model
    gc.collect()

del dtest
gc.collect()

xgb_oof_r2 = r2_score(y_train, xgb_oof)
print(f"\nXGBoost OOF R²: {xgb_oof_r2:.6f}")
np.save('outputs/oof_predictions/xgb_oof.npy',        xgb_oof)
np.save('outputs/oof_predictions/xgb_test_preds.npy', xgb_test_preds)
print("SAVED: outputs/oof_predictions/xgb_oof.npy")
print("SAVED: outputs/oof_predictions/xgb_test_preds.npy")

# ── 3. CatBoost ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL 3: CatBoost (RMSE, GroupKFold 3-fold)")
print("=" * 70)

cb_params = dict(
    iterations          = 3000,
    learning_rate       = 0.03,
    depth               = 6,
    l2_leaf_reg         = 3.0,
    random_strength     = 1.0,
    bagging_temperature = 1.0,
    od_type             = 'Iter',
    od_wait             = 100,
    loss_function       = 'RMSE',
    eval_metric         = 'R2',
    random_seed         = RANDOM_SEED,
    thread_count        = -1,
    verbose             = 200,
)

cb_oof        = np.zeros(len(y_train), dtype=np.float32)
cb_test_preds = np.zeros(len(X_test_tree), dtype=np.float64)

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")
    X_tr, X_va = X_train_tree[tr_idx], X_train_tree[va_idx]
    y_tr, y_va = y_train[tr_idx],      y_train[va_idx]

    pool_tr = Pool(X_tr, label=y_tr, feature_names=tree_feats)
    pool_va = Pool(X_va, label=y_va, feature_names=tree_feats)

    model = CatBoostRegressor(**cb_params)
    model.fit(pool_tr, eval_set=pool_va, use_best_model=True)

    cb_oof[va_idx] = model.predict(X_va).astype(np.float32)
    cb_test_preds += model.predict(X_test_tree) / N_FOLDS

    fold_r2 = r2_score(y_va, cb_oof[va_idx])
    print(f"    Best iter: {model.best_iteration_}  |  Fold R²: {fold_r2:.6f}")

    model.save_model(f'models/checkpoints/catboost_fold{fold_idx+1}.cbm')
    del X_tr, X_va, pool_tr, pool_va, model
    gc.collect()

cb_oof_r2 = r2_score(y_train, cb_oof)
print(f"\nCatBoost OOF R²: {cb_oof_r2:.6f}")
np.save('outputs/oof_predictions/catboost_oof.npy',        cb_oof)
np.save('outputs/oof_predictions/catboost_test_preds.npy', cb_test_preds)
print("SAVED: outputs/oof_predictions/catboost_oof.npy")
print("SAVED: outputs/oof_predictions/catboost_test_preds.npy")

# ── 4. Ridge ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL 4: Ridge (baseline, linear_features, GroupKFold 3-fold)")
print("=" * 70)

ridge_oof        = np.zeros(len(y_train), dtype=np.float32)
ridge_test_preds = np.zeros(len(X_test_linear), dtype=np.float64)

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")
    X_tr, X_va = X_train_linear[tr_idx], X_train_linear[va_idx]
    y_tr, y_va = y_train[tr_idx],        y_train[va_idx]

    model = Ridge(alpha=100.0, random_state=RANDOM_SEED)
    model.fit(X_tr, y_tr)

    ridge_oof[va_idx] = model.predict(X_va).astype(np.float32)
    ridge_test_preds += model.predict(X_test_linear) / N_FOLDS

    fold_r2 = r2_score(y_va, ridge_oof[va_idx])
    print(f"    Fold R²: {fold_r2:.6f}")

    with open(f'models/checkpoints/ridge_fold{fold_idx+1}.pkl', 'wb') as fh:
        pickle.dump(model, fh)
    del X_tr, X_va, model
    gc.collect()

ridge_oof_r2 = r2_score(y_train, ridge_oof)
print(f"\nRidge OOF R²: {ridge_oof_r2:.6f}")
np.save('outputs/oof_predictions/ridge_oof.npy',        ridge_oof)
np.save('outputs/oof_predictions/ridge_test_preds.npy', ridge_test_preds)
print("SAVED: outputs/oof_predictions/ridge_oof.npy")
print("SAVED: outputs/oof_predictions/ridge_test_preds.npy")

# ── Summary & Submissions ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODELLING SUMMARY")
print("=" * 70)
print(f"  LightGBM  OOF R²: {lgb_oof_r2:.6f}  (loaded from v1)")
print(f"  XGBoost   OOF R²: {xgb_oof_r2:.6f}")
print(f"  CatBoost  OOF R²: {cb_oof_r2:.6f}")
print(f"  Ridge     OOF R²: {ridge_oof_r2:.6f}")

sample_sub = pd.read_csv('data/raw/sample_submission.csv')

def make_submission(test_ids_arr, preds, path):
    sub = pd.DataFrame({'ID': test_ids_arr, 'TARGET': preds})
    sub = sample_sub[['ID']].merge(sub, on='ID', how='left')
    sub.to_csv(path, index=False)
    print(f"SAVED: {path}  ({len(sub)} rows)")

make_submission(test_ids, lgb_test_preds,   'outputs/submissions/lgbm_v1.csv')
make_submission(test_ids, xgb_test_preds,   'outputs/submissions/xgb_v1.csv')
make_submission(test_ids, cb_test_preds,    'outputs/submissions/catboost_v1.csv')
make_submission(test_ids, ridge_test_preds, 'outputs/submissions/ridge_v1.csv')

print()
print("  All OOF arrays  : outputs/oof_predictions/")
print("  Model checkpoints: models/checkpoints/")
print("  Submissions      : outputs/submissions/")
print()
print("  Next step: notebooks/04_hyperparameter_tuning.ipynb")
print("=" * 70)
