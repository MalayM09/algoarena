# ================================================================
# VALIDATE cross_sectional_v1 on train-001.parquet
# Runs the EXACT same logic as cross_sectional_v1.py
# Saves to cross_sectional_v1_validate.csv
# Compares with saved cross_sectional_v1.csv
# ================================================================

import os, gc, time, warnings, sys
sys.stdout.reconfigure(line_buffering=True)
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
SAVED_CSV  = os.path.join(OUT_DIR, 'cross_sectional_v1.csv')
t0 = time.time()

N_FOLDS    = 5
NUM_LEAVES = 63
N_EST      = 2000
LR         = 0.05
ES_ROUNDS  = 50

print("Loading data...", flush=True)
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET'}]
print(f"  train={train.shape}  test={test.shape}  feat_cols={len(feat_cols)}", flush=True)

train_time_ids = train['SO3_T'].round(5).astype(str).values
test_time_ids  = test['SO3_T'].round(5).astype(str).values

y_train  = train['TARGET'].values.astype(np.float32)
test_ids = test['ID'].values

print("\nCS z-score...", flush=True)
t1 = time.time()
train_feat = train[feat_cols].fillna(0).values.astype(np.float32)
test_feat  = test.reindex(columns=feat_cols, fill_value=0).values.astype(np.float32)

train_norm = np.zeros_like(train_feat)
for tid in np.unique(train_time_ids):
    mask = train_time_ids == tid
    x = train_feat[mask]; m = x.mean(0); s = x.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    train_norm[mask] = (x - m) / s

test_norm = np.zeros_like(test_feat)
for tid in np.unique(test_time_ids):
    mask = test_time_ids == tid
    x = test_feat[mask]; m = x.mean(0); s = x.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    test_norm[mask] = (x - m) / s

print(f"  Done in {time.time()-t1:.1f}s", flush=True)
del train_feat, test_feat, train, test; gc.collect()

X_train = train_norm.astype(np.float32)
X_test  = test_norm.astype(np.float32)
del train_norm, test_norm; gc.collect()

# GroupKFold on z-scored SO3_T quintiles (same as original)
so3t_idx  = feat_cols.index('SO3_T')
so3t_vals = X_train[:, so3t_idx]
groups    = pd.qcut(pd.Series(so3t_vals), q=N_FOLDS,
                    labels=False, duplicates='drop').values.astype(np.int32)
n_folds   = len(np.unique(groups))
gkf       = GroupKFold(n_splits=n_folds)
folds     = list(gkf.split(X_train, y_train, groups=groups))

print(f"\nGroupKFold: {n_folds} folds on z-scored SO3_T")
for i, (tr, va) in enumerate(folds):
    print(f"  Fold {i+1}: train={len(tr):,}  val={len(va):,}", flush=True)

lgb_params = dict(
    objective='regression', metric='rmse', num_leaves=63,
    learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, min_child_samples=50, lambda_l1=0.1, lambda_l2=1.0,
    n_jobs=-1, verbose=-1, seed=42,
)

oof_preds  = np.zeros(len(y_train), dtype=np.float64)
test_preds = np.zeros(len(X_test),  dtype=np.float64)
fold_r2s   = []
best_iters = []

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    tf = time.time()
    print(f"\nFOLD {fold_idx+1}/{n_folds}", flush=True)
    X_tr = X_train[tr_idx].copy()
    y_tr = y_train[tr_idx].astype(np.float64)
    y_va = y_train[va_idx].astype(np.float64)
    lo, hi = np.percentile(y_tr, 1), np.percentile(y_tr, 99)
    y_tr   = np.clip(y_tr, lo, hi)
    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
    dval   = lgb.Dataset(X_train[va_idx], label=y_va, reference=dtrain, free_raw_data=True)
    del X_tr; gc.collect()
    model = lgb.train(lgb_params, dtrain, num_boost_round=N_EST, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False),
                                 lgb.log_evaluation(200)])
    best_iter = model.best_iteration
    best_iters.append(best_iter)
    oof_preds[va_idx] = model.predict(X_train[va_idx], num_iteration=best_iter)
    test_preds       += model.predict(X_test, num_iteration=best_iter) / n_folds
    fold_r2 = r2_score(y_va, oof_preds[va_idx])
    fold_r2s.append(fold_r2)
    print(f"  best_iter={best_iter}  fold_R²={fold_r2:+.6f}  ({time.time()-tf:.0f}s)", flush=True)
    del dtrain, dval, model; gc.collect()

oof_r2 = r2_score(y_train, oof_preds)
print(f"\n{'='*55}")
print(f"RESULTS")
print(f"{'='*55}")
print(f"  OOF R²     : {oof_r2:+.6f}")
print(f"  Best iters : {best_iters}")
print(f"  Test std   : {test_preds.std():.6f}", flush=True)

# Save validation output
sample_sub = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/sample_submission.csv'))[['ID']]
sub_new = sample_sub.merge(pd.DataFrame({'ID': test_ids, 'TARGET': test_preds}), on='ID', how='left').fillna(0.0)
out_path = os.path.join(OUT_DIR, 'cross_sectional_v1_validate.csv')
sub_new.to_csv(out_path, index=False)
print(f"  Saved: {out_path}", flush=True)

# ── COMPARE with saved CSV ─────────────────────────────────────
print(f"\n{'='*55}")
print("COMPARISON vs saved cross_sectional_v1.csv")
print(f"{'='*55}")
saved = pd.read_csv(SAVED_CSV)
merged = sample_sub.merge(saved.rename(columns={'TARGET':'TARGET_saved'}), on='ID', how='left').fillna(0)
merged = merged.merge(sub_new.rename(columns={'TARGET':'TARGET_new'}), on='ID', how='left').fillna(0)

diff = (merged['TARGET_new'] - merged['TARGET_saved']).abs()
corr = merged['TARGET_new'].corr(merged['TARGET_saved'])
print(f"  Max abs diff  : {diff.max():.10f}")
print(f"  Mean abs diff : {diff.mean():.10f}")
print(f"  Correlation   : {corr:.8f}")
print(f"  Saved std     : {merged['TARGET_saved'].std():.6f}")
print(f"  New std       : {merged['TARGET_new'].std():.6f}")

if diff.max() < 1e-6:
    print(f"\n  PASS — predictions match within 1e-6")
else:
    print(f"\n  MISMATCH — investigating...")
    worst = merged.loc[diff.idxmax()]
    print(f"  Worst ID: {worst['ID']}  saved={worst['TARGET_saved']:.8f}  new={worst['TARGET_new']:.8f}")

print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f} min")
