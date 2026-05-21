# ================================================================
# fold_safe_v4c — LOW-DRIFT FEATURES + inv_SO3_T + NO REGIME WEIGHTS
# ================================================================
# Changes vs fold_safe_v1 (baseline OOF=+0.000544):
#   + inv_SO3_T added as engineered feature (1 / SO3_T)
#   + Only 352 low-drift features (KS ≤ 0.20) instead of all 445
#   - NO regime sample_weight (root cause of best_iter=1 in v4)
#
# Why inv_SO3_T: SO3_T is a normalisation factor buried in TARGET.
# TARGET = raw_return / SO3_T  (approximately). Giving the model
# 1/SO3_T directly lets it learn the back-scaling relationship.
#
# Why low-drift only: 93/445 features have KS>0.2; they shift heavily
# between train and test. Removing them gives a cleaner signal that
# actually generalises to the test distribution.
#
# Architecture: identical to fold_safe_v1
#   - GroupKFold(3) on SO3_T quintiles
#   - Original TARGET as label (no y_clean)
#   - StandardScaler inside fold (anti-leakage)
#   - Winsorisation at p1/p99 of train fold
#   - float16 storage + free_raw_data (memory)
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR    = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH  = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH   = os.path.join(BASE_DIR, 'data/raw/test.parquet')
DRIFT_PATH  = os.path.join(BASE_DIR, 'outputs/analysis/drift_report.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'outputs/submissions')
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS    = 5
NUM_LEAVES = 63
N_EST      = 2000
LR         = 0.05
ES_ROUNDS  = 50

t0 = time.time()

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)

all_feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET'}]
print(f"  Total features available: {len(all_feat_cols)}")

# ── Low-drift feature filter ─────────────────────────────────────────────────
drift_df   = pd.read_csv(DRIFT_PATH)
low_drift  = set(drift_df[drift_df['ks_statistic'] <= 0.20]['feature'].tolist())
feat_cols  = [c for c in all_feat_cols if c in low_drift]
print(f"  Low-drift features (KS≤0.20): {len(feat_cols)}")

# ── Add inv_SO3_T ─────────────────────────────────────────────────────────────
so3_tr = train['SO3_T'].values.clip(1e-6, None)
so3_te = test['SO3_T'].values.clip(1e-6, None)
train['inv_SO3_T'] = (1.0 / so3_tr).astype(np.float32)
test['inv_SO3_T']  = (1.0 / so3_te).astype(np.float32)
feat_cols = feat_cols + ['inv_SO3_T']
print(f"  Final feature set (low-drift + inv_SO3_T): {len(feat_cols)}")

# ── Build matrices (float16 for memory) ──────────────────────────────────────
y_train  = train['TARGET'].values.astype(np.float32)
test_ids = test['ID'].values

X_train = train[feat_cols].fillna(0).values.astype(np.float32)
X_test  = test.reindex(columns=feat_cols, fill_value=0.0).values.astype(np.float32)
print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"  X_train memory: {X_train.nbytes/1e9:.2f} GB")

del train, test
gc.collect()

# ── GroupKFold on SO3_T quintiles ─────────────────────────────────────────────
# SO3_T is at index of feat_cols (if it's low-drift) or we extract from X_train
so3t_col_name = 'SO3_T'
if so3t_col_name in feat_cols:
    so3t_idx  = feat_cols.index(so3t_col_name)
    so3t_vals = X_train[:, so3t_idx].astype(np.float32)
else:
    # SO3_T was filtered out — use inv_SO3_T inversely
    inv_idx   = feat_cols.index('inv_SO3_T')
    so3t_vals = (1.0 / X_train[:, inv_idx].astype(np.float64).clip(1e-9, None)).astype(np.float32)

groups  = pd.qcut(pd.Series(so3t_vals), q=N_FOLDS,
                  labels=False, duplicates='drop').values.astype(np.int32)
n_folds = len(np.unique(groups))
gkf     = GroupKFold(n_splits=n_folds)
folds   = list(gkf.split(X_train, y_train, groups=groups))

print(f"\nGroupKFold: {n_folds} folds on SO3_T quintiles")
for i, (tr, va) in enumerate(folds):
    print(f"  Fold {i+1}: train={len(tr):,}  val={len(va):,}  "
          f"group={sorted(np.unique(groups[va]).tolist())}")

# ── LightGBM params ───────────────────────────────────────────────────────────
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

# ── Training loop ─────────────────────────────────────────────────────────────
oof_preds   = np.zeros(len(y_train), dtype=np.float64)
test_preds  = np.zeros(len(X_test),  dtype=np.float64)
fold_r2s    = []
best_iters  = []

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    tf = time.time()
    print(f"\n{'─'*55}")
    print(f"FOLD {fold_idx+1}/{n_folds}")

    X_tr = X_train[tr_idx].copy()
    y_tr = y_train[tr_idx].astype(np.float64)
    y_va = y_train[va_idx].astype(np.float64)

    # Anti-leakage winsorisation
    lo, hi = np.percentile(y_tr, 1), np.percentile(y_tr, 99)
    y_tr   = np.clip(y_tr, lo, hi)

    # Anti-leakage StandardScaler
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)

    X_va = scaler.transform(X_train[va_idx])

    # LightGBM datasets (free_raw_data to save RAM)
    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain, free_raw_data=True)

    del X_tr
    gc.collect()

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round       = N_EST,
        valid_sets            = [dval],
        callbacks             = [
            lgb.early_stopping(ES_ROUNDS, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    best_iter = model.best_iteration
    best_iters.append(best_iter)

    oof_preds[va_idx] = model.predict(X_va, num_iteration=best_iter)

    # Test predictions (scale X_test with this fold's scaler)
    X_te_scaled = scaler.transform(X_test)
    test_preds  += model.predict(X_te_scaled, num_iteration=best_iter) / n_folds

    fold_r2 = r2_score(y_va, oof_preds[va_idx])
    fold_r2s.append(fold_r2)

    print(f"  best_iter={best_iter}  fold_R²={fold_r2:+.6f}  ({time.time()-tf:.0f}s)")

    del X_va, X_te_scaled, dtrain, dval, model, scaler
    gc.collect()

# ── OOF summary ───────────────────────────────────────────────────────────────
oof_r2   = r2_score(y_train, oof_preds)
min_r2   = min(fold_r2s)
pred_std = oof_preds.std()
pct_pos  = (oof_preds > 0).mean()

print(f"\n{'='*55}")
print(f"fold_safe_v4c RESULTS")
print(f"{'='*55}")
print(f"  OOF R²        : {oof_r2:+.6f}")
print(f"  Per-fold R²   : {[f'{r:+.6f}' for r in fold_r2s]}")
print(f"  Min fold R²   : {min_r2:+.6f}")
print(f"  Best iters    : {best_iters}")
print(f"  OOF pred std  : {pred_std:.6f}")
print(f"  OOF pct_pos   : {pct_pos*100:.1f}%")
print(f"\n  Reference — fold_safe_v1:  OOF=+0.000544  LB=+0.00005  std=0.000624")
print(f"  Delta vs v1   : {oof_r2 - 0.000544:+.6f}")

# ── Test submission stats ─────────────────────────────────────────────────────
print(f"\n  Test pred std : {test_preds.std():.6f}")
print(f"  Test pct_pos  : {(test_preds > 0).mean()*100:.1f}%")
print(f"  Test skew     : {pd.Series(test_preds).skew():+.3f}")

# ── Save submission ───────────────────────────────────────────────────────────
sample_sub_path = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
if os.path.exists(sample_sub_path):
    id_col = pd.read_csv(sample_sub_path)[['ID']]
else:
    id_col = pd.DataFrame({'ID': test_ids})

sub = pd.DataFrame({'ID': test_ids, 'TARGET': test_preds})
sub = id_col.merge(sub, on='ID', how='left').fillna(0.0)

out_path = os.path.join(OUT_DIR, 'fold_safe_v4c.csv')
sub.to_csv(out_path, index=False)
print(f"\n  Saved → {out_path}")
print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")
