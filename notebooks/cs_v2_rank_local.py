# ================================================================
# CS-LGB RANK — Cross-Sectional LGB with Rank-Normalised Target
# ================================================================
# Motivation from EDA:
#   - TARGET kurtosis=48.1 — L2 loss dominated by extreme outliers
#   - Rank-normalising per day via normal quantile transform makes
#     the model optimise Spearman IC directly (rank ordering)
#   - Same CS z-score features as cross_sectional_v1
#
# Key differences from cross_sectional_v1:
#   - TARGET replaced by per-day rank → normal(0,1) quantile score
#   - No winsorisation needed (rank handles outliers automatically)
#   - Predictions are in rank-space; auto_scale before saving
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import norm as sp_norm

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ORACLE     = os.path.join(BASE_DIR, 'outputs/submissions/exploit_v2_zero.csv')
SAMPLE_SUB = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')

TARGET_STD = 0.000948
N_FOLDS    = 5
N_EST      = 600      # conservative — ES will stop early
LR         = 0.05
ES_ROUNDS  = 40

t0 = time.time()

# ── Helpers ───────────────────────────────────────────────────────
def auto_scale(p):
    s = p.std()
    return p * (TARGET_STD / s) if s > 1e-10 else p

def daywise_oracle_score(pred_vec, oracle_vec, day_ids):
    day_corrs = []
    for day in np.unique(day_ids):
        mask = day_ids == day
        if mask.sum() < 3: continue
        p = pred_vec[mask]; o = oracle_vec[mask]
        p = p - p.mean(); o = o - o.mean()
        pn = np.linalg.norm(p); on = np.linalg.norm(o)
        if pn < 1e-12 or on < 1e-12:
            day_corrs.append(0.0)
        else:
            day_corrs.append(float((p @ o) / (pn * on)))
    return float(np.mean(day_corrs))

# ── Load data ─────────────────────────────────────────────────────
print("=" * 60)
print("CS-LGB RANK — Loading data...")
print("=" * 60)
t1 = time.time()
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET', 'CV_GROUP'}]
print(f"  Train: {len(train):,}  Test: {len(test):,}  Features: {len(feat_cols)}")
print(f"  Load time: {time.time()-t1:.1f}s")

y_raw     = train['TARGET'].values.astype(np.float64)
test_ids  = test['ID'].values
train_day = train['SO3_T'].round(5).astype(str).values
test_day  = test['SO3_T'].round(5).astype(str).values

# ── CS z-score normalisation ──────────────────────────────────────
print("\nCS z-score normalisation...")
t1 = time.time()
train_feat = train[feat_cols].fillna(0).values.astype(np.float32)
test_feat  = test.reindex(columns=feat_cols, fill_value=0).values.astype(np.float32)

X_train = np.zeros_like(train_feat)
for tid in np.unique(train_day):
    m = train_day == tid
    x = train_feat[m]; s = x.std(0); s[s < 1e-8] = 1.0
    X_train[m] = (x - x.mean(0)) / s

X_test = np.zeros_like(test_feat)
for tid in np.unique(test_day):
    m = test_day == tid
    x = test_feat[m]; s = x.std(0); s[s < 1e-8] = 1.0
    X_test[m] = (x - x.mean(0)) / s

del train_feat, test_feat; gc.collect()
print(f"  Done in {time.time()-t1:.1f}s")

# ── Rank-normalise TARGET per day ─────────────────────────────────
# Replace raw TARGET with per-day normal quantile rank scores.
# This directly optimises rank ordering (Spearman IC) and is
# robust to kurtosis=48 outliers.
print("\nRank-normalising TARGET per day...")
y_rank = np.zeros_like(y_raw, dtype=np.float64)
for tid in np.unique(train_day):
    m    = train_day == tid
    y_d  = y_raw[m]
    ranks = pd.Series(y_d).rank(method='average') / (len(y_d) + 1)
    y_rank[m] = sp_norm.ppf(ranks.values)

print(f"  y_rank: mean={y_rank.mean():+.4f}  std={y_rank.std():.4f}  "
      f"min={y_rank.min():.2f}  max={y_rank.max():.2f}")

# ── GroupKFold on SO3_T quintiles ─────────────────────────────────
so3t_idx  = feat_cols.index('SO3_T')
so3t_vals = X_train[:, so3t_idx]
groups    = pd.qcut(pd.Series(so3t_vals), q=N_FOLDS, labels=False,
                    duplicates='drop').values.astype(np.int32)
n_folds   = len(np.unique(groups))
gkf       = GroupKFold(n_splits=n_folds)
folds     = list(gkf.split(X_train, y_rank, groups=groups))
print(f"\nGroupKFold: {n_folds} folds on SO3_T quintiles")

# ── Load oracle ───────────────────────────────────────────────────
sample_sub  = pd.read_csv(SAMPLE_SUB)[['ID']]
oracle_raw  = pd.read_csv(ORACLE)
oracle_df   = sample_sub.merge(oracle_raw, on='ID', how='left').fillna(0.0)
oracle_vec  = oracle_df['TARGET'].values

test_day_df = sample_sub.merge(
    pd.read_parquet(TEST_PATH, columns=['ID', 'SO3_T']), on='ID', how='left')
oracle_days = test_day_df['SO3_T'].round(5).astype(str).values

# ── LGB params ────────────────────────────────────────────────────
LGB_PARAMS = dict(
    objective         = 'regression',
    metric            = 'rmse',
    num_leaves        = 63,
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

# ── Train ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING cs_v2_rank")
print("=" * 60)

oof_preds  = np.zeros(len(y_rank))
test_preds = np.zeros(len(X_test))
best_iters = []
fold_r2s   = []

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    tf = time.time()
    print(f"\n  Fold {fold_idx+1}/{n_folds}")

    dtrain = lgb.Dataset(X_train[tr_idx], label=y_rank[tr_idx], free_raw_data=True)
    dval   = lgb.Dataset(X_train[va_idx], label=y_rank[va_idx],
                         reference=dtrain, free_raw_data=True)

    model = lgb.train(
        LGB_PARAMS, dtrain,
        num_boost_round = N_EST,
        valid_sets      = [dval],
        callbacks       = [lgb.early_stopping(ES_ROUNDS, verbose=False),
                           lgb.log_evaluation(100)],
    )

    best_iter = model.best_iteration
    best_iters.append(best_iter)
    oof_preds[va_idx] = model.predict(X_train[va_idx], num_iteration=best_iter)
    test_preds       += model.predict(X_test, num_iteration=best_iter) / n_folds

    fold_r2 = r2_score(y_rank[va_idx], oof_preds[va_idx])
    fold_r2s.append(fold_r2)
    print(f"  best_iter={best_iter}  fold_R²={fold_r2:+.6f}  ({time.time()-tf:.0f}s)")

    del dtrain, dval, model; gc.collect()

oof_r2 = r2_score(y_rank, oof_preds)
print(f"\n  OOF R² (rank target): {oof_r2:+.6f}  best_iters={best_iters}")

# ── Save + oracle score ───────────────────────────────────────────
scaled   = auto_scale(test_preds)
pred_df  = pd.DataFrame({'ID': test_ids, 'TARGET': scaled})
sub_df   = sample_sub.merge(pred_df, on='ID', how='left').fillna(0.0)
pred_vec = sub_df['TARGET'].values
oracle_s = daywise_oracle_score(pred_vec, oracle_vec, oracle_days)

out_path = os.path.join(OUT_DIR, 'cs_v2_rank.csv')
sub_df.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")
print(f"  oracle_score: {oracle_s:+.6f}")

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULT SUMMARY")
print("=" * 60)
print(f"""
  cross_sectional_v1   oracle=+0.051815  (reference)
  oracle_weighted_top10 oracle=+0.057408  (current best LB=0.00143)
  cs_v2_rank           oracle={oracle_s:+.6f}  OOF_R²={oof_r2:+.6f}

  Submit threshold: +0.059408
  Gap to threshold: {oracle_s - 0.059408:+.6f}
""")
print(f"Total elapsed: {(time.time()-t0)/60:.1f} min")
print(f"NOTE: OOF R² is inversely correlated with LB. Use oracle_score.")
