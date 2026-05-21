# ================================================================
# ADVERSARIAL-WEIGHTED RIDGE — TEST-ALIGNED TRAINING
# ================================================================
# Key insight: 99.8% of train rows are NOT test-like.
# Training a model that weights test-like rows 1000× should force
# it to learn only what works in the test distribution.
#
# Variants:
#   V1: Top 10% by p_global_rank (66k rows) — Ridge, all features
#   V2: Top 20% by p_global_rank (132k rows) — Ridge, all features
#   V3: final_weight_sq as sample_weight — Ridge on all 661k rows
#   V4: final_weight_sq — LightGBM on all 661k rows, random KFold
#       (the version that failed in v4 used GroupKFold — this fixes it)
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import lightgbm as lgb

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR    = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH  = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH   = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ADV_PATH    = os.path.join(BASE_DIR, 'outputs/analysis/adversarial_scores.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'outputs/submissions')
t0 = time.time()

print("Loading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
adv   = pd.read_csv(ADV_PATH).set_index('ID')

feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET'}]
y_all     = train['TARGET'].values.astype(np.float64)
test_ids  = test['ID'].values
X_all     = train[feat_cols].fillna(0).values.astype(np.float32)
X_te      = test.reindex(columns=feat_cols, fill_value=0).values.astype(np.float32)

# Merge adversarial weights
adv_aligned = adv.loc[train['ID'].values]
p_global_rank  = adv_aligned['p_global_rank'].values
final_w_sq     = adv_aligned['final_weight_sq'].values
print(f"  X_all: {X_all.shape}  |  p_global_rank: min={p_global_rank.min():.4f} max={p_global_rank.max():.4f}")

del train, test
gc.collect()

sample_sub = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/sample_submission.csv'))[['ID']]

def save_sub(test_ids, preds, name):
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': preds})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    path = os.path.join(OUT_DIR, f'{name}.csv')
    sub.to_csv(path, index=False)
    t = sub['TARGET']
    print(f"  saved {name}.csv  std={t.std():.7f}  mean={t.mean():+.8f}  pct_pos={(t>0).mean()*100:.1f}%")


# ================================================================
# V1 & V2: Train on top-k% most test-like rows — Ridge
# ================================================================
for pct_cutoff, vname in [(0.90, 'V1_top10pct'), (0.80, 'V2_top20pct')]:
    mask     = p_global_rank >= pct_cutoff
    n_rows   = mask.sum()
    X_sub    = X_all[mask].astype(np.float64)
    y_sub    = y_all[mask]
    X_te_f64 = X_te.astype(np.float64)

    print(f"\n{'='*60}")
    print(f"RIDGE {vname}: {n_rows:,} rows ({(1-pct_cutoff)*100:.0f}% most test-like)")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(n_rows)
    te_preds = np.zeros(len(X_te_f64))
    fold_r2s = []

    for fold_i, (tr, va) in enumerate(kf.split(X_sub)):
        lo, hi = np.percentile(y_sub[tr], 1), np.percentile(y_sub[tr], 99)
        y_tr   = np.clip(y_sub[tr], lo, hi)

        model = Ridge(alpha=10.0, fit_intercept=True)
        model.fit(X_sub[tr], y_tr)
        oof[va]   = model.predict(X_sub[va])
        te_preds += model.predict(X_te_f64) / 5

        r2 = r2_score(y_sub[va], oof[va])
        fold_r2s.append(r2)
        print(f"  Fold {fold_i+1}  R²={r2:+.6f}")

    oof_r2 = r2_score(y_sub, oof)
    print(f"  OOF R²={oof_r2:+.6f}  (note: OOF on these test-like rows only)")
    save_sub(test_ids, te_preds, f'adv_{vname}_ridge10')

    del X_sub, y_sub, oof, te_preds
    gc.collect()


# ================================================================
# V3: full 661k rows, Ridge, sample_weight = final_weight_sq
# ================================================================
print(f"\n{'='*60}")
print("RIDGE V3: All 661k rows, sample_weight = final_weight_sq")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_v3   = np.zeros(len(y_all))
te_v3    = np.zeros(len(X_te))
fold_r2s = []
X_f64    = X_all.astype(np.float64)
X_te_f64 = X_te.astype(np.float64)

for fold_i, (tr, va) in enumerate(kf.split(X_f64)):
    lo, hi = np.percentile(y_all[tr], 1), np.percentile(y_all[tr], 99)
    y_tr   = np.clip(y_all[tr], lo, hi)
    w_tr   = final_w_sq[tr]

    model = Ridge(alpha=10.0, fit_intercept=True)
    model.fit(X_f64[tr], y_tr, sample_weight=w_tr)
    oof_v3[va] = model.predict(X_f64[va])
    te_v3     += model.predict(X_te_f64) / 5

    r2 = r2_score(y_all[va], oof_v3[va])
    fold_r2s.append(r2)
    print(f"  Fold {fold_i+1}  R²={r2:+.6f}")

oof_r2 = r2_score(y_all, oof_v3)
print(f"  OOF R²={oof_r2:+.6f}  vs fold_safe_v1 baseline +0.000544")
save_sub(test_ids, te_v3, 'adv_V3_allrows_ridge_weighted')
del X_f64, oof_v3, te_v3
gc.collect()


# ================================================================
# V4: LightGBM with final_weight_sq, random KFold (NOT GroupKFold)
# This is what v4/v4c tried to do but failed with GroupKFold
# ================================================================
print(f"\n{'='*60}")
print("LGB V4: All 661k rows, sample_weight = final_weight_sq, random KFold")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_v4 = np.zeros(len(y_all))
te_v4  = np.zeros(len(X_te))
fold_r2s_v4 = []

lgb_params = dict(
    objective        = 'regression',
    metric           = 'rmse',
    num_leaves       = 63,
    learning_rate    = 0.05,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq     = 1,
    min_child_samples= 50,
    lambda_l1        = 0.1,
    lambda_l2        = 1.0,
    n_jobs           = -1,
    verbose          = -1,
    seed             = 42,
)

for fold_i, (tr, va) in enumerate(kf.split(X_all)):
    tf = time.time()
    y_tr = y_all[tr].copy()
    lo, hi = np.percentile(y_tr, 1), np.percentile(y_tr, 99)
    y_tr = np.clip(y_tr, lo, hi)
    w_tr = final_w_sq[tr]

    dtrain = lgb.Dataset(X_all[tr], label=y_tr, weight=w_tr, free_raw_data=True)
    dval   = lgb.Dataset(X_all[va], label=y_all[va], reference=dtrain, free_raw_data=True)

    model = lgb.train(
        lgb_params, dtrain,
        num_boost_round = 1000,
        valid_sets      = [dval],
        callbacks       = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    oof_v4[va] = model.predict(X_all[va], num_iteration=model.best_iteration)
    te_v4     += model.predict(X_te,      num_iteration=model.best_iteration) / 5

    r2 = r2_score(y_all[va], oof_v4[va])
    fold_r2s_v4.append(r2)
    print(f"  Fold {fold_i+1}  best_iter={model.best_iteration}  R²={r2:+.6f}  ({time.time()-tf:.0f}s)")

    del dtrain, dval, model
    gc.collect()

oof_r2_v4 = r2_score(y_all, oof_v4)
print(f"  OOF R²={oof_r2_v4:+.6f}  vs fold_safe_v1 baseline +0.000544")
save_sub(test_ids, te_v4, 'adv_V4_lgb_advweight_randomkfold')
del oof_v4, te_v4
gc.collect()


print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
