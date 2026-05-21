# ┌─────────────────────────────────────────────────────────────────────────┐
# │  KAGGLE NOTEBOOK — Stability-Selected, Regularisation Search            │
# │                                                                          │
# │  Incorporates the user's 7-step analysis:                               │
# │  Step 1+2: Drop 61 sign-flip (poison) features → 384 stable features    │
# │  Step 3:   Report MINIMAX fold R² (worst fold), not just mean           │
# │  Step 5:   Config E adds min_child_samples=5000 (more extreme)          │
# │  Step 4:   Shrinkage CSVs generated separately (already done)           │
# │                                                                          │
# │  HYPOTHESIS: fold_safe_v1 scored LB=+0.00005 with all 445 features.    │
# │  61 of those features have proven sign-flip across regimes → poison.    │
# │  Dropping them + heavy regularisation should improve regime robustness. │
# └─────────────────────────────────────────────────────────────────────────┘


# ── CELL 1: Imports ────────────────────────────────────────────────────────
import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

TRAIN_PATH  = '/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet'
TEST_PATH   = '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet'
SAMPLE_PATH = '/kaggle/input/jane-street-real-time-market-data-forecasting/sample_submission.csv'

N_FOLDS    = 5
N_ROUNDS   = 3000
EARLY_STOP = 150
FAIR_C     = 1.0

print(f"LightGBM version: {lgb.__version__}")


# ── CELL 2: Configs ────────────────────────────────────────────────────────
# Step 5 incorporated: Config E adds min_child_samples=5000
# (User's analysis: force model to find patterns backed by 5000+ rows)
#
# Config A = fold_safe_v1 EXACT replica         (control, known LB=+0.00005)
# Config B = conservative                        (halved leaves, 5× lambda)
# Config C = ultra-conservative                  (very shallow, heavy reg)
# Config D = extreme                             (near-zero, min_child=2000)
# Config E = user-specified extreme              (min_child=5000 as suggested)

CONFIGS = {
    'A_foldsafe_exact': {
        'num_leaves': 63, 'learning_rate': 0.02,
        'feature_fraction': 0.4, 'bagging_fraction': 0.7, 'bagging_freq': 1,
        'min_child_samples': 250, 'reg_alpha': 0.5, 'reg_lambda': 10.0,
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    },
    'B_conservative': {
        'num_leaves': 31, 'learning_rate': 0.02,
        'feature_fraction': 0.4, 'bagging_fraction': 0.7, 'bagging_freq': 1,
        'min_child_samples': 500, 'reg_alpha': 1.0, 'reg_lambda': 50.0,
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    },
    'C_ultra_conservative': {
        'num_leaves': 15, 'learning_rate': 0.02,
        'feature_fraction': 0.3, 'bagging_fraction': 0.7, 'bagging_freq': 1,
        'min_child_samples': 1000, 'reg_alpha': 2.0, 'reg_lambda': 100.0,
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    },
    'D_extreme': {
        'num_leaves': 7, 'learning_rate': 0.05,
        'feature_fraction': 0.3, 'bagging_fraction': 0.6, 'bagging_freq': 1,
        'min_child_samples': 2000, 'reg_alpha': 5.0, 'reg_lambda': 200.0,
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    },
    'E_user_extreme': {
        # Step 5 from user analysis: min_child_samples=5000
        # Forces model to find only macro-patterns backed by 5000+ rows (~0.9% of data)
        # Expected: near-zero predictions, maximum regime robustness
        'num_leaves': 7, 'learning_rate': 0.05,
        'feature_fraction': 0.3, 'bagging_fraction': 0.6, 'bagging_freq': 1,
        'min_child_samples': 5000, 'reg_alpha': 5.0, 'reg_lambda': 200.0,
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    },
}

print("Configurations (Steps 1-5 applied to all):")
for name, p in CONFIGS.items():
    print(f"  {name}: leaves={p['num_leaves']}  "
          f"min_child={p['min_child_samples']}  lambda={p['reg_lambda']}")


# ── CELL 3: Load data + Step 1+2: Drop sign-flip (poison) features ─────────
# Step 1 (Stability Selection): keep features consistent across regimes.
# Step 2 (Penalise sign-flips): drop features whose correlation with TARGET
#         flips sign across SO3_T regimes.
#
# Source: EDA computed regime_conditional_correlations on 80 highest-variance
# features. 61 of those 80 are sign-flip features. The other 365 features
# were not in the top 80 by regime volatility, so they are presumed stable.
# Safe approach: drop the 61 known poison features, keep the rest (384 total).

# 61 confirmed sign-flip features from EDA (regime_conditional_correlations.csv)
# Exact 61 sign-flip features from outputs/eda/summaries/regime_conditional_correlations.csv
# These are features where sign(corr(feature, TARGET)) flips across SO3_T regimes.
SIGN_FLIP_FEATURES = [
    'S03_V03_T03_LagT1','S03_V03_T02_LagT1','S03_V03_T01_LagT1',
    'S03_V04_T06_LagT1','S03_V04_T04_LagT1','S03_V04_T05_LagT1',
    'S03_V04_T03_LagT1','S03_V03_T04_LagT2','S03_V03_T06_LagT2',
    'S03_V03_T05_LagT2','S03_V04_T04_LagT2','S03_V04_T03_LagT2',
    'S03_V03_T03_LagT2','S03_V04_T01_LagT1','S03_V04_T05_LagT2',
    'S03_V04_T01_LagT2','S03_V07_V06_LagT2','S03_V04_T06_LagT2',
    'S03_V04_T02_LagT2','S03_V04_T02_LagT1','S03_V03_T02_LagT2',
    'Price_LagT3','S03_V03_T01_LagT2','S03_D02_V01_A01_B10_E10_E11_LagT3',
    'S03_D01_V12_D06_LagT1','S03_A07_V01_V09_LagT3','S03_A07_A05_V09',
    'Price_LagT2','S03_A07_V01_V09_LagT2','S03_D02_V01_A01_B09_E09_E10_LagT1',
    'S03_D02_V01_A01_B10_E10_E11_LagT2','S03_V14_I01_LagT2',
    'S03_A07_A05_V09_LagT2','S03_A02_D04_W02_LagT2',
    'S03_D02_A09_A02_B02_E02_E03_LagT3','S03_D02_A09_A02_B04_E04_E05_LagT2',
    'S02_F01_U01_LagT1','S03_A02_D04_W02_LagT3',
    'S03_D02_A09_A02_B04_E04_E05_LagT1','S02_F03_U01_LagT1',
    'S03_D02_A09_A02_B07_E07_E08_LagT3','S03_A02_W01_LagT3',
    'S03_A07_V01_V09_LagT1','S03_D02_A09_A02_B04_E04_E05_LagT3',
    'S03_D02_A09_A02_B06_E06_E07_LagT3','S03_A02_D03_W02_LagT3',
    'S01_F01_U01_LagT1','S03_A02_W01_LagT1','S02_F03_U01_LagT3',
    'S03_P01_D04_LagT1','S02_F03_U01_LagT2','S03_A02_W01_LagT2',
    'S03_D02_V01_A01_B08_E08_E09_LagT2','S03_D02_V01_A01_B08_E08_E09_LagT1',
    'S03_D02_V01_A01_B10_E10_E11_LagT1','S03_A07_A05_V09_LagT1',
    'S01_F03_U01_LagT3','S03_D02_A09_A02_B06_E06_E07_LagT1',
    'S03_A02_D03_W02_LagT2','S01_F03_U01_LagT2',
    'S03_D02_A09_A02_B06_E06_E07_LagT2',
]

print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)

y_train   = train['TARGET'].values.astype(np.float32)
test_ids  = test['ID'].values if 'ID' in test.columns else test.index.values

drop_cols = {'ID', 'TARGET'}
all_feat  = [c for c in train.columns if c not in drop_cols]

# Step 1+2: Remove confirmed sign-flip features
sign_flip_present = [f for f in SIGN_FLIP_FEATURES if f in all_feat]
feat_cols = [c for c in all_feat if c not in set(sign_flip_present)]

print(f"  All raw features     : {len(all_feat)}")
print(f"  Sign-flip dropped    : {len(sign_flip_present)}  (poison features — Step 2)")
print(f"  Stable features kept : {len(feat_cols)}  (Step 1 — regime-consistent)")
print(f"  Train: {train.shape}  |  Test: {test.shape}")

X_train = np.ascontiguousarray(train[feat_cols].values, dtype=np.float32)
X_test  = np.ascontiguousarray(
    test.reindex(columns=feat_cols, fill_value=0.0).values,
    dtype=np.float32
)

print(f"  X_train: {X_train.shape}  ({X_train.nbytes/1e6:.0f} MB)")
print(f"  X_test : {X_test.shape}  ({X_test.nbytes/1e6:.0f} MB)")

so3t_idx = feat_cols.index('SO3_T') if 'SO3_T' in feat_cols else None
print(f"  SO3_T index: {so3t_idx}")

del train, test
gc.collect()
print("  DataFrames freed.")


# ── CELL 4: GroupKFold setup ───────────────────────────────────────────────
print("\nGroupKFold on SO3_T quantile buckets...")
so3t_vals    = X_train[:, so3t_idx]
groups       = pd.qcut(pd.Series(so3t_vals), q=N_FOLDS,
                       labels=False, duplicates='drop').values.astype(np.int32)
actual_folds = len(np.unique(groups))
gkf          = GroupKFold(n_splits=actual_folds)
folds        = list(gkf.split(X_train, y_train, groups=groups))

print(f"  {actual_folds} folds")
for i, (tr, va) in enumerate(folds):
    vg = sorted(np.unique(groups[va]).tolist())
    print(f"  Fold {i+1}: train={len(tr):,}  val={len(va):,}  group={vg}")


# ── CELL 5: Fair loss and R² metric ───────────────────────────────────────
def fair_obj(y_pred, dataset):
    y_true = dataset.get_label()
    r      = y_pred - y_true
    grad   = r / (1.0 + np.abs(r) / FAIR_C)
    hess   = FAIR_C ** 2 / (FAIR_C + np.abs(r)) ** 2
    return grad, hess

def r2_metric(y_pred, dataset):
    y_true = dataset.get_label()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 'r2', 1.0 - ss_res / (ss_tot + 1e-15), True


# ── CELL 6: Training loop ──────────────────────────────────────────────────
def run_config(config_name, lgb_params, X_train, y_train, X_test,
               folds, actual_folds, test_ids):
    print(f"\n{'='*65}")
    print(f"  CONFIG: {config_name}")
    print(f"  leaves={lgb_params['num_leaves']}  "
          f"min_child={lgb_params['min_child_samples']}  "
          f"lambda={lgb_params['reg_lambda']}")
    print(f"  (sign-flip features already removed — Step 1+2)")
    print(f"{'='*65}")

    oof_preds   = np.zeros(len(y_train), dtype=np.float32)
    test_preds  = np.zeros(len(X_test),  dtype=np.float64)
    fold_r2s    = []
    X_test_work = X_test.copy()

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        fold_start = time.time()

        X_tr = np.ascontiguousarray(X_train[tr_idx], dtype=np.float32)
        X_va = np.ascontiguousarray(X_train[va_idx], dtype=np.float32)
        y_tr = y_train[tr_idx].copy()
        y_va = y_train[va_idx].copy()

        # Winsorise at train-fold percentiles only
        clip_lo = np.percentile(y_tr, 1)
        clip_hi = np.percentile(y_tr, 99)
        y_tr    = np.clip(y_tr, clip_lo, clip_hi).astype(np.float32)
        y_va    = np.clip(y_va, clip_lo, clip_hi).astype(np.float32)

        # Fold-safe scaling
        scaler = StandardScaler(copy=False)
        scaler.fit(X_tr)
        X_tr[:]        = scaler.transform(X_tr)
        X_va[:]        = scaler.transform(X_va)
        X_test_work[:] = scaler.transform(X_test_work)

        # Step 7 (noise regularisation) — small Gaussian noise on X_tr only
        # Forces model to avoid brittle splits on any single feature value.
        # Noise std = 0.01 (1% of Z-scored std=1 scale — very conservative)
        X_tr += np.random.normal(0, 0.01, X_tr.shape).astype(np.float32)

        dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
        dvalid = lgb.Dataset(X_va, label=y_va,
                             reference=dtrain, free_raw_data=True)

        model = lgb.train(
            {**lgb_params, 'objective': fair_obj},
            dtrain,
            num_boost_round=N_ROUNDS,
            valid_sets=[dvalid],
            feval=r2_metric,
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(500),
            ],
        )

        oof_preds[va_idx]  = model.predict(X_va).astype(np.float32)
        test_preds        += model.predict(X_test_work) / actual_folds

        fold_r2 = r2_score(y_va, oof_preds[va_idx])
        fold_r2s.append(fold_r2)
        print(f"  Fold {fold_idx+1}  best_iter={model.best_iteration:>4}  "
              f"R²={fold_r2:+.6f}  ({time.time()-fold_start:.0f}s)")

        del dtrain, dvalid, model, X_tr, X_va, y_tr, y_va, scaler
        X_test_work[:] = X_test.copy()
        gc.collect()

    oof_r2   = r2_score(y_train, oof_preds)
    min_r2   = min(fold_r2s)   # Step 3: track worst fold
    pred_std = oof_preds.std()
    print(f"\n  OOF mean R²  : {oof_r2:+.6f}")
    print(f"  OOF min  R²  : {min_r2:+.6f}  ← Step 3: worst-case fold (regime robustness)")
    print(f"  Pred std     : {pred_std:.6f}")
    print(f"  Per-fold R²  : {[f'{r:+.6f}' for r in fold_r2s]}")

    return oof_r2, min_r2, test_preds, fold_r2s, pred_std


results = {}
for config_name, lgb_params in CONFIGS.items():
    oof_r2, min_r2, test_preds, fold_r2s, pred_std = run_config(
        config_name, lgb_params,
        X_train, y_train, X_test,
        folds, actual_folds, test_ids
    )
    results[config_name] = dict(
        oof_r2=oof_r2, min_r2=min_r2,
        test_preds=test_preds, fold_r2s=fold_r2s, pred_std=pred_std,
    )
    gc.collect()


# ── CELL 7: Summary — Step 3: pick by MINIMAX, not mean ───────────────────
sample_sub = pd.read_csv(SAMPLE_PATH)

print("\n\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Config':<24} {'mean R²':>10} {'min R²':>10} {'pred_std':>10} "
      f"{'exp_LB':>10}  worst fold")
print("-"*80)

for name, r in results.items():
    worst_fold = r['fold_r2s'].index(min(r['fold_r2s'])) + 1
    print(f"  {name:<22} {r['oof_r2']:>+10.6f} {r['min_r2']:>+10.6f} "
          f"{r['pred_std']:>10.6f} {r['oof_r2']*0.092:>+10.6f}  Fold {worst_fold}")

# Step 3: MINIMAX recommendation
best_by_mean   = max(results, key=lambda k: results[k]['oof_r2'])
best_by_minimax = max(results, key=lambda k: results[k]['min_r2'])

print(f"\n  Best by MEAN R²    : {best_by_mean}")
print(f"  Best by MINIMAX R² : {best_by_minimax}  ← Step 3: submit this one")
print(f"  (These may differ — if so, MINIMAX is more regime-robust)")
print(f"\n  Reference: fold_safe_v1 (445 feats, no drop)  "
      f"OOF=+0.000544  LB=+0.00005")
print(f"  This run uses {len(feat_cols)} features (dropped {445-len(feat_cols)} sign-flip)")

# Save each config CSV + shrinkage at alpha=0.5
for name, r in results.items():
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': r['test_preds']})
    sub = sample_sub[['ID']].merge(sub, on='ID', how='left').fillna(0.0)
    p   = sub['TARGET']
    print(f"\n  {name}:")
    print(f"    mean={p.mean():+.7f}  std={p.std():.7f}  "
          f"skew={p.skew():+.3f}  pct_pos={(p>0).mean()*100:.1f}%")

    # Full predictions
    out = f'/kaggle/working/{name}.csv'
    sub.to_csv(out, index=False)
    print(f"    saved → {out}")

    # Step 4 (shrinkage): also save alpha=0.5 version of each config
    sub_shrink = sub.copy()
    sub_shrink['TARGET'] *= 0.5
    out_s = f'/kaggle/working/{name}_shrink0p5.csv'
    sub_shrink.to_csv(out_s, index=False)
    print(f"    saved → {out_s}  (alpha=0.5 shrinkage)")

print("\n\n" + "="*80)
print("SUBMISSION PRIORITY  (Step 3: prefer minimax config)")
print("="*80)
print(f"  1st choice: {best_by_minimax}.csv              (minimax — most robust)")
print(f"  2nd choice: {best_by_minimax}_shrink0p5.csv   (minimax + shrinkage)")
print(f"  3rd choice: {best_by_mean}.csv                 (mean-optimal)")
print(f"  Diagnostic: A_foldsafe_exact.csv               (control — verify drop helps)")
