# ================================================================
# FOLD-SAFE REGULARISATION + OPTIMISATION SEARCH
# ================================================================
# Identical architecture to fold_safe_v1 (which gave LB = +0.00005):
#   - ALL 445 features (no dropping)
#   - GroupKFold(5) on SO3_T quintiles
#   - StandardScaler fitted inside fold (anti-leakage)
#   - Fair Loss objective (c=1.0)
#   - Winsorisation at train-fold p1/p99
#
# What we're searching:
#   - num_leaves       : tree complexity
#   - min_child_samples: minimum leaf population
#   - reg_lambda       : L2 regularisation
#   - reg_alpha        : L1 regularisation
#   - feature_fraction : column sub-sampling
#   - learning_rate    : step size (with more rounds for slower lr)
#
# Goal: find config whose OOF R² ≈ fold_safe_v1 (+0.000544) but with
#       lower pred_std or better minimax fold score.
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

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS   = 5
FAIR_C    = 1.0

# ── Config grid ────────────────────────────────────────────────────────────
# A = fold_safe_v1 EXACT replica (control — must reproduce OOF ≈ +0.000544)
# B = halved leaves, more regularisation
# C = ultra-conservative (very shallow + heavy L2)
# D = slower learning rate (lr=0.01, more rounds) — optimisation axis
# E = aggressive col sampling (feature_fraction=0.2)
# F = high min_child_samples (large-leaf regime)
# G = combined best guesses across axes

CONFIGS = {
    'A_exact_foldsafe': {
        # fold_safe_v1 exact params — LB=+0.00005 known
        'num_leaves'        : 63,
        'learning_rate'     : 0.02,
        'n_estimators'      : 3000,
        'early_stopping'    : 150,
        'feature_fraction'  : 0.4,
        'bagging_fraction'  : 0.7,
        'bagging_freq'      : 1,
        'min_child_samples' : 250,
        'reg_alpha'         : 0.5,
        'reg_lambda'        : 10.0,
        'n_jobs'            : -1,
        'verbose'           : -1,
        'random_state'      : 42,
    },
    'B_more_reg': {
        # Halve leaves, 5x lambda → reduces overfitting
        'num_leaves'        : 31,
        'learning_rate'     : 0.02,
        'n_estimators'      : 3000,
        'early_stopping'    : 150,
        'feature_fraction'  : 0.4,
        'bagging_fraction'  : 0.7,
        'bagging_freq'      : 1,
        'min_child_samples' : 500,
        'reg_alpha'         : 1.0,
        'reg_lambda'        : 50.0,
        'n_jobs'            : -1,
        'verbose'           : -1,
        'random_state'      : 42,
    },
    'C_ultra_conservative': {
        # Very shallow trees, heavy L2
        'num_leaves'        : 15,
        'learning_rate'     : 0.02,
        'n_estimators'      : 3000,
        'early_stopping'    : 150,
        'feature_fraction'  : 0.3,
        'bagging_fraction'  : 0.7,
        'bagging_freq'      : 1,
        'min_child_samples' : 1000,
        'reg_alpha'         : 2.0,
        'reg_lambda'        : 100.0,
        'n_jobs'            : -1,
        'verbose'           : -1,
        'random_state'      : 42,
    },
    'D_slow_lr': {
        # Slower learning rate (0.01 vs 0.02) + more rounds — optimisation axis
        # More granular gradient steps, often improves generalisation
        'num_leaves'        : 63,
        'learning_rate'     : 0.01,
        'n_estimators'      : 5000,
        'early_stopping'    : 200,
        'feature_fraction'  : 0.4,
        'bagging_fraction'  : 0.7,
        'bagging_freq'      : 1,
        'min_child_samples' : 250,
        'reg_alpha'         : 0.5,
        'reg_lambda'        : 10.0,
        'n_jobs'            : -1,
        'verbose'           : -1,
        'random_state'      : 42,
    },
    'E_aggressive_col_sample': {
        # Lower feature_fraction → less correlated trees → better ensemble
        'num_leaves'        : 63,
        'learning_rate'     : 0.02,
        'n_estimators'      : 3000,
        'early_stopping'    : 150,
        'feature_fraction'  : 0.2,
        'bagging_fraction'  : 0.6,
        'bagging_freq'      : 1,
        'min_child_samples' : 250,
        'reg_alpha'         : 0.5,
        'reg_lambda'        : 10.0,
        'n_jobs'            : -1,
        'verbose'           : -1,
        'random_state'      : 42,
    },
    'F_large_leaves': {
        # High min_child_samples — only macro patterns
        'num_leaves'        : 31,
        'learning_rate'     : 0.02,
        'n_estimators'      : 3000,
        'early_stopping'    : 150,
        'feature_fraction'  : 0.4,
        'bagging_fraction'  : 0.7,
        'bagging_freq'      : 1,
        'min_child_samples' : 2000,
        'reg_alpha'         : 1.0,
        'reg_lambda'        : 50.0,
        'n_jobs'            : -1,
        'verbose'           : -1,
        'random_state'      : 42,
    },
    'G_combined': {
        # Combines: slow lr + shallow leaves + low feature_fraction + heavy reg
        # Best-guess combination from axes B, D, E
        'num_leaves'        : 31,
        'learning_rate'     : 0.01,
        'n_estimators'      : 5000,
        'early_stopping'    : 200,
        'feature_fraction'  : 0.3,
        'bagging_fraction'  : 0.7,
        'bagging_freq'      : 1,
        'min_child_samples' : 500,
        'reg_alpha'         : 1.0,
        'reg_lambda'        : 50.0,
        'n_jobs'            : -1,
        'verbose'           : -1,
        'random_state'      : 42,
    },
}

# ── Fair loss + R² metric ───────────────────────────────────────────────────
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

# ── Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)

y_train  = train['TARGET'].values.astype(np.float32)
test_ids = test['ID'].values
feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET'}]

X_train = np.ascontiguousarray(train[feat_cols].values, dtype=np.float32)
X_test  = np.ascontiguousarray(
    test.reindex(columns=feat_cols, fill_value=0.0).values, dtype=np.float32
)
print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"  Features: {len(feat_cols)} (ALL — identical to fold_safe_v1)")

del train, test
gc.collect()

# ── GroupKFold on SO3_T quintiles ───────────────────────────────────────────
so3t_idx  = feat_cols.index('SO3_T')
so3t_vals = X_train[:, so3t_idx]
groups    = pd.qcut(pd.Series(so3t_vals), q=N_FOLDS,
                    labels=False, duplicates='drop').values.astype(np.int32)
n_folds   = len(np.unique(groups))
gkf       = GroupKFold(n_splits=n_folds)
folds     = list(gkf.split(X_train, y_train, groups=groups))

print(f"\nGroupKFold: {n_folds} folds on SO3_T quintiles")
for i, (tr, va) in enumerate(folds):
    print(f"  Fold {i+1}: train={len(tr):,}  val={len(va):,}  "
          f"group={sorted(np.unique(groups[va]).tolist())}")

# ── Training loop ───────────────────────────────────────────────────────────
def run_config(name, cfg, X_train, y_train, X_test, folds, n_folds, test_ids):
    n_est    = cfg['n_estimators']
    es       = cfg['early_stopping']
    lgb_p    = {k: v for k, v in cfg.items()
                if k not in {'n_estimators', 'early_stopping'}}

    print(f"\n{'='*68}")
    print(f"  CONFIG: {name}")
    print(f"  leaves={cfg['num_leaves']}  lr={cfg['learning_rate']}  "
          f"min_child={cfg['min_child_samples']}  "
          f"lambda={cfg['reg_lambda']}  ff={cfg['feature_fraction']}")
    print(f"{'='*68}")

    oof_preds  = np.zeros(len(y_train), dtype=np.float32)
    test_preds = np.zeros(len(X_test),  dtype=np.float64)
    fold_r2s   = []
    X_test_work = X_test.copy()

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        t0 = time.time()

        X_tr = np.ascontiguousarray(X_train[tr_idx], dtype=np.float32)
        X_va = np.ascontiguousarray(X_train[va_idx], dtype=np.float32)
        y_tr = y_train[tr_idx].copy()
        y_va = y_train[va_idx].copy()

        # Anti-leakage winsorisation: bounds from y_tr only
        lo, hi = np.percentile(y_tr, 1), np.percentile(y_tr, 99)
        y_tr   = np.clip(y_tr, lo, hi).astype(np.float32)
        y_va   = np.clip(y_va, lo, hi).astype(np.float32)

        # Anti-leakage scaling: scaler fitted on X_tr only
        scaler = StandardScaler(copy=False)
        scaler.fit(X_tr)
        X_tr[:]         = scaler.transform(X_tr)
        X_va[:]         = scaler.transform(X_va)
        X_test_work[:]  = scaler.transform(X_test_work)

        dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, free_raw_data=True)

        model = lgb.train(
            {**lgb_p, 'objective': fair_obj},
            dtrain,
            num_boost_round=n_est,
            valid_sets=[dvalid],
            feval=r2_metric,
            callbacks=[
                lgb.early_stopping(es, verbose=False),
                lgb.log_evaluation(500),
            ],
        )

        oof_preds[va_idx] = model.predict(X_va).astype(np.float32)
        test_preds       += model.predict(X_test_work) / n_folds

        fold_r2 = r2_score(y_va, oof_preds[va_idx])
        fold_r2s.append(fold_r2)
        print(f"  Fold {fold_idx+1}  best_iter={model.best_iteration:>5}  "
              f"R²={fold_r2:+.6f}  ({time.time()-t0:.0f}s)")

        # Restore X_test_work for next fold
        del dtrain, dvalid, model, X_tr, X_va, y_tr, y_va, scaler
        X_test_work[:] = X_test.copy()
        gc.collect()

    oof_r2   = r2_score(y_train, oof_preds)
    min_r2   = min(fold_r2s)
    pred_std = oof_preds.std()
    print(f"\n  OOF mean R²  : {oof_r2:+.6f}")
    print(f"  OOF min  R²  : {min_r2:+.6f}  (worst fold — minimax metric)")
    print(f"  Pred std     : {pred_std:.6f}")
    print(f"  Per-fold     : {[f'{r:+.6f}' for r in fold_r2s]}")

    del X_test_work
    gc.collect()
    return oof_r2, min_r2, test_preds, fold_r2s, pred_std


results = {}
for name, cfg in CONFIGS.items():
    oof_r2, min_r2, test_preds, fold_r2s, pred_std = run_config(
        name, cfg, X_train, y_train, X_test, folds, n_folds, test_ids
    )
    results[name] = dict(
        oof_r2=oof_r2, min_r2=min_r2,
        test_preds=test_preds, fold_r2s=fold_r2s, pred_std=pred_std,
    )
    gc.collect()


# ── Summary ─────────────────────────────────────────────────────────────────
print("\n\n" + "="*80)
print("SUMMARY  (reference: fold_safe_v1  OOF=+0.000544  LB=+0.00005  std=0.000624)")
print("="*80)
print(f"{'Config':<26} {'OOF R²':>10} {'min R²':>10} {'pred_std':>10} "
      f"{'exp_LB(9.2%)':>13}  worst_fold")
print("-"*80)

for name, r in results.items():
    worst = r['fold_r2s'].index(min(r['fold_r2s'])) + 1
    better = '✓' if r['oof_r2'] > 0.000544 else ' '
    print(f"  {name:<24} {r['oof_r2']:>+10.6f} {r['min_r2']:>+10.6f} "
          f"{r['pred_std']:>10.6f} {r['oof_r2']*0.092:>+13.6f} {better}  Fold {worst}")

best_mean    = max(results, key=lambda k: results[k]['oof_r2'])
best_minimax = max(results, key=lambda k: results[k]['min_r2'])
best_std_match = min(results, key=lambda k: abs(results[k]['pred_std'] - 0.000624))

# ── Composite score ─────────────────────────────────────────────────────────
# score = oof_r2
#         - 4 * |pred_std - 0.0006|   penalises amplitude drift from fold_safe_v1
#         - 2 * |pct_pos  - 0.5|      penalises directional bias
#         + 2 * min_r2                rewards regime robustness (minimax)
#
# fold_safe_v1 reference score:
#   oof_r2=+0.000544  std=0.000624  pct_pos≈0.491  min_r2=unknown
#   score ≈ 0.000544 - 4*|0.000624-0.0006| - 2*|0.491-0.5| + 0 ≈ +0.000296
print(f"\n\n{'Config':<26} {'score':>10}  (components)")
print("-"*80)

composite_scores = {}
for name, r in results.items():
    # need pct_pos from test_preds — use oof_preds as proxy
    pct_pos_oof = (r['test_preds'] > 0).mean()
    score = (
          r['oof_r2']
        - 4 * abs(r['pred_std'] - 0.0006)
        - 2 * abs(pct_pos_oof   - 0.5)
        + 2 * r['min_r2']
    )
    composite_scores[name] = score
    print(f"  {name:<24} {score:>+10.6f}  "
          f"(oof={r['oof_r2']:+.6f}  std_pen={4*abs(r['pred_std']-0.0006):.6f}  "
          f"dir_pen={2*abs(pct_pos_oof-0.5):.6f}  minimax={2*r['min_r2']:+.6f})")

best_composite = max(composite_scores, key=composite_scores.get)
# fold_safe_v1 reference composite (min_r2 unknown → 0)
fs_ref_score = 0.000544 - 4*abs(0.000624-0.0006) - 2*abs(0.491-0.5) + 0.0
print(f"\n  Best composite score: {best_composite}  ({composite_scores[best_composite]:+.6f})")
print(f"  fold_safe_v1 reference composite ≈ {fs_ref_score:+.6f}  (min_r2 unknown → 0)")

print(f"\n  Best by OOF mean R²  : {best_mean}")
print(f"  Best by minimax R²   : {best_minimax}")
print(f"  Best by composite    : {best_composite}")
print(f"  Closest std to 0.000624: {best_std_match}")
print(f"\n  Reference: fold_safe_v1  OOF=+0.000544  LB=+0.00005  std=0.000624")


# ── Save CSVs ───────────────────────────────────────────────────────────────
# Load sample submission to align IDs
sample_sub_path = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
if os.path.exists(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path)
    id_col = sample_sub[['ID']]
else:
    id_col = pd.DataFrame({'ID': test_ids})

print("\n\nSAVING CSVs:")
print("-"*60)

SHRINK_ALPHAS = [1.0, 0.5, 0.3]

for name, r in results.items():
    sub_base = pd.DataFrame({'ID': test_ids, 'TARGET': r['test_preds']})
    sub_base = id_col.merge(sub_base, on='ID', how='left').fillna(0.0)
    t = sub_base['TARGET']

    print(f"\n  {name}:")
    print(f"    OOF={r['oof_r2']:+.6f}  min={r['min_r2']:+.6f}  "
          f"std={t.std():.6f}  pct_pos={(t>0).mean()*100:.1f}%")

    for alpha in SHRINK_ALPHAS:
        s = sub_base.copy()
        s['TARGET'] *= alpha
        suffix = '' if alpha == 1.0 else f'_shrink{str(alpha).replace(".", "p")}'
        out = os.path.join(OUT_DIR, f'fs_{name}{suffix}.csv')
        s.to_csv(out, index=False)
        tag = f'  (alpha={alpha})' if alpha < 1.0 else ''
        print(f"    saved → fs_{name}{suffix}.csv{tag}")


# ── Submission priority ──────────────────────────────────────────────────────
print("\n\n" + "="*80)
print("SUBMISSION PRIORITY")
print("="*80)
print(f"  1. fs_{best_composite}.csv            ← best composite score (primary)")
print(f"  2. fs_{best_composite}_shrink0p5.csv  ← composite winner + shrinkage")
print(f"  3. fs_{best_mean}.csv                 ← best OOF mean")
print(f"  4. fs_{best_minimax}.csv              ← best minimax (regime-robust)")
print(f"\n  COMPARE: fold_safe_v1  OOF=+0.000544  LB=+0.00005  std=0.000624")
print(f"  If composite score > {fs_ref_score:+.6f} (fold_safe_v1 ref) → strong submit candidate.")
print(f"  If any config has OOF > 0.000544 AND std ≤ 0.000624 → submit immediately.")
