# ================================================================
# LINEAR MODEL PIPELINE — Ridge + ElasticNet
# ================================================================
# Identical anti-leakage architecture to fold_safe_v1:
#   - ALL 445 features
#   - GroupKFold(5) on SO3_T quintiles
#   - StandardScaler fitted inside fold only
#   - Winsorisation at train-fold p1/p99
#
# Why linear models after LightGBM failures:
#   - Trees learn regime-specific thresholds → fail under dist shift
#   - Linear models have weaker inductive bias → less regime overfitting
#   - Ridge/ElasticNet regularisation is mathematically cleaner than
#     LGB's min_child_samples / lambda combination
#   - No early stopping, no boosting rounds to overfit
#
# Configs:
#   Ridge_A  : light regularisation (alpha=1)
#   Ridge_B  : medium regularisation (alpha=10)
#   Ridge_C  : heavy regularisation (alpha=100)
#   Ridge_D  : very heavy (alpha=1000)
#   EN_A     : ElasticNet l1_ratio=0.1 (mostly Ridge, some L1)
#   EN_B     : ElasticNet l1_ratio=0.5 (balanced)
#   EN_C     : ElasticNet l1_ratio=0.9 (mostly Lasso — aggressive selection)
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS = 5

# ── Config grid ─────────────────────────────────────────────────────────────
CONFIGS = {
    'Ridge_A_alpha1':    {'model': 'ridge',   'alpha': 1.0,   'l1_ratio': None},
    'Ridge_B_alpha10':   {'model': 'ridge',   'alpha': 10.0,  'l1_ratio': None},
    'Ridge_C_alpha100':  {'model': 'ridge',   'alpha': 100.0, 'l1_ratio': None},
    'Ridge_D_alpha1000': {'model': 'ridge',   'alpha': 1000.0,'l1_ratio': None},
    'EN_A_l1r01':        {'model': 'elasticnet', 'alpha': 0.001, 'l1_ratio': 0.1},
    'EN_B_l1r05':        {'model': 'elasticnet', 'alpha': 0.001, 'l1_ratio': 0.5},
    'EN_C_l1r09':        {'model': 'elasticnet', 'alpha': 0.001, 'l1_ratio': 0.9},
}

# ── Load data ────────────────────────────────────────────────────────────────
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
print(f"  Features: {len(feat_cols)}")

del train, test
gc.collect()

# ── GroupKFold on SO3_T quintiles (identical to fold_safe_v1) ────────────────
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


# ── Training loop ─────────────────────────────────────────────────────────────
def build_model(cfg):
    if cfg['model'] == 'ridge':
        return Ridge(alpha=cfg['alpha'], fit_intercept=True)
    else:
        return ElasticNet(
            alpha=cfg['alpha'],
            l1_ratio=cfg['l1_ratio'],
            fit_intercept=True,
            max_iter=5000,
            tol=1e-4,
        )


def run_config(name, cfg, X_train, y_train, X_test, folds, n_folds, test_ids):
    print(f"\n{'='*65}")
    print(f"  CONFIG: {name}")
    if cfg['model'] == 'ridge':
        print(f"  Ridge  alpha={cfg['alpha']}")
    else:
        print(f"  ElasticNet  alpha={cfg['alpha']}  l1_ratio={cfg['l1_ratio']}")
    print(f"{'='*65}")

    oof_preds   = np.zeros(len(y_train), dtype=np.float64)
    test_preds  = np.zeros(len(X_test),  dtype=np.float64)
    fold_r2s    = []
    X_test_work = X_test.copy().astype(np.float64)

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        t0 = time.time()

        X_tr = X_train[tr_idx].astype(np.float64)
        X_va = X_train[va_idx].astype(np.float64)
        y_tr = y_train[tr_idx].astype(np.float64)
        y_va = y_train[va_idx].astype(np.float64)

        # Anti-leakage winsorisation: bounds from y_tr only
        lo, hi = np.percentile(y_tr, 1), np.percentile(y_tr, 99)
        y_tr   = np.clip(y_tr, lo, hi)
        y_va   = np.clip(y_va, lo, hi)

        # Anti-leakage scaling: scaler fitted on X_tr only
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_va   = scaler.transform(X_va)
        X_test_work[:] = scaler.transform(X_test_work)

        model = build_model(cfg)
        model.fit(X_tr, y_tr)

        oof_preds[va_idx] = model.predict(X_va)
        test_preds       += model.predict(X_test_work) / n_folds

        fold_r2 = r2_score(y_va, oof_preds[va_idx])
        fold_r2s.append(fold_r2)

        n_nonzero = np.sum(np.abs(model.coef_) > 1e-10) if hasattr(model, 'coef_') else 'N/A'
        print(f"  Fold {fold_idx+1}  R²={fold_r2:+.6f}  "
              f"nonzero_coefs={n_nonzero}  ({time.time()-t0:.1f}s)")

        del X_tr, X_va, y_tr, y_va, model, scaler
        X_test_work[:] = X_test.copy()
        gc.collect()

    oof_r2   = r2_score(y_train, oof_preds)
    min_r2   = min(fold_r2s)
    pred_std = oof_preds.std()

    print(f"\n  OOF mean R²  : {oof_r2:+.6f}")
    print(f"  OOF min  R²  : {min_r2:+.6f}  (worst fold)")
    print(f"  OOF pred std : {pred_std:.6f}")
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


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n\n" + "="*80)
print("SUMMARY  (reference: fold_safe_v1  OOF=+0.000544  LB=+0.00005  std=0.000624)")
print("="*80)
print(f"{'Config':<26} {'OOF R²':>10} {'min R²':>10} {'pred_std':>10} {'score':>10}  beat_ref")
print("-"*80)

ref_oof = 0.000544

composite_scores = {}
for name, r in results.items():
    pct_pos = (r['test_preds'] > 0).mean()
    score = (
          r['oof_r2']
        - 4 * abs(r['pred_std'] - 0.0006)
        - 2 * abs(pct_pos - 0.5)
        + 2 * r['min_r2']
    )
    composite_scores[name] = score
    beat = '✓' if r['oof_r2'] > ref_oof else ' '
    worst = r['fold_r2s'].index(min(r['fold_r2s'])) + 1
    print(f"  {name:<24} {r['oof_r2']:>+10.6f} {r['min_r2']:>+10.6f} "
          f"{r['pred_std']:>10.6f} {score:>+10.6f} {beat}  Fold {worst}")

best_oof       = max(results, key=lambda k: results[k]['oof_r2'])
best_minimax   = max(results, key=lambda k: results[k]['min_r2'])
best_composite = max(composite_scores, key=composite_scores.get)

print(f"\n  Best OOF       : {best_oof}")
print(f"  Best minimax   : {best_minimax}")
print(f"  Best composite : {best_composite}")
print(f"\n  Reference: fold_safe_v1  OOF=+0.000544  LB=+0.00005  std=0.000624")


# ── Save CSVs ─────────────────────────────────────────────────────────────────
sample_sub_path = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
id_col = pd.read_csv(sample_sub_path)[['ID']] if os.path.exists(sample_sub_path) \
         else pd.DataFrame({'ID': test_ids})

print("\n\nSAVING CSVs:")
print("-"*60)

for name, r in results.items():
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': r['test_preds']})
    sub = id_col.merge(sub, on='ID', how='left').fillna(0.0)
    t   = sub['TARGET']

    print(f"\n  {name}:")
    print(f"    OOF={r['oof_r2']:+.6f}  min={r['min_r2']:+.6f}  "
          f"std={t.std():.6f}  pct_pos={(t>0).mean()*100:.1f}%  skew={t.skew():+.3f}")

    for alpha in [1.0, 0.5, 0.3]:
        s      = sub.copy()
        s['TARGET'] *= alpha
        suffix = '' if alpha == 1.0 else f'_shrink{str(alpha).replace(".", "p")}'
        path   = os.path.join(OUT_DIR, f'lin_{name}{suffix}.csv')
        s.to_csv(path, index=False)
        tag    = f'  (alpha={alpha})' if alpha < 1.0 else ''
        print(f"    saved → lin_{name}{suffix}.csv{tag}")


# ── Submission priority ───────────────────────────────────────────────────────
print("\n\n" + "="*80)
print("SUBMISSION PRIORITY")
print("="*80)
print(f"  1. lin_{best_composite}.csv          ← best composite score")
print(f"  2. lin_{best_oof}.csv                ← best raw OOF")
print(f"  3. lin_{best_minimax}.csv            ← best minimax (regime-robust)")
print(f"\n  If any lin_ CSV has OOF > +0.000544 AND std ≈ 0.000624 → top priority submit.")
print(f"  Linear models scoring better than fold_safe_v1 OOF = paradigm shift confirmed.")
