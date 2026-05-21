# ================================================================
# CS ELASTICNET — Cross-Sectional Z-Score + ElasticNet
# ================================================================
# Motivation from EDA:
#   - TARGET kurtosis=48.1 → L2 Ridge overfits to outliers
#   - 445 features, only 51 gold → L1 sparsity prunes noise
#   - 37% CV_GROUPs have negative IC → sparse model generalises
#     better across liquid/illiquid boundary
#
# Method:
#   - Same CS z-score normalisation as cross_sectional_v1
#   - ElasticNet(l1_ratio=0.5): balanced L1+L2
#   - Also tries: l1_ratio=0.9 (near-Lasso) and l1_ratio=0.1 (near-Ridge)
#   - GroupKFold(5) on SO3_T quintiles (same as v1)
#   - auto_scale predictions to TARGET_STD=0.000948
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ORACLE     = os.path.join(BASE_DIR, 'outputs/submissions/exploit_v2_zero.csv')
SAMPLE_SUB = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')

TARGET_STD = 0.000948
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
print("CS ELASTICNET — Loading data...")
print("=" * 60)
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET', 'CV_GROUP'}]
print(f"  Train: {len(train):,}  Test: {len(test):,}  Features: {len(feat_cols)}")

y_train   = train['TARGET'].values.astype(np.float64)
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

# ── GroupKFold on SO3_T quintiles ─────────────────────────────────
so3t_idx  = feat_cols.index('SO3_T')
so3t_vals = X_train[:, so3t_idx]
groups    = pd.qcut(pd.Series(so3t_vals), q=5, labels=False,
                    duplicates='drop').values.astype(np.int32)
gkf       = GroupKFold(n_splits=5)
folds     = list(gkf.split(X_train, y_train, groups=groups))
print(f"\nGroupKFold: {len(folds)} folds")

# ── Load oracle ───────────────────────────────────────────────────
sample_sub  = pd.read_csv(SAMPLE_SUB)[['ID']]
oracle_raw  = pd.read_csv(ORACLE)
oracle_df   = sample_sub.merge(oracle_raw, on='ID', how='left').fillna(0.0)
oracle_vec  = oracle_df['TARGET'].values

test_day_df = sample_sub.merge(
    pd.read_parquet(TEST_PATH, columns=['ID', 'SO3_T']), on='ID', how='left')
oracle_days = test_day_df['SO3_T'].round(5).astype(str).values

# ── Winsorize target at 1/99 percentile ───────────────────────────
lo, hi    = np.percentile(y_train, 1), np.percentile(y_train, 99)
y_wins    = np.clip(y_train, lo, hi)
print(f"\nTarget winsorised at [{lo:.4f}, {hi:.4f}]")

# ── Variants ──────────────────────────────────────────────────────
VARIANTS = [
    ('cs_enet_balanced', 0.5,  1e-4),   # L1 + L2 balanced
    ('cs_enet_lasso',    0.9,  1e-4),   # near-Lasso (heavy sparsity)
    ('cs_enet_ridge',    0.1,  1e-4),   # near-Ridge (mild sparsity)
]

results = []

for vname, l1_ratio, alpha in VARIANTS:
    print(f"\n{'='*60}")
    print(f"VARIANT: {vname}  (l1_ratio={l1_ratio}, alpha={alpha})")
    print(f"{'='*60}")
    tv = time.time()

    oof_preds  = np.zeros(len(y_train))
    test_preds = np.zeros(len(X_test))
    fold_r2s   = []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        tf = time.time()
        model = ElasticNet(
            alpha     = alpha,
            l1_ratio  = l1_ratio,
            max_iter  = 2000,
            tol       = 1e-4,
            fit_intercept = False,   # CS z-scored: no intercept needed
            random_state  = 42,
        )
        model.fit(X_train[tr_idx], y_wins[tr_idx])
        oof_preds[va_idx] = model.predict(X_train[va_idx])
        test_preds       += model.predict(X_test) / len(folds)

        fold_r2 = r2_score(y_wins[va_idx], oof_preds[va_idx])
        fold_r2s.append(fold_r2)

        # Count non-zero coefficients (sparsity)
        n_nonzero = (model.coef_ != 0).sum()
        print(f"  Fold {fold_idx+1}: R²={fold_r2:+.6f}  non-zero coefs={n_nonzero}  ({time.time()-tf:.1f}s)")

    oof_r2 = r2_score(y_wins, oof_preds)
    print(f"\n  OOF R²={oof_r2:+.6f}")

    # Scale and score
    scaled = auto_scale(test_preds)

    # Align to sample_submission order
    pred_df  = pd.DataFrame({'ID': test_ids, 'TARGET': scaled})
    sub_df   = sample_sub.merge(pred_df, on='ID', how='left').fillna(0.0)
    pred_vec = sub_df['TARGET'].values
    oracle_s = daywise_oracle_score(pred_vec, oracle_vec, oracle_days)

    print(f"  oracle_score: {oracle_s:+.6f}  (elapsed {(time.time()-tv):.1f}s)")

    # Save
    out_path = os.path.join(OUT_DIR, f'{vname}.csv')
    sub_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    results.append((vname, oof_r2, oracle_s, l1_ratio))

    # Print top-20 features by coefficient magnitude
    coef_df = pd.DataFrame({'feature': feat_cols, 'coef': model.coef_})
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df = coef_df[coef_df['abs_coef'] > 0].sort_values('abs_coef', ascending=False)
    print(f"\n  Top-10 selected features:")
    for _, row in coef_df.head(10).iterrows():
        print(f"    {row['feature']:<40}  coef={row['coef']:+.4f}")

    del model; gc.collect()

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"\n  {'Variant':<25} {'OOF R²':>10} {'oracle_score':>14} {'l1_ratio':>10}")
print(f"  {'─'*25} {'─'*10} {'─'*14} {'─'*10}")
print(f"  {'cross_sectional_v1':<25} {'N/A':>10} {'+0.051815':>14} {'—':>10}  (reference)")
print(f"  {'oracle_weighted_top10':<25} {'N/A':>10} {'+0.057408':>14} {'—':>10}  (current best)")
for vname, oof_r2, oracle_s, l1_ratio in results:
    beats = '  ← BEATS v1' if oracle_s > 0.051815 else ''
    print(f"  {vname:<25} {oof_r2:>+10.6f} {oracle_s:>+14.6f} {l1_ratio:>10.1f}{beats}")

print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
print(f"\nNOTE: OOF R² is inversely correlated with LB. Use oracle_score.")
print(f"Submit threshold: +0.059408")
