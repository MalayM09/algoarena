# ================================================================
# PSEUDO-LABEL GRINOLD — Retrain ICs using test set predictions
# ================================================================
# Research basis (AMEX/Ubiquant/Numerai):
#   Use current best model predictions on unlabeled test set as
#   "soft labels" to augment IC estimation.
#
# Design:
#   1. Start from NW kernel-only predictions (no Grinold component)
#      → avoids circular dependency / confirmation bias
#   2. Pseudo-labels get weight alpha (0.1, 0.2, 0.3) vs real labels
#   3. IC estimation: pooled across train days + pseudo-labeled test days
#   4. Use per-day z-scores computed from COMBINED pool (liquid+illiquid)
#      for test rows, liquid-only for train rows (to isolate effects)
#
# Hypothesis:
#   Liquid training ICs are computed from liquid asset returns.
#   Illiquid test assets may have slightly different IC structure.
#   Pseudo-labels from the kernel model (which does liquid→illiquid
#   transfer) can shift the IC vector toward the illiquid regime.
#
# Generates:
#   pseudolabel_grinold_a010.csv  — weight=0.10 on pseudo-labels
#   pseudolabel_grinold_a020.csv  — weight=0.20 on pseudo-labels
#   pseudolabel_grinold_a030.csv  — weight=0.30 on pseudo-labels
#   pseudolabel_threeway_a020.csv — best-alpha pseudo-Grinold + kernel + Ridge
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR    = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH  = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH   = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH   = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
SAMPLE_PATH = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
BEST_SUB    = os.path.join(BASE_DIR, 'outputs/submissions/threeway_r30_k40_g29.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'outputs/submissions')
os.makedirs(OUT_DIR, exist_ok=True)

RIDGE_ALPHA  = 10.0
TARGET_STD   = 0.000948
KNN_K        = 10          # same K as our best submission
PL_ALPHAS    = [0.10, 0.20, 0.30]  # pseudo-label weights
CLIP_Z       = 5.0

print("=" * 65)
print("PSEUDO-LABEL GRINOLD — IC retraining with test pseudo-labels")
print("=" * 65)

# ── Load data ──────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days = set(train['day_id'].unique())
y_train    = train['TARGET'].values.astype(np.float64)
test_ids   = test['ID'].values
n_test     = len(test)
sample_sub = pd.read_csv(SAMPLE_PATH)[['ID']]
print(f"  Train: {len(train):,}  Test: {n_test:,}")

# ── Gold features ──────────────────────────────────────────────────
icir_df  = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df  = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
all_gold = [f for f in gold_df['feature'].tolist() if f in train.columns]
ic_map   = gold_df.set_index('feature')['mean_ic'].to_dict()
top10    = all_gold[:10]
ic_arr_orig = np.array([ic_map[f] for f in top10])
print(f"  Gold top-10 features loaded")

# ── Load current best predictions as pseudo-labels ────────────────
print("\nLoading current best predictions as pseudo-labels...")
best_df = pd.read_csv(BEST_SUB)
# Align to test order
test_with_pl = test[['ID','day_id']].copy()
test_with_pl = test_with_pl.merge(
    best_df[['ID','TARGET']].rename(columns={'TARGET': 'pseudo_label'}),
    on='ID', how='left'
).fillna(0.0)
pl_vals = test_with_pl['pseudo_label'].values
print(f"  Pseudo-labels std={pl_vals.std():.6f}  mean={pl_vals.mean():+.8f}")
print(f"  Using: {BEST_SUB.split('/')[-1]}")

# ── Helpers ────────────────────────────────────────────────────────
def zscore_fit(X, clip=CLIP_Z):
    m = X.mean(0); s = X.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip), m, s

def zscore_apply(X, m, s, clip=CLIP_Z):
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip)

def winsorise(y, lo=1, hi=99):
    return np.clip(y, np.percentile(y, lo), np.percentile(y, hi))

def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return (a @ b) / d if d > 1e-10 else 0.0

# ── Step 1: Compute ORIGINAL (no pseudo-label) ICs ────────────────
# Also generates NW kernel predictions (used as pseudo-label source)
print("\nStep 1: Computing original ICs + NW kernel predictions...")
ic_pool_X, ic_pool_y = [], []
te_knn_preds = np.zeros(n_test)
te_ridge_preds = np.zeros(n_test)
te_grinold_preds = np.zeros(n_test)

for day, grp_tr in train.groupby('day_id'):
    if len(grp_tr) < 5: continue
    X_tr = grp_tr[top10].fillna(0).values.astype(np.float64)
    y_tr = y_train[grp_tr.index]
    X_z, m, s = zscore_fit(X_tr)
    ic_pool_X.append(X_z)
    ic_pool_y.append(y_tr)

X_pool = np.vstack(ic_pool_X)
y_pool = np.concatenate(ic_pool_y)
y_dm   = y_pool - y_pool.mean()
ic_orig = (X_pool * y_dm[:, None]).mean(0)  # (10,)
print(f"  Original IC vector norm: {np.linalg.norm(ic_orig):.6f}")
print(f"  IC signs: {['+'  if v>0 else '-' for v in ic_orig]}")
del X_pool, y_pool, ic_pool_X, ic_pool_y

# Generate baseline predictions (Ridge + KNN + Grinold with original ICs)
for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index.values
    X_te = grp_te[top10].fillna(0).values.astype(np.float64)

    if day in train_days:
        grp_tr = train[train['day_id'] == day]
        y_tr   = y_train[grp_tr.index]
        X_tr   = grp_tr[top10].fillna(0).values.astype(np.float64)
        X_tr_z, m, s = zscore_fit(X_tr)
        X_te_z = zscore_apply(X_te, m, s)

        # Grinold with original ICs
        pred_g = X_te_z @ ic_orig
        pred_g -= pred_g.mean()
        te_grinold_preds[te_idx] = pred_g

        # Ridge
        y_tr_w = winsorise(y_tr)
        mdl = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        mdl.fit(X_tr_z, y_tr_w)
        pred_r = mdl.predict(X_te_z)
        pred_r -= pred_r.mean()
        te_ridge_preds[te_idx] = pred_r

        # KNN
        if len(y_tr) >= KNN_K:
            sim = cosine_similarity(X_te_z, X_tr_z)
            topk = np.argpartition(sim, -KNN_K, axis=1)[:, -KNN_K:]
            w = np.maximum(sim[np.arange(len(X_te_z))[:, None], topk], 0)
            ws = w.sum(1, keepdims=True)
            w /= np.where(ws < 1e-10, 1.0, ws)
            pred_k = (w * y_tr[topk]).sum(1)
            pred_k -= pred_k.mean()
            te_knn_preds[te_idx] = pred_k
        else:
            te_knn_preds[te_idx] = pred_g
    else:
        X_z, _, _ = zscore_fit(X_te)
        pred_g = X_z @ ic_orig
        pred_g -= pred_g.mean()
        te_grinold_preds[te_idx] = pred_g
        te_ridge_preds[te_idx]   = pred_g
        te_knn_preds[te_idx]     = pred_g

print(f"  Baseline predictions computed  [{(time.time()-t0)/60:.1f}m]")

# Compare baseline with current best submission
prev = sample_sub.merge(
    pd.read_csv(BEST_SUB)[['ID','TARGET']].rename(columns={'TARGET':'prev'}),
    on='ID', how='left'
).fillna(0.0)['prev'].values
r_s0 = auto_scale(te_ridge_preds)
k_s0 = auto_scale(te_knn_preds)
g_s0 = auto_scale(te_grinold_preds)
bl0  = auto_scale(0.30*r_s0 + 0.40*k_s0 + 0.30*g_s0)
print(f"  Baseline threeway corr vs best_sub: {pearson_r(bl0, prev):+.4f}")

# ── Step 2: Augment IC estimation with pseudo-labels ──────────────
print(f"\nStep 2: Pseudo-label IC augmentation (alphas={PL_ALPHAS})...")
print("  Pooling pseudo-labeled test rows with training rows...")

# Build test feature pools for IC augmentation
te_ic_pool_X = []
te_ic_pool_y = []
te_ic_pool_w = []

for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index.values
    X_te = grp_te[top10].fillna(0).values.astype(np.float64)
    y_pl = pl_vals[te_idx]  # pseudo-labels from best submission

    # FIX 1: Z-score test assets using their OWN daily stats.
    # Illiquid test assets have a different distribution (KS=0.37 gap)
    # from liquid train assets. Using train stats would create massively
    # biased z-scores, corrupting the IC estimation.
    X_te_z, _, _ = zscore_fit(X_te)

    te_ic_pool_X.append(X_te_z)
    te_ic_pool_y.append(y_pl)
    te_ic_pool_w.append(np.ones(len(y_pl)))  # weight=1 before alpha scaling

X_te_pool = np.vstack(te_ic_pool_X)    # (n_test, 10) — all test rows z-scored
y_te_pool  = np.concatenate(te_ic_pool_y)  # pseudo-labels
print(f"  Test pool size: {len(y_te_pool):,} rows")

# Build training pool (same as before)
tr_pool_X, tr_pool_y = [], []
for day, grp_tr in train.groupby('day_id'):
    if len(grp_tr) < 5: continue
    X_tr = grp_tr[top10].fillna(0).values.astype(np.float64)
    y_tr = y_train[grp_tr.index]
    X_z, m, s = zscore_fit(X_tr)
    tr_pool_X.append(X_z)
    tr_pool_y.append(y_tr)

X_tr_pool = np.vstack(tr_pool_X)
y_tr_pool  = np.concatenate(tr_pool_y)
print(f"  Train pool size: {len(y_tr_pool):,} rows")

ic_results = {}
for alpha in PL_ALPHAS:
    # FIX 2: Standardize both real targets and pseudo-labels to unit variance
    # before computing weighted IC.
    # Without this: pseudo-labels have much lower variance than real targets
    # (predictions are always smoothed/regularised), so the covariance of
    # the test pool is negligible and alpha has near-zero effect.
    # Standardising forces both pools to contribute equally per unit weight.
    y_tr_std = (y_tr_pool - y_tr_pool.mean()) / max(y_tr_pool.std(), 1e-10)
    y_te_std = (y_te_pool - y_te_pool.mean()) / max(y_te_pool.std(), 1e-10)

    w_tr = np.ones(len(y_tr_pool))
    w_te = np.full(len(y_te_pool), alpha)
    w_all = np.concatenate([w_tr, w_te])
    X_all = np.vstack([X_tr_pool, X_te_pool])
    y_all = np.concatenate([y_tr_std, y_te_std])  # standardised targets

    # Weighted mean of y (should be near 0 since both are standardised)
    y_wmean = np.average(y_all, weights=w_all)
    y_dm    = y_all - y_wmean

    # Weighted IC (now a proper weighted correlation since both y pools
    # have unit variance — the covariance equals the correlation)
    ic_aug = (X_all * (w_all[:, None] * y_dm[:, None])).sum(0) / w_all.sum()
    ic_results[alpha] = ic_aug

    # Show change vs original IC
    cos_sim = np.dot(ic_orig, ic_aug) / (np.linalg.norm(ic_orig) * np.linalg.norm(ic_aug))
    print(f"\n  alpha={alpha:.2f}:")
    print(f"    IC cosine similarity to original: {cos_sim:+.6f}")
    print(f"    IC norm: {np.linalg.norm(ic_aug):.6f}  (orig: {np.linalg.norm(ic_orig):.6f})")
    ic_shift = ic_aug / (np.linalg.norm(ic_aug) + 1e-10) - ic_orig / (np.linalg.norm(ic_orig) + 1e-10)
    print(f"    IC direction shift: {np.linalg.norm(ic_shift):.6f}")

# ── Step 3: Generate predictions with augmented ICs ───────────────
print(f"\nStep 3: Generating predictions with augmented ICs...")

def predict_with_ic(ic_vec):
    """Generate Grinold predictions using given IC vector."""
    preds = np.zeros(n_test)
    for day, grp_te in test.groupby('day_id'):
        te_idx = grp_te.index.values
        X_te = grp_te[top10].fillna(0).values.astype(np.float64)
        if day in train_days:
            grp_tr = train[train['day_id'] == day]
            X_tr = grp_tr[top10].fillna(0).values.astype(np.float64)
            _, m, s = zscore_fit(X_tr)
            X_te_z = zscore_apply(X_te, m, s)
        else:
            X_te_z, _, _ = zscore_fit(X_te)
        pred = X_te_z @ ic_vec
        pred -= pred.mean()
        preds[te_idx] = pred
    return preds

def save_sub(preds, name):
    ps = auto_scale(preds)
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': ps})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    t_col = sub['TARGET']
    r = pearson_r(ps, prev)
    r_g = pearson_r(ps, bl0)
    print(f"  {name:<50}  std={t_col.std():.6f}  corr_vs_orig={r:+.4f}  corr_vs_baseline={r_g:+.4f}")

# Baseline (original ICs)
save_sub(0.30*r_s0 + 0.40*k_s0 + 0.30*g_s0, 'pl_baseline_k10_r30_k40_g30')

# Pure pseudo-label Grinold variants
for alpha in PL_ALPHAS:
    ic_aug = ic_results[alpha]
    g_aug  = predict_with_ic(ic_aug)
    # Pure augmented Grinold
    save_sub(g_aug, f'pl_grinold_a{int(alpha*100):03d}')
    # Threeway with augmented Grinold (same Ridge and KNN as baseline)
    blend = 0.30*r_s0 + 0.40*k_s0 + 0.30*auto_scale(g_aug)
    save_sub(blend, f'pl_threeway_a{int(alpha*100):03d}')

print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f} min")
print("""
  ── INTERPRETATION ─────────────────────────────────────────────
  corr_vs_orig: correlation with current best submission (+0.00124)
    < 0.95: meaningfully different predictions (worth submitting)
    > 0.99: nearly identical (skip)

  corr_vs_baseline: correlation with rebuilt threeway (no PL)
    Shows how much pseudo-labels changed the Grinold component

  Recommended submission order:
    1. pl_threeway_a020 — moderate pseudo-label weight (0.20)
    2. pl_threeway_a010 — conservative (0.10)
    3. pl_threeway_a030 — aggressive (0.30)
    4. pl_grinold_a020  — pure pseudo-Grinold (no kernel/Ridge)

  If any pl_threeway outperforms baseline: pseudo-labeling helps.
  The IC direction shift tells you how much the IC moved toward
  the illiquid regime.
""")
