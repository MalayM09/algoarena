# ================================================================
# THREEWAY REBUILD — g_top15 Grinold
# ================================================================
# Same structure as +0.00124 (threeway_r30_k40_g29):
#   30% same-day Ridge  (top-10 gold features, alpha=10)
#   40% cosine KNN      (top-10 gold features, K=10)
#   30% Grinold         → UPGRADED from top-10 to top-15
#
# PI OOF confirmed g_top15 beats g_top10 by +0.00024 (reliable).
# All other components kept identical to isolate the Grinold change.
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR    = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH  = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH   = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH   = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
SAMPLE_PATH = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'outputs/submissions')

RIDGE_ALPHA = 10.0
KNN_K       = 10      # same K as used in original threeway
TARGET_STD  = 0.000948

print("=" * 65)
print("THREEWAY REBUILD — Grinold top-10 → top-15")
print("=" * 65)

# ── Load data ──────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days      = set(train['day_id'].unique())
overlap_days    = train_days & set(test['day_id'].unique())
new_days        = set(test['day_id'].unique()) - train_days
y_train         = train['TARGET'].values.astype(np.float64)
test_ids        = test['ID'].values
print(f"  Train: {len(train):,}  Test: {len(test):,}")
print(f"  Overlap: {len(overlap_days)}  New: {len(new_days)}")

# ── Gold features ──────────────────────────────────────────────────
icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
all_gold  = [f for f in gold_df['feature'].tolist() if f in train.columns]
ic_map    = gold_df.set_index('feature')['mean_ic'].to_dict()

top10 = all_gold[:10]
top15 = all_gold[:15]
ic_top10 = np.array([ic_map[f] for f in top10])
ic_top15 = np.array([ic_map[f] for f in top15])
print(f"  Gold features — top10: {len(top10)}  top15: {len(top15)}")

# ── Helpers ────────────────────────────────────────────────────────
def zscore(X, m=None, s=None, clip=5.0):
    if m is None: m = X.mean(0)
    if s is None: s = X.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip), m, s

def winsorise(y, lo=1, hi=99):
    return np.clip(y, np.percentile(y, lo), np.percentile(y, hi))

def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

def knn_predict(X_q, X_r, y_r, K):
    sim  = cosine_similarity(X_q, X_r)
    K    = min(K, sim.shape[1])
    topk = np.argpartition(sim, -K, axis=1)[:, -K:]
    w    = np.maximum(sim[np.arange(len(X_q))[:, None], topk], 0)
    ws   = w.sum(1, keepdims=True)
    w   /= np.where(ws < 1e-10, 1.0, ws)
    return (w * y_r[topk]).sum(1)

# ── PI OOF: confirm g_top15 > g_top10 on this run ─────────────────
print("\nRunning PI OOF confirmation (g_top10 vs g_top15)...")
b_near = [c for c in train.columns if 'Lag' not in c and any(f'_B0{i}' in c for i in range(5))]
b_far  = [c for c in train.columns if 'Lag' not in c and any(f'_B{i}' in c for i in ['06','07','08','09','10'])]
train['bs'] = (train[b_near].fillna(0).sum(1) - train[b_far].fillna(0).sum(1)).values

ics10, ics15 = [], []
for day, grp in train.groupby('day_id'):
    if len(grp) < 20: continue
    y_d    = y_train[grp.index]
    bs     = grp['bs'].values
    illiq  = bs < np.median(bs)
    if illiq.sum() < 5: continue
    for feats, ic_arr, store in [(top10, ic_top10, ics10), (top15, ic_top15, ics15)]:
        X  = grp[feats].fillna(0).values.astype(np.float64)
        Xz, _, _ = zscore(X)
        p  = Xz @ ic_arr; p -= p.mean()
        r, _ = spearmanr(y_d[illiq], p[illiq])
        if not np.isnan(r):
            store.append(r)

med10 = np.nanmedian(ics10)
med15 = np.nanmedian(ics15)
print(f"  g_top10 PI OOF Med IC: {med10:+.5f}")
print(f"  g_top15 PI OOF Med IC: {med15:+.5f}  (Δ {med15-med10:+.5f})")
winner = 'g_top15' if med15 > med10 else 'g_top10'
print(f"  Winner: {winner}")

# Use winner for Grinold component
g_feats  = top15 if med15 >= med10 else top10
g_ic_arr = ic_top15 if med15 >= med10 else ic_top10
print(f"  Using {winner} for Grinold component in threeway.")

# ── Generate test predictions ──────────────────────────────────────
print("\nGenerating test predictions...")
n_test   = len(test)
te_ridge   = np.zeros(n_test)
te_knn     = np.zeros(n_test)
te_grinold = np.zeros(n_test)    # g_top15 (or top10 if top10 wins)
te_grinold_orig = np.zeros(n_test)  # always g_top10 (for comparison)

test['bs'] = (test[b_near].fillna(0).sum(1) - test[b_far].fillna(0).sum(1)).values

for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index
    X_te10 = grp_te[top10].fillna(0).values.astype(np.float64)
    X_te_g = grp_te[g_feats].fillna(0).values.astype(np.float64)

    # Grinold — always per-day z-score on test assets
    Xz_g, _, _  = zscore(X_te_g)
    pred_g       = Xz_g @ g_ic_arr; pred_g -= pred_g.mean()
    te_grinold[te_idx] = pred_g

    Xz_10, _, _ = zscore(X_te10)
    pred_g10     = Xz_10 @ ic_top10; pred_g10 -= pred_g10.mean()
    te_grinold_orig[te_idx] = pred_g10

    if day in train_days:
        grp_tr  = train[train['day_id'] == day]
        y_tr    = y_train[grp_tr.index]          # ALL training assets
        X_tr10  = grp_tr[top10].fillna(0).values.astype(np.float64)
        y_tr_w  = winsorise(y_tr)                # winsorise ALL targets

        # Ridge: z-score using ALL training assets for this day
        # (no bookshape split — matches original rank_knn_boost.py)
        _, m_tr, s_tr = zscore(X_tr10)
        X_tr_z  = zscore(X_tr10, m_tr, s_tr)[0]
        X_te_z  = zscore(X_te10, m_tr, s_tr)[0]
        ridge   = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        ridge.fit(X_tr_z, y_tr_w)               # fit on ALL train assets
        pred_r  = ridge.predict(X_te_z); pred_r -= pred_r.mean()
        te_ridge[te_idx] = pred_r

        # Cosine KNN — ALL training assets as reference
        pred_knn = knn_predict(X_te_z, X_tr_z, y_tr, KNN_K)
        pred_knn -= pred_knn.mean()
        te_knn[te_idx] = pred_knn

    else:
        # New day: Grinold fallback
        te_ridge[te_idx] = pred_g
        te_knn[te_idx]   = pred_g

elapsed = (time.time() - t0) / 60
print(f"  Done.  [{elapsed:.1f}m elapsed]")

# ── Signal correlation check ───────────────────────────────────────
def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return (a @ b) / d if d > 1e-10 else 0.0

print(f"\n  Component correlations:")
print(f"  Ridge    vs Grinold-{winner}: {pearson_r(te_ridge,   te_grinold):+.4f}")
print(f"  KNN      vs Grinold-{winner}: {pearson_r(te_knn,     te_grinold):+.4f}")
print(f"  Ridge    vs KNN:             {pearson_r(te_ridge,   te_knn):+.4f}")
print(f"  Grinold-top15 vs top10:      {pearson_r(te_grinold, te_grinold_orig):+.4f}")

# ── Build and save submissions ─────────────────────────────────────
print("\nBuilding submissions...")
sample_sub = pd.read_csv(SAMPLE_PATH)[['ID']]

r_s = auto_scale(te_ridge)
k_s = auto_scale(te_knn)
g_s = auto_scale(te_grinold)
g0_s = auto_scale(te_grinold_orig)

def save_sub(preds, name):
    preds_s = auto_scale(preds)
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': preds_s})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    t = sub['TARGET']
    print(f"  {name:<50}  std={t.std():.6f}  mean={t.mean():+.7f}")

# Exact replica of original threeway but with g_top15
threeway_g15 = 0.30 * r_s + 0.40 * k_s + 0.30 * g_s
save_sub(threeway_g15, 'threeway_g15_r30_k40_g30')

# Also save the original threeway reconstruction (should match +0.00124 file closely)
threeway_g10 = 0.30 * r_s + 0.40 * k_s + 0.30 * g0_s
save_sub(threeway_g10, 'threeway_g10_rebuild_r30_k40_g30')

# Additional weight variants with g_top15 (in case weights need slight retuning)
save_sub(0.25 * r_s + 0.45 * k_s + 0.30 * g_s, 'threeway_g15_r25_k45_g30')
save_sub(0.30 * r_s + 0.45 * k_s + 0.25 * g_s, 'threeway_g15_r30_k45_g25')

print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f} min")
print(f"""
  ── SUBMISSION GUIDE ─────────────────────────────────────────────
  PRIMARY:   threeway_g15_r30_k40_g30.csv
             Same structure as +0.00124 but Grinold top10 → top15.
             Expected: marginally above +0.00124 (~+0.00125).

  SANITY:    threeway_g10_rebuild_r30_k40_g30.csv
             Should score near +0.00124 (reconstructed original).
             If far off, K or other setting differs from original.

  VARIANTS:  threeway_g15_r25_k45/r30_k45 — slightly different weights.
             Submit only if primary doesn't improve over +0.00124.
""")
