# ================================================================
# KNN SWEEP SUBMISSION — test different K values via LB feedback
# ================================================================
# CV cannot evaluate the NW Kernel component because CV_GROUP
# tests liquid→liquid, while the competition requires liquid→illiquid.
# We must use LB feedback to find the optimal K.
#
# Replicates threeway_g15_rebuild.py exactly, varying only K.
# All other settings identical to the +0.00122 submission.
#
# Current: K=10  →  +0.00122
# Testing: K=5, 15, 20, 30, 50
#
# Also tests one weight variant: shift weight from Ridge to KNN
# (since KNN is the liquid→illiquid component and Ridge is not).
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
os.makedirs(OUT_DIR, exist_ok=True)

RIDGE_ALPHA = 10.0
TARGET_STD  = 0.000948

print("=" * 65)
print("KNN SWEEP SUBMISSION — K-value search for NW Kernel")
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
print(f"  Train: {len(train):,}  Test: {len(test):,}")
print(f"  Train days: {len(train_days)}")

# ── Gold features ──────────────────────────────────────────────────
icir_df  = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df  = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
all_gold = [f for f in gold_df['feature'].tolist() if f in train.columns]
ic_map   = gold_df.set_index('feature')['mean_ic'].to_dict()

top10    = all_gold[:10]
ic_arr10 = np.array([ic_map[f] for f in top10])
print(f"  Gold features: {len(all_gold)}  using top-10 for all components")

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

def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return (a @ b) / d if d > 1e-10 else 0.0

def knn_predict(X_q, X_r, y_r, K):
    sim  = cosine_similarity(X_q, X_r)
    K    = min(K, sim.shape[1])
    topk = np.argpartition(sim, -K, axis=1)[:, -K:]
    w    = np.maximum(sim[np.arange(len(X_q))[:, None], topk], 0)
    ws   = w.sum(1, keepdims=True)
    w   /= np.where(ws < 1e-10, 1.0, ws)
    return (w * y_r[topk]).sum(1)

# ── Generate test predictions for all K values simultaneously ─────
print("\nGenerating test predictions (Ridge + Grinold + KNN×6)...")
K_VALUES = [5, 10, 15, 20, 30, 50]
n_test = len(test)

te_ridge   = np.zeros(n_test)
te_grinold = np.zeros(n_test)
te_knn     = {K: np.zeros(n_test) for K in K_VALUES}

for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index.values
    X_te10 = grp_te[top10].fillna(0).values.astype(np.float64)

    # Grinold: z-score test assets within day, dot with IC
    Xz_te, _, _ = zscore(X_te10)
    pred_g = Xz_te @ ic_arr10
    pred_g -= pred_g.mean()
    te_grinold[te_idx] = pred_g

    if day in train_days:
        grp_tr = train[train['day_id'] == day]
        y_tr   = y_train[grp_tr.index]
        X_tr10 = grp_tr[top10].fillna(0).values.astype(np.float64)

        # Ridge: z-score with train stats, fit on winsorised targets
        _, m_tr, s_tr = zscore(X_tr10)
        X_tr_z = zscore(X_tr10, m_tr, s_tr)[0]
        X_te_z = zscore(X_te10, m_tr, s_tr)[0]
        y_tr_w = winsorise(y_tr)
        mdl = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        mdl.fit(X_tr_z, y_tr_w)
        pred_r = mdl.predict(X_te_z)
        pred_r -= pred_r.mean()
        te_ridge[te_idx] = pred_r

        # KNN for all K values (reuse same sim matrix via argpartition)
        sim = cosine_similarity(X_te_z, X_tr_z)   # (n_test_day, n_train_day)
        max_K = min(max(K_VALUES), sim.shape[1])
        # Get top-max_K indices once, then slice for each K
        topk_all = np.argpartition(sim, -max_K, axis=1)[:, -max_K:]
        sim_topk = sim[np.arange(len(X_te_z))[:, None], topk_all]

        for K in K_VALUES:
            K_eff = min(K, sim.shape[1])
            # Take top-K from the already-partitioned top-max_K
            if K_eff < max_K:
                # Find top-K within topk_all
                local_idx = np.argpartition(sim_topk, -K_eff, axis=1)[:, -K_eff:]
                topk_K = topk_all[np.arange(len(X_te_z))[:, None], local_idx]
                sim_K  = sim_topk[np.arange(len(X_te_z))[:, None], local_idx]
            else:
                topk_K = topk_all
                sim_K  = sim_topk

            w  = np.maximum(sim_K, 0)
            ws = w.sum(1, keepdims=True)
            w /= np.where(ws < 1e-10, 1.0, ws)
            pred_k = (w * y_tr[topk_K]).sum(1)
            pred_k -= pred_k.mean()
            te_knn[K][te_idx] = pred_k
    else:
        # New test day — Grinold fallback for Ridge and KNN
        te_ridge[te_idx] = pred_g
        for K in K_VALUES:
            te_knn[K][te_idx] = pred_g

print(f"  Done.  [{(time.time()-t0)/60:.1f}m]")

# ── Component correlations ─────────────────────────────────────────
print(f"\n  Component correlations (all K vs Ridge and Grinold):")
print(f"  Ridge vs Grinold: {pearson_r(te_ridge, te_grinold):+.4f}")
for K in K_VALUES:
    r_k = pearson_r(te_knn[K], te_grinold)
    r_r = pearson_r(te_knn[K], te_ridge)
    print(f"  KNN(K={K:2d}) vs Grinold: {r_k:+.4f}   vs Ridge: {r_r:+.4f}")

# ── Build and save submissions ─────────────────────────────────────
print("\nBuilding submissions...")
sample_sub = pd.read_csv(SAMPLE_PATH)[['ID']]

def save_sub(preds, name):
    ps = auto_scale(preds)
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': ps})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    t = sub['TARGET']
    print(f"  {name:<55}  std={t.std():.6f}  mean={t.mean():+.7f}")

r_s = auto_scale(te_ridge)
g_s = auto_scale(te_grinold)

# K sweep — same weights as original (0.30/0.40/0.30)
for K in K_VALUES:
    k_s = auto_scale(te_knn[K])
    save_sub(0.30 * r_s + 0.40 * k_s + 0.30 * g_s,
             f'threeway_k{K:02d}_r30_k40_g30')

print()

# Best K candidates with higher KNN weight (KNN is the key liquid→illiquid component)
# Original K=10, weight=0.40. Try heavier KNN weight for K=20/30 (larger K → smoother)
for K, wk, wr, wg in [(20, 0.50, 0.25, 0.25),
                       (30, 0.50, 0.25, 0.25),
                       (20, 0.45, 0.30, 0.25),
                       (30, 0.45, 0.30, 0.25)]:
    k_s = auto_scale(te_knn[K])
    save_sub(wr * r_s + wk * k_s + wg * g_s,
             f'threeway_k{K:02d}_r{int(wr*100)}_k{int(wk*100)}_g{int(wg*100)}')

print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f} min")
print(f"""
  ── SUBMISSION GUIDE ─────────────────────────────────────────
  BASELINE (K=10):   threeway_k10_r30_k40_g30.csv
                     Should match +0.00122. Confirms rebuild is correct.

  K SWEEP:           threeway_k05 through threeway_k50
                     Test one by one to find optimal K.
                     Larger K = smoother, more stable prediction.
                     Smaller K = noisier but more specific match.

  HEAVIER KNN:       threeway_k20_r25_k50_g25, threeway_k30_r25_k50_g25
                     KNN is the key liquid→illiquid component.
                     Giving it more weight may help.

  RECOMMENDED ORDER:
    1. threeway_k10_r30_k40_g30   ← sanity check (should = +0.00122)
    2. threeway_k20_r30_k40_g30   ← most likely improvement
    3. threeway_k30_r30_k40_g30   ← smoother, less noise
    4. threeway_k20_r25_k50_g25   ← heavier KNN weight
""")
