# ================================================================
# GAUSSIAN NW KERNEL REBUILD
# ================================================================
# The original hybrid_grinold_kernel.csv (scored +0.00115 standalone,
# +0.00124 in threeway) was a Nadaraya-Watson kernel with Gaussian
# weights: w_ij = exp(-||z_i - z_j||^2 / (2*h^2))
#
# From time_aligned_nw.py comment:
#   "Original NW kernel: dist(test.features, train.features)"
#   i.e. direct feature matching (no temporal alignment)
#
# This script reconstructs that kernel, sweeps bandwidth h,
# finds which h best matches the original file (via correlation),
# then builds threeway submissions for the best h values.
#
# Feature normalisation: z-score using LIQUID training stats
# (matches perday_ridge.py and the original threeway pipeline)
#
# For new (non-overlap) days: Grinold fallback (same as original).
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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
ORIG_HYBRID = os.path.join(OUT_DIR, 'hybrid_grinold_kernel.csv')

print("=" * 65)
print("GAUSSIAN NW KERNEL REBUILD — bandwidth sweep + threeway")
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
print(f"  Overlap days: {len(train_days & set(test['day_id']))}")

# ── Gold features ──────────────────────────────────────────────────
icir_df  = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df  = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
all_gold = [f for f in gold_df['feature'].tolist() if f in train.columns]
ic_map   = gold_df.set_index('feature')['mean_ic'].to_dict()
top10    = all_gold[:10]
ic_arr10 = np.array([ic_map[f] for f in top10])
print(f"  Gold features: {len(all_gold)}  top-10: {len(top10)}")

# ── Original kernel for comparison ────────────────────────────────
sample_sub = pd.read_csv(SAMPLE_PATH)[['ID']]
orig_df    = pd.read_csv(ORIG_HYBRID)
orig_df    = sample_sub.merge(
    orig_df[['ID','TARGET']].rename(columns={'TARGET':'orig'}),
    on='ID', how='left'
).fillna(0.0)
orig_vals = orig_df['orig'].values
print(f"\n  Original hybrid_grinold_kernel: std={orig_vals.std():.7f}")

# ── Helpers ────────────────────────────────────────────────────────
def zscore_fit(X, clip=5.0):
    m = X.mean(0); s = X.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip), m, s

def zscore_apply(X, m, s, clip=5.0):
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

def gaussian_nw(X_test_z, X_train_z, y_train, h):
    """
    Nadaraya-Watson with Gaussian kernel.
    X_test_z, X_train_z: z-scored with SAME (train) stats.
    h: bandwidth (std of Gaussian kernel in z-score space).
    """
    # Squared L2 distances: (n_test, n_train)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a@b^T
    a2 = (X_test_z  ** 2).sum(1, keepdims=True)   # (n_test, 1)
    b2 = (X_train_z ** 2).sum(1, keepdims=True).T  # (1, n_train)
    ab = X_test_z @ X_train_z.T                    # (n_test, n_train)
    dist2 = np.maximum(a2 + b2 - 2 * ab, 0)        # clip numerical noise

    # Gaussian weights
    w = np.exp(-dist2 / (2 * h * h))               # (n_test, n_train)
    ws = w.sum(1, keepdims=True)
    ws = np.where(ws < 1e-12, 1.0, ws)
    return (w @ y_train) / ws.ravel()

# ── Sweep bandwidths — compute NW for all h values in one pass ────
# Also compute Ridge once (same for all h)
BANDWIDTHS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]

n_test     = len(test)
te_ridge   = np.zeros(n_test)
te_grinold = np.zeros(n_test)
te_nw      = {h: np.zeros(n_test) for h in BANDWIDTHS}

print(f"\nGenerating predictions (Ridge + Grinold + NW×{len(BANDWIDTHS)})...")

for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index.values
    X_te10 = grp_te[top10].fillna(0).values.astype(np.float64)

    # Grinold: z-score test-only stats
    Xz_te, _, _ = zscore_fit(X_te10)
    pred_g = Xz_te @ ic_arr10
    pred_g -= pred_g.mean()
    te_grinold[te_idx] = pred_g

    if day in train_days:
        grp_tr = train[train['day_id'] == day]
        y_tr   = y_train[grp_tr.index]
        X_tr10 = grp_tr[top10].fillna(0).values.astype(np.float64)

        # Z-score using LIQUID (train) stats — shared across Ridge and NW
        _, m_tr, s_tr = zscore_fit(X_tr10)
        X_tr_z = zscore_apply(X_tr10, m_tr, s_tr)
        X_te_z = zscore_apply(X_te10, m_tr, s_tr)

        # Ridge
        y_tr_w = winsorise(y_tr)
        mdl = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        mdl.fit(X_tr_z, y_tr_w)
        pred_r = mdl.predict(X_te_z)
        pred_r -= pred_r.mean()
        te_ridge[te_idx] = pred_r

        # NW for all bandwidths — reuse same z-scored matrices
        y_tr_f64 = y_tr.astype(np.float64)
        for h in BANDWIDTHS:
            pred_nw = gaussian_nw(X_te_z, X_tr_z, y_tr_f64, h)
            pred_nw -= pred_nw.mean()
            te_nw[h][te_idx] = pred_nw
    else:
        # New day: Grinold fallback
        te_ridge[te_idx] = pred_g
        for h in BANDWIDTHS:
            te_nw[h][te_idx] = pred_g

elapsed = (time.time() - t0) / 60
print(f"  Done.  [{elapsed:.1f}m]")

# ── Find which bandwidth best matches the original kernel ──────────
print("\n" + "=" * 65)
print("BANDWIDTH CALIBRATION — correlation vs original hybrid_kernel")
print("=" * 65)

g_s = auto_scale(te_grinold)
r_s = auto_scale(te_ridge)

nw_corr = {}
for h in BANDWIDTHS:
    # Build hybrid: 70% NW + 30% Grinold (same structure as original)
    nw_s = auto_scale(te_nw[h])
    hybrid = auto_scale(0.70 * nw_s + 0.30 * g_s)
    r = pearson_r(hybrid, orig_vals)
    nw_corr[h] = r
    print(f"  h={h:<5.1f}  hybrid_corr_vs_orig={r:.5f}  NW_std={te_nw[h].std():.7f}")

best_h     = max(nw_corr, key=nw_corr.get)
best_corr  = nw_corr[best_h]
print(f"\n  Best match: h={best_h}  corr={best_corr:.5f}")
print(f"  (>0.98 = good match, confirms we've reproduced the original kernel)")

# ── Build submissions ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("BUILDING SUBMISSIONS")
print("=" * 65)

def save_sub(preds, name):
    ps = auto_scale(preds)
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': ps})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    t_col = sub['TARGET']
    print(f"  {name:<55}  std={t_col.std():.6f}  corr_vs_orig={pearson_r(ps, orig_vals):+.4f}")

# Best bandwidth
nw_best   = auto_scale(te_nw[best_h])
hybrid_best = auto_scale(0.70 * nw_best + 0.30 * g_s)
save_sub(0.30 * r_s + 0.40 * hybrid_best + 0.30 * g_s,
         f'gnw_threeway_h{best_h:.1f}')

# Also try second-best bandwidth
sorted_h = sorted(nw_corr, key=nw_corr.get, reverse=True)
if len(sorted_h) >= 2:
    h2 = sorted_h[1]
    nw_h2 = auto_scale(te_nw[h2])
    hybrid_h2 = auto_scale(0.70 * nw_h2 + 0.30 * g_s)
    save_sub(0.30 * r_s + 0.40 * hybrid_h2 + 0.30 * g_s,
             f'gnw_threeway_h{h2:.1f}')

# Fine-grained bandwidth sweep around best_h
print("\n  Fine-grained sweep around best h:")
fine_hs = np.round(np.linspace(max(0.1, best_h * 0.5), best_h * 2.0, 8), 2)
for h_fine in fine_hs:
    nw_f = auto_scale(te_nw.get(h_fine, np.zeros(n_test)))
    if h_fine not in te_nw:
        # Compute on-the-fly for fine-grained h
        te_nw_fine = np.zeros(n_test)
        for day, grp_te in test.groupby('day_id'):
            te_idx = grp_te.index.values
            X_te10 = grp_te[top10].fillna(0).values.astype(np.float64)
            if day in train_days:
                grp_tr = train[train['day_id'] == day]
                y_tr   = y_train[grp_tr.index].astype(np.float64)
                X_tr10 = grp_tr[top10].fillna(0).values.astype(np.float64)
                _, m_tr, s_tr = zscore_fit(X_tr10)
                X_tr_z = zscore_apply(X_tr10, m_tr, s_tr)
                X_te_z = zscore_apply(X_te10, m_tr, s_tr)
                pred_nw = gaussian_nw(X_te_z, X_tr_z, y_tr, h_fine)
                pred_nw -= pred_nw.mean()
                te_nw_fine[te_idx] = pred_nw
            else:
                X_te_z2, _, _ = zscore_fit(X_te10)
                pred_g2 = X_te_z2 @ ic_arr10; pred_g2 -= pred_g2.mean()
                te_nw_fine[te_idx] = pred_g2
        nw_f = auto_scale(te_nw_fine)
        hybrid_fine = auto_scale(0.70 * nw_f + 0.30 * g_s)
        r_fine = pearson_r(hybrid_fine, orig_vals)
        print(f"    h={h_fine:.2f}  corr={r_fine:.5f}")
        save_sub(0.30 * r_s + 0.40 * hybrid_fine + 0.30 * g_s,
                 f'gnw_threeway_h{h_fine:.2f}')

print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f} min")
print(f"""
  ── SUBMISSION GUIDE ─────────────────────────────────────────
  If corr_vs_orig > 0.98: kernel is reproduced correctly.
    → Submit gnw_threeway_h{{best_h}} first (should ≈ +0.00124)
    → Then try fine-grained variants near best_h

  If corr_vs_orig < 0.90: bandwidth didn't match.
    → The original used different features or normalization
    → Try checking if features were NOT z-scored with train stats
    → Or if original used different feature set (all51 vs top10)
""")
