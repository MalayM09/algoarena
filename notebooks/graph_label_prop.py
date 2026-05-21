# ================================================================
# GRAPH LABEL PROPAGATION — Semi-supervised Liquid→Illiquid
# ================================================================
# Hypothesis:
#   Liquid training assets (labeled) and illiquid test assets
#   share a common feature manifold. We can propagate TARGET
#   labels from liquid → illiquid via feature-space similarity,
#   using the normalized graph Laplacian.
#
# Algorithm (per overlap day):
#   1. Combine liquid (train) + illiquid (test) into one joint pool
#   2. Z-score features using liquid stats
#   3. Build sparse k-NN similarity graph W over joint pool
#      (Gaussian kernel: W_ij = exp(-||xi - xj||^2 / sigma^2))
#   4. Normalize: S = D^{-1/2} W D^{-1/2}  (symmetric)
#   5. Label vector Y: y_liquid for labeled nodes, 0 for unlabeled
#   6. Iterate: F = alpha * (S @ F) + (1-alpha) * Y
#      until convergence (or fixed iterations)
#   7. Read F at test node positions → illiquid predictions
#
# Non-overlap days: Grinold IC fallback
#
# Fixes applied:
#   - top10 sorted by ICIR descending (not alphabetical LagT1)
#   - train_day_set = set() for O(1) lookup
#   - Grinold fallback for non-overlap days
#   - auto_scale before saving
#   - Winsorize labels before propagation
#
# Generates:
#   glp_k15_a08.csv       — k=15, alpha=0.8 (baseline)
#   glp_k10_a08.csv       — k=10, alpha=0.8
#   glp_k20_a08.csv       — k=20, alpha=0.8
#   glp_k15_a09.csv       — k=15, alpha=0.9 (more propagation)
#   glp_k15_a06.csv       — k=15, alpha=0.6 (less propagation)
#   glp_ens_best.csv      — best GLP blended with best ensemble
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR    = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH  = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH   = os.path.join(BASE_DIR, 'data/raw/test.parquet')
SAMPLE_PATH = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
ICIR_PATH   = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
BEST_ENS    = os.path.join(BASE_DIR, 'outputs/submissions/ens_tw35_hyb30_g35.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'outputs/submissions')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_STD = 0.000948
CLIP_Z     = 5.0

# Sweep parameters
K_VALUES    = [10, 15, 20]
ALPHA_VALUES= [0.6, 0.8, 0.9]

print("=" * 65)
print("GRAPH LABEL PROPAGATION — Semi-supervised Liquid→Illiquid")
print("=" * 65)

# ── Load data ──────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_day_set   = set(train['day_id'].unique())   # O(1) lookup (FIX: was slow)
y_train         = train['TARGET'].values.astype(np.float64)
test_ids        = test['ID'].values
sample_sub      = pd.read_csv(SAMPLE_PATH)[['ID']]
n_test          = len(test)
print(f"  Train: {len(train):,}  Test: {n_test:,}")

# ── Gold top-10 features (FIX: sorted by ICIR, not alphabetical) ──
icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = ((icir_df['abs_icir'] >= 3) &
             ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0)))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False)  # FIX
top10     = [f for f in gold_df['feature'].tolist()[:10] if f in train.columns]
ic_arr    = np.array([gold_df.set_index('feature')['mean_ic'].to_dict()[f] for f in top10])
print(f"  Gold top-10 (by ICIR desc): {top10[:3]}...  [{len(top10)} total]")

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
    return float((a @ b) / d) if d > 1e-12 else 0.0


def label_propagation(X_all, y_liq_winsorised, n_liq, n_te,
                       k=15, alpha=0.8, n_iter=30):
    """
    Semi-supervised label propagation on joint feature space.

    Parameters
    ----------
    X_all          : (n_liq + n_te, n_feat) — z-scored features, liquid first
    y_liq_winsorised: (n_liq,) — winsorised labels for liquid assets
    n_liq          : number of liquid (labeled) assets
    n_te           : number of test (unlabeled) assets
    k              : number of nearest neighbours for graph
    alpha          : propagation weight (0=no propagation, 1=full propagation)
    n_iter         : number of Jacobi iterations

    Returns
    -------
    F_te : (n_te,) — propagated predictions for test assets
    """
    n_all = n_liq + n_te

    # Build k-NN graph (mutual k-NN for symmetry)
    knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean', n_jobs=-1)
    knn.fit(X_all)
    distances, indices = knn.kneighbors(X_all)

    # distances[:, 0] = 0 (self), skip
    # Gaussian kernel: sigma = median of non-zero distances
    nonzero_dists = distances[:, 1:].ravel()
    sigma = np.median(nonzero_dists) + 1e-8

    # Build sparse weight matrix
    W = lil_matrix((n_all, n_all), dtype=np.float64)
    for i in range(n_all):
        for jj in range(1, k + 1):   # skip self (index 0)
            j = indices[i, jj]
            d = distances[i, jj]
            w = np.exp(-(d ** 2) / (sigma ** 2))
            W[i, j] = w
            W[j, i] = w   # symmetrize

    W = W.tocsr()

    # Degree matrix and normalized Laplacian: S = D^{-1/2} W D^{-1/2}
    deg = np.array(W.sum(1)).ravel()
    deg_inv_sqrt = np.where(deg > 1e-12, deg ** -0.5, 0.0)
    D_inv_sqrt = diags(deg_inv_sqrt)
    S = D_inv_sqrt @ W @ D_inv_sqrt   # (n_all, n_all) sparse

    # Label vector Y: liquid labels, 0 for unlabeled
    Y = np.zeros(n_all, dtype=np.float64)
    Y[:n_liq] = y_liq_winsorised

    # Propagation: F = alpha * S @ F + (1-alpha) * Y
    F = Y.copy()
    for _ in range(n_iter):
        F = alpha * (S @ F) + (1 - alpha) * Y

    return F[n_liq:]   # return test predictions only


# ── Grinold baseline for fallback and comparison ───────────────────
print("\nComputing Grinold baseline (for non-overlap day fallback)...")
grinold_preds = np.zeros(n_test)
for day, grp_te in test.groupby('day_id'):
    te_idx   = grp_te.index.values
    X_te_raw = grp_te[top10].fillna(0).values.astype(np.float64)
    if day in train_day_set:
        grp_tr   = train[train['day_id'] == day]
        X_tr_raw = grp_tr[top10].fillna(0).values.astype(np.float64)
        _, m, s  = zscore_fit(X_tr_raw)
        X_te_z   = zscore_apply(X_te_raw, m, s)
    else:
        X_te_z, _, _ = zscore_fit(X_te_raw)
    pred = X_te_z @ ic_arr
    pred -= pred.mean()
    grinold_preds[te_idx] = pred
grinold_s = auto_scale(grinold_preds)
print(f"  Grinold done  [{(time.time()-t0)/60:.1f}m]")

# ── Load best ensemble ─────────────────────────────────────────────
best_ens_raw = (sample_sub
                .merge(pd.read_csv(BEST_ENS)[['ID','TARGET']]
                       .rename(columns={'TARGET':'b'}), on='ID', how='left')
                .fillna(0.0)['b'].values)
best_s = auto_scale(best_ens_raw)

# ── Main GLP loop (per overlap day, sweep k and alpha) ────────────
# Collect predictions for each (k, alpha) pair
configs   = [(k, a) for k in K_VALUES for a in ALPHA_VALUES]
all_preds = {cfg: np.zeros(n_test) for cfg in configs}

overlap_days = [d for d in test['day_id'].unique() if d in train_day_set]
future_days  = [d for d in test['day_id'].unique() if d not in train_day_set]
print(f"\nOverlap days: {len(overlap_days)}  Future days: {len(future_days)}")
print(f"Running GLP over {len(configs)} configs × {len(overlap_days)} days...")

for di, day in enumerate(sorted(overlap_days)):
    grp_tr = train[train['day_id'] == day]
    grp_te = test[test['day_id'] == day]
    te_idx = grp_te.index.values

    if len(grp_tr) < 10 or len(grp_te) < 1:
        # Fallback: Grinold
        for cfg in configs:
            all_preds[cfg][te_idx] = grinold_preds[te_idx]
        continue

    y_liq    = winsorise(y_train[grp_tr.index])
    X_tr_raw = grp_tr[top10].fillna(0).values.astype(np.float64)
    X_te_raw = grp_te[top10].fillna(0).values.astype(np.float64)

    # Z-score using liquid (training) stats
    X_tr_z, m, s = zscore_fit(X_tr_raw)
    X_te_z        = zscore_apply(X_te_raw, m, s)

    # Joint feature matrix: liquid first, then test
    X_all = np.vstack([X_tr_z, X_te_z])
    n_liq = len(grp_tr)
    n_te  = len(grp_te)

    for (k, alpha) in configs:
        try:
            f_te = label_propagation(X_all, y_liq, n_liq, n_te, k=k, alpha=alpha)
            f_te -= f_te.mean()
            all_preds[(k, alpha)][te_idx] = f_te
        except Exception as e:
            # Fallback on error
            all_preds[(k, alpha)][te_idx] = grinold_preds[te_idx]

    if (di + 1) % 50 == 0 or (di + 1) == len(overlap_days):
        elapsed = (time.time() - t0) / 60
        print(f"  Day {di+1}/{len(overlap_days)}  [{elapsed:.1f}m]")

# FIX: Grinold fallback for non-overlap (future) days
for day in future_days:
    grp_te = test[test['day_id'] == day]
    te_idx = grp_te.index.values
    for cfg in configs:
        all_preds[cfg][te_idx] = grinold_preds[te_idx]

print(f"\nGLP complete  [{(time.time()-t0)/60:.1f}m]")

# ── Save and report ────────────────────────────────────────────────
print(f"\n{'Config':<15}  {'corr_vs_grinold':>16}  {'corr_vs_ens':>12}  File")
print("-" * 70)

best_corr_ens = -999
best_cfg_name = None
best_cfg_preds = None

for (k, alpha), preds in all_preds.items():
    ps     = auto_scale(preds)   # FIX: auto_scale before saving
    c_grin = pearson_r(ps, grinold_s)
    c_ens  = pearson_r(ps, best_s)
    name   = f'glp_k{k}_a{int(alpha*10):02d}'
    sub    = pd.DataFrame({'ID': test_ids, 'TARGET': ps})
    sub    = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    print(f"  k={k} a={alpha}       {c_grin:+.4f}           {c_ens:+.4f}       {name}.csv")

    if c_ens < best_corr_ens or best_cfg_name is None:
        # Track most diverse config (lowest corr with best_ens is highest reward)
        pass
    if c_grin > 0.85 and c_ens > best_corr_ens:
        best_corr_ens = c_ens
        best_cfg_name = name
        best_cfg_preds = ps

# ── Best GLP blended with best ensemble ───────────────────────────
print("\nBuilding blends (GLP + best_ens)...")
# Use k=15, alpha=0.8 as primary
primary_preds = auto_scale(all_preds[(15, 0.8)])
for w in [0.15, 0.20, 0.30]:
    blend   = (1 - w) * best_s + w * primary_preds
    blend_s = auto_scale(blend)
    c       = pearson_r(blend_s, best_s)
    name    = f'glp_k15_a08_w{int(w*100)}_ens{int((1-w)*100)}'
    sub     = pd.DataFrame({'ID': test_ids, 'TARGET': blend_s})
    sub     = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    print(f"  {name:<50}  corr_vs_best={c:+.4f}")

print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
print("""
── INTERPRETATION ─────────────────────────────────────────────
GLP propagates labels from liquid → illiquid via feature proximity.
This is complementary to KNN/NW kernel (which directly matches
liquid→illiquid using feature similarity) because GLP also uses
inter-illiquid similarity (multi-hop propagation).

corr_vs_grinold:
  GLP should score higher corr with grinold (~0.85+) if the
  feature manifold is informative.

corr_vs_ens:
  < 0.95: GLP adds genuinely new information — worth submitting
  > 0.98: GLP is nearly identical to ensemble — skip

KEY: Higher alpha = more propagation = smoother predictions.
  If alpha=0.9 variants differ from alpha=0.6 variants (low mutual corr)
  → alpha matters → tune it for LB.

If GLP corr_vs_ens > 0.98 for all configs:
  → GLP is just rediscovering KNN signal (already in threeway ensemble)
  → Not worth pursuing further
""")
