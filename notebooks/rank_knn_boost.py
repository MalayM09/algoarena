# ================================================================
# RANK NORMALIZATION + EXPLICIT TOP-K KNN
# ================================================================
# Two new ideas on top of the confirmed best (threeway +0.00124):
#
#   1. RANK NORMALIZATION — post-process any prediction per-day
#      rankdata(pred) / N - 0.5 → uniform distribution
#      Aligns submission distribution with IC (Spearman) objective
#
#   2. EXPLICIT TOP-K KNN — harder cutoff than Gaussian kernel
#      Find K most similar liquid assets by cosine similarity
#      Weighted average of their returns → neighbor prediction
#      Test K = 5, 10, 20, 50
#
# Both tested in PI OOF, then applied to test predictions.
# All variants blended with Grinold and saved for submission.
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
t0 = time.time()

print("=" * 70)
print("RANK NORMALIZATION + EXPLICIT TOP-K KNN ANALYSIS")
print("=" * 70)

# ── Feature selection ──────────────────────────────────────────────────────
icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False).copy()
all51     = gold_df['feature'].tolist()
ic_weights   = gold_df.set_index('feature')['mean_ic'].to_dict()
icir_weights = gold_df.set_index('feature')['abs_icir'].to_dict()

# ── Load data ──────────────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)

all_cols = set(train.columns) - {'ID', 'TARGET'}
all51    = [f for f in all51 if f in all_cols]
top10    = all51[:10]

train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days = set(train['day_id'].unique())
overlap    = train_days & set(test['day_id'].unique())
new_days   = set(test['day_id'].unique()) - train_days

y_train  = train['TARGET'].values.astype(np.float64)
test_ids = test['ID'].values

# BookShape for PI OOF split
b_near = [c for c in all_cols if 'Lag' not in c and
          any(f'_B0{i}' in c for i in range(5))]
b_far  = [c for c in all_cols if 'Lag' not in c and
          any(f'_B{i}' in c for i in ['06','07','08','09','10'])]
train['bookshape'] = (train[b_near].fillna(0).sum(1) -
                      train[b_far].fillna(0).sum(1)).astype(np.float64)
test['bookshape']  = (test[b_near].fillna(0).sum(1) -
                      test[b_far].fillna(0).sum(1)).astype(np.float64)

ic_arr_top10   = np.array([ic_weights[f]   for f in top10])
icir_arr_top10 = np.array([icir_weights[f] * np.sign(ic_weights[f]) for f in top10])

print(f"Train: {len(train):,} rows  Test: {len(test):,} rows")
print(f"Overlap: {len(overlap)} days  Future: {len(new_days)} days")

# ── Helpers ────────────────────────────────────────────────────────────────
def zscore_fit(X, clip=5.0):
    m = X.mean(0); s = X.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip), m, s

def zscore_apply(X, m, s, clip=5.0):
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip)

def winsorise(y, lo=1, hi=99):
    return np.clip(y, np.percentile(y, lo), np.percentile(y, hi))

def per_day_ic(y_true, y_pred):
    if len(y_true) < 5: return np.nan
    return spearmanr(y_true, y_pred)[0]

def rank_norm(pred):
    """Per-vector rank normalization → uniform in [-0.5, +0.5]."""
    if len(pred) < 2: return pred
    return rankdata(pred) / len(pred) - 0.5

def knn_predict(X_query, X_ref, y_ref, K):
    """Weighted KNN: predict y for query using top-K neighbors from ref."""
    sim = cosine_similarity(X_query, X_ref)          # (n_query, n_ref)
    K   = min(K, sim.shape[1])
    topk_idx = np.argpartition(sim, -K, axis=1)[:, -K:]  # (n_query, K)
    # Gather similarity weights
    rows = np.arange(len(X_query))[:, None]
    w    = sim[rows, topk_idx]                        # (n_query, K)
    w    = np.maximum(w, 0)                           # no negative weights
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum < 1e-10, 1.0, w_sum)
    w    = w / w_sum                                  # normalize
    y_pred = (w * y_ref[topk_idx]).sum(axis=1)
    return y_pred


# ================================================================
# PART 1: PI OOF VALIDATION
# ================================================================
print("\n" + "=" * 70)
print("PART 1: PI OOF — Grinold vs Top-K KNN vs Rank-Normalized variants")
print("=" * 70)

K_values   = [5, 10, 20, 50]
ridge_alpha = 10.0   # confirmed best from perday_ridge.py

ic_store = {
    'grinold':          [],
    'ridge_top10':      [],
}
for K in K_values:
    ic_store[f'knn_K{K}']         = []
    ic_store[f'knn_K{K}_ranked']  = []

ic_store['grinold_ranked']      = []
ic_store['ridge_ranked']        = []

# Threeway variants (will be built from components)
# 30% Ridge + 40% KNN-best + 30% Grinold  (ranked and unranked)
# assembled after we know the best K

day_count = 0
for day, grp in train.groupby('day_id'):
    n = len(grp)
    if n < 20: continue

    y_day  = y_train[grp.index]
    bs     = grp['bookshape'].values
    bs_med = np.median(bs)

    liq_mask   = bs >= bs_med
    illiq_mask = bs <  bs_med
    if liq_mask.sum() < 10 or illiq_mask.sum() < 5: continue

    y_liq   = y_day[liq_mask]
    y_illiq = y_day[illiq_mask]
    y_liq_w = winsorise(y_liq)

    X_all = grp[top10].fillna(0).values.astype(np.float64)
    X_all_z, m_all, s_all = zscore_fit(X_all)

    X_liq_z   = X_all_z[liq_mask]
    X_illiq_z = X_all_z[illiq_mask]

    # ── Grinold ──────────────────────────────────────────────────
    pred_g = X_all_z @ ic_arr_top10
    pred_g -= pred_g.mean()
    ic_store['grinold'].append(per_day_ic(y_illiq, pred_g[illiq_mask]))
    ic_store['grinold_ranked'].append(
        per_day_ic(y_illiq, rank_norm(pred_g[illiq_mask])))

    # ── Per-day Ridge ─────────────────────────────────────────────
    # z-score using liquid stats
    _, m_liq, s_liq = zscore_fit(X_all[liq_mask])
    X_liq_lnorm   = zscore_apply(X_all[liq_mask],   m_liq, s_liq)
    X_illiq_lnorm = zscore_apply(X_all[illiq_mask], m_liq, s_liq)
    X_all_lnorm   = zscore_apply(X_all,             m_liq, s_liq)

    ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
    ridge.fit(X_liq_lnorm, y_liq_w)
    pred_r = ridge.predict(X_all_lnorm)
    pred_r -= pred_r.mean()
    ic_store['ridge_top10'].append(per_day_ic(y_illiq, pred_r[illiq_mask]))
    ic_store['ridge_ranked'].append(
        per_day_ic(y_illiq, rank_norm(pred_r[illiq_mask])))

    # ── Top-K KNN ─────────────────────────────────────────────────
    for K in K_values:
        pred_knn = knn_predict(X_illiq_z, X_liq_z, y_liq, K)
        pred_knn -= pred_knn.mean()
        ic_store[f'knn_K{K}'].append(per_day_ic(y_illiq, pred_knn))
        ic_store[f'knn_K{K}_ranked'].append(
            per_day_ic(y_illiq, rank_norm(pred_knn)))

    day_count += 1

elapsed = (time.time() - t0) / 60
print(f"\n  Validated on {day_count} training days [{elapsed:.1f}m elapsed]")
print(f"\n  {'Model':<28}  {'Med IC':>10}  {'%pos':>8}  {'p25':>10}  {'p75':>10}")
print(f"  {'-' * 72}")

results_summary = {}
best_ic, best_model = -999, None
for k, v in ic_store.items():
    arr  = np.array([x for x in v if not np.isnan(x)])
    if len(arr) == 0: continue
    med  = np.nanmedian(arr)
    ppos = (arr > 0).mean() * 100
    p25, p75 = np.percentile(arr, [25, 75])
    results_summary[k] = {'med': med, 'ppos': ppos}
    marker = ' ←' if med > best_ic else ''
    if med > best_ic:
        best_ic = med; best_model = k
    print(f"  {k:<28}  {med:+10.5f}  {ppos:7.1f}%  {p25:+10.5f}  {p75:+10.5f}{marker}")

print(f"\n  WINNER: {best_model}  (Med IC = {best_ic:+.5f})")


# ================================================================
# PART 2: RANK NORMALIZATION IMPACT ANALYSIS
# ================================================================
print("\n" + "=" * 70)
print("PART 2: RANK NORMALIZATION — Does It Help Each Model?")
print("=" * 70)

pairs = [
    ('grinold',     'grinold_ranked'),
    ('ridge_top10', 'ridge_ranked'),
]
for K in K_values:
    pairs.append((f'knn_K{K}', f'knn_K{K}_ranked'))

print(f"\n  {'Model':<15}  {'Raw IC':>10}  {'Ranked IC':>10}  {'Delta':>10}  {'Verdict'}")
print(f"  {'-' * 65}")
for raw_key, ranked_key in pairs:
    raw_med    = results_summary.get(raw_key,    {}).get('med', np.nan)
    ranked_med = results_summary.get(ranked_key, {}).get('med', np.nan)
    delta      = ranked_med - raw_med
    verdict    = 'HELPS' if delta > 0.0005 else ('HURTS' if delta < -0.0005 else 'NEUTRAL')
    print(f"  {raw_key:<15}  {raw_med:+10.5f}  {ranked_med:+10.5f}  {delta:+10.5f}  {verdict}")


# ================================================================
# PART 3: GENERATE TEST PREDICTIONS
# ================================================================
print("\n" + "=" * 70)
print("PART 3: GENERATING TEST PREDICTIONS")
print("=" * 70)

# Find best K
best_K = max(K_values,
             key=lambda K: results_summary.get(f'knn_K{K}', {}).get('med', -999))
best_K_ranked = max(K_values,
                    key=lambda K: results_summary.get(f'knn_K{K}_ranked', {}).get('med', -999))
print(f"\n  Best K (raw):    K={best_K}  "
      f"Med IC={results_summary[f'knn_K{best_K}']['med']:+.5f}")
print(f"  Best K (ranked): K={best_K_ranked}  "
      f"Med IC={results_summary[f'knn_K{best_K_ranked}_ranked']['med']:+.5f}")

te_preds_grinold = np.zeros(len(test))
te_preds_ridge   = np.zeros(len(test))
te_preds_knn     = {K: np.zeros(len(test)) for K in K_values}

n_overlap = 0
for day, grp_te in test.groupby('day_id'):
    te_idx     = grp_te.index
    X_te_top10 = grp_te[top10].fillna(0).values.astype(np.float64)

    if day in train_days:
        grp_tr = train[train['day_id'] == day]
        y_liq  = y_train[grp_tr.index]
        y_liq_w = winsorise(y_liq)
        X_tr   = grp_tr[top10].fillna(0).values.astype(np.float64)

        # Grinold: z-score test assets
        X_te_z, _, _ = zscore_fit(X_te_top10)
        pred_g = X_te_z @ ic_arr_top10
        pred_g -= pred_g.mean()
        te_preds_grinold[te_idx] = pred_g

        # Ridge: z-score using liquid (train) stats
        _, m_liq, s_liq = zscore_fit(X_tr)
        X_tr_z  = zscore_apply(X_tr,       m_liq, s_liq)
        X_te_lz = zscore_apply(X_te_top10, m_liq, s_liq)
        ridge   = Ridge(alpha=ridge_alpha, fit_intercept=False)
        ridge.fit(X_tr_z, y_liq_w)
        pred_r  = ridge.predict(X_te_lz)
        pred_r -= pred_r.mean()
        te_preds_ridge[te_idx] = pred_r

        # Top-K KNN
        # z-score combined (test day stats include only illiquid test assets)
        # Use liquid (train) stats to normalize both
        X_tr_z_knn = zscore_apply(X_tr,       m_liq, s_liq)
        X_te_z_knn = zscore_apply(X_te_top10, m_liq, s_liq)
        for K in K_values:
            pred_knn = knn_predict(X_te_z_knn, X_tr_z_knn, y_liq, K)
            pred_knn -= pred_knn.mean()
            te_preds_knn[K][te_idx] = pred_knn

        n_overlap += 1

    else:
        # Future day: Grinold fallback for all
        X_te_z, _, _ = zscore_fit(X_te_top10)
        pred_g = X_te_z @ ic_arr_top10
        pred_g -= pred_g.mean()
        te_preds_grinold[te_idx] = pred_g
        te_preds_ridge[te_idx]   = pred_g
        for K in K_values:
            te_preds_knn[K][te_idx] = pred_g

elapsed = (time.time() - t0) / 60
print(f"  Overlap days: {n_overlap}  Future days: {len(new_days)}  [{elapsed:.1f}m]")


# ================================================================
# PART 4: BUILD ALL SUBMISSION VARIANTS
# ================================================================
print("\n" + "=" * 70)
print("PART 4: SUBMISSION VARIANTS")
print("=" * 70)

TARGET_STD = 0.000948
sample_sub = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/sample_submission.csv'))[['ID']]

def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

def rank_norm_test(preds_arr, test_day_ids):
    """Apply per-day rank normalization to test predictions."""
    out = preds_arr.copy()
    for day in np.unique(test_day_ids):
        mask = test_day_ids == day
        if mask.sum() < 2: continue
        out[mask] = rank_norm(preds_arr[mask])
    return out

def save_sub(preds, name):
    preds_s = auto_scale(preds)
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': preds_s})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    t = sub['TARGET']
    return t.std(), (t > 0).mean() * 100

test_day_ids = test['day_id'].values

g_s = auto_scale(te_preds_grinold)
r_s = auto_scale(te_preds_ridge)
k_s = {K: auto_scale(te_preds_knn[K]) for K in K_values}

# Load existing kernel (Gaussian) predictions for comparison
kernel_path = os.path.join(OUT_DIR, 'hybrid_grinold_kernel.csv')
has_kernel  = os.path.exists(kernel_path)
if has_kernel:
    kdf = pd.read_csv(kernel_path)
    kdf = sample_sub.merge(kdf[['ID','TARGET']].rename(
        columns={'TARGET':'k_pred'}), on='ID', how='left').fillna(0.0)
    gauss_kernel_preds = kdf['k_pred'].values
    print(f"\n  Loaded existing Gaussian kernel hybrid for comparison.")

print(f"\n  {'Submission name':<45}  {'std':>10}  {'%pos':>8}")
print(f"  {'-' * 68}")

submissions_info = []

# 1. Pure top-K KNN variants (raw and ranked)
for K in K_values:
    std, ppos = save_sub(te_preds_knn[K], f'knn_K{K}_pure')
    print(f"  knn_K{K}_pure                                  {std:10.7f}  {ppos:7.1f}%")

    pred_rk = rank_norm_test(te_preds_knn[K], test_day_ids)
    std, ppos = save_sub(pred_rk, f'knn_K{K}_pure_ranked')
    print(f"  knn_K{K}_pure_ranked                           {std:10.7f}  {ppos:7.1f}%")

# 2. KNN + Grinold hybrids (best K, mirroring kernel hybrid success)
print()
for alpha in [0.3, 0.5, 0.7]:
    for K in [best_K, best_K_ranked]:
        blend = alpha * te_preds_knn[K] + (1 - alpha) * te_preds_grinold
        name  = f'knn_K{K}_a{int(alpha*100)}_grinold'
        std, ppos = save_sub(blend, name)
        if alpha == 0.7:  # only print key ones
            print(f"  {name:<45}  {std:10.7f}  {ppos:7.1f}%")

# 3. Threeway: Ridge + KNN (best K) + Grinold
print()
for r_w, k_w in [(0.30, 0.40), (0.25, 0.45), (0.20, 0.50), (0.35, 0.35)]:
    g_w = 1.0 - r_w - k_w
    if g_w < 0: continue
    blend = r_w * te_preds_ridge + k_w * te_preds_knn[best_K] + g_w * te_preds_grinold
    name  = f'tw_r{int(r_w*100)}_k{int(k_w*100)}K{best_K}_g{int(round(g_w*100))}'
    std, ppos = save_sub(blend, name)
    print(f"  {name:<45}  {std:10.7f}  {ppos:7.1f}%")

# 4. Rank-normalized threeway (apply rank norm to best existing blend)
print()
# Best existing: 30% Ridge + 40% Gauss-kernel + 30% Grinold = +0.00124
# Rebuild it and apply rank norm
if has_kernel:
    # Extract pure kernel from hybrid_grinold_kernel (= 0.7*kernel + 0.3*grinold)
    # We can approximate: just rank-normalize the existing threeway prediction
    # Load threeway_r30_k40_g29 and rank-normalize it
    tw_path = os.path.join(OUT_DIR, 'threeway_r30_k40_g29.csv')
    if os.path.exists(tw_path):
        tw_df = pd.read_csv(tw_path)
        tw_df = sample_sub.merge(tw_df[['ID','TARGET']].rename(
            columns={'TARGET': 'tw_pred'}), on='ID', how='left').fillna(0.0)
        tw_preds = tw_df['tw_pred'].values

        tw_ranked = rank_norm_test(tw_preds, test_day_ids)
        std, ppos = save_sub(tw_ranked, 'threeway_r30_k40_g29_ranked')
        print(f"  threeway_r30_k40_g29_ranked (CURRENT BEST + rank norm)")
        print(f"  {'':45}  {std:10.7f}  {ppos:7.1f}%")

        # Also: rebuild threeway with new top-K KNN instead of Gauss kernel, then rank-norm
        blend_new = 0.30 * te_preds_ridge + 0.40 * te_preds_knn[best_K] + 0.30 * te_preds_grinold
        blend_new_rk = rank_norm_test(blend_new, test_day_ids)
        std, ppos = save_sub(blend_new_rk, f'tw_r30_K{best_K}knn_g30_ranked')
        print(f"  tw_r30_K{best_K}knn_g30_ranked (new KNN + rank norm)")
        print(f"  {'':45}  {std:10.7f}  {ppos:7.1f}%")


# ================================================================
# PART 5: COMPREHENSIVE ANALYSIS
# ================================================================
print("\n" + "=" * 70)
print("PART 5: COMPREHENSIVE ANALYSIS")
print("=" * 70)

# Correlation matrix: Grinold, Ridge, KNN variants, Gauss-kernel blend
from numpy.linalg import norm
def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = norm(a) * norm(b)
    return (a @ b) / d if d > 1e-10 else 0.0

print(f"\n  Correlation of signals with Grinold:")
print(f"  {'Signal':<30}  {'Corr vs Grinold':>16}  {'Sign agree':>12}")
print(f"  {'-' * 62}")
g_raw = te_preds_grinold
for K in K_values:
    k_raw = te_preds_knn[K]
    corr  = pearson_r(g_raw, k_raw)
    sa    = (np.sign(k_raw) == np.sign(g_raw)).mean() * 100
    print(f"  knn_K{K:<25}  {corr:+16.4f}  {sa:11.1f}%")

corr_r = pearson_r(g_raw, te_preds_ridge)
sa_r   = (np.sign(te_preds_ridge) == np.sign(g_raw)).mean() * 100
print(f"  ridge_top10                     {corr_r:+16.4f}  {sa_r:11.1f}%")

if has_kernel:
    corr_gk = pearson_r(g_raw, gauss_kernel_preds)
    sa_gk   = (np.sign(gauss_kernel_preds) == np.sign(g_raw)).mean() * 100
    print(f"  gauss_kernel_hybrid             {corr_gk:+16.4f}  {sa_gk:11.1f}%")

print(f"\n  Cross-correlations (KNN variants vs Ridge):")
print(f"  {'Signal':<30}  {'Corr vs Ridge':>14}")
print(f"  {'-' * 48}")
for K in K_values:
    k_raw = te_preds_knn[K]
    corr  = pearson_r(te_preds_ridge, k_raw)
    print(f"  knn_K{K:<25}  {corr:+14.4f}")


# ================================================================
# PART 6: SUBMISSION PRIORITY GUIDE
# ================================================================
print("\n" + "=" * 70)
print("PART 6: SUBMISSION PRIORITY GUIDE")
print("=" * 70)

grinold_pi  = results_summary.get('grinold',     {}).get('med', 0)
ridge_pi    = results_summary.get('ridge_top10', {}).get('med', 0)
best_knn_pi = results_summary.get(f'knn_K{best_K}', {}).get('med', 0)
best_knn_rk_pi = results_summary.get(f'knn_K{best_K_ranked}_ranked', {}).get('med', 0)

print(f"""
  ── CONFIRMED LB SCORES ─────────────────────────────────────────
  grinold_allday_top10_probe_005      +0.00096  (pure Grinold)
  hybrid_grinold_kernel               +0.00115  (70% Gauss kernel + 30% Grinold)
  ridge_hybrid_a070                   +0.00086  (70% Ridge + 30% Grinold — HURTS)
  threeway_r30_k40_g29                +0.00124  ← CURRENT BEST

  ── PI OOF THIS SESSION ─────────────────────────────────────────
  Grinold top10:                      {grinold_pi:+.5f}  (baseline)
  Per-day Ridge:                      {ridge_pi:+.5f}
  Best KNN (K={best_K}):               {best_knn_pi:+.5f}
  Best KNN ranked (K={best_K_ranked}):          {best_knn_rk_pi:+.5f}

  ── SUBMISSION PRIORITY ─────────────────────────────────────────
  NOTE: PI OOF is unreliable for same-day return methods
        (kernel proof: PI OOF +0.022 but LB contributed to +0.00124).
        Submit in order below and let LB decide.

  1. threeway_r30_k40_g29_ranked.csv
     → HIGHEST PRIORITY. Apply rank norm to current best (+0.00124).
        If rank norm helps at all, this beats +0.00124.
        If neutral/hurts, we learn rank norm is not useful.

  2. tw_r30_K{best_K}knn_g30_ranked.csv
     → Replace Gaussian kernel with explicit K={best_K} KNN + rank norm.
        Tests both new ideas simultaneously.

  3. knn_K{best_K}_a70_grinold.csv
     → 70% KNN(K={best_K}) + 30% Grinold. Direct analog of kernel hybrid.
        Tells us if explicit KNN > Gaussian kernel at same blend ratio.

  Total elapsed: {(time.time()-t0)/60:.1f} min
""")
