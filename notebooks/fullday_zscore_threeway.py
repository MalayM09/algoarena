# ================================================================
# FULL-DAY Z-SCORE THREEWAY — Combined liquid+illiquid normalization
# ================================================================
# Research finding (Numerai / Ubiquant winners):
#   Cross-sectional normalization should use ALL rows in a day
#   (both labeled liquid train + unlabeled illiquid test) to
#   accurately capture the true cross-section.
#
# Why this matters:
#   Current approach: liquid rows z-scored against liquid pool;
#   illiquid rows z-scored against illiquid pool SEPARATELY.
#   Problem: z-score means are biased to their own group.
#   The systematic offset between liquid and illiquid distributions
#   is normalized away — KNN can't see the distributional difference.
#
#   Combined approach: all assets on a day share the same mean/std.
#   This preserves the cross-sectional offset between liquid/illiquid,
#   giving KNN cosine similarity a richer signal.
#
# Note: For Grinold (linear IC dot product), z-score rankings are
# invariant to affine rescaling → Grinold predictions are identical.
# Impact is purely on the KNN and Ridge components.
#
# Generates:
#   fullday_threeway_k10.csv — standard K=10 with full-day z-score
#   fullday_threeway_k05.csv — K=5 with full-day z-score
#   fullday_threeway_k20.csv — K=20 with full-day z-score
#   fullday_threeway_k30.csv — K=30 with full-day z-score
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
OUT_DIR     = os.path.join(BASE_DIR, 'outputs/submissions')
os.makedirs(OUT_DIR, exist_ok=True)

RIDGE_ALPHA = 10.0
TARGET_STD  = 0.000948
K_VALUES    = [5, 10, 20, 30]

print("=" * 65)
print("FULL-DAY Z-SCORE THREEWAY — Combined liquid+illiquid normalization")
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
print(f"  Train: {len(train):,}  Test: {n_test:,}")
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
print(f"  Gold features: {len(all_gold)}  using top-10")

# ── Helpers ────────────────────────────────────────────────────────
def zscore_combined(X_liq, X_illiq, clip=5.0):
    """
    Z-score both liquid and illiquid using COMBINED stats.
    Returns (X_liq_z, X_illiq_z, mean, std).
    """
    X_all = np.vstack([X_liq, X_illiq])
    m = X_all.mean(0)
    s = X_all.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    X_liq_z   = np.clip((X_liq   - m) / s, -clip, clip)
    X_illiq_z = np.clip((X_illiq - m) / s, -clip, clip)
    return X_liq_z, X_illiq_z, m, s

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

# ── Predictions ────────────────────────────────────────────────────
print(f"\nGenerating predictions (Ridge + Grinold + KNN×{len(K_VALUES)})...")

te_ridge   = np.zeros(n_test)
te_grinold = np.zeros(n_test)
te_knn     = {K: np.zeros(n_test) for K in K_VALUES}

for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index.values
    X_te10 = grp_te[top10].fillna(0).values.astype(np.float64)

    if day in train_days:
        grp_tr = train[train['day_id'] == day]
        y_tr   = y_train[grp_tr.index]
        X_tr10 = grp_tr[top10].fillna(0).values.astype(np.float64)

        # ── FULL-DAY Z-SCORE: combined liquid + illiquid stats ──
        X_tr_z, X_te_z, m_comb, s_comb = zscore_combined(X_tr10, X_te10)

        # Grinold: uses liquid-only IC applied to test z-scores
        # (z-score rankings preserved, same predictions as before)
        pred_g = X_te_z @ ic_arr10
        pred_g -= pred_g.mean()
        te_grinold[te_idx] = pred_g

        # Ridge: fit on liquid rows (combined z-scored), predict test
        y_tr_w = winsorise(y_tr)
        mdl = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        mdl.fit(X_tr_z, y_tr_w)
        pred_r = mdl.predict(X_te_z)
        pred_r -= pred_r.mean()
        te_ridge[te_idx] = pred_r

        # KNN: cosine similarity now uses COMBINED z-scores
        # Liquid z-scores have positive offset, illiquid have negative offset
        # → KNN sees distributional difference, finds more appropriate neighbors
        sim = cosine_similarity(X_te_z, X_tr_z)
        max_K = min(max(K_VALUES), sim.shape[1])
        topk_all = np.argpartition(sim, -max_K, axis=1)[:, -max_K:]
        sim_topk = sim[np.arange(len(X_te_z))[:, None], topk_all]

        for K in K_VALUES:
            K_eff = min(K, sim.shape[1])
            if K_eff < max_K:
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
        # New day (no train data): use test-only z-score for Grinold
        Xz_te, _, _ = zscore_fit(X_te10)
        pred_g = Xz_te @ ic_arr10
        pred_g -= pred_g.mean()
        te_grinold[te_idx] = pred_g
        te_ridge[te_idx]   = pred_g
        for K in K_VALUES:
            te_knn[K][te_idx] = pred_g

elapsed = (time.time() - t0) / 60
print(f"  Done.  [{elapsed:.1f}m]")

# ── Component stats ────────────────────────────────────────────────
print(f"\n  Component correlations:")
print(f"  Ridge vs Grinold: {pearson_r(te_ridge, te_grinold):+.4f}")
for K in K_VALUES:
    print(f"  KNN(K={K:2d}) vs Grinold: {pearson_r(te_knn[K], te_grinold):+.4f}"
          f"   vs Ridge: {pearson_r(te_knn[K], te_ridge):+.4f}")

# ── Save submissions ───────────────────────────────────────────────
print("\nBuilding submissions...")
sample_sub = pd.read_csv(SAMPLE_PATH)[['ID']]

def save_sub(preds, name):
    ps = auto_scale(preds)
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': ps})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    t_col = sub['TARGET']
    print(f"  {name:<50}  std={t_col.std():.6f}")

r_s = auto_scale(te_ridge)
g_s = auto_scale(te_grinold)

# Load previous best for correlation check
prev_best_path = os.path.join(OUT_DIR, 'threeway_r30_k40_g29.csv')
if os.path.exists(prev_best_path):
    prev = pd.read_csv(prev_best_path)
    prev = sample_sub.merge(prev[['ID','TARGET']].rename(columns={'TARGET':'prev'}),
                            on='ID', how='left').fillna(0.0)
    prev_vals = prev['prev'].values
    print(f"\n  Prev best (threeway_r30_k40_g29) std={prev_vals.std():.6f}")

# Threeway 30R/40K/30G for each K
for K in K_VALUES:
    k_s = auto_scale(te_knn[K])
    blend = 0.30 * r_s + 0.40 * k_s + 0.30 * g_s
    save_sub(blend, f'fullday_threeway_k{K:02d}')
    if os.path.exists(prev_best_path):
        ps = auto_scale(blend)
        print(f"    corr_vs_prev: {pearson_r(ps, prev_vals):+.4f}")

# Also: pure KNN variants (no Ridge, Grinold as safety)
for K in [10, 20]:
    k_s = auto_scale(te_knn[K])
    blend = 0.40 * k_s + 0.60 * g_s
    save_sub(blend, f'fullday_knn{K:02d}_grinold60')
    if os.path.exists(prev_best_path):
        ps = auto_scale(blend)
        print(f"    corr_vs_prev: {pearson_r(ps, prev_vals):+.4f}")

print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f} min")
print("""
  ── SUBMISSION GUIDE ─────────────────────────────────────────
  fullday_threeway_k10.csv   ← primary (same K as best LB +0.00124)
                               Difference: KNN uses combined z-scores
                               If corr_vs_prev < 0.98: meaningfully different
                               If LB improves: full-day z-score helps

  fullday_threeway_k05.csv   ← smaller K, higher specificity
  fullday_threeway_k20.csv   ← larger K, smoother
  fullday_threeway_k30.csv   ← largest K tested

  Submit fullday_threeway_k10 first (sanity check vs +0.00124).
""")
