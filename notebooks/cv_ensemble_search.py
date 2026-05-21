# ================================================================
# CV ENSEMBLE SEARCH — NW Kernel + Weight Optimization
# ================================================================
# Stage 1: NW Kernel search
#   K values: 3, 5, 10, 15, 20, 30
#   Feature sets: top-5, top-10, top-15, top-20 gold
#   For each config: per-day R² on held-out val assets (CV_GROUP)
#
# Stage 2: Weight optimization
#   Components: Ridge (top-10) + best NW KNN + Grinold (top-10)
#   Grid search w_r, w_k, w_g with w_r + w_k + w_g = 1.0
#   Find the blend that maximises per-day R² on held-out val assets
#
# Key design (leak-free):
#   - Grinold IC computed inside fold (same as cv_grinold_search.py)
#   - Ridge fit per-day on TRAIN fold assets only
#   - KNN uses TRAIN fold assets as reference, predicts VAL fold
#   - All normalisation stats from FULL day (train+val assets)
#     for features (matches production where all assets are present)
#
# Distribution shift note:
#   CV measures liquid→liquid (all assets in train are liquid).
#   Test is liquid→illiquid (KS=0.37 gap).
#   Linear/simple models transfer better; prefer simpler configs.
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/cv_results')
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS     = 5
CLIP_Z      = 5.0
RIDGE_ALPHA = 10.0

print("=" * 70)
print("CV ENSEMBLE SEARCH — NW Kernel K/Feature + Weight Optimisation")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
print(f"  Rows: {len(train):,} | Days: {train['day_id'].nunique()}")

icir_df  = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df  = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
all_gold = [f for f in gold_df['feature'].tolist() if f in train.columns]
print(f"  Gold features: {len(all_gold)}")

groups_sorted = sorted(train['CV_GROUP'].unique())
fold_map = {g: i % N_FOLDS for i, g in enumerate(groups_sorted)}
train['_fold'] = train['CV_GROUP'].map(fold_map)

y_all = train['TARGET'].values.astype(np.float32)

# ── Pre-group data per day ────────────────────────────────────────────
print("\nPre-grouping per day (float32, all gold features)...")

def zscore_day(X, clip=CLIP_Z):
    m = X.mean(0, keepdims=True)
    s = X.std(0, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip).astype(np.float32)

day_keys  = []
day_X_z   = []   # z-scored, all gold features, float32
day_y     = []
day_folds = []

for day, grp in train.groupby('day_id'):
    if len(grp) < 10: continue
    X_raw = grp[all_gold].fillna(0).values.astype(np.float32)
    day_keys.append(day)
    day_X_z.append(zscore_day(X_raw))
    day_y.append(y_all[grp.index])
    day_folds.append(grp['_fold'].values.astype(np.int8))

n_days = len(day_keys)
print(f"  Grouped {n_days} days  [{(time.time()-t0)/60:.1f}m]")

# ── Per-day R² ────────────────────────────────────────────────────────
def compute_perday_r2(preds, targets, days):
    r2s = []
    df = pd.DataFrame({'p': preds, 't': targets, 'd': days})
    for _, grp in df.groupby('d'):
        p = grp['p'].values; t = grp['t'].values
        if len(p) < 3: continue
        p_dm = p - p.mean(); t_dm = t - t.mean()
        ss_tot = (t_dm**2).sum()
        if ss_tot < 1e-12: continue
        r2s.append(1.0 - ((p_dm - t_dm)**2).sum() / ss_tot)
    return (np.mean(r2s), np.std(r2s), len(r2s)) if r2s else (np.nan, np.nan, 0)

# ── KNN prediction helper ─────────────────────────────────────────────
def knn_predict(X_q, X_r, y_r, K):
    sim   = cosine_similarity(X_q, X_r)
    K     = min(K, sim.shape[1])
    topk  = np.argpartition(sim, -K, axis=1)[:, -K:]
    w     = np.maximum(sim[np.arange(len(X_q))[:, None], topk], 0)
    ws    = w.sum(1, keepdims=True)
    w    /= np.where(ws < 1e-10, 1.0, ws)
    return (w * y_r[topk]).sum(1)

# ── Main CV loop — compute ALL component OOF predictions ──────────────
# Feature sets to test for KNN
KNN_FEAT_SETS = {
    'f05': list(range(5)),
    'f10': list(range(10)),
    'f15': list(range(15)),
    'f20': list(range(20)),
}
KNN_K_VALUES = [3, 5, 10, 15, 20, 30]
GRINOLD_FEATS = list(range(10))   # top-10 gold
RIDGE_FEATS   = list(range(10))   # top-10 gold

print("\nRunning CV loop (all 5 folds)...")

# Storage: lists per component
oof = {
    'grinold': {'p': [], 't': [], 'd': []},
    'ridge':   {'p': [], 't': [], 'd': []},
}
for fname in KNN_FEAT_SETS:
    for K in KNN_K_VALUES:
        oof[f'knn_{fname}_k{K:02d}'] = {'p': [], 't': [], 'd': []}

for fold in range(N_FOLDS):
    print(f"  Fold {fold}...", end=' ', flush=True)

    # ── Step 1: Compute in-fold Grinold ICs (pooled train assets) ────
    g_pool_X, g_pool_y = [], []
    for X_day, y_day, folds_day in zip(day_X_z, day_y, day_folds):
        tr = folds_day != fold
        if tr.sum() < 5: continue
        Xg = X_day[:, GRINOLD_FEATS]
        g_pool_X.append(Xg[tr])
        g_pool_y.append(y_day[tr])
    X_pool = np.vstack(g_pool_X)
    y_pool = np.concatenate(g_pool_y)
    y_dm   = y_pool - y_pool.mean()
    ic_arr = (X_pool * y_dm[:, None]).mean(0)   # (10,), vectorised
    del X_pool, y_pool, g_pool_X, g_pool_y

    # ── Step 2: Per-day predictions ──────────────────────────────────
    for day_key, X_day, y_day, folds_day in zip(day_keys, day_X_z, day_y, day_folds):
        tr = folds_day != fold
        va = folds_day == fold
        if va.sum() == 0: continue

        # ── Grinold (in-fold IC) ──────────────────────────────────
        X_va_g = X_day[va][:, GRINOLD_FEATS]
        p_g    = (X_va_g @ ic_arr).astype(np.float32)
        oof['grinold']['p'].extend(p_g.tolist())
        oof['grinold']['t'].extend(y_day[va].tolist())
        oof['grinold']['d'].extend([day_key] * int(va.sum()))

        # ── Ridge (per-day fit on train assets) ───────────────────
        if tr.sum() >= 5:
            X_tr_r = X_day[tr][:, RIDGE_FEATS].astype(np.float64)
            X_va_r = X_day[va][:, RIDGE_FEATS].astype(np.float64)
            y_tr   = y_day[tr].astype(np.float64)
            # Winsorise targets (same as threeway)
            lo, hi = np.percentile(y_tr, 1), np.percentile(y_tr, 99)
            y_tr_w = np.clip(y_tr, lo, hi)
            mdl = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
            mdl.fit(X_tr_r, y_tr_w)
            p_r = mdl.predict(X_va_r).astype(np.float32)
            oof['ridge']['p'].extend(p_r.tolist())
            oof['ridge']['t'].extend(y_day[va].tolist())
            oof['ridge']['d'].extend([day_key] * int(va.sum()))

        # ── KNN (per feature set) ─────────────────────────────────
        if tr.sum() >= max(KNN_K_VALUES):
            y_tr_knn = y_day[tr]
            for fname, fidx in KNN_FEAT_SETS.items():
                X_tr_k = X_day[tr][:, fidx]
                X_va_k = X_day[va][:, fidx]
                # Compute full sim matrix once, reuse for all K
                sim = cosine_similarity(X_va_k, X_tr_k)  # (n_val, n_tr)
                n_tr = sim.shape[1]
                for K in KNN_K_VALUES:
                    K_eff = min(K, n_tr)
                    topk  = np.argpartition(sim, -K_eff, axis=1)[:, -K_eff:]
                    w     = np.maximum(sim[np.arange(len(X_va_k))[:, None], topk], 0)
                    ws    = w.sum(1, keepdims=True)
                    w    /= np.where(ws < 1e-10, 1.0, ws)
                    p_k   = (w * y_tr_knn[topk]).sum(1).astype(np.float32)
                    key   = f'knn_{fname}_k{K:02d}'
                    oof[key]['p'].extend(p_k.tolist())
                    oof[key]['t'].extend(y_day[va].tolist())
                    oof[key]['d'].extend([day_key] * int(va.sum()))

    print(f"done  [{(time.time()-t0)/60:.1f}m]")

# ── Convert to arrays ─────────────────────────────────────────────────
for key in oof:
    oof[key]['p'] = np.array(oof[key]['p'], dtype=np.float32)
    oof[key]['t'] = np.array(oof[key]['t'], dtype=np.float32)

print(f"\n  OOF sizes — grinold: {len(oof['grinold']['p']):,}  "
      f"ridge: {len(oof['ridge']['p']):,}")

# ── Stage 1: NW Kernel search ─────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 1 — NW Kernel search (K × feature set)")
print("=" * 70)

knn_results = []
for fname in KNN_FEAT_SETS:
    key0 = f'knn_{fname}_k{KNN_K_VALUES[0]:02d}'
    if not oof[key0]['p'].size: continue
    for K in KNN_K_VALUES:
        key = f'knn_{fname}_k{K:02d}'
        r2, std, nd = compute_perday_r2(oof[key]['p'], oof[key]['t'], oof[key]['d'])
        knn_results.append({'config': key, 'feats': fname,
                            'K': K, 'r2': r2, 'std': std})
        print(f"  {key:<25}  R²={r2:.5f}  ±{std:.5f}")

knn_df  = pd.DataFrame(knn_results).sort_values('r2', ascending=False)
best_knn_key  = knn_df.iloc[0]['config']
best_knn_feat = knn_df.iloc[0]['feats']
best_knn_K    = int(knn_df.iloc[0]['K'])
best_knn_r2   = knn_df.iloc[0]['r2']

# Grinold and Ridge baselines
g_r2, _, _ = compute_perday_r2(oof['grinold']['p'], oof['grinold']['t'], oof['grinold']['d'])
r_r2, _, _ = compute_perday_r2(oof['ridge']['p'],   oof['ridge']['t'],   oof['ridge']['d'])
print(f"\n  Grinold-10 baseline R²:   {g_r2:.5f}")
print(f"  Ridge-10   baseline R²:   {r_r2:.5f}")
print(f"  Best KNN config:          {best_knn_key}  R²={best_knn_r2:.5f}")

# ── Stage 2: Weight optimisation ──────────────────────────────────────
print("\n" + "=" * 70)
print(f"STAGE 2 — Weight grid search  (Ridge + {best_knn_key} + Grinold)")
print("=" * 70)

# Align all three components to the same index (grinold as reference)
# They were all collected in the same day order but ridge/knn might skip
# some days. Use grinold's day list as the universe and match.
g_days = oof['grinold']['d']
g_p    = oof['grinold']['p']
g_t    = oof['grinold']['t']

r_df  = pd.DataFrame({'p': oof['ridge']['p'],   'd': oof['ridge']['d'],   't': oof['ridge']['t']})
k_df  = pd.DataFrame({'p': oof[best_knn_key]['p'], 'd': oof[best_knn_key]['d']})
g_df  = pd.DataFrame({'p': g_p, 't': g_t, 'd': g_days})

# Restrict to days where ALL three components have predictions.
# Grinold collects every val row; Ridge/KNN skip days where train < threshold.
# Align by intersecting days (same loop order guarantees row order within days).
common_days = set(g_df['d']) & set(r_df['d']) & set(k_df['d'])
g_df = g_df[g_df['d'].isin(common_days)].reset_index(drop=True)
r_df = r_df[r_df['d'].isin(common_days)].reset_index(drop=True)
k_df = k_df[k_df['d'].isin(common_days)].reset_index(drop=True)
print(f"  Aligned to {len(common_days)} common days  "
      f"(g={len(g_df)}, r={len(r_df)}, k={len(k_df)} rows)")

p_g = g_df['p'].values.astype(np.float32)
p_r = r_df['p'].values.astype(np.float32)
p_k = k_df['p'].values.astype(np.float32)
t   = g_df['t'].values.astype(np.float32)
d   = g_df['d'].values

# Grid search — step of 0.05
weight_results = []
steps = np.arange(0.10, 0.65, 0.05)
for wr in steps:
    for wk in steps:
        wg = 1.0 - wr - wk
        if wg < 0.05 or wg > 0.85: continue
        p_blend = wr * p_r + wk * p_k + wg * p_g
        r2, std, nd = compute_perday_r2(p_blend, t, d)
        weight_results.append({'wr': round(wr, 2), 'wk': round(wk, 2),
                                'wg': round(wg, 2), 'r2': r2, 'std': std})

w_df = pd.DataFrame(weight_results).sort_values('r2', ascending=False)

print("\n  Top-20 weight combinations:")
print(f"  {'wr':>6}  {'wk':>6}  {'wg':>6}  {'R²':>10}  {'±std':>10}")
for _, row in w_df.head(20).iterrows():
    print(f"  {row['wr']:>6.2f}  {row['wk']:>6.2f}  {row['wg']:>6.2f}  "
          f"{row['r2']:>10.5f}  {row['std']:>10.5f}")

best_w = w_df.iloc[0]
# Current threeway weights (effective): 0.30/~0.28/~0.41
cur_g_r2 = 0.00229   # from cv_grinold_search.py

print(f"\n  Current threeway effective weights: r≈0.30  k≈0.28  g≈0.41")
print(f"  Best weights found:  r={best_w['wr']:.2f}  "
      f"k={best_w['wk']:.2f}  g={best_w['wg']:.2f}  "
      f"R²={best_w['r2']:.5f}")
print(f"  Current threeway R² (est): ~{cur_g_r2:.5f}")
print(f"  Improvement: {best_w['r2'] - cur_g_r2:+.5f}")

# ── Save ───────────────────────────────────────────────────────────────
knn_df.to_csv(os.path.join(OUT_DIR, 'cv_nw_search.csv'), index=False)
w_df.to_csv(os.path.join(OUT_DIR, 'cv_weight_opt.csv'), index=False)

print(f"\n  Saved: cv_nw_search.csv  cv_weight_opt.csv")
print(f"  Total: {(time.time()-t0)/60:.1f} min")

print(f"""
  ── SUMMARY ──────────────────────────────────────────────────
  Best NW Kernel:    K={best_knn_K}, feats={best_knn_feat}  R²={best_knn_r2:.5f}
  Best blend:        r={best_w['wr']:.2f}  k={best_w['wk']:.2f}  g={best_w['wg']:.2f}
  Best blend R²:     {best_w['r2']:.5f}

  Use these in the submission script to rebuild threeway with
  optimised NW kernel and weights.
""")
