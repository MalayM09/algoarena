# ================================================================
# REGIME-TRANSFER VALIDATOR
# ================================================================
# Competition host hint: "CV_GROUPs should be treated as independent
# regimes. They are not guaranteed to represent strictly ordered or
# contiguous time blocks."
#
# Key findings from EDA:
#   - CV_GROUP is an ASSET-LEVEL attribute (97.6% of assets always
#     belong to the same group across all training days)
#   - CV_GROUPs correlate strongly with BookShape (liquidity proxy):
#     Groups {40,31,52,33,22,...} = illiquid-like (negative BookShape)
#     Groups {35,38,63,58,69,...} = liquid-like (positive BookShape)
#   - Test set mean BookShape = -2.48M vs Train mean = +4.97M
#   - CV_GROUPs closest to test = {65,49,14,61,9,62,32,17,3,53}
#
# Validation design:
#   - TRAIN: liquid-like CV_GROUPs (positive mean BookShape)
#   - VALIDATE: illiquid-like CV_GROUPs (negative mean BookShape)
#   This mimics the actual competition structure (liquid→illiquid)
#   and should give more trustworthy KNN K / blend weight estimates.
#
# Experiments:
#   1. Grinold feature count on regime-transfer R²
#   2. KNN K sweep and feature count
#   3. Ridge + KNN + Grinold weight grid on regime-transfer
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
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/cv_results')
os.makedirs(OUT_DIR, exist_ok=True)

RIDGE_ALPHA = 10.0
CLIP_Z      = 5.0

print("=" * 70)
print("REGIME-TRANSFER VALIDATOR — liquid-like train → illiquid-like val")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
y_all = train['TARGET'].values.astype(np.float32)
print(f"  Train: {len(train):,}  Test: {len(test):,}")

# ── Gold features ──────────────────────────────────────────────────────
icir_df  = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df  = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
all_gold = [f for f in gold_df['feature'].tolist() if f in train.columns]
ic_map   = gold_df.set_index('feature')['mean_ic'].to_dict()
print(f"  Gold features: {len(all_gold)}")

# ── Compute BookShape per CV_GROUP ─────────────────────────────────────
print("\nComputing CV_GROUP BookShape (liquidity regime)...")
b_near = [c for c in train.columns if any(f'_B0{i}_' in c for i in range(5))]
b_far  = [c for c in train.columns if any(f'_B0{i}_' in c for i in [6,7,8,9]) or '_B10_' in c]
train['_bs'] = train[b_near].sum(1) - train[b_far].sum(1)
test['_bs']  = test[b_near].sum(1)  - test[b_far].sum(1)

cvg_bs = train.groupby('CV_GROUP')['_bs'].mean()
te_bs_mean = test['_bs'].mean()
print(f"  Test BookShape mean: {te_bs_mean:.0f}")
print(f"  Train BookShape mean: {train['_bs'].mean():.0f}")

# Split: illiquid-like (negative mean BS) vs liquid-like (positive)
illiq_groups = set(cvg_bs[cvg_bs < 0].index.tolist())
liquid_groups = set(cvg_bs[cvg_bs >= 0].index.tolist())

# Sort illiquid groups by proximity to test mean (closest first)
illiq_sorted = (cvg_bs[cvg_bs < 0] - te_bs_mean).abs().sort_values().index.tolist()
print(f"\n  Liquid-like groups ({len(liquid_groups)}): {sorted(liquid_groups)}")
print(f"  Illiquid-like groups ({len(illiq_groups)}): {sorted(illiq_groups)}")
print(f"  Closest to test: {illiq_sorted[:10]}")

# Assign regime labels
train['_regime'] = train['CV_GROUP'].apply(
    lambda g: 'val' if g in illiq_groups else 'train'
)
liq_rows  = (train['_regime'] == 'train').sum()
illiq_rows = (train['_regime'] == 'val').sum()
print(f"\n  Liquid-like (train): {liq_rows:,} rows ({liq_rows/len(train)*100:.1f}%)")
print(f"  Illiquid-like (val): {illiq_rows:,} rows ({illiq_rows/len(train)*100:.1f}%)")

# ── Per-day R² helper ─────────────────────────────────────────────────
def compute_perday_r2(preds, targets, days_list):
    r2s = []
    df = pd.DataFrame({'p': preds, 't': targets, 'd': days_list})
    for _, grp in df.groupby('d'):
        p = grp['p'].values; t = grp['t'].values
        if len(p) < 3: continue
        p_dm = p - p.mean(); t_dm = t - t.mean()
        ss = (t_dm ** 2).sum()
        if ss < 1e-12: continue
        r2s.append(1.0 - ((p_dm - t_dm) ** 2).sum() / ss)
    if not r2s: return np.nan, np.nan, 0
    return np.mean(r2s), np.std(r2s), len(r2s)

# ── Pre-group by day ───────────────────────────────────────────────────
print("\nPre-grouping by day...")
day_keys, day_X, day_y, day_regime = [], [], [], []

for day, grp in train.groupby('day_id'):
    if len(grp) < 5: continue
    day_keys.append(day)
    day_X.append(grp[all_gold].fillna(0).values.astype(np.float32))
    day_y.append(y_all[grp.index])
    day_regime.append(grp['_regime'].values)

print(f"  Grouped {len(day_keys)} days  [{(time.time()-t0)/60:.1f}m]")

# ── Z-score helper ────────────────────────────────────────────────────
def zscore(X, clip=CLIP_Z):
    m = X.mean(0, keepdims=True)
    s = X.std(0, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip).astype(np.float32)

# ── Compute in-regime Grinold ICs (pooled over all days, train regime only) ─
print("\nComputing in-regime Grinold ICs...")
grinold_X_pool, grinold_y_pool = [], []
for X_day, y_day, regime_day in zip(day_X, day_y, day_regime):
    tr_mask = (regime_day == 'train')
    if tr_mask.sum() < 5: continue
    X_z = zscore(X_day)
    grinold_X_pool.append(X_z[tr_mask])
    grinold_y_pool.append(y_day[tr_mask])
X_pool = np.vstack(grinold_X_pool)
y_pool = np.concatenate(grinold_y_pool)
y_dm   = y_pool - y_pool.mean()
ic_infold = (X_pool * y_dm[:, None]).mean(0)   # (n_gold,)
del X_pool, y_pool, grinold_X_pool, grinold_y_pool
print(f"  In-regime ICs computed  [{(time.time()-t0)/60:.1f}m]")

# ── Per-day OOF predictions ───────────────────────────────────────────
print("\nGenerating per-day OOF predictions (train→val regime transfer)...")

K_VALUES = [3, 5, 10, 15, 20, 30, 50]

oof_g  = {'p': [], 't': [], 'd': []}    # Grinold
oof_r  = {'p': [], 't': [], 'd': []}    # Ridge
oof_k  = {K: {'p': [], 't': [], 'd': []} for K in K_VALUES}  # KNN per K

for day_key, X_day, y_day, regime_day in zip(day_keys, day_X, day_y, day_regime):
    tr_mask = (regime_day == 'train')
    va_mask = (regime_day == 'val')
    if va_mask.sum() == 0: continue

    X_z = zscore(X_day)     # z-score using ALL assets on this day (train+val)
    X_va_z = X_z[va_mask]
    X_tr_z = X_z[tr_mask]
    y_tr = y_day[tr_mask]
    y_va = y_day[va_mask]
    n_day = len(y_day)

    # ── Grinold (top-10 in-regime IC) ──────────────────────────────
    g10_ic = ic_infold[:10]
    pred_g = (X_va_z[:, :10] @ g10_ic)
    pred_g -= pred_g.mean()
    oof_g['p'].extend(pred_g.tolist())
    oof_g['t'].extend(y_va.tolist())
    oof_g['d'].extend([day_key] * int(va_mask.sum()))

    if tr_mask.sum() < 5:
        # No train assets this day — skip Ridge and KNN
        for K in K_VALUES:
            oof_k[K]['p'].extend(pred_g.tolist())
            oof_k[K]['t'].extend(y_va.tolist())
            oof_k[K]['d'].extend([day_key] * int(va_mask.sum()))
        oof_r['p'].extend(pred_g.tolist())
        oof_r['t'].extend(y_va.tolist())
        oof_r['d'].extend([day_key] * int(va_mask.sum()))
        continue

    # ── Ridge (per-day, fit on train regime assets) ─────────────────
    lo, hi = np.percentile(y_tr, 1), np.percentile(y_tr, 99)
    y_tr_w = np.clip(y_tr, lo, hi)
    mdl = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    mdl.fit(X_tr_z[:, :10].astype(np.float64), y_tr_w.astype(np.float64))
    pred_r = mdl.predict(X_va_z[:, :10].astype(np.float64)).astype(np.float32)
    pred_r -= pred_r.mean()
    oof_r['p'].extend(pred_r.tolist())
    oof_r['t'].extend(y_va.tolist())
    oof_r['d'].extend([day_key] * int(va_mask.sum()))

    # ── KNN (cosine similarity, train→val, top-10 features) ─────────
    if tr_mask.sum() >= max(K_VALUES):
        sim = cosine_similarity(X_va_z[:, :10], X_tr_z[:, :10])
        max_K = min(max(K_VALUES), sim.shape[1])
        topk_all = np.argpartition(sim, -max_K, axis=1)[:, -max_K:]
        sim_topk = sim[np.arange(len(X_va_z))[:, None], topk_all]

        for K in K_VALUES:
            K_eff = min(K, sim.shape[1])
            if K_eff < max_K:
                local_idx = np.argpartition(sim_topk, -K_eff, axis=1)[:, -K_eff:]
                sim_K = sim_topk[np.arange(len(X_va_z))[:, None], local_idx]
                topk_K = topk_all[np.arange(len(X_va_z))[:, None], local_idx]
            else:
                topk_K = topk_all
                sim_K  = sim_topk
            w  = np.maximum(sim_K, 0)
            ws = w.sum(1, keepdims=True)
            w /= np.where(ws < 1e-10, 1.0, ws)
            pred_k = (w * y_tr[topk_K]).sum(1).astype(np.float32)
            pred_k -= pred_k.mean()
            oof_k[K]['p'].extend(pred_k.tolist())
            oof_k[K]['t'].extend(y_va.tolist())
            oof_k[K]['d'].extend([day_key] * int(va_mask.sum()))
    else:
        for K in K_VALUES:
            oof_k[K]['p'].extend(pred_g.tolist())
            oof_k[K]['t'].extend(y_va.tolist())
            oof_k[K]['d'].extend([day_key] * int(va_mask.sum()))

print(f"  Done  [{(time.time()-t0)/60:.1f}m]")
print(f"  Grinold OOF rows: {len(oof_g['p']):,}")

# ── Stage 1: Component R² ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 1 — Component R² on illiquid-like val regime")
print("=" * 70)

g_r2, g_std, g_nd = compute_perday_r2(oof_g['p'], oof_g['t'], oof_g['d'])
r_r2, r_std, r_nd = compute_perday_r2(oof_r['p'], oof_r['t'], oof_r['d'])
print(f"  Grinold-10 (in-regime IC)  R²={g_r2:+.5f}  ±{g_std:.5f}  days={g_nd}")
print(f"  Ridge-10   (per-day fit)   R²={r_r2:+.5f}  ±{r_std:.5f}  days={r_nd}")

knn_r2s = {}
for K in K_VALUES:
    k_r2, k_std, k_nd = compute_perday_r2(oof_k[K]['p'], oof_k[K]['t'], oof_k[K]['d'])
    knn_r2s[K] = k_r2
    print(f"  KNN K={K:<3}                    R²={k_r2:+.5f}  ±{k_std:.5f}  days={k_nd}")

best_K = max(knn_r2s, key=knn_r2s.get)
print(f"\n  Best KNN K: {best_K}  R²={knn_r2s[best_K]:+.5f}")

# ── Stage 2: Weight grid search ───────────────────────────────────────
print("\n" + "=" * 70)
print(f"STAGE 2 — Weight grid search (Ridge + KNN[K={best_K}] + Grinold)")
print("=" * 70)

p_g = np.array(oof_g['p'], dtype=np.float32)
p_r = np.array(oof_r['p'], dtype=np.float32)
p_k = np.array(oof_k[best_K]['p'], dtype=np.float32)
t   = np.array(oof_g['t'], dtype=np.float32)
d   = oof_g['d']

weight_results = []
steps = np.arange(0.05, 0.70, 0.05)
for wr in steps:
    for wk in steps:
        wg = 1.0 - wr - wk
        if wg < 0.05 or wg > 0.90: continue
        p_blend = wr * p_r + wk * p_k + wg * p_g
        r2, std, nd = compute_perday_r2(p_blend, t, d)
        weight_results.append({'wr': round(wr,2), 'wk': round(wk,2),
                                'wg': round(wg,2), 'r2': r2, 'std': std})

w_df = pd.DataFrame(weight_results).sort_values('r2', ascending=False)

print("\n  Top-20 weight combinations:")
print(f"  {'wr':>6}  {'wk':>6}  {'wg':>6}  {'R²':>10}  {'±std':>10}")
for _, row in w_df.head(20).iterrows():
    print(f"  {row['wr']:>6.2f}  {row['wk']:>6.2f}  {row['wg']:>6.2f}  "
          f"{row['r2']:>10.5f}  {row['std']:>10.5f}")

best_w = w_df.iloc[0]
print(f"\n  Current best submission weights: r=0.30  k=0.40  g=0.30  (LB=+0.00124)")
print(f"  Regime-transfer optimal:         r={best_w['wr']:.2f}  k={best_w['wk']:.2f}  g={best_w['wg']:.2f}  R²={best_w['r2']:+.5f}")

# ── Stage 3: Also check K sweep with optimal weights ──────────────────
print("\n" + "=" * 70)
print("STAGE 3 — All K values with optimal weights")
print("=" * 70)
for K in K_VALUES:
    p_k_k = np.array(oof_k[K]['p'], dtype=np.float32)
    p_blend = best_w['wr'] * p_r + best_w['wk'] * p_k_k + best_w['wg'] * p_g
    r2, std, nd = compute_perday_r2(p_blend, t, d)
    print(f"  K={K:<3}  blend r={best_w['wr']:.2f}/k={best_w['wk']:.2f}/g={best_w['wg']:.2f}  R²={r2:+.5f}  ±{std:.5f}")

# ── Save results ──────────────────────────────────────────────────────
knn_res = pd.DataFrame({'K': K_VALUES, 'r2': [knn_r2s[K] for K in K_VALUES]})
knn_res.to_csv(os.path.join(OUT_DIR, 'regime_knn_search.csv'), index=False)
w_df.to_csv(os.path.join(OUT_DIR, 'regime_weight_opt.csv'), index=False)

print(f"\n  Saved: regime_knn_search.csv  regime_weight_opt.csv")
print(f"  Total: {(time.time()-t0)/60:.1f} min")

print(f"""
  ── INTERPRETATION ─────────────────────────────────────────────
  This validator uses liquid-like CV_GROUPs as training reference
  and illiquid-like groups as validation targets.
  This mimics the actual competition structure better than uniform
  5-fold CV (which mixes all regimes equally).

  If KNN R² > 0: kernel adds genuine cross-regime signal
  If KNN R² < 0: kernel hurts even in regime-transfer (unexpected)

  The optimal weights found here should be more faithful to LB
  than the uniform CV results (which showed all-Grinold as best).
""")
