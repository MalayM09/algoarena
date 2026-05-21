# ================================================================
# METHOD 1: Cross-Day Ridge
# METHOD 2: Grinold with More Gold Features
# ================================================================
#
# METHOD 1 — Cross-Day Ridge
#   Standard per-day Ridge uses only same-day liquid assets (~1,900).
#   Cross-day Ridge also pools liquid assets from the K nearest
#   training days (by SO3_T proximity), weighted by temporal distance.
#   More training data per model → more stable per-day beta.
#   Legitimate approach: temporal continuity is a standard assumption
#   in finance and general time-series domains.
#
# METHOD 2 — Grinold with More Gold Features
#   Current best uses top-10 gold features (ICIR≥3, never-flip).
#   The top-10 cut was chosen by PI OOF on an arbitrary threshold.
#   Grinold PI OOF is RELIABLE (no per-day fitting → no OOF inversion).
#   Testing top-10/15/20/30/51 to find the true optimal.
#
# BOTH methods blended with each other and with existing components.
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

BASE_DIR    = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH  = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH   = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH   = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
SAMPLE_PATH = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'outputs/submissions')

TARGET_STD  = 0.000948
RIDGE_ALPHA = 10.0

print("=" * 70)
print("METHOD 1: CROSS-DAY RIDGE  |  METHOD 2: GRINOLD MORE FEATURES")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)

train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)

# Day-level SO3_T mapping (for proximity)
train_day_t  = train.groupby('day_id')['SO3_T'].mean().to_dict()
test_day_t   = test.groupby('day_id')['SO3_T'].mean().to_dict()
all_train_days = sorted(train_day_t.keys(), key=lambda d: train_day_t[d])

train_days   = set(train['day_id'].unique())
overlap_days = train_days & set(test['day_id'].unique())
new_days     = set(test['day_id'].unique()) - train_days
y_train      = train['TARGET'].values.astype(np.float64)
test_ids     = test['ID'].values

print(f"  Train: {len(train):,}  Test: {len(test):,}")
print(f"  Overlap: {len(overlap_days)} days  New: {len(new_days)} days")

# ── Gold features ──────────────────────────────────────────────────
icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False).copy()
all_gold  = [f for f in gold_df['feature'].tolist() if f in train.columns]
ic_map    = gold_df.set_index('feature')['mean_ic'].to_dict()

print(f"  Total gold features (ICIR≥3, never-flip): {len(all_gold)}")

# Grinold subsets to test
g_subsets = {
    'g_top10': all_gold[:10],
    'g_top15': all_gold[:15],
    'g_top20': all_gold[:20],
    'g_top30': all_gold[:30],
    'g_all51': all_gold,
}
g_ic_arrs = {k: np.array([ic_map[f] for f in v]) for k, v in g_subsets.items()}

# ── BookShape for PI OOF split ─────────────────────────────────────
all_cols = set(train.columns) - {'ID', 'TARGET', 'SO3_T', 'day_id'}
b_near = [c for c in all_cols if 'Lag' not in c and any(f'_B0{i}' in c for i in range(5))]
b_far  = [c for c in all_cols if 'Lag' not in c and any(f'_B{i}' in c for i in ['06','07','08','09','10'])]
train['bookshape'] = (train[b_near].fillna(0).sum(1) - train[b_far].fillna(0).sum(1)).values
test['bookshape']  = (test[b_near].fillna(0).sum(1)  - test[b_far].fillna(0).sum(1)).values

# ── Helpers ────────────────────────────────────────────────────────
def zscore(X, m=None, s=None, clip=5.0):
    if m is None: m = X.mean(0)
    if s is None: s = X.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip), m, s

def winsorise(y, lo=1, hi=99):
    return np.clip(y, np.percentile(y, lo), np.percentile(y, hi))

def per_day_ic(y_true, y_pred):
    if len(y_true) < 5: return np.nan
    return spearmanr(y_true, y_pred)[0]

def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

# ── Pre-index train by day for fast lookup ─────────────────────────
train_by_day = {day: grp for day, grp in train.groupby('day_id')}

# ── Cross-day pooling helper ───────────────────────────────────────
def get_cross_day_pool(anchor_day, K_days, sigma, feat_list):
    """
    Pool liquid assets from the K nearest training days to anchor_day,
    weighted by temporal proximity: w = exp(-|SO3T_diff|^2 / sigma^2).
    Returns X_pool (z-scored), y_pool (winsorized), sample_weights.
    Uses anchor_day liquid-asset stats for z-scoring (prevents data leakage).
    """
    t_anchor = train_day_t.get(anchor_day) or test_day_t.get(anchor_day)
    if t_anchor is None:
        return None, None, None, None, None

    # Sort training days by proximity to anchor
    nearby = sorted(all_train_days,
                    key=lambda d: abs(train_day_t[d] - t_anchor))
    nearby = [d for d in nearby if d != anchor_day][:K_days]

    # Also include the anchor day itself if it's a training day
    days_to_pool = ([anchor_day] if anchor_day in train_days else []) + nearby

    X_parts, y_parts, w_parts = [], [], []
    for d in days_to_pool:
        grp = train_by_day.get(d)
        if grp is None: continue
        bs    = grp['bookshape'].values
        liq_m = bs >= np.median(bs)
        if liq_m.sum() < 5: continue
        X_d = grp[feat_list].fillna(0).values.astype(np.float64)[liq_m]
        y_d = y_train[grp.index][liq_m]
        t_d = train_day_t[d]
        w_d = np.exp(-((t_d - t_anchor) ** 2) / (sigma ** 2))
        X_parts.append(X_d)
        y_parts.append(y_d)
        w_parts.append(np.full(len(y_d), w_d))

    if not X_parts:
        return None, None, None, None, None

    X_pool = np.vstack(X_parts)
    y_pool = np.concatenate(y_parts)
    w_pool = np.concatenate(w_parts)

    # Z-score using anchor-day liquid stats if available; else pool stats
    if anchor_day in train_days:
        grp_anchor = train_by_day[anchor_day]
        bs_a = grp_anchor['bookshape'].values
        liq_a = bs_a >= np.median(bs_a)
        X_anchor_liq = grp_anchor[feat_list].fillna(0).values.astype(np.float64)[liq_a]
        _, m_ref, s_ref = zscore(X_anchor_liq)
    else:
        _, m_ref, s_ref = zscore(X_pool)

    X_pool_z = zscore(X_pool, m_ref, s_ref)[0]
    y_pool_w  = winsorise(y_pool)

    return X_pool_z, y_pool_w, w_pool, m_ref, s_ref


# ================================================================
# PART 1: PI OOF — METHOD 2 (Grinold more features)
# ================================================================
# Grinold OOF is reliable: formula is fixed, no per-day fitting.
print("\n" + "=" * 70)
print("PART 1: PI OOF — METHOD 2  (Grinold feature count)")
print("=" * 70)
print("  NOTE: Grinold PI OOF is RELIABLE — no fitting, no OOF inversion.\n")

g_oof = {k: [] for k in g_subsets}
ridge_oof = []   # baseline same-day Ridge
cross_day_oof = {f'cd_K{K}_s{str(s).replace(".", "")}': []
                 for K in [5, 10, 20] for s in [0.01, 0.05, 0.1]}

day_count = 0
for day, grp in train.groupby('day_id'):
    n = len(grp)
    if n < 20: continue
    y_day = y_train[grp.index]
    bs    = grp['bookshape'].values
    bs_med = np.median(bs)
    liq_m   = bs >= bs_med
    illiq_m = bs <  bs_med
    if liq_m.sum() < 10 or illiq_m.sum() < 5: continue

    y_liq    = y_day[liq_m]
    y_illiq  = y_day[illiq_m]
    y_liq_w  = winsorise(y_liq)
    top10_f  = g_subsets['g_top10']

    # ── Method 2: Grinold variants ────────────────────────────
    for k, feats in g_subsets.items():
        X_day  = grp[feats].fillna(0).values.astype(np.float64)
        X_z, _, _ = zscore(X_day)
        pred   = X_z @ g_ic_arrs[k]
        pred  -= pred.mean()
        g_oof[k].append(per_day_ic(y_illiq, pred[illiq_m]))

    # ── Baseline: same-day Ridge (top10) ──────────────────────
    X_g = grp[top10_f].fillna(0).values.astype(np.float64)
    _, m_liq, s_liq = zscore(X_g[liq_m])
    X_liq_z   = zscore(X_g[liq_m],   m_liq, s_liq)[0]
    X_illiq_z = zscore(X_g[illiq_m], m_liq, s_liq)[0]
    X_all_z   = zscore(X_g,          m_liq, s_liq)[0]
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
    ridge.fit(X_liq_z, y_liq_w)
    pred_r = ridge.predict(X_all_z); pred_r -= pred_r.mean()
    ridge_oof.append(per_day_ic(y_illiq, pred_r[illiq_m]))

    # ── Method 1: Cross-day Ridge ─────────────────────────────
    for K in [5, 10, 20]:
        for sigma in [0.01, 0.05, 0.1]:
            key = f'cd_K{K}_s{str(sigma).replace(".", "")}'
            X_pool_z, y_pool_w, w_pool, m_ref, s_ref = \
                get_cross_day_pool(day, K, sigma, top10_f)
            if X_pool_z is None:
                cross_day_oof[key].append(np.nan)
                continue
            ridge_cd = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
            ridge_cd.fit(X_pool_z, y_pool_w, sample_weight=w_pool)
            X_illiq_z = zscore(X_g[illiq_m], m_ref, s_ref)[0]
            X_all_z_  = zscore(X_g,          m_ref, s_ref)[0]
            pred_cd = ridge_cd.predict(X_all_z_); pred_cd -= pred_cd.mean()
            cross_day_oof[key].append(per_day_ic(y_illiq, pred_cd[illiq_m]))

    day_count += 1

elapsed = (time.time() - t0) / 60
print(f"  Validated on {day_count} training days  [{elapsed:.1f}m elapsed]")

# ── Print Method 2 results ────────────────────────────────────────
print(f"\n  {'Model':<20}  {'Med IC':>10}  {'%pos':>8}  {'p25':>10}  {'p75':>10}")
print(f"  {'-' * 65}")

def summarise(arr):
    a = np.array([x for x in arr if not np.isnan(x)])
    if len(a) == 0: return np.nan, 0, np.nan, np.nan
    return np.nanmedian(a), (a > 0).mean()*100, np.percentile(a,25), np.percentile(a,75)

# Baseline
med, ppos, p25, p75 = summarise(ridge_oof)
print(f"  {'ridge_sameday (base)':<20}  {med:+10.5f}  {ppos:7.1f}%  {p25:+10.5f}  {p75:+10.5f}")
print()

best_g_med, best_g_key = -999, None
for k in g_subsets:
    med, ppos, p25, p75 = summarise(g_oof[k])
    marker = ''
    if med > best_g_med:
        best_g_med, best_g_key = med, k
        marker = ' ◄'
    print(f"  {k:<20}  {med:+10.5f}  {ppos:7.1f}%  {p25:+10.5f}  {p75:+10.5f}{marker}")

print(f"\n  Best Grinold config: {best_g_key}  (Med IC={best_g_med:+.5f})")
print(f"  vs g_top10: {best_g_med - summarise(g_oof['g_top10'])[0]:+.5f} delta")

# ── Print Method 1 results ────────────────────────────────────────
print(f"\n  {'Cross-Day Ridge':<25}  {'Med IC':>10}  {'%pos':>8}  vs sameday")
print(f"  {'-' * 55}")
base_med = summarise(ridge_oof)[0]
best_cd_med, best_cd_key = -999, None
for key, vals in cross_day_oof.items():
    med, ppos, p25, p75 = summarise(vals)
    delta  = med - base_med
    marker = ' ◄' if med > best_cd_med else ''
    if med > best_cd_med:
        best_cd_med, best_cd_key = med, key
    print(f"  {key:<25}  {med:+10.5f}  {ppos:7.1f}%  {delta:+.5f}{marker}")

print(f"\n  Best Cross-Day Ridge: {best_cd_key}  (Med IC={best_cd_med:+.5f})")
print(f"  vs same-day Ridge: {best_cd_med - base_med:+.5f} delta")


# ================================================================
# PART 2: GENERATE TEST PREDICTIONS
# ================================================================
print("\n" + "=" * 70)
print("PART 2: GENERATING TEST PREDICTIONS")
print("=" * 70)

# Parse best cross-day config
# key format: cd_K{K}_s{sigma_str}
parts    = best_cd_key.split('_')
best_K   = int(parts[1].replace('K', ''))
sig_str  = parts[2].replace('s', '')
best_sig = float(sig_str[0] + '.' + sig_str[1:]) if len(sig_str) > 1 else float('0.' + sig_str)
print(f"  Method 1 config: K={best_K}  sigma={best_sig}")
print(f"  Method 2 config: {best_g_key} ({len(g_subsets[best_g_key])} features)")

n_test = len(test)
te_grinold_top10 = np.zeros(n_test)   # current baseline
te_grinold_best  = np.zeros(n_test)   # method 2 best
te_ridge_sameday = np.zeros(n_test)   # baseline same-day Ridge
te_ridge_crossday = np.zeros(n_test)  # method 1

top10_f    = g_subsets['g_top10']
best_g_f   = g_subsets[best_g_key]
best_g_ic  = g_ic_arrs[best_g_key]

n_overlap = 0
for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index

    # ── Grinold (both top10 and best) ─────────────────────────
    # Per-day z-score on test assets (same-day, as per hard rule)
    for feat_k, ic_k, out_arr in [
        ('g_top10',  g_ic_arrs['g_top10'], te_grinold_top10),
        (best_g_key, best_g_ic,            te_grinold_best),
    ]:
        feats = g_subsets[feat_k]
        X_te  = grp_te[feats].fillna(0).values.astype(np.float64)
        X_z, _, _ = zscore(X_te)
        pred  = X_z @ ic_k; pred -= pred.mean()
        out_arr[te_idx] = pred

    if day in train_days:
        grp_tr  = train_by_day[day]
        y_liq   = y_train[grp_tr.index]
        y_liq_w = winsorise(y_liq)
        X_tr    = grp_tr[top10_f].fillna(0).values.astype(np.float64)
        X_te_g  = grp_te[top10_f].fillna(0).values.astype(np.float64)

        # Same-day Ridge
        bs_tr   = grp_tr['bookshape'].values
        liq_m   = bs_tr >= np.median(bs_tr)
        _, m_liq, s_liq = zscore(X_tr[liq_m])
        X_tr_z  = zscore(X_tr,   m_liq, s_liq)[0]
        X_te_z  = zscore(X_te_g, m_liq, s_liq)[0]
        ridge   = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        ridge.fit(X_tr_z[liq_m], y_liq_w[liq_m] if len(y_liq_w) > len(y_liq_w[liq_m])
                  else winsorise(y_liq[liq_m]))
        pred_r  = ridge.predict(X_te_z); pred_r -= pred_r.mean()
        te_ridge_sameday[te_idx] = pred_r

        # Cross-day Ridge
        X_pool_z, y_pool_w, w_pool, m_ref, s_ref = \
            get_cross_day_pool(day, best_K, best_sig, top10_f)
        if X_pool_z is not None:
            ridge_cd = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
            ridge_cd.fit(X_pool_z, y_pool_w, sample_weight=w_pool)
            X_te_z_cd = zscore(X_te_g, m_ref, s_ref)[0]
            pred_cd = ridge_cd.predict(X_te_z_cd); pred_cd -= pred_cd.mean()
            te_ridge_crossday[te_idx] = pred_cd
        else:
            te_ridge_crossday[te_idx] = pred_r  # fallback to same-day

        n_overlap += 1

    else:
        # New day: Grinold is already computed above; Ridge = Grinold fallback
        X_te_g = grp_te[top10_f].fillna(0).values.astype(np.float64)
        X_z, _, _ = zscore(X_te_g)
        pred_g = X_z @ g_ic_arrs['g_top10']; pred_g -= pred_g.mean()
        te_ridge_sameday[te_idx]  = pred_g
        te_ridge_crossday[te_idx] = pred_g

elapsed = (time.time() - t0) / 60
print(f"  Overlap days processed: {n_overlap}  [{elapsed:.1f}m elapsed]")


# ================================================================
# PART 3: BUILD AND SAVE SUBMISSIONS
# ================================================================
print("\n" + "=" * 70)
print("PART 3: SUBMISSION VARIANTS")
print("=" * 70)

sample_sub = pd.read_csv(SAMPLE_PATH)[['ID']]

def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return (a @ b) / d if d > 1e-10 else 0.0

print(f"\n  Component correlations:")
print(f"  CrossDay-Ridge vs SameDay-Ridge:  {pearson_r(te_ridge_crossday, te_ridge_sameday):+.4f}")
print(f"  Grinold-best   vs Grinold-top10:  {pearson_r(te_grinold_best, te_grinold_top10):+.4f}")
print(f"  CrossDay-Ridge vs Grinold-best:   {pearson_r(te_ridge_crossday, te_grinold_best):+.4f}")
print(f"  SameDay-Ridge  vs Grinold-top10:  {pearson_r(te_ridge_sameday, te_grinold_top10):+.4f}")
print()

def save_sub(preds, name):
    preds_s = auto_scale(preds)
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': preds_s})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    t = sub['TARGET']
    print(f"  {name:<50}  std={t.std():.6f}")

# ── Pure components ───────────────────────────────────────────────
save_sub(te_grinold_best,   f'm2_{best_g_key}_pure')
save_sub(te_ridge_crossday, f'm1_crossday_K{best_K}_pure')

# ── Method 2 blends: Grinold best + top10 hybrid ─────────────────
print()
for alpha in [0.5, 0.7]:
    blend = alpha * te_grinold_best + (1-alpha) * te_grinold_top10
    save_sub(blend, f'm2_blend_best{int(alpha*100)}_top10{int((1-alpha)*100)}')

# ── Method 1 blends: Cross-day Ridge + Grinold ───────────────────
print()
for alpha in [0.3, 0.5, 0.7]:
    blend = alpha * te_ridge_crossday + (1-alpha) * te_grinold_best
    save_sub(blend, f'm1_crossday_a{int(alpha*100)}_grinold')

# ── Threeway: Cross-day Ridge + Grinold-best + (existing kernel) ──
# Mirror the +0.00124 threeway but replace sameday-Ridge with cross-day-Ridge
# and replace Grinold-top10 with Grinold-best
print()
for r_w, g_w in [(0.30, 0.30), (0.25, 0.30), (0.30, 0.25)]:
    k_w = round(1.0 - r_w - g_w, 2)
    if k_w < 0: continue
    # Load existing Gaussian kernel predictions (from threeway +0.00124)
    kernel_path = os.path.join(OUT_DIR, 'hybrid_grinold_kernel.csv')
    if os.path.exists(kernel_path):
        kdf = pd.read_csv(kernel_path)
        kdf = sample_sub.merge(kdf[['ID','TARGET']].rename(
                  columns={'TARGET': 'kp'}), on='ID', how='left').fillna(0.0)
        kernel_pred = kdf['kp'].values
        blend = r_w * te_ridge_crossday + k_w * kernel_pred + g_w * te_grinold_best
        save_sub(blend, f'm1m2_tw_r{int(r_w*100)}_k{int(k_w*100)}_g{int(g_w*100)}')

# Best upgrade of current +0.00124 threeway:
# Replace same-day Ridge with cross-day Ridge
# Replace Grinold-top10 with Grinold-best
kernel_path = os.path.join(OUT_DIR, 'hybrid_grinold_kernel.csv')
if os.path.exists(kernel_path):
    print()
    kdf = pd.read_csv(kernel_path)
    kdf = sample_sub.merge(kdf[['ID','TARGET']].rename(
              columns={'TARGET': 'kp'}), on='ID', how='left').fillna(0.0)
    kernel_pred = kdf['kp'].values
    # Current best: 0.30 Ridge + 0.40 NW-kernel + 0.30 Grinold-top10
    # Upgraded:    0.30 CrossRidge + 0.40 NW-kernel + 0.30 Grinold-best
    upgraded = 0.30 * te_ridge_crossday + 0.40 * kernel_pred + 0.30 * te_grinold_best
    save_sub(upgraded, 'm1m2_upgraded_threeway')
    print(f"  [KEY] m1m2_upgraded_threeway: CrossRidge(30%) + Kernel(40%) + GrinoldBest(30%)")
    print(f"        Direct upgrade of +0.00124 threeway with both methods applied.")


# ================================================================
# PART 4: SUMMARY + SUBMISSION RECOMMENDATIONS
# ================================================================
print("\n" + "=" * 70)
print("PART 4: SUMMARY & SUBMISSION RECOMMENDATIONS")
print("=" * 70)

g_top10_med = summarise(g_oof['g_top10'])[0]
g_best_med  = summarise(g_oof[best_g_key])[0]
cd_best_med = summarise(cross_day_oof[best_cd_key])[0]
sd_med      = summarise(ridge_oof)[0]

print(f"""
  ── PI OOF RESULTS (Grinold is RELIABLE; Ridge has some inversion) ──
  Grinold top10 (current):    {g_top10_med:+.5f}
  Grinold {best_g_key} (best):  {g_best_med:+.5f}  (Δ {g_best_med - g_top10_med:+.5f})

  SameDay Ridge (baseline):   {sd_med:+.5f}
  CrossDay Ridge (best):      {cd_best_med:+.5f}  (Δ {cd_best_med - sd_med:+.5f})

  ── LB REFERENCE ────────────────────────────────────────────────────
  grinold_top10 pure:     +0.00096
  hybrid_grinold_kernel:  +0.00115
  threeway_r30_k40_g29:   +0.00124  ← CURRENT BEST CLEAN ML

  ── SUBMISSION PRIORITY ─────────────────────────────────────────────
  1. m1m2_upgraded_threeway.csv  ← HIGHEST PRIORITY
     Same structure as +0.00124 (R30 + K40 + G30) but:
       - Ridge → CrossDay Ridge (more stable beta)
       - Grinold top10 → Grinold {best_g_key} (more features)
     If both methods help, this captures both improvements.

  2. m2_{best_g_key}_pure.csv
     Pure Grinold test. Grinold PI OOF is reliable.
     If {best_g_key} PI OOF > top10 PI OOF, LB improvement is likely real.

  3. m1_crossday_K{best_K}_pure.csv  (or m1_crossday_a70_grinold.csv)
     Tests cross-day Ridge in isolation vs Grinold blend.
     Confirms whether pooling neighboring days helps.

  Total elapsed: {(time.time()-t0)/60:.1f} min
""")
