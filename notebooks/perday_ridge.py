# ================================================================
# PER-DAY RIDGE — Same-Day Liquid Return Regression
# ================================================================
# Motivation: kernel regression (70% blend) scored +0.00115 vs
#   Grinold's +0.00096, proving same-day liquid returns carry
#   real signal for illiquid return prediction.
#
# This script replaces the nonparametric kernel with an explicit
# linear per-day Ridge:
#   - For each overlap test day:
#       1. Z-score features using liquid assets' stats
#       2. Fit Ridge(X_liq_z, y_liq_winsorized)
#       3. Apply to illiquid assets using same day's liquid stats
#       4. Market neutral: subtract cross-sectional mean
#   - For future days (no liquid returns): Grinold fallback
#
# Also tests ICIR-weighted Grinold (one-line change from IC-weighted).
#
# PI OOF protocol: split each training day by BookShape median.
#   Top 50% = pseudo-liquid (train Ridge). Bottom 50% = pseudo-illiquid (validate).
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
t0 = time.time()

print("=" * 70)
print("PER-DAY RIDGE — Same-Day Liquid→Illiquid Regression")
print("=" * 70)

# ── Feature selection ──────────────────────────────────────────────────────
icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False).copy()
all51     = gold_df['feature'].tolist()
ic_weights   = gold_df.set_index('feature')['mean_ic'].to_dict()
icir_weights = gold_df.set_index('feature')['abs_icir'].to_dict()

print(f"Gold features (ICIR>=3, never flip sign): {len(all51)}")

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

print(f"Train rows: {len(train):,}  Test rows: {len(test):,}")
print(f"Train days: {len(train_days)}  Overlap: {len(overlap)}  Future: {len(new_days)}")

# ── BookShape (for PI OOF split) ───────────────────────────────────────────
b_near = [c for c in all_cols if 'Lag' not in c and
          any(f'_B0{i}' in c for i in range(5))]
b_far  = [c for c in all_cols if 'Lag' not in c and
          any(f'_B{i}' in c for i in ['06','07','08','09','10'])]

train['bookshape'] = (train[b_near].fillna(0).sum(1) -
                      train[b_far].fillna(0).sum(1)).astype(np.float64)
test['bookshape']  = (test[b_near].fillna(0).sum(1) -
                      test[b_far].fillna(0).sum(1)).astype(np.float64)

# ── IC/ICIR weight arrays ──────────────────────────────────────────────────
ic_arr_top10   = np.array([ic_weights[f]   for f in top10])
ic_arr_all51   = np.array([ic_weights[f]   for f in all51])
# ICIR-weighted: sign from mean_IC, magnitude from ICIR
icir_arr_top10 = np.array([icir_weights[f] * np.sign(ic_weights[f]) for f in top10])
icir_arr_all51 = np.array([icir_weights[f] * np.sign(ic_weights[f]) for f in all51])

# ── Helpers ────────────────────────────────────────────────────────────────
def zscore_fit(X, clip=5.0):
    """Compute z-score stats from X. Returns (X_z, mean, std)."""
    m = X.mean(0)
    s = X.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip), m, s

def zscore_apply(X, m, s, clip=5.0):
    """Apply pre-computed z-score stats to X."""
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip)

def winsorise(y, lo_pct=1, hi_pct=99):
    lo = np.percentile(y, lo_pct)
    hi = np.percentile(y, hi_pct)
    return np.clip(y, lo, hi)

def per_day_ic(y_true, y_pred):
    if len(y_true) < 5: return np.nan
    r, _ = spearmanr(y_true, y_pred)
    return r


# ================================================================
# PI OOF VALIDATION
# ================================================================
# Compare Grinold variants vs Per-day Ridge variants
# All validated on bottom-50% BookShape (pseudo-illiquid)
# ================================================================
print("\n" + "=" * 70)
print("PI OOF VALIDATION — Grinold vs ICIR-Grinold vs Per-Day Ridge")
print("=" * 70)

# Ridge regularization strengths to test
ridge_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

ic_store = {
    'grinold_top10':  [],
    'grinold_icir10': [],
    'grinold_all51':  [],
    'grinold_icir51': [],
}
for ra in ridge_alphas:
    ic_store[f'ridge_top10_a{ra}'] = []
    ic_store[f'ridge_all51_a{ra}'] = []

day_count = 0
for day, grp in train.groupby('day_id'):
    n = len(grp)
    if n < 20:
        continue

    y_day  = y_train[grp.index]
    bs_day = grp['bookshape'].values
    bs_med = np.median(bs_day)

    liq_mask   = bs_day >= bs_med
    illiq_mask = bs_day <  bs_med

    n_liq   = liq_mask.sum()
    n_illiq = illiq_mask.sum()
    if n_liq < 10 or n_illiq < 5:
        continue

    y_liq   = y_day[liq_mask]
    y_illiq = y_day[illiq_mask]
    y_liq_w = winsorise(y_liq)

    # ── Feature matrices ─────────────────────────────────────────
    X_all_top10 = grp[top10].fillna(0).values.astype(np.float64)
    X_all_all51 = grp[all51].fillna(0).values.astype(np.float64)

    # Grinold: z-score all assets together (standard approach)
    X_z_top10_all, _, _ = zscore_fit(X_all_top10)
    X_z_all51_all, _, _ = zscore_fit(X_all_all51)

    # Per-day Ridge: z-score using LIQUID stats only, apply to all
    X_liq_top10 = X_all_top10[liq_mask]
    X_liq_all51 = X_all_all51[liq_mask]
    X_z_liq_top10, m10, s10 = zscore_fit(X_liq_top10)
    X_z_liq_all51, m51, s51 = zscore_fit(X_liq_all51)
    # Apply liquid stats to all assets
    X_z_all_top10_liqnorm = zscore_apply(X_all_top10, m10, s10)
    X_z_all_all51_liqnorm = zscore_apply(X_all_all51, m51, s51)

    # ── Grinold variants ─────────────────────────────────────────
    for arr, name in [(ic_arr_top10, 'grinold_top10'),
                      (icir_arr_top10, 'grinold_icir10'),
                      (ic_arr_all51,   'grinold_all51'),
                      (icir_arr_all51, 'grinold_icir51')]:
        if 'top10' in name or 'icir10' in name:
            X_z = X_z_top10_all
        else:
            X_z = X_z_all51_all
        pred = X_z @ arr
        pred -= pred.mean()
        ic_store[name].append(per_day_ic(y_illiq, pred[illiq_mask]))

    # ── Per-day Ridge ─────────────────────────────────────────────
    for ra in ridge_alphas:
        # top10
        ridge = Ridge(alpha=ra, fit_intercept=True)
        ridge.fit(X_z_liq_top10, y_liq_w)
        pred_r10 = ridge.predict(X_z_all_top10_liqnorm)
        pred_r10 -= pred_r10.mean()
        ic_store[f'ridge_top10_a{ra}'].append(per_day_ic(y_illiq, pred_r10[illiq_mask]))

        # all51
        ridge51 = Ridge(alpha=ra, fit_intercept=True)
        ridge51.fit(X_z_liq_all51, y_liq_w)
        pred_r51 = ridge51.predict(X_z_all_all51_liqnorm)
        pred_r51 -= pred_r51.mean()
        ic_store[f'ridge_all51_a{ra}'].append(per_day_ic(y_illiq, pred_r51[illiq_mask]))

    day_count += 1

elapsed = (time.time() - t0) / 60
print(f"\n  Validated on {day_count} training days [{elapsed:.1f}m elapsed]")
print(f"\n  {'Model':<30}  {'Med IC':>10}  {'%pos':>8}  {'p25':>10}  {'p75':>10}  {'N':>6}")
print(f"  {'-' * 80}")

best_model  = None
best_med_ic = -999
for k, v in ic_store.items():
    arr  = np.array([x for x in v if not np.isnan(x)])
    if len(arr) == 0:
        continue
    med  = np.nanmedian(arr)
    ppos = (arr > 0).mean() * 100
    p25, p75 = np.percentile(arr, [25, 75])
    marker = ' ← BEST' if med > best_med_ic else ''
    if med > best_med_ic:
        best_med_ic = med
        best_model  = k
    print(f"  {k:<30}  {med:+10.5f}  {ppos:7.1f}%  {p25:+10.5f}  {p75:+10.5f}  {len(arr):6d}{marker}")

print(f"\n  WINNER: {best_model}  (Med IC = {best_med_ic:+.5f})")


# ================================================================
# DETERMINE BEST RIDGE CONFIGURATION
# ================================================================
# Find best (feature_set, alpha) for per-day Ridge
best_ridge_model = None
best_ridge_ic    = -999
for k in ic_store:
    if not k.startswith('ridge_'):
        continue
    arr = np.array([x for x in ic_store[k] if not np.isnan(x)])
    med = np.nanmedian(arr)
    if med > best_ridge_ic:
        best_ridge_ic    = med
        best_ridge_model = k

# Parse best config
if best_ridge_model:
    parts = best_ridge_model.split('_')  # ridge_top10_a10.0 or ridge_all51_a1.0
    best_feats     = top10 if 'top10' in best_ridge_model else all51
    best_feat_name = 'top10' if 'top10' in best_ridge_model else 'all51'
    best_ridge_alpha = float(best_ridge_model.split('_a')[1])
    print(f"\n  Best Ridge config: feats={best_feat_name}  alpha={best_ridge_alpha}")
    print(f"  Best Ridge Med IC: {best_ridge_ic:+.5f}")
else:
    best_feats       = top10
    best_feat_name   = 'top10'
    best_ridge_alpha = 1.0
    print("\n  WARNING: No Ridge model found, defaulting to top10 alpha=1.0")

# Also store best ICIR-Grinold IC for reference
icir_arr_best = icir_arr_top10  # will update below based on all51 comparison
grinold_top10_med = np.nanmedian([x for x in ic_store['grinold_top10'] if not np.isnan(x)])
icir10_med        = np.nanmedian([x for x in ic_store['grinold_icir10'] if not np.isnan(x)])
icir51_med        = np.nanmedian([x for x in ic_store['grinold_icir51'] if not np.isnan(x)])

print(f"\n  ICIR-weighted Grinold improvement over IC-weighted:")
print(f"    top10: IC-wtd={grinold_top10_med:+.5f}  ICIR-wtd={icir10_med:+.5f}  "
      f"delta={icir10_med - grinold_top10_med:+.5f}")


# ================================================================
# GENERATE TEST PREDICTIONS
# ================================================================
print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

te_preds_grinold  = np.zeros(len(test))   # baseline (IC-weighted, top10)
te_preds_icir     = np.zeros(len(test))   # ICIR-weighted Grinold, top10
te_preds_ridge    = np.zeros(len(test))   # best per-day Ridge config

n_overlap_processed = 0
n_future_processed  = 0

for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index

    X_te_top10 = grp_te[top10].fillna(0).values.astype(np.float64)
    X_te_feats = grp_te[best_feats].fillna(0).values.astype(np.float64)

    if day in train_days:
        # ── Overlap day ──────────────────────────────────────────
        grp_tr = train[train['day_id'] == day]
        y_liq  = y_train[grp_tr.index]
        y_liq_w = winsorise(y_liq)

        X_tr_top10 = grp_tr[top10].fillna(0).values.astype(np.float64)
        X_tr_feats = grp_tr[best_feats].fillna(0).values.astype(np.float64)

        # ── Grinold: z-score test assets (as in original engine) ─
        X_te_z_top10, _, _ = zscore_fit(X_te_top10)
        pred_g  = X_te_z_top10 @ ic_arr_top10
        pred_g -= pred_g.mean()
        te_preds_grinold[te_idx] = pred_g

        pred_icir  = X_te_z_top10 @ icir_arr_top10
        pred_icir -= pred_icir.mean()
        te_preds_icir[te_idx] = pred_icir

        # ── Per-day Ridge: z-score using LIQUID (train) stats ────
        X_tr_z, m_tr, s_tr = zscore_fit(X_tr_feats)
        X_te_z_ridge = zscore_apply(X_te_feats, m_tr, s_tr)

        ridge = Ridge(alpha=best_ridge_alpha, fit_intercept=True)
        ridge.fit(X_tr_z, y_liq_w)
        pred_r  = ridge.predict(X_te_z_ridge)
        pred_r -= pred_r.mean()
        te_preds_ridge[te_idx] = pred_r

        n_overlap_processed += 1

    else:
        # ── Future day — Grinold fallback ─────────────────────────
        X_te_z_top10, _, _ = zscore_fit(X_te_top10)
        pred_g  = X_te_z_top10 @ ic_arr_top10
        pred_g -= pred_g.mean()
        te_preds_grinold[te_idx] = pred_g
        te_preds_icir[te_idx]    = pred_g   # fallback same as Grinold
        te_preds_ridge[te_idx]   = pred_g   # Ridge falls back to Grinold

        n_future_processed += 1

elapsed = (time.time() - t0) / 60
print(f"  Overlap days processed (Ridge + Grinold): {n_overlap_processed} [{elapsed:.1f}m]")
print(f"  Future days processed (Grinold fallback): {n_future_processed}")


# ================================================================
# HYBRID BLENDS
# ================================================================
# alpha = fraction of Ridge, (1-alpha) = fraction of Grinold
# Test: 0.0, 0.3, 0.5, 0.7, 0.85, 1.0
# ================================================================
print("\n" + "=" * 70)
print("HYBRID BLENDS — Per-Day Ridge + Grinold")
print("=" * 70)

TARGET_STD = 0.000948   # confirmed optimal scale from probe_005

def auto_scale(preds, target_std=TARGET_STD):
    s = preds.std()
    if s < 1e-10:
        return preds
    return preds * (target_std / s)

alphas = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]

# Load sample submission for alignment
sample_sub = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/sample_submission.csv'))[['ID']]

submissions = {}

print(f"\n  {'Alpha':<8}  {'Description':<35}  {'std':>10}  {'mean':>14}  {'%pos':>8}")
print(f"  {'-' * 80}")

for alpha in alphas:
    preds_blend = alpha * te_preds_ridge + (1 - alpha) * te_preds_grinold
    preds_scaled = auto_scale(preds_blend)

    desc = f'{int(alpha*100)}% Ridge + {int((1-alpha)*100)}% Grinold'
    std  = preds_scaled.std()
    mean = preds_scaled.mean()
    ppos = (preds_scaled > 0).mean() * 100
    print(f"  {alpha:<8.2f}  {desc:<35}  {std:10.7f}  {mean:+14.10f}  {ppos:7.1f}%")

    fname = f'ridge_hybrid_a{int(alpha*100):03d}'
    sub   = pd.DataFrame({'ID': test_ids, 'TARGET': preds_scaled})
    sub   = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    path  = os.path.join(OUT_DIR, f'{fname}.csv')
    sub.to_csv(path, index=False)
    submissions[fname] = path

# Also save ICIR-weighted Grinold standalone
preds_icir_scaled = auto_scale(te_preds_icir)
fname_icir = 'grinold_icir_top10_p005'
sub_icir = pd.DataFrame({'ID': test_ids, 'TARGET': preds_icir_scaled})
sub_icir = sample_sub.merge(sub_icir, on='ID', how='left').fillna(0.0)
sub_icir.to_csv(os.path.join(OUT_DIR, f'{fname_icir}.csv'), index=False)
print(f"\n  ICIR-Grinold saved: {fname_icir}  std={preds_icir_scaled.std():.7f}")

# Also save 3-way blend: Ridge + Kernel + Grinold (if kernel file exists)
kernel_path = os.path.join(OUT_DIR, 'hybrid_grinold_kernel.csv')
if os.path.exists(kernel_path):
    kernel_df = pd.read_csv(kernel_path)
    kernel_df = sample_sub.merge(
        kernel_df[['ID', 'TARGET']].rename(columns={'TARGET': 'kernel_pred'}),
        on='ID', how='left'
    ).fillna(0.0)
    kernel_preds_raw = kernel_df['kernel_pred'].values

    print(f"\n  3-way blends (Ridge + Kernel + Grinold):")
    print(f"  Note: hybrid_grinold_kernel.csv = 70% kernel + 30% Grinold = +0.00115 LB")
    print(f"\n  {'Blend':<50}  {'std':>10}  {'%pos':>8}")
    print(f"  {'-' * 72}")

    # Extract pure Grinold from hybrid_grinold_kernel to get kernel-only signal
    # hybrid = 0.7*kernel + 0.3*grinold → kernel_only = (hybrid - 0.3*grinold) / 0.7
    # But we need aligned grinold. Use te_preds_grinold instead.
    # Better: just blend directly at the submission level.

    for r_alpha, k_alpha in [(0.5, 0.3), (0.3, 0.4), (0.4, 0.3), (0.35, 0.35)]:
        g_alpha = 1.0 - r_alpha - k_alpha
        if g_alpha < 0: continue
        blend = (r_alpha * auto_scale(te_preds_ridge) +
                 k_alpha * kernel_preds_raw +
                 g_alpha * auto_scale(te_preds_grinold))
        blend_s = auto_scale(blend)
        ppos = (blend_s > 0).mean() * 100
        desc = f'Ridge={r_alpha:.2f} Kernel={k_alpha:.2f} Grinold={g_alpha:.2f}'
        print(f"  {desc:<50}  {blend_s.std():10.7f}  {ppos:7.1f}%")
        fname3 = f'threeway_r{int(r_alpha*100):02d}_k{int(k_alpha*100):02d}_g{int(g_alpha*100):02d}'
        sub3 = pd.DataFrame({'ID': test_ids, 'TARGET': blend_s})
        sub3 = sample_sub.merge(sub3, on='ID', how='left').fillna(0.0)
        sub3.to_csv(os.path.join(OUT_DIR, f'{fname3}.csv'), index=False)


# ================================================================
# CORRELATION ANALYSIS
# ================================================================
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

g_s = auto_scale(te_preds_grinold)
r_s = auto_scale(te_preds_ridge)
i_s = preds_icir_scaled

from numpy.linalg import norm

def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = (norm(a) * norm(b))
    return (a @ b) / denom if denom > 1e-10 else 0.0

print(f"\n  Correlation matrix:")
print(f"  Grinold ↔ Per-day Ridge:  {pearson_r(g_s, r_s):+.4f}")
print(f"  Grinold ↔ ICIR-Grinold:   {pearson_r(g_s, i_s):+.4f}")
print(f"  Ridge   ↔ ICIR-Grinold:   {pearson_r(r_s, i_s):+.4f}")

sign_agree_ri = (np.sign(r_s) == np.sign(g_s)).mean() * 100
sign_agree_ii = (np.sign(i_s) == np.sign(g_s)).mean() * 100
print(f"\n  Sign agreement vs Grinold:")
print(f"    Per-day Ridge:  {sign_agree_ri:.1f}%")
print(f"    ICIR-Grinold:   {sign_agree_ii:.1f}%")


# ================================================================
# SUBMISSION PRIORITY
# ================================================================
print("\n" + "=" * 70)
print("SUBMISSION PRIORITY ORDER")
print("=" * 70)

grinold_top10_pi = np.nanmedian([x for x in ic_store['grinold_top10'] if not np.isnan(x)])
icir10_pi        = np.nanmedian([x for x in ic_store['grinold_icir10'] if not np.isnan(x)])
ridge_pi         = np.nanmedian([x for x in ic_store[best_ridge_model] if not np.isnan(x)])

print(f"""
  CONFIRMED LB scores:
    grinold_allday_top10_probe_005 → +0.00096  (pure Grinold)
    hybrid_grinold_kernel          → +0.00115  (70% kernel + 30% Grinold)  ← BEST
    kernel_noint_best              → +0.00054  (pure kernel — worse than Grinold)

  PI OOF this session:
    Grinold top10:    {grinold_top10_pi:+.5f}  (baseline)
    ICIR-Grinold:     {icir10_pi:+.5f}
    Best per-day Ridge ({best_ridge_model}): {ridge_pi:+.5f}

  SUBMISSION PRIORITY:
    1. ridge_hybrid_a070.csv    — 70% Ridge + 30% Grinold  (mirrors best kernel blend ratio)
    2. ridge_hybrid_a050.csv    — 50% Ridge + 50% Grinold  (conservative)
    3. grinold_icir_top10_p005  — ICIR-weighted Grinold    (quick test, 1 slot)
    4. ridge_hybrid_a030.csv    — 30% Ridge + 70% Grinold  (mild blend)

  3-way blends (Ridge + Kernel + Grinold) also saved — try after testing pure Ridge hybrids.

  IMPORTANT: PI OOF is unreliable for ranking these (kernel result proved this).
  Let the LB be the judge. Submit in priority order above.

  Total elapsed: {(time.time()-t0)/60:.1f} min
""")
