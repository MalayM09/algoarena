# ================================================================
# PSEUDO-ILLIQUID OOF — Fix the Validation Problem
# ================================================================
# Core idea: the standard within-day OOF validates on liquid assets
# (same population as training). But test assets are illiquid.
# That's why OOF is always inflated and inversely correlated with LB.
#
# Fix: within each training day, split assets by BookShape:
#   - Pseudo-liquid   (top 50% BookShape)  → train set
#   - Pseudo-illiquid (bottom 50% BookShape) → validation set
#
# This mimics the actual competition structure: train on liquid,
# predict illiquid. If our signal generalises (ICIR 94% preserved),
# this OOF should correlate with actual LB performance.
#
# Experiments run in parallel:
#   A. gold_z    : gold features + z-score norm
#   B. gold_r    : gold features + rank norm
#   C. silver_z  : silver features + z-score norm
#   D. Grinold   : IC-weighted direct alpha (no model fitting)
#
# ALSO run standard 80/20 OOF for same-day comparison.
#
# Key question answered: does pseudo-illiquid OOF correlate
# with known LB scores better than standard OOF?
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
t0 = time.time()


# ── Feature selection ─────────────────────────────────────────────
icir_df = pd.read_csv(ICIR_PATH)

gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_feats   = icir_df[gold_mask].sort_values('abs_icir', ascending=False)['feature'].tolist()
gold_ic_map  = dict(zip(icir_df['feature'], icir_df['mean_ic']))   # for Grinold

silver_feats = icir_df[icir_df['abs_icir'] >= 2].sort_values(
    'abs_icir', ascending=False)['feature'].tolist()

print(f"Gold features  (ICIR>=3, never flip): {len(gold_feats)}")
print(f"Silver features (ICIR>=2):             {len(silver_feats)}")


# ── Load data ─────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)

all_cols     = set(train.columns) - {'ID', 'TARGET'}
gold_feats   = [f for f in gold_feats   if f in all_cols]
silver_feats = [f for f in silver_feats if f in all_cols]
print(f"Gold in dataset:   {len(gold_feats)}")
print(f"Silver in dataset: {len(silver_feats)}")

# Day identifier
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days = set(train['day_id'].unique())
overlap    = train_days & set(test['day_id'].unique())
new_days   = set(test['day_id'].unique()) - train_days
print(f"Train days: {len(train_days)}  |  Overlap: {len(overlap)}  |  New: {len(new_days)}")

y_train  = train['TARGET'].values.astype(np.float64)
test_ids = test['ID'].values


# ── BookShape computation ─────────────────────────────────────────
# Near-mid bins (B00-B04) vs far-from-mid bins (B06-B10)
# BookShape > 0 = liquid (volume near mid)
# BookShape < 0 = illiquid (volume deep in book)
print("\nComputing BookShape proxy...")

near_mid_bins = [c for c in all_cols if
                 any(f'_B0{i}_' in c or c.endswith(f'_B0{i}') for i in range(5))
                 and 'Lag' not in c]
far_from_bins = [c for c in all_cols if
                 any(f'_B0{i}_' in c or c.endswith(f'_B0{i}') for i in range(6, 10))
                 and '_B10_' not in c and not c.endswith('_B10')
                 and 'Lag' not in c]
far_from_bins += [c for c in all_cols if ('_B10_' in c or c.endswith('_B10')) and 'Lag' not in c]

# Also try direct B-name matching
b_near = [c for c in all_cols if 'Lag' not in c and
          any(f'_B0{i}' in c for i in range(5))]
b_far  = [c for c in all_cols if 'Lag' not in c and
          any(f'_B{i}' in c for i in ['06', '07', '08', '09', '10'])]

print(f"  Near-mid bin features: {len(b_near)}")
print(f"  Far-from-mid features: {len(b_far)}")
print(f"  Near examples: {b_near[:3]}")
print(f"  Far  examples: {b_far[:3]}")

if len(b_near) > 0 and len(b_far) > 0:
    bookshape = (train[b_near].fillna(0).sum(axis=1).values -
                 train[b_far].fillna(0).sum(axis=1).values)
    use_bookshape = True
    print(f"  BookShape computed. Mean={bookshape.mean():.0f}  Std={bookshape.std():.0f}")
    print(f"  BookShape percentiles: p25={np.percentile(bookshape,25):.0f}  "
          f"p50={np.percentile(bookshape,50):.0f}  p75={np.percentile(bookshape,75):.0f}")
else:
    print("  WARNING: B-bin features not found. Falling back to SO3_T as proxy.")
    use_bookshape = False

train['bookshape'] = bookshape if use_bookshape else train['SO3_T'].values


# ── Normalization helpers ──────────────────────────────────────────
def zscore_day(X_tr, X_te=None, clip=5.0):
    m = X_tr.mean(axis=0)
    s = X_tr.std(axis=0)
    s = np.where(s < 1e-8, 1.0, s)
    Xtz = np.clip((X_tr - m) / s, -clip, clip)
    Xez = np.clip((X_te - m) / s, -clip, clip) if X_te is not None else None
    return Xtz, Xez

def rank_day(X_tr, X_te=None):
    if X_te is None:
        n = X_tr.shape[0]
        return np.apply_along_axis(
            lambda col: rankdata(col, method='average') / n, 0, X_tr), None
    n_tr = X_tr.shape[0]
    X_all = np.vstack([X_tr, X_te])
    n_all = X_all.shape[0]
    X_rank = np.apply_along_axis(
        lambda col: rankdata(col, method='average') / n_all, 0, X_all)
    return X_rank[:n_tr], X_rank[n_tr:]

def winsorise(y, pct=5):
    lo, hi = np.percentile(y, pct), np.percentile(y, 100 - pct)
    return np.clip(y, lo, hi)

def fit_predict_ridge(X_tr, y_tr, X_te, alpha=100):
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_tr, winsorise(y_tr))
    return model.predict(X_te)

def per_day_ic(y_true, y_pred):
    if len(y_true) < 5:
        return np.nan
    corr, _ = spearmanr(y_true, y_pred)
    return corr


# ================================================================
# MAIN LOOP — Pseudo-Illiquid OOF vs Standard 80/20 OOF
# ================================================================

rng = np.random.default_rng(42)

experiments = {
    'gold_z'   : (gold_feats,   'zscore'),
    'gold_r'   : (gold_feats,   'rank'),
    'silver_z' : (silver_feats, 'zscore'),
}

print("\n" + "="*65)
print("RUNNING PSEUDO-ILLIQUID OOF vs STANDARD 80/20 OOF")
print("="*65)
print("(Training on top-50% BookShape, validating on bottom-50%)")
print()

# Storage
pi_oof     = {k: np.full(len(train), np.nan) for k in experiments}  # pseudo-illiquid OOF
std_oof    = {k: np.zeros(len(train)) for k in experiments}          # standard 80/20 OOF

pi_day_ics  = {k: {} for k in experiments}   # per-day IC on pseudo-illiquid set
std_day_ics = {k: {} for k in experiments}   # per-day IC on standard validation set
pi_day_r2s  = {k: {} for k in experiments}
std_day_r2s = {k: {} for k in experiments}

for exp_name, (feat_cols, norm_type) in experiments.items():
    print(f"\n  [{exp_name}]  {len(feat_cols)} feats  norm={norm_type}")
    t_exp = time.time()

    days_skipped = 0
    days_ok      = 0

    for day, grp in train.groupby('day_id'):
        n = len(grp)
        if n < 20:
            days_skipped += 1
            continue

        y_day   = y_train[grp.index]
        bs_day  = grp['bookshape'].values

        # ── Pseudo-illiquid split (by BookShape) ──────────────────
        bs_median    = np.median(bs_day)
        liq_mask     = bs_day >= bs_median      # pseudo-liquid = train
        illiq_mask   = ~liq_mask                 # pseudo-illiquid = validation

        n_liq   = liq_mask.sum()
        n_illiq = illiq_mask.sum()

        if n_liq < 10 or n_illiq < 5:
            days_skipped += 1
            continue

        liq_idx   = grp.index[liq_mask]
        illiq_idx = grp.index[illiq_mask]

        X_liq_raw   = grp.loc[liq_idx,   feat_cols].fillna(0).values.astype(np.float64)
        X_illiq_raw = grp.loc[illiq_idx, feat_cols].fillna(0).values.astype(np.float64)
        y_liq       = y_train[liq_idx]
        y_illiq     = y_train[illiq_idx]

        if norm_type == 'zscore':
            X_liq_n, X_illiq_n = zscore_day(X_liq_raw, X_illiq_raw)
        else:
            X_liq_n, X_illiq_n = rank_day(X_liq_raw, X_illiq_raw)

        pi_preds = fit_predict_ridge(X_liq_n, y_liq, X_illiq_n)
        pi_oof[exp_name][illiq_idx] = pi_preds

        if len(y_illiq) >= 5:
            pi_day_r2s[exp_name][day] = r2_score(y_illiq, pi_preds)
            pi_day_ics[exp_name][day] = per_day_ic(y_illiq, pi_preds)

        # ── Standard 80/20 OOF ────────────────────────────────────
        perm   = rng.permutation(n)
        n_tr   = int(n * 0.8)
        tr_idx = grp.index[perm[:n_tr]]
        va_idx = grp.index[perm[n_tr:]]

        X_tr_raw = grp.loc[tr_idx, feat_cols].fillna(0).values.astype(np.float64)
        X_va_raw = grp.loc[va_idx, feat_cols].fillna(0).values.astype(np.float64)
        y_tr     = y_train[tr_idx]
        y_va     = y_train[va_idx]

        if norm_type == 'zscore':
            X_tr_n, X_va_n = zscore_day(X_tr_raw, X_va_raw)
        else:
            X_tr_n, X_va_n = rank_day(X_tr_raw, X_va_raw)

        std_preds = fit_predict_ridge(X_tr_n, y_tr, X_va_n)
        std_oof[exp_name][va_idx] = std_preds

        if len(y_va) >= 5:
            std_day_r2s[exp_name][day] = r2_score(y_va, std_preds)
            std_day_ics[exp_name][day] = per_day_ic(y_va, std_preds)

        days_ok += 1

    # ── Summary ───────────────────────────────────────────────────
    # Pseudo-illiquid OOF: only on illiquid-half rows
    pi_mask   = ~np.isnan(pi_oof[exp_name])
    pi_r2     = r2_score(y_train[pi_mask], pi_oof[exp_name][pi_mask]) if pi_mask.sum() > 0 else np.nan
    std_r2    = r2_score(y_train, std_oof[exp_name])

    pi_med_ic  = np.nanmedian(list(pi_day_ics[exp_name].values()))
    std_med_ic = np.nanmedian(list(std_day_ics[exp_name].values()))
    pi_med_r2  = np.nanmedian(list(pi_day_r2s[exp_name].values()))
    std_med_r2 = np.nanmedian(list(std_day_r2s[exp_name].values()))

    print(f"    Days OK / skipped: {days_ok} / {days_skipped}")
    print(f"    Pseudo-illiquid OOF:")
    print(f"      Global R²  = {pi_r2:+.6f}")
    print(f"      Median day R² = {pi_med_r2:+.6f}")
    print(f"      Median day IC = {pi_med_ic:+.6f}")
    print(f"    Standard 80/20 OOF:")
    print(f"      Global R²  = {std_r2:+.6f}")
    print(f"      Median day R² = {std_med_r2:+.6f}")
    print(f"      Median day IC = {std_med_ic:+.6f}")
    print(f"    OOF inflation ratio (std/pi): {std_r2/pi_r2:.2f}x  ({time.time()-t_exp:.0f}s)")


# ================================================================
# GRINOLD DIRECT ALPHA — Parameter-free prediction
# ================================================================
# alpha_i = mean_IC_j * z_i_j  summed over gold features
# This is the theoretically optimal linear combination.
# ================================================================
print("\n" + "="*65)
print("GRINOLD DIRECT ALPHA (no model fitting)")
print("="*65)

gold_ics    = np.array([gold_ic_map.get(f, 0.0) for f in gold_feats])
print(f"  Using {len(gold_feats)} gold features, IC range: [{gold_ics.min():.4f}, {gold_ics.max():.4f}]")

grinold_pi_r2s  = {}
grinold_pi_ics  = {}
grinold_std_r2s = {}

for day, grp in train.groupby('day_id'):
    n = len(grp)
    if n < 20:
        continue

    X_day = grp[gold_feats].fillna(0).values.astype(np.float64)
    y_day = y_train[grp.index]
    bs_day = grp['bookshape'].values

    # Z-score within day
    m, s = X_day.mean(0), X_day.std(0)
    s    = np.where(s < 1e-8, 1.0, s)
    Z    = np.clip((X_day - m) / s, -5, 5)

    # Grinold: prediction = Z @ mean_IC (weighted sum of z-scores)
    grinold_pred = Z @ gold_ics   # alpha_i = sum_j(IC_j * z_ij)

    # Standard OOF (just measure on all — since no model, use full day)
    ic_full = per_day_ic(y_day, grinold_pred)
    grinold_std_r2s[day] = r2_score(y_day, grinold_pred)

    # Pseudo-illiquid: measure only on illiquid half
    illiq_mask = bs_day < np.median(bs_day)
    if illiq_mask.sum() >= 5:
        grinold_pi_r2s[day] = r2_score(y_day[illiq_mask], grinold_pred[illiq_mask])
        grinold_pi_ics[day] = per_day_ic(y_day[illiq_mask], grinold_pred[illiq_mask])

grinold_pi_med_r2  = np.nanmedian(list(grinold_pi_r2s.values()))
grinold_pi_med_ic  = np.nanmedian(list(grinold_pi_ics.values()))
grinold_std_med_r2 = np.nanmedian(list(grinold_std_r2s.values()))

print(f"  Grinold pseudo-illiquid:  median day R²={grinold_pi_med_r2:+.6f}  median IC={grinold_pi_med_ic:+.6f}")
print(f"  Grinold full-day:         median day R²={grinold_std_med_r2:+.6f}")


# ================================================================
# KEY DIAGNOSTIC: Day R² distribution under pseudo-illiquid OOF
# ================================================================
print("\n" + "="*65)
print("DIAGNOSTIC: Day-level IC distribution (pseudo-illiquid)")
print("="*65)
print(f"  {'Experiment':<12}  {'p10':>8}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'p90':>8}  "
      f"{'%neg':>8}  {'%pos':>8}")
print(f"  {'-'*74}")

for name in list(experiments.keys()) + ['grinold']:
    if name == 'grinold':
        ics = list(grinold_pi_ics.values())
    else:
        ics = list(pi_day_ics[name].values())

    ics_arr = np.array([x for x in ics if not np.isnan(x)])
    if len(ics_arr) == 0:
        continue
    p10, p25, p50, p75, p90 = np.percentile(ics_arr, [10,25,50,75,90])
    pct_neg = (ics_arr < 0).mean() * 100
    pct_pos = (ics_arr > 0).mean() * 100
    print(f"  {name:<12}  {p10:+8.4f}  {p25:+8.4f}  {p50:+8.4f}  {p75:+8.4f}  {p90:+8.4f}  "
          f"{pct_neg:7.1f}%  {pct_pos:7.1f}%")


# ================================================================
# SUMMARY TABLE — What does pseudo-illiquid OOF vs standard tell us?
# ================================================================
print("\n" + "="*65)
print("SUMMARY: OOF COMPARISON (Pseudo-Illiquid vs Standard 80/20)")
print("="*65)
print(f"\n  Known LB scores for reference:")
print(f"    fold_safe_v1           : LB = +0.00005")
print(f"    transductive_v4_005    : LB = +0.00003")
print(f"    knn_K3_3pct            : LB = -0.00042  (OOF was +0.093 — worst inverse)")
print(f"    lgbm_baseline_v1       : LB = -0.00002")
print()

print(f"  {'Experiment':<12}  {'PI R²':>12}  {'Std R²':>12}  {'PI med IC':>12}  {'Std med IC':>12}  {'Inflation':>10}")
print(f"  {'-'*72}")
for name, (feat_cols, norm_type) in experiments.items():
    pi_mask  = ~np.isnan(pi_oof[name])
    pi_r2    = r2_score(y_train[pi_mask], pi_oof[name][pi_mask]) if pi_mask.sum() > 0 else np.nan
    std_r2   = r2_score(y_train, std_oof[name])
    pi_med   = np.nanmedian(list(pi_day_ics[name].values()))
    std_med  = np.nanmedian(list(std_day_ics[name].values()))
    ratio    = std_r2 / pi_r2 if pi_r2 != 0 else np.nan
    print(f"  {name:<12}  {pi_r2:+12.6f}  {std_r2:+12.6f}  {pi_med:+12.6f}  {std_med:+12.6f}  {ratio:10.2f}x")

print()
print(f"  Grinold (no model):")
print(f"  {'grinold':<12}  {grinold_pi_med_r2:+12.6f}  {grinold_std_med_r2:+12.6f}  "
      f"{grinold_pi_med_ic:+12.6f}  {'n/a':>12}  {'n/a':>10}")

print(f"""
INTERPRETATION GUIDE:
──────────────────────
If PI R² < Standard R²:
  → OOF inflation is confirmed. Standard OOF is too optimistic.
  → PI R² is a better proxy for LB performance.

If PI R² is positive and small (similar to LB scale ~0.00005):
  → We are seeing the real signal. Pseudo-illiquid OOF is valid.

If PI R² is negative:
  → Signal does NOT generalise from liquid to illiquid in this model.
  → Need covariate shift reweighting before this works.

The Grinold alpha:
  → Zero parameters. Only reflects signal quality, not model fitting.
  → If Grinold PI R² > 0 but model PI R² < 0, it's a calibration problem.
""")

print(f"Total elapsed: {(time.time()-t0)/60:.1f} min")
