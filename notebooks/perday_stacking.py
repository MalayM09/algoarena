# ================================================================
# PER-DAY STACKING — Day-specific meta-model for signal combination
# ================================================================
# Hypothesis:
#   The optimal blend of Grinold IC-signal and Ridge signal may
#   vary by market regime (day). On volatile days Ridge may
#   outperform; on calm days Grinold may dominate.
#
#   Per-day stacking learns day-specific combination weights
#   using liquid assets as a training population, then applies
#   those weights to illiquid test assets.
#
# Method per overlap day:
#   1. Z-score features using day's training population stats
#   2. Compute signal_A = Grinold IC-weighted z-score (top-10)
#   3. Compute signal_B = Ridge(alpha=10).predict on test assets
#      (fitted on liquid assets of that day)
#   4. Meta-model: OLS on [signal_A, signal_B] → returns
#      using LIQUID assets (in-sample for that day)
#   5. Apply day-specific meta-weights to TEST asset predictions
#
# Note: meta-model is fitted on same population used to fit Ridge
# → Grinold vs Ridge weight is calibrated in-sample per day.
#   The regime adaptation (daily weight variation) is the key value-add.
#
# Also implements: global OLS meta-weight (calibrated across all days)
# as a sanity check against the threeway ensemble.
#
# Generates:
#   perday_stack_insample.csv       — day-specific meta-weights (in-sample)
#   perday_stack_global.csv         — global fixed meta-weights
#   perday_stack_oos_bs.csv         — OOF: fit on high-BS, predict low-BS
#   perday_stack_ens.csv            — best stacking blended with best ensemble
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression

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

RIDGE_ALPHA = 10.0
TARGET_STD  = 0.000948
CLIP_Z      = 5.0

print("=" * 65)
print("PER-DAY STACKING — Day-Specific Signal Combination")
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
sample_sub = pd.read_csv(SAMPLE_PATH)[['ID']]
n_test     = len(test)
print(f"  Train: {len(train):,}  Test: {n_test:,}")

# ── Gold top-10 features ───────────────────────────────────────────
icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = ((icir_df['abs_icir'] >= 3) &
             ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0)))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
top10     = [f for f in gold_df['feature'].tolist()[:10] if f in train.columns]
ic_arr    = np.array([gold_df.set_index('feature')['mean_ic'].to_dict()[f] for f in top10])
print(f"  Gold top-10: {len(top10)} features")

# ── BookShape proxy (for OOF split) ───────────────────────────────
all_cols = set(train.columns) - {'ID', 'TARGET', 'CV_GROUP', 'SO3_T', 'day_id'}
b_near   = [c for c in all_cols if 'Lag' not in c and
            any(f'_B0{i}' in c for i in range(1, 6))]
b_far    = [c for c in all_cols if 'Lag' not in c and
            any(f'_B{i}' in c for i in ['06', '07', '08', '09', '10'])]
train['bookshape'] = (train[b_near].fillna(0).sum(1) -
                      train[b_far].fillna(0).sum(1)).astype(np.float64)

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

# ── Load best ensemble ─────────────────────────────────────────────
best_ens_raw = (sample_sub
                .merge(pd.read_csv(BEST_ENS)[['ID','TARGET']]
                       .rename(columns={'TARGET':'b'}), on='ID', how='left')
                .fillna(0.0)['b'].values)
best_s = auto_scale(best_ens_raw)

# ================================================================
# Pass 1: Calibrate global meta-weights using all training days
# ================================================================
# For each training day: compute signal_A (Grinold) and signal_B (Ridge)
# Collect all (signal_A, signal_B, y) pairs → fit global OLS meta-model
print("\nPass 1: Calibrating global meta-weights...")

global_sig_A = []
global_sig_B = []
global_y     = []
day_meta_weights = {}  # day → (w_A, w_B, intercept)

for day, grp in train.groupby('day_id'):
    n = len(grp)
    if n < 20:
        continue
    y_d   = y_train[grp.index]
    y_d_w = winsorise(y_d)
    X_raw = grp[top10].fillna(0).values.astype(np.float64)

    # Z-score using all-day stats (standard Grinold approach)
    X_z, m, s = zscore_fit(X_raw)

    # Signal A: Grinold IC-weighted
    sig_A = X_z @ ic_arr
    sig_A -= sig_A.mean()

    # Signal B: Per-day Ridge (fitted on same day's liquid-like assets)
    # Use top-50% BookShape as pseudo-liquid
    bs_d = grp['bookshape'].values
    bs_med = np.median(bs_d)
    hi_mask = bs_d >= bs_med
    lo_mask = ~hi_mask
    if hi_mask.sum() < 10:
        continue

    X_hi_z = X_z[hi_mask]
    y_hi_w = winsorise(y_d[hi_mask])
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge.fit(X_hi_z, y_hi_w)
    sig_B = ridge.predict(X_z)
    sig_B -= sig_B.mean()

    # Collect for global calibration
    global_sig_A.extend(sig_A.tolist())
    global_sig_B.extend(sig_B.tolist())
    global_y.extend(y_d_w.tolist())

    # Also fit per-day meta-weights (in-sample, for variance analysis)
    X_meta = np.column_stack([sig_A, sig_B])
    meta = LinearRegression(fit_intercept=True)
    meta.fit(X_meta, y_d_w)
    day_meta_weights[day] = (meta.coef_[0], meta.coef_[1], meta.intercept_)

print(f"  Days with meta-weights: {len(day_meta_weights)}")

# Global OLS
global_X = np.column_stack([global_sig_A, global_sig_B])
global_y_arr = np.array(global_y)
global_meta = LinearRegression(fit_intercept=True)
global_meta.fit(global_X, global_y_arr)
w_A_global = global_meta.coef_[0]
w_B_global = global_meta.coef_[1]
print(f"  Global meta-weights: w_Grinold={w_A_global:.4f}  w_Ridge={w_B_global:.4f}")
print(f"  (compare to threeway: Ridge≈0.30, Grinold≈0.30)")

# Distribution of per-day weights
wa_vals = [v[0] for v in day_meta_weights.values()]
wb_vals = [v[1] for v in day_meta_weights.values()]
print(f"\n  Per-day w_Grinold: mean={np.mean(wa_vals):.4f}  std={np.std(wa_vals):.4f}  "
      f"min={np.min(wa_vals):.4f}  max={np.max(wa_vals):.4f}")
print(f"  Per-day w_Ridge:   mean={np.mean(wb_vals):.4f}  std={np.std(wb_vals):.4f}  "
      f"min={np.min(wb_vals):.4f}  max={np.max(wb_vals):.4f}")
print(f"  → Weight std / mean = {np.std(wa_vals)/max(abs(np.mean(wa_vals)),1e-8):.2f} for Grinold, "
      f"{np.std(wb_vals)/max(abs(np.mean(wb_vals)),1e-8):.2f} for Ridge")
print(f"  (If std/mean > 0.5: regime variation exists → day-specific weights add value)")

# ================================================================
# Pass 2: Generate test predictions
# ================================================================
print("\nPass 2: Generating test predictions...")

preds_insample = np.zeros(n_test)   # day-specific meta-weights (in-sample)
preds_global   = np.zeros(n_test)   # global fixed meta-weights
preds_grinold  = np.zeros(n_test)   # pure Grinold (baseline)

for day, grp_te in test.groupby('day_id'):
    te_idx   = grp_te.index.values
    X_te_raw = grp_te[top10].fillna(0).values.astype(np.float64)

    if day in train_days:
        grp_tr   = train[train['day_id'] == day]
        X_tr_raw = grp_tr[top10].fillna(0).values.astype(np.float64)
        y_tr     = winsorise(y_train[grp_tr.index])

        _, m, s  = zscore_fit(X_tr_raw)
        X_tr_z   = zscore_apply(X_tr_raw, m, s)
        X_te_z   = zscore_apply(X_te_raw, m, s)

        # Signal A: Grinold
        sig_A_te = X_te_z @ ic_arr
        sig_A_te -= sig_A_te.mean()

        # Signal B: Ridge (fitted on all training assets for this day)
        if len(grp_tr) >= 10:
            ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
            ridge.fit(X_tr_z, y_tr)
            sig_B_te = ridge.predict(X_te_z)
            sig_B_te -= sig_B_te.mean()
        else:
            sig_B_te = sig_A_te.copy()

        # In-sample day-specific meta
        if day in day_meta_weights:
            wa, wb, intercept = day_meta_weights[day]
        else:
            wa, wb, intercept = w_A_global, w_B_global, global_meta.intercept_

        pred_insample = wa * sig_A_te + wb * sig_B_te + intercept
        pred_insample -= pred_insample.mean()
        preds_insample[te_idx] = pred_insample

        # Global meta
        pred_global = w_A_global * sig_A_te + w_B_global * sig_B_te + global_meta.intercept_
        pred_global -= pred_global.mean()
        preds_global[te_idx] = pred_global

        # Pure Grinold
        pred_grin = sig_A_te.copy()
        preds_grinold[te_idx] = pred_grin

    else:
        # No training data — Grinold only
        X_te_z, _, _ = zscore_fit(X_te_raw)
        pred = X_te_z @ ic_arr
        pred -= pred.mean()
        preds_insample[te_idx] = pred
        preds_global[te_idx]   = pred
        preds_grinold[te_idx]  = pred

print("  Done.")

# ================================================================
# Pass 3: OOF validation using BookShape split
# ================================================================
# Fit Ridge + meta on HIGH-BS liquid, validate on LOW-BS pseudo-illiquid
print("\nPass 3: OOF validation (high-BS → low-BS prediction)...")

oof_preds_stack  = np.zeros(len(train))
oof_preds_grinold = np.zeros(len(train))
oof_counts = np.zeros(len(train))

for day, grp in train.groupby('day_id'):
    if len(grp) < 30:
        continue
    y_d   = y_train[grp.index]
    bs_d  = grp['bookshape'].values
    X_raw = grp[top10].fillna(0).values.astype(np.float64)
    bs_med = np.median(bs_d)
    hi_mask = bs_d >= bs_med
    lo_mask = ~hi_mask
    if hi_mask.sum() < 10 or lo_mask.sum() < 5:
        continue

    # Z-score using high-BS stats (liquid population)
    X_hi_raw = X_raw[hi_mask]
    X_hi_z, m, s = zscore_fit(X_hi_raw)
    X_lo_z = zscore_apply(X_raw[lo_mask], m, s)
    X_all_z = zscore_apply(X_raw, m, s)

    y_hi_w = winsorise(y_d[hi_mask])

    # Signal A: Grinold
    sig_A_lo = X_lo_z @ ic_arr
    sig_A_lo -= sig_A_lo.mean()

    # Signal B: Ridge
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    ridge.fit(X_hi_z, y_hi_w)
    sig_B_all = ridge.predict(X_all_z)
    sig_B_lo  = sig_B_all[lo_mask]
    sig_B_lo -= sig_B_lo.mean()

    # Meta prediction (global weights)
    pred_stack = w_A_global * sig_A_lo + w_B_global * sig_B_lo
    pred_stack -= pred_stack.mean()

    idx_lo = grp.index[lo_mask]
    oof_preds_stack[idx_lo]  = pred_stack
    oof_preds_grinold[idx_lo]= sig_A_lo
    oof_counts[idx_lo]       = 1

valid_mask = oof_counts > 0
oof_stack_r  = pearson_r(oof_preds_stack[valid_mask],   y_train[valid_mask])
oof_grinold_r= pearson_r(oof_preds_grinold[valid_mask], y_train[valid_mask])
print(f"  OOF Pearson (stacking):  {oof_stack_r:+.4f}")
print(f"  OOF Pearson (Grinold):   {oof_grinold_r:+.4f}")
print(f"  (Note: OOF is inversely correlated with LB — use only for sign check)")

# ================================================================
# Save and report
# ================================================================
grinold_s = auto_scale(preds_grinold)

print(f"\n{'Variant':<25}  {'corr_vs_grinold':>16}  {'corr_vs_ens':>12}  File")
print("-" * 70)

variants = {
    'perday_stack_insample': preds_insample,
    'perday_stack_global':   preds_global,
    'perday_stack_grinold':  preds_grinold,
}

saved = {}
for vname, preds in variants.items():
    ps     = auto_scale(preds)
    c_grin = pearson_r(ps, grinold_s)
    c_ens  = pearson_r(ps, best_s)
    sub    = pd.DataFrame({'ID': test_ids, 'TARGET': ps})
    sub    = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{vname}.csv'), index=False)
    saved[vname] = ps
    print(f"  {vname:<25}  {c_grin:+.4f}           {c_ens:+.4f}       {vname}.csv")

# Blends with best ensemble
print("\nBuilding blends (stacking + best_ens)...")
for vname in ['perday_stack_insample', 'perday_stack_global']:
    rs = saved[vname]
    for w in [0.20, 0.30, 0.40]:
        blend   = (1 - w) * best_s + w * rs
        blend_s = auto_scale(blend)
        c       = pearson_r(blend_s, best_s)
        name    = f'{vname}_w{int(w*100)}_ens{int((1-w)*100)}'
        sub     = pd.DataFrame({'ID': test_ids, 'TARGET': blend_s})
        sub     = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
        sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
        print(f"  {name:<55}  corr_vs_best={c:+.4f}")

print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
print(f"""
── INTERPRETATION ─────────────────────────────────────────────
Global meta-weights: w_Grinold={w_A_global:.4f}  w_Ridge={w_B_global:.4f}
  Expected from threeway: each ≈ 0.25-0.35

Per-day weight std/mean for Grinold: {np.std(wa_vals)/max(abs(np.mean(wa_vals)),1e-8):.2f}
Per-day weight std/mean for Ridge:   {np.std(wb_vals)/max(abs(np.mean(wb_vals)),1e-8):.2f}
  > 0.5: regime variation exists → day-specific weights add value
  < 0.3: weights are stable → fixed global weights are sufficient

corr_vs_ens for perday_stack_insample:
  > 0.99: stacking rediscovers same signal as ensemble → no gain
  0.95-0.99: some adaptation → worth submitting
  < 0.95: significantly different → high upside potential

KEY: If day-specific meta-weights vary substantially (std/mean > 0.5),
  the stacking is learning real regime information. If they're stable,
  it's just replicating the threeway ensemble with extra steps.
""")
