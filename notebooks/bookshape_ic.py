# ================================================================
# BOOKSHAPE-CONDITIONED IC — Domain-adapted Grinold
# ================================================================
# Hypothesis:
#   Liquid training assets have high BookShape (near-best-bid/ask).
#   Illiquid test assets have low BookShape.
#   The IC of a feature computed on high-BS liquid assets may
#   differ from the IC computed on low-BS liquid assets.
#   Low-BS liquid assets better resemble illiquid test assets.
#
# Method:
#   1. Build BookShape proxy for all assets (same as perday_ridge.py):
#      BS = sum(near_bid/ask levels) - sum(far bid/ask levels)
#   2. Per day, split liquid training assets into high/low BookShape halves
#   3. Compute IC on each half separately → IC_high, IC_low
#   4. For test assets: apply IC_low (match closer to illiquid distribution)
#   5. Also try weighted combination: IC = alpha*IC_low + (1-alpha)*IC_high
#
# Comparison baseline: static Grinold (long-run mean IC on all assets)
#
# Generates:
#   bs_ic_low.csv       — IC from low-BS liquid half only
#   bs_ic_high.csv      — IC from high-BS liquid half only
#   bs_ic_global.csv    — IC from all liquid assets (Grinold baseline recomputed)
#   bs_ic_weighted.csv  — weighted blend of low/high IC
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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

print("=" * 65)
print("BOOKSHAPE-CONDITIONED IC — Domain-Adapted Grinold")
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
ic_longrun = np.array([gold_df.set_index('feature')['mean_ic'].to_dict()[f] for f in top10])
print(f"  Gold top-10: {len(top10)} features")

# ── BookShape proxy ────────────────────────────────────────────────
# Near levels: B01..B05 (first 5 bid/ask levels)
# Far levels:  B06..B10 (next 5 levels)
all_cols = set(train.columns) - {'ID', 'TARGET', 'CV_GROUP', 'SO3_T', 'day_id'}
b_near   = [c for c in all_cols if 'Lag' not in c and
            any(f'_B0{i}' in c for i in range(1, 6))]
b_far    = [c for c in all_cols if 'Lag' not in c and
            any(f'_B{i}' in c for i in ['06', '07', '08', '09', '10'])]
print(f"  BookShape proxy — near cols: {len(b_near)}  far cols: {len(b_far)}")

train['bookshape'] = (train[b_near].fillna(0).sum(1) -
                      train[b_far].fillna(0).sum(1)).astype(np.float64)
test['bookshape']  = (test[b_near].fillna(0).sum(1) -
                      test[b_far].fillna(0).sum(1)).astype(np.float64)

# ── Helpers ────────────────────────────────────────────────────────
def zscore_fit(X, clip=CLIP_Z):
    m = X.mean(0); s = X.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip), m, s

def zscore_apply(X, m, s, clip=CLIP_Z):
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip)

def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d > 1e-12 else 0.0

# ── Step 1: Compute per-day IC by BookShape stratum ───────────────
print("\nComputing per-day BookShape-stratified ICs...")

ic_high_list   = []   # per-day IC from high-BS half
ic_low_list    = []   # per-day IC from low-BS half
ic_global_list = []   # per-day IC from all assets (baseline)
day_count = 0

for day, grp in train.groupby('day_id'):
    if len(grp) < 20:
        continue
    y_d  = y_train[grp.index]
    bs_d = grp['bookshape'].values
    X_d  = grp[top10].fillna(0).values.astype(np.float64)

    bs_med = np.median(bs_d)
    hi_mask = bs_d >= bs_med
    lo_mask = bs_d <  bs_med

    if hi_mask.sum() < 5 or lo_mask.sum() < 5:
        continue

    def compute_ic(X, y):
        ics = np.zeros(len(top10))
        for j in range(len(top10)):
            if len(y) < 5:
                ics[j] = np.nan
            else:
                r, _ = spearmanr(X[:, j], y)
                ics[j] = r
        return ics

    ic_hi  = compute_ic(X_d[hi_mask],  y_d[hi_mask])
    ic_lo  = compute_ic(X_d[lo_mask],  y_d[lo_mask])
    ic_all = compute_ic(X_d,           y_d)

    ic_high_list.append(ic_hi)
    ic_low_list.append(ic_lo)
    ic_global_list.append(ic_all)
    day_count += 1

print(f"  Days processed: {day_count}")

ic_high_arr   = np.array(ic_high_list)    # (n_days, 10)
ic_low_arr    = np.array(ic_low_list)
ic_global_arr = np.array(ic_global_list)

# Replace NaN with 0 before averaging
ic_high_arr   = np.where(np.isnan(ic_high_arr),   0, ic_high_arr)
ic_low_arr    = np.where(np.isnan(ic_low_arr),     0, ic_low_arr)
ic_global_arr = np.where(np.isnan(ic_global_arr),  0, ic_global_arr)

ic_high_mean   = ic_high_arr.mean(0)
ic_low_mean    = ic_low_arr.mean(0)
ic_global_mean = ic_global_arr.mean(0)

print(f"\n  IC comparison (mean across days, per feature):")
print(f"  {'Feature':<45}  {'IC_high':>8}  {'IC_low':>8}  {'IC_global':>10}  {'LongRun':>8}")
for j, feat in enumerate(top10):
    print(f"  {feat:<45}  {ic_high_mean[j]:+.4f}    {ic_low_mean[j]:+.4f}    {ic_global_mean[j]:+.4f}    {ic_longrun[j]:+.4f}")

# Sign-lock: ensure all IC vectors agree in sign with long-run IC
# (low-BS IC on small subsample is noisy, can flip)
def sign_lock(ic_vec, reference=ic_longrun):
    sign_match = np.sign(ic_vec) == np.sign(reference)
    return np.where(sign_match, ic_vec, reference)

ic_high_locked   = sign_lock(ic_high_mean)
ic_low_locked    = sign_lock(ic_low_mean)
ic_global_locked = sign_lock(ic_global_mean)

print(f"\n  After sign-locking to long-run IC:")
print(f"  IC_high_locked:   {ic_high_locked.round(4)}")
print(f"  IC_low_locked:    {ic_low_locked.round(4)}")
print(f"  IC_global_locked: {ic_global_locked.round(4)}")
print(f"  IC_longrun:       {ic_longrun.round(4)}")

# ── Step 2: Score test assets with each IC vector ─────────────────
print("\nScoring test assets...")

def score_grinold(ic_vec, label=""):
    preds = np.zeros(n_test)
    for day, grp_te in test.groupby('day_id'):
        te_idx   = grp_te.index.values
        X_te_raw = grp_te[top10].fillna(0).values.astype(np.float64)
        if day in train_days:
            grp_tr   = train[train['day_id'] == day]
            X_tr_raw = grp_tr[top10].fillna(0).values.astype(np.float64)
            _, m, s  = zscore_fit(X_tr_raw)
            X_te_z   = zscore_apply(X_te_raw, m, s)
        else:
            X_te_z, _, _ = zscore_fit(X_te_raw)
        pred = X_te_z @ ic_vec
        pred -= pred.mean()
        preds[te_idx] = pred
    return preds

preds_high   = score_grinold(ic_high_locked,   "high-BS")
preds_low    = score_grinold(ic_low_locked,    "low-BS")
preds_global = score_grinold(ic_global_locked, "global")
preds_longrun= score_grinold(ic_longrun,        "longrun")
print("  Done scoring all IC variants.")

# ── Step 3: Test-day BookShape-adaptive weighting ──────────────────
# For each test asset, weight IC_low more if its BookShape is low
print("\nComputing BookShape-adaptive IC predictions...")

preds_adaptive = np.zeros(n_test)
for day, grp_te in test.groupby('day_id'):
    te_idx   = grp_te.index.values
    X_te_raw = grp_te[top10].fillna(0).values.astype(np.float64)
    bs_te    = grp_te['bookshape'].values

    if day in train_days:
        grp_tr   = train[train['day_id'] == day]
        X_tr_raw = grp_tr[top10].fillna(0).values.astype(np.float64)
        _, m, s  = zscore_fit(X_tr_raw)
        X_te_z   = zscore_apply(X_te_raw, m, s)
        # BookShape reference: training day median (liquid assets)
        bs_ref = grp_tr['bookshape'].median()
    else:
        X_te_z, _, _ = zscore_fit(X_te_raw)
        bs_ref = test['bookshape'].median()  # global test median

    # Weight: sigmoid-smoothed by relative BookShape
    # assets with BS < bs_ref → get more weight on IC_low
    # assets with BS >= bs_ref → get more weight on IC_high
    bs_norm = np.clip((bs_te - bs_ref) / (np.abs(bs_ref) + 1e-8), -3, 3)
    w_high  = 1 / (1 + np.exp(-bs_norm))  # sigmoid: 1 if very high BS, 0 if very low
    w_low   = 1 - w_high

    # Compute predictions from each IC
    pred_hi_day = X_te_z @ ic_high_locked
    pred_lo_day = X_te_z @ ic_low_locked

    # Per-asset adaptive blend
    pred = w_high * pred_hi_day + w_low * pred_lo_day
    pred -= pred.mean()
    preds_adaptive[te_idx] = pred

# ── Step 4: Load best ensemble ─────────────────────────────────────
best_ens_raw = (sample_sub
                .merge(pd.read_csv(BEST_ENS)[['ID','TARGET']]
                       .rename(columns={'TARGET':'b'}), on='ID', how='left')
                .fillna(0.0)['b'].values)
best_s = auto_scale(best_ens_raw)
grinold_s = auto_scale(preds_longrun)

# ── Step 5: Save and report ────────────────────────────────────────
print(f"\n{'Variant':<20}  {'corr_vs_grinold':>16}  {'corr_vs_ens':>12}  File")
print("-" * 70)

variants = {
    'bs_ic_high':     preds_high,
    'bs_ic_low':      preds_low,
    'bs_ic_global':   preds_global,
    'bs_ic_adaptive': preds_adaptive,
    'bs_ic_longrun':  preds_longrun,
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
    print(f"  {vname:<20}  {c_grin:+.4f}           {c_ens:+.4f}       {vname}.csv")

# ── Step 6: Blends with best ensemble ─────────────────────────────
print("\nBuilding blends (bs_ic + best_ens)...")
focus_variants = ['bs_ic_low', 'bs_ic_adaptive']  # most interesting
for vname in focus_variants:
    rs = saved[vname]
    for w in [0.20, 0.30, 0.40]:
        blend   = (1 - w) * best_s + w * rs
        blend_s = auto_scale(blend)
        c       = pearson_r(blend_s, best_s)
        name    = f'{vname}_w{int(w*100)}_ens{int((1-w)*100)}'
        sub     = pd.DataFrame({'ID': test_ids, 'TARGET': blend_s})
        sub     = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
        sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
        print(f"  {name:<50}  corr_vs_best={c:+.4f}")

print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
print("""
── INTERPRETATION ─────────────────────────────────────────────
Key question: Does IC computed on LOW-BookShape liquid assets
  generalize better to illiquid test assets than IC computed on
  HIGH-BookShape liquid assets?

corr_vs_grinold for bs_ic_low:
  > 0.99: IC is identical regardless of BookShape split → no gain
  < 0.99: IC differs by stratum → worth testing on LB

If bs_ic_low scores higher than static Grinold on LB:
  → Feature-return relationship is BookShape-dependent
  → Low-BS assets are better proxies for illiquid test population

Adaptive variant uses per-asset BookShape weighting → best of both worlds
but only if the two IC vectors are genuinely different.
""")
