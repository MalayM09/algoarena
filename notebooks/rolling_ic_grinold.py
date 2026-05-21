# ================================================================
# ROLLING IC GRINOLD — Adaptive IC using recent N-day window
# ================================================================
# Hypothesis:
#   Fixed long-run mean IC may average over different regimes.
#   Per-day ICs computed on each training day, then rolling-averaged
#   over the N most recent days, should better adapt to the current
#   market regime on each test day.
#
# Method:
#   For each training day: compute Spearman IC for each gold feature
#   For each test day: find N most recent training days → rolling mean IC
#   Use rolling IC as weights in Grinold scoring
#
# Sweep: N = [20, 30, 50, 100, ALL]
#
# Generates:
#   rolling_ic_n20.csv, rolling_ic_n30.csv, ..., rolling_ic_all.csv
#   rolling_ens_tw35_hyb30_g35_n?.csv  (blend with best ensemble)
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
WINDOWS    = [20, 30, 50, 100, None]  # None = all training days

print("=" * 65)
print("ROLLING IC GRINOLD — Adaptive N-Day Window Sweep")
print("=" * 65)

# ── Load data ──────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
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
print(f"  Gold top-10: {top10}")
print(f"  Long-run mean IC: {ic_longrun.round(4)}")

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

def spearman_ic(y, f):
    """Spearman IC between feature f and target y. Returns scalar."""
    if len(y) < 5:
        return np.nan
    r, _ = spearmanr(f, y)
    return r

# ── Step 1: Compute per-day IC for each top-10 feature ────────────
print("\nComputing per-day ICs on training data...")
train_day_list = sorted(train['day_id'].unique())   # chronological order

# per_day_ics[d][f] = Spearman IC of feature f on day d
per_day_ics = []   # list of (day_id, np.array shape (10,))
for day in train_day_list:
    grp = train[train['day_id'] == day]
    if len(grp) < 10:
        continue
    y_d = y_train[grp.index]
    X_d = grp[top10].fillna(0).values.astype(np.float64)
    ics = np.array([spearman_ic(y_d, X_d[:, j]) for j in range(len(top10))])
    per_day_ics.append((day, ics))

print(f"  Days with ICs computed: {len(per_day_ics)}")
ic_days  = [d for d, _ in per_day_ics]
ic_array = np.array([ic for _, ic in per_day_ics])  # shape (n_days, 10)

# Summary stats on raw per-day ICs
nan_frac = np.isnan(ic_array).mean()
ic_array_clean = np.where(np.isnan(ic_array), 0.0, ic_array)
print(f"  IC array shape: {ic_array.shape}  NaN frac: {nan_frac:.3f}")
print(f"  Long-run mean IC (from scratch):  {ic_array_clean.mean(0).round(4)}")
print(f"  Long-run mean IC (from ICIR file): {ic_longrun.round(4)}")

# ── Step 2: For each test day, compute rolling IC ──────────────────
test_day_order = {}  # day_id → index in ic_days for "last day before this test day"
# For each test day, find all training days (by index in ic_days)
# Rolling window = last N training days relative to test day position

# Build sorted training day index
train_day_set = set(ic_days)

print("\nBuilding rolling IC for each test day...")
# For each test day, we need to know which training days are "prior"
# Since train/test overlap in time, and we know SO3_T is temporal,
# we use: rolling window = N most recent training days overall
# (not filtered by date — we just use the last N rows in ic_days)
# This is safe: train ICs are computed on labeled liquid assets, no leakage.

# ── Step 3: Generate predictions for each window size ─────────────
train_day_idx = {d: i for i, d in enumerate(ic_days)}

all_preds = {}  # window → np.array(n_test,)

for window in WINDOWS:
    wname = f"n{window}" if window else "all"
    print(f"\n  Window={wname}...")

    preds = np.zeros(n_test)

    for day, grp_te in test.groupby('day_id'):
        te_idx   = grp_te.index.values
        X_te_raw = grp_te[top10].fillna(0).values.astype(np.float64)

        # Determine rolling IC for this day
        if day in train_day_idx:
            # Overlap day: this day exists in ic_days
            # Use N days *before* this day (exclude same day to avoid leakage)
            d_pos = train_day_idx[day]
            if window is None:
                ic_slice = ic_array_clean[:d_pos]  # all prior days
            else:
                ic_slice = ic_array_clean[max(0, d_pos - window):d_pos]
        else:
            # Future day: use last N training days
            if window is None:
                ic_slice = ic_array_clean
            else:
                ic_slice = ic_array_clean[-window:]

        if len(ic_slice) == 0:
            # No prior data: fall back to long-run IC
            ic_use = ic_longrun
        else:
            ic_use = ic_slice.mean(0)
            # Preserve sign from long-run IC (rolling IC might flip on small samples)
            # Only use rolling IC if it agrees in sign with long-run IC
            sign_match = np.sign(ic_use) == np.sign(ic_longrun)
            ic_use = np.where(sign_match, ic_use, ic_longrun)

        # Z-score features on test day population
        if day in train_day_set:
            grp_tr = train[train['day_id'] == day]
            X_tr_raw = grp_tr[top10].fillna(0).values.astype(np.float64)
            _, m, s = zscore_fit(X_tr_raw)
            X_te_z = zscore_apply(X_te_raw, m, s)
        else:
            X_te_z, _, _ = zscore_fit(X_te_raw)

        pred = X_te_z @ ic_use
        pred -= pred.mean()
        preds[te_idx] = pred

    all_preds[wname] = preds
    ps = auto_scale(preds)
    # Compare to long-run Grinold
    print(f"    std={ps.std():.6f}")

# ── Step 4: Load best ensemble for comparison ──────────────────────
best_ens_raw = (sample_sub
                .merge(pd.read_csv(BEST_ENS)[['ID','TARGET']]
                       .rename(columns={'TARGET':'b'}), on='ID', how='left')
                .fillna(0.0)['b'].values)
best_s = auto_scale(best_ens_raw)

# Also compute static Grinold (long-run IC) for reference
print("\nComputing static Grinold (long-run IC) for reference...")
grinold_preds = np.zeros(n_test)
for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index.values
    X_te_raw = grp_te[top10].fillna(0).values.astype(np.float64)
    if day in train_day_set:
        grp_tr = train[train['day_id'] == day]
        X_tr_raw = grp_tr[top10].fillna(0).values.astype(np.float64)
        _, m, s = zscore_fit(X_tr_raw)
        X_te_z = zscore_apply(X_te_raw, m, s)
    else:
        X_te_z, _, _ = zscore_fit(X_te_raw)
    pred = X_te_z @ ic_longrun
    pred -= pred.mean()
    grinold_preds[te_idx] = pred
grinold_s = auto_scale(grinold_preds)

# ── Step 5: Save and report ────────────────────────────────────────
print(f"\n{'Window':<8}  {'corr_vs_grinold':>16}  {'corr_vs_ens':>12}  File")
print("-" * 65)

for wname, preds in all_preds.items():
    ps       = auto_scale(preds)
    c_grin   = pearson_r(ps, grinold_s)
    c_ens    = pearson_r(ps, best_s)
    fname    = f'rolling_ic_{wname}'
    sub      = pd.DataFrame({'ID': test_ids, 'TARGET': ps})
    sub      = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{fname}.csv'), index=False)
    print(f"  {wname:<8}  {c_grin:+.4f}           {c_ens:+.4f}       {fname}.csv")

# ── Step 6: Build blends with best ensemble ────────────────────────
print("\nBuilding blends (rolling_ic + best_ens)...")
for wname, preds in all_preds.items():
    rs = auto_scale(preds)
    for w in [0.20, 0.30, 0.40]:
        blend   = (1 - w) * best_s + w * rs
        blend_s = auto_scale(blend)
        c       = pearson_r(blend_s, best_s)
        name    = f'ric_{wname}_w{int(w*100)}_ens{int((1-w)*100)}'
        sub     = pd.DataFrame({'ID': test_ids, 'TARGET': blend_s})
        sub     = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
        sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
        print(f"  {name:<45}  corr_vs_best={c:+.4f}")

print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
print("""
── INTERPRETATION ─────────────────────────────────────────────
Rolling IC works if recent IC signal has improved predictive power.
corr_vs_grinold:
  > 0.99: rolling IC is nearly identical to fixed IC (regime is stable)
  0.90-0.99: some adaptation occurring
  < 0.90: significant regime shift in feature importance

KEY CHECK: Is rolling IC meaningfully different from long-run IC?
If corr_vs_grinold > 0.99 for all windows: feature ICs are stable
  → rolling IC won't help, skip
If some windows show corr < 0.98: submit those for LB test
""")
