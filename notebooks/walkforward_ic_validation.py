# ================================================================
# WALK-FORWARD IC VALIDATION — True Out-of-Sample Signal Test
# ================================================================
# Tests the Grinold formula with NO look-ahead contamination.
#
# Protocol:
#   For each day T (starting after burn-in of MIN_DAYS):
#     1. Compute IC_j = mean(daily_IC[j, 0:T]) over all features
#        (expanding window — uses only history before day T)
#     2. Predict Day T labeled assets:
#        pred_i = sum_j( IC_j × z_score(feature_j, day_T) )
#     3. Measure Rank IC and R² against actual TARGET for day T
#     4. Step to T+1
#
# Three variants compared:
#   top3  — 3 gold features
#   top10 — 10 gold features (best pseudo-illiquid IC = +0.059)
#   all51 — 51 gold features
#
# Also computes:
#   oracle   — uses ALL 428-day IC (look-ahead contaminated — upper bound)
#   rolling  — IC from last ROLL_DAYS days only (regime-adaptive)
#
# Key question: does walk-forward IC approach the oracle signal?
# If yes → the IC estimate stabilises quickly and our Grinold
#   submissions are using a good IC estimate.
# If no → the oracle IC is inflating our confidence and the real
#   signal may be weaker than we think.
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/eda/summaries')
t0 = time.time()

MIN_DAYS  = 30   # burn-in: need this many days before first prediction
ROLL_DAYS = 50   # rolling window size

print("=" * 65)
print("WALK-FORWARD IC VALIDATION")
print("=" * 65)

# ── Feature selection ─────────────────────────────────────────────
icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
all51     = gold_df['feature'].tolist()

print(f"Gold features: {len(all51)}")

# ── Load training data ────────────────────────────────────────────
print("\nLoading training data...")
train = pd.read_parquet(TRAIN_PATH)
train = train.reset_index(drop=True)

all_cols = set(train.columns) - {'ID', 'TARGET'}
all51    = [f for f in all51 if f in all_cols]

train['day_id']  = train['SO3_T'].round(5).astype(str)
day_order        = train.groupby('day_id')['SO3_T'].mean().sort_values().index.tolist()
n_days           = len(day_order)
y_all            = train['TARGET'].values.astype(np.float64)

print(f"Gold features in dataset : {len(all51)}")
print(f"Training days            : {n_days}")
print(f"Burn-in                  : {MIN_DAYS} days")
print(f"Walk-forward days        : {n_days - MIN_DAYS}")

# Feature subsets
subsets = {
    'top3' : all51[:3],
    'top10': all51[:10],
    'all51': all51,
}

# ================================================================
# STEP 1: COMPUTE PER-DAY IC FOR ALL FEATURES
# ================================================================
# We need the daily IC for each feature across all training days
# to build the expanding/rolling IC estimates.
# ================================================================
print("\n" + "="*65)
print("STEP 1: Computing per-day IC for all gold features...")
print("="*65)

n_feats    = len(all51)
daily_ic   = np.full((n_days, n_feats), np.nan)   # shape (428, 51)
day_sizes  = []

for d_idx, day in enumerate(day_order):
    grp = train[train['day_id'] == day]
    n   = len(grp)
    day_sizes.append(n)

    if n < 10:
        continue

    y_day = y_all[grp.index]
    X_day = grp[all51].fillna(0).values.astype(np.float64)

    for f_idx in range(n_feats):
        r, _ = spearmanr(X_day[:, f_idx], y_day)
        daily_ic[d_idx, f_idx] = r

day_sizes = np.array(day_sizes)
print(f"  Per-day IC matrix computed: shape={daily_ic.shape}")
print(f"  Mean IC per feature (full window): min={np.nanmean(daily_ic, 0).min():+.5f}  "
      f"max={np.nanmean(daily_ic, 0).max():+.5f}")
print(f"  Average daily N per day: {day_sizes.mean():.0f}")

# Oracle IC (uses all 428 days — contaminated, upper bound)
oracle_ic = np.nanmean(daily_ic, axis=0)   # shape (n_feats,)
print(f"\n  Oracle IC by subset (look-ahead, upper bound):")
for name, feats in subsets.items():
    fidx = [all51.index(f) for f in feats]
    ics  = oracle_ic[fidx]
    print(f"    {name:<8}: mean|IC|={np.abs(ics).mean():.5f}  "
          f"sum|IC|={np.abs(ics).sum():.5f}")


# ================================================================
# STEP 2: WALK-FORWARD PREDICTION + EVALUATION
# ================================================================
print("\n" + "="*65)
print("STEP 2: Walk-forward prediction (expanding + rolling windows)")
print("="*65)

def grinold_predict(X_day, ic_vec):
    """
    X_day : (n_assets, n_feats) — raw feature values
    ic_vec: (n_feats,)          — IC weights
    Returns z-scored predictions, already mean-zero.
    """
    m = X_day.mean(0);  s = X_day.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    Z    = np.clip((X_day - m) / s, -5, 5)
    pred = Z @ ic_vec
    pred -= pred.mean()   # ensure market-neutral
    return pred

results = {}   # keyed by (variant, window_type)

variant_configs = [
    ('top3',  'expanding'),
    ('top3',  'rolling'),
    ('top3',  'oracle'),
    ('top10', 'expanding'),
    ('top10', 'rolling'),
    ('top10', 'oracle'),
    ('all51', 'expanding'),
    ('all51', 'rolling'),
    ('all51', 'oracle'),
]

for subset_name, window_type in variant_configs:
    feats   = subsets[subset_name]
    f_idx   = [all51.index(f) for f in feats]
    tag     = f"{subset_name}_{window_type}"

    day_ics_wf  = []   # per-day Rank IC
    day_r2s_wf  = []   # per-day R²
    day_idxs    = []   # day index for plotting

    for d_idx in range(MIN_DAYS, n_days):
        day = day_order[d_idx]
        grp = train[train['day_id'] == day]
        if len(grp) < 10:
            continue

        y_day = y_all[grp.index]
        X_day = grp[feats].fillna(0).values.astype(np.float64)

        # Determine IC weights from history (no look-ahead)
        if window_type == 'expanding':
            hist_ic = np.nanmean(daily_ic[:d_idx, f_idx], axis=0)  # days 0..T-1
        elif window_type == 'rolling':
            start   = max(0, d_idx - ROLL_DAYS)
            hist_ic = np.nanmean(daily_ic[start:d_idx, f_idx], axis=0)
        elif window_type == 'oracle':
            hist_ic = oracle_ic[f_idx]

        if np.any(np.isnan(hist_ic)):
            continue

        pred = grinold_predict(X_day, hist_ic)

        # Rank IC
        r, _ = spearmanr(y_day, pred)

        # R² (raw, not rank-transformed — will be small/negative due to kurtosis)
        r2 = r2_score(y_day, pred)

        day_ics_wf.append(r)
        day_r2s_wf.append(r2)
        day_idxs.append(d_idx)

    results[tag] = {
        'day_ics': np.array(day_ics_wf),
        'day_r2s': np.array(day_r2s_wf),
        'day_idx': np.array(day_idxs),
    }


# ================================================================
# STEP 3: RESULTS SUMMARY
# ================================================================
print("\n" + "="*65)
print("STEP 3: WALK-FORWARD RESULTS")
print("="*65)

print(f"\n  {'Variant':<22}  {'Med IC':>9}  {'Mean IC':>9}  {'ICIR':>8}  "
      f"{'%pos':>7}  {'Med R²':>10}  {'%pos R²':>9}")
print(f"  {'-'*82}")

for tag in [k for k in results if 'expanding' in k] + \
           [k for k in results if 'rolling'   in k] + \
           [k for k in results if 'oracle'    in k]:
    res  = results[tag]
    ics  = res['day_ics']
    r2s  = res['day_r2s']
    if len(ics) == 0:
        continue

    med_ic   = np.median(ics)
    mean_ic  = np.mean(ics)
    std_ic   = np.std(ics)
    icir_wf  = mean_ic / std_ic if std_ic > 0 else 0
    ppos_ic  = (ics > 0).mean() * 100
    med_r2   = np.median(r2s)
    ppos_r2  = (r2s > 0).mean() * 100

    marker = ' ← ORACLE (look-ahead)' if 'oracle' in tag else ''
    print(f"  {tag:<22}  {med_ic:+9.5f}  {mean_ic:+9.5f}  {icir_wf:+8.3f}  "
          f"{ppos_ic:6.1f}%  {med_r2:+10.6f}  {ppos_r2:8.1f}%{marker}")

# ================================================================
# STEP 4: CONVERGENCE ANALYSIS
# ================================================================
# How quickly does walk-forward IC converge to oracle IC?
# Plot IC in 3 phases: early (days 30-100), mid (100-250), late (250-428)
# ================================================================
print("\n" + "="*65)
print("STEP 4: CONVERGENCE — Does Walk-Forward IC Stabilise?")
print("="*65)

for subset_name in ['top3', 'top10', 'all51']:
    tag_exp    = f"{subset_name}_expanding"
    tag_oracle = f"{subset_name}_oracle"

    ics_exp    = results[tag_exp]['day_ics']
    day_idxs   = results[tag_exp]['day_idx']

    # Phase split
    early_mask = day_idxs < 100
    mid_mask   = (day_idxs >= 100) & (day_idxs < 250)
    late_mask  = day_idxs >= 250

    ics_oracle = results[tag_oracle]['day_ics']

    print(f"\n  {subset_name} — expanding window IC by phase:")
    for mask, label in [(early_mask, 'early (30-100)'),
                         (mid_mask,   'mid  (100-250)'),
                         (late_mask,  'late (250-428)')]:
        if mask.sum() == 0:
            continue
        med = np.median(ics_exp[mask])
        ppos = (ics_exp[mask] > 0).mean() * 100
        print(f"    {label}: Med IC={med:+.5f}  %pos={ppos:.1f}%  n={mask.sum()}")

    print(f"    oracle (all 428d) : Med IC={np.median(ics_oracle):+.5f}  "
          f"%pos={(ics_oracle>0).mean()*100:.1f}%")


# ================================================================
# STEP 5: COMPARISON — Walk-Forward vs Pseudo-Illiquid OOF
# ================================================================
print("\n" + "="*65)
print("STEP 5: SIGNAL QUALITY — Walk-Forward vs Previous Benchmarks")
print("="*65)

print(f"""
Known benchmarks (pseudo-illiquid OOF from previous run):
  Grinold all51 pseudo-illiquid  : Med IC = +0.05604   76.5% pos
  Grinold top10 pseudo-illiquid  : Med IC = +0.05939   78.1% pos
  Ridge gold_r  pseudo-illiquid  : Med IC = +0.02913   64.0% pos

These measured performance on ILLIQUID half of training data.
Walk-forward below measures performance on ALL labeled training
assets (LIQUID). Given AUC=0.498, the two should be comparable.

Key questions answered by walk-forward:
  Q1: Does expanding IC converge to oracle? (IC stability)
  Q2: Is the signal consistent early in the history? (regime risk)
  Q3: Does top10 beat top3 and all51 OOS? (feature count validation)
  Q4: Is rolling (adaptive) better than expanding? (regime sensitivity)
""")

# ================================================================
# STEP 6: R² ANALYSIS — Why Is R² Negative?
# ================================================================
print("="*65)
print("STEP 6: R² DEEP DIVE — Scale Analysis")
print("="*65)

# For the best variant (top10 expanding), look at R² distribution
tag = 'top10_expanding'
r2s = results[tag]['day_r2s']
ics = results[tag]['day_ics']

print(f"\n  top10 expanding — R² distribution:")
for pct in [5, 10, 25, 50, 75, 90, 95]:
    print(f"    p{pct:2d} = {np.percentile(r2s, pct):+.6f}")

# Days where IC>0 vs IC<0 — does sign agree with R²?
pos_ic_mask = ics > 0
neg_ic_mask = ics < 0
r2_when_ic_pos = r2s[pos_ic_mask]
r2_when_ic_neg = r2s[neg_ic_mask]

print(f"\n  R² when IC > 0 (correct direction): "
      f"med={np.median(r2_when_ic_pos):+.6f}  "
      f"%pos_R2={(r2_when_ic_pos > 0).mean()*100:.1f}%  n={pos_ic_mask.sum()}")
print(f"  R² when IC < 0 (wrong direction):   "
      f"med={np.median(r2_when_ic_neg):+.6f}  "
      f"%pos_R2={(r2_when_ic_neg > 0).mean()*100:.1f}%  n={neg_ic_mask.sum()}")

# Theoretical R² from IC: R² ≈ IC² for linear predictions under normality
# With fat tails, actual R² < IC² (outliers inflate SS_tot)
print(f"\n  IC → R² gap analysis (top10 expanding):")
print(f"    Mean IC²          (theoretical R²) = {np.mean(ics**2):+.6f}")
print(f"    Mean actual R²    (walk-forward)   = {np.mean(r2s):+.6f}")
print(f"    R² attrition      (IC²/R²-1)       = "
      f"{(np.mean(ics**2)/np.mean(r2s) - 1)*100 if np.mean(r2s) != 0 else float('inf'):.0f}%")
print(f"    Cause: kurtosis={pd.Series(y_all).kurtosis():.1f} inflates SS_tot denominator")

# ── Save per-day results for reference ───────────────────────────
rows = []
for tag, res in results.items():
    subset, wtype = tag.rsplit('_', 1)
    for i, d_idx in enumerate(res['day_idx']):
        rows.append({
            'day_index': d_idx,
            'day_id':    day_order[d_idx],
            'subset':    subset,
            'window':    wtype,
            'rank_ic':   res['day_ics'][i],
            'r2':        res['day_r2s'][i],
        })

df_wf = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, 'walkforward_results.csv')
df_wf.to_csv(out_path, index=False)
print(f"\n  Saved per-day results: {out_path}  ({len(df_wf):,} rows)")

print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
