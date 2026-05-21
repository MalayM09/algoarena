# ================================================================
# STRUCTURAL DEEP DIVE — Three New Hypotheses
# ================================================================
# Motivated by dataset documentation reveal: LagT1 = feature[t] - feature[t-T1]
#
# HYPOTHESIS 1: Price Fingerprinting for Asset Identity
#   If liquid and illiquid assets share identical Price + Price_LagT1,
#   they are the SAME instrument. Direct return copy = near-perfect prediction.
#   This likely explains the 75× gap to 1st place.
#
# HYPOTHESIS 2: Reconstructed Raw Lags as New Features
#   BASE - LagT1 = feature[t-T1] = raw lagged state (NOT in current feature set)
#   Could carry significant IC as a different temporal slice.
#
# HYPOTHESIS 3: Multi-Lag Momentum Ratios
#   LagT1 = recent change (velocity)
#   LagT2 - LagT1 = older change (velocity between T1 and T2)
#   Ratio / sign agreement = momentum acceleration signal
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
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
print("STRUCTURAL DEEP DIVE — Price Identity + Reconstructed Lags + Momentum")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)

train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days = set(train['day_id'].unique())
overlap    = train_days & set(test['day_id'].unique())
new_days   = set(test['day_id'].unique()) - train_days

y_train  = train['TARGET'].values.astype(np.float64)
test_ids = test['ID'].values
all_cols = set(train.columns) - {'ID', 'TARGET', 'day_id'}

print(f"Train: {len(train):,}  Test: {len(test):,}")
print(f"Overlap: {len(overlap)} days  Future: {len(new_days)} days")

# ── Feature inventory ──────────────────────────────────────────────────────
lag1_cols = [c for c in all_cols if c.endswith('_LagT1')]
lag2_cols = [c for c in all_cols if c.endswith('_LagT2')]
lag3_cols = [c for c in all_cols if c.endswith('_LagT3')]

# For each LagT1 col, find its BASE (same name minus _LagT1)
base_of_lag1 = {}
for lc in lag1_cols:
    base = lc.replace('_LagT1', '')
    if base in all_cols:
        base_of_lag1[lc] = base

# Find pairs where LagT1 and LagT2 both exist for same base
lag1_of_base = {v: k for k, v in base_of_lag1.items()}
lag2_of_base = {}
for lc in lag2_cols:
    base = lc.replace('_LagT2', '')
    if base in all_cols:
        lag2_of_base[base] = lc

print(f"\nFeature inventory:")
print(f"  LagT1 features: {len(lag1_cols)}")
print(f"  LagT2 features: {len(lag2_cols)}")
print(f"  LagT3 features: {len(lag3_cols)}")
print(f"  Base features with both LagT1 and LagT2: "
      f"{sum(1 for b in base_of_lag1.values() if b in lag2_of_base)}")

# ── Load IC/ICIR table ─────────────────────────────────────────────────────
icir_df = pd.read_csv(ICIR_PATH)
icir_map = icir_df.set_index('feature')['abs_icir'].to_dict()
ic_map   = icir_df.set_index('feature')['mean_ic'].to_dict()

gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df  = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
top10    = gold_df['feature'].tolist()[:10]
ic_arr   = np.array([ic_map[f] for f in top10])

def per_day_ic(y_true, y_pred):
    if len(y_true) < 5: return np.nan
    return spearmanr(y_true, y_pred)[0]

def zscore_fit(X, clip=5.0):
    m = X.mean(0); s = X.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip), m, s


# ================================================================
# HYPOTHESIS 1: PRICE FINGERPRINTING FOR ASSET IDENTITY
# ================================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 1: PRICE FINGERPRINTING — Can we identify illiquid = liquid?")
print("=" * 70)

price_feat = 'Price' if 'Price' in all_cols else None
price_lag1 = 'Price_LagT1' if 'Price_LagT1' in all_cols else None
price_lag2 = 'Price_LagT2' if 'Price_LagT2' in all_cols else None

print(f"\n  Price feature:      {price_feat}")
print(f"  Price_LagT1:        {price_lag1}")
print(f"  Price_LagT2:        {price_lag2}")

if price_feat and price_lag1:
    # On overlap test days, try to match illiquid test assets to liquid train assets
    # by (Price, Price_LagT1) within various tolerances

    tolerances = [0.0, 1e-6, 1e-4, 1e-3, 0.01]  # relative tolerance
    n_sample_days = 20

    sample_days = sorted(list(overlap))[:n_sample_days]

    match_results = {tol: {'n_illiq': 0, 'n_matched': 0, 'ics': []} for tol in tolerances}

    for day in sample_days:
        tr_day = train[train['day_id'] == day]
        te_day = test[test['day_id']   == day]
        if len(tr_day) < 5 or len(te_day) < 5:
            continue

        y_liq = y_train[tr_day.index]

        p_liq  = tr_day[price_feat].values.astype(np.float64)
        p1_liq = tr_day[price_lag1].values.astype(np.float64)
        p_ill  = te_day[price_feat].values.astype(np.float64)
        p1_ill = te_day[price_lag1].values.astype(np.float64)

        for tol in tolerances:
            preds = np.zeros(len(te_day))
            matched = np.zeros(len(te_day), dtype=bool)

            for i, (pi, p1i) in enumerate(zip(p_ill, p1_ill)):
                if tol == 0.0:
                    mask = (p_liq == pi) & (p1_liq == p1i)
                else:
                    mask = (np.abs(p_liq - pi) / (np.abs(pi) + 1e-10) <= tol) & \
                           (np.abs(p1_liq - p1i) / (np.abs(p1i) + 1e-10) <= tol)

                if mask.sum() > 0:
                    preds[i] = y_liq[mask].mean()
                    matched[i] = True

            match_results[tol]['n_illiq']   += len(te_day)
            match_results[tol]['n_matched'] += matched.sum()

            if matched.sum() >= 5:
                ic = per_day_ic(y_train[te_day.index[matched]],
                                preds[matched])
                match_results[tol]['ics'].append(ic)

    print(f"\n  Results over {n_sample_days} sample overlap days:")
    print(f"  {'Tolerance':>12}  {'Match rate':>12}  {'Matched/Illiq':>14}  {'Med IC (matched)':>18}")
    print(f"  {'-' * 62}")

    for tol in tolerances:
        r = match_results[tol]
        rate = r['n_matched'] / max(r['n_illiq'], 1) * 100
        ics  = [x for x in r['ics'] if not np.isnan(x)]
        med_ic = np.nanmedian(ics) if ics else np.nan
        print(f"  {tol:>12.6f}  {rate:>11.2f}%  {r['n_matched']:>6}/{r['n_illiq']:<6}  "
              f"  {med_ic:+18.5f}")

    # Add Price_LagT2 as third fingerprint column
    if price_lag2:
        print(f"\n  Using 3-column fingerprint (Price + Price_LagT1 + Price_LagT2):")
        tol = 1e-4
        n_illiq, n_matched = 0, 0
        ics3 = []
        for day in sample_days:
            tr_day = train[train['day_id'] == day]
            te_day = test[test['day_id']   == day]
            if len(tr_day) < 5 or len(te_day) < 5: continue

            y_liq  = y_train[tr_day.index]
            p_liq  = tr_day[price_feat].values.astype(np.float64)
            p1_liq = tr_day[price_lag1].values.astype(np.float64)
            p2_liq = tr_day[price_lag2].values.astype(np.float64)
            p_ill  = te_day[price_feat].values.astype(np.float64)
            p1_ill = te_day[price_lag1].values.astype(np.float64)
            p2_ill = te_day[price_lag2].values.astype(np.float64)

            preds   = np.zeros(len(te_day))
            matched = np.zeros(len(te_day), dtype=bool)
            for i in range(len(te_day)):
                mask = ((np.abs(p_liq  - p_ill[i])  / (np.abs(p_ill[i])  + 1e-10) <= tol) &
                        (np.abs(p1_liq - p1_ill[i]) / (np.abs(p1_ill[i]) + 1e-10) <= tol) &
                        (np.abs(p2_liq - p2_ill[i]) / (np.abs(p2_ill[i]) + 1e-10) <= tol))
                if mask.sum() > 0:
                    preds[i]   = y_liq[mask].mean()
                    matched[i] = True

            n_illiq   += len(te_day)
            n_matched += matched.sum()
            if matched.sum() >= 5:
                ics3.append(per_day_ic(y_train[te_day.index[matched]], preds[matched]))

        rate = n_matched / max(n_illiq, 1) * 100
        med  = np.nanmedian(ics3) if ics3 else np.nan
        print(f"  3-col fingerprint (tol={tol}): match rate={rate:.2f}%  "
              f"n={n_matched}/{n_illiq}  Med IC={med:+.5f}")
else:
    print("  Price or Price_LagT1 not found — skipping fingerprinting.")

elapsed = (time.time() - t0) / 60
print(f"\n  [Hypothesis 1 complete — {elapsed:.1f}m]")


# ================================================================
# HYPOTHESIS 2: RECONSTRUCTED RAW LAGS AS NEW FEATURES
# ================================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 2: RECONSTRUCTED RAW LAGS — BASE - LagT1 = feature[t-T1]")
print("=" * 70)
print("  LagT1 = feature[t] - feature[t-T1]")
print("  BASE  = feature[t]")
print("  BASE - LagT1 = feature[t-T1]  ← raw lagged state, NOT currently used")

# Compute IC of reconstructed raw lags across all training days
reco_ic_records = []

for lag1_col, base_col in list(base_of_lag1.items())[:50]:  # test first 50
    reco_col_name = f'{base_col}_raw_lag1'

    day_ics = []
    for day, grp in train.groupby('day_id'):
        if len(grp) < 10: continue
        y_day = y_train[grp.index]

        base_vals = grp[base_col].fillna(0).values.astype(np.float64)
        lag1_vals = grp[lag1_col].fillna(0).values.astype(np.float64)
        reco      = base_vals - lag1_vals  # = feature[t-T1]

        # Normalize within day
        s = reco.std()
        if s < 1e-10: continue
        reco_z = (reco - reco.mean()) / s

        ic = per_day_ic(y_day, reco_z)
        day_ics.append(ic)

    if not day_ics: continue
    arr = np.array([x for x in day_ics if not np.isnan(x)])
    if len(arr) == 0: continue

    med_ic   = np.nanmedian(arr)
    mean_ic  = np.nanmean(arr)
    std_ic   = arr.std()
    icir     = mean_ic / (std_ic + 1e-10)
    ppos     = (arr > 0).mean()
    pneg     = (arr < 0).mean()
    never_flip = ppos == 1.0 or pneg == 1.0

    reco_ic_records.append({
        'reconstructed': reco_col_name,
        'base': base_col,
        'lag1': lag1_col,
        'lag1_icir': icir_map.get(lag1_col, 0),
        'lag1_mean_ic': ic_map.get(lag1_col, 0),
        'reco_med_ic': med_ic,
        'reco_mean_ic': mean_ic,
        'reco_icir': abs(icir),
        'reco_ppos': ppos,
        'reco_pneg': pneg,
        'never_flip': never_flip,
    })

reco_df = pd.DataFrame(reco_ic_records).sort_values('reco_icir', ascending=False)

print(f"\n  Reconstructed raw lag IC analysis ({len(reco_df)} features):")
print(f"\n  {'Reconstructed feature':<45}  {'Reco ICIR':>10}  {'LagT1 ICIR':>11}  {'Never flip':>11}")
print(f"  {'-' * 82}")
for _, row in reco_df.head(15).iterrows():
    print(f"  {row['reconstructed']:<45}  {row['reco_icir']:+10.3f}  "
          f"{row['lag1_icir']:+11.3f}  {str(row['never_flip']):>11}")

print(f"\n  Summary statistics:")
print(f"  Reco ICIR > 3.0:  {(reco_df['reco_icir'] > 3.0).sum()}")
print(f"  Reco ICIR > 1.0:  {(reco_df['reco_icir'] > 1.0).sum()}")
print(f"  Reco ICIR > 0.5:  {(reco_df['reco_icir'] > 0.5).sum()}")
print(f"  Reco never flip:  {reco_df['never_flip'].sum()}")
print(f"  Avg reco ICIR:    {reco_df['reco_icir'].mean():.3f}")
print(f"  Avg LagT1 ICIR:   {reco_df['lag1_icir'].mean():.3f}")
print(f"  Ratio (reco/lag1): {(reco_df['reco_icir'] / (reco_df['lag1_icir'] + 1e-6)).mean():.3f}")

elapsed = (time.time() - t0) / 60
print(f"\n  [Hypothesis 2 complete — {elapsed:.1f}m]")


# ================================================================
# HYPOTHESIS 3: MULTI-LAG MOMENTUM RATIOS
# ================================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 3: MULTI-LAG MOMENTUM RATIOS")
print("=" * 70)
print("  LagT1 = velocity (short-term change)")
print("  LagT2 - LagT1 = velocity between T1 and T2 (older momentum component)")
print("  LagT1 / LagT2 = momentum acceleration ratio")
print("  sign(LagT1) == sign(LagT2) → momentum continuation")
print("  sign(LagT1) != sign(LagT2) → momentum reversal")

# Find features with both LagT1 and LagT2 in training data
bases_with_both = [b for b in base_of_lag1.values() if b in lag2_of_base][:30]

ratio_records = []

for base_col in bases_with_both:
    lag1_col = lag1_of_base.get(base_col, '')
    lag2_col = lag2_of_base.get(base_col, '')
    if not lag1_col or not lag2_col: continue
    if lag1_col not in train.columns or lag2_col not in train.columns: continue

    day_ics_ratio = []
    day_ics_older = []
    day_ics_sign  = []
    day_ics_accel = []

    for day, grp in train.groupby('day_id'):
        if len(grp) < 10: continue
        y_day = y_train[grp.index]

        l1 = grp[lag1_col].fillna(0).values.astype(np.float64)
        l2 = grp[lag2_col].fillna(0).values.astype(np.float64)

        # Older component: feature[t-T1] - feature[t-T2] = LagT2 - LagT1
        older = l2 - l1

        # Ratio: LagT1 / (LagT2 + eps)  — relative acceleration
        safe_l2 = np.where(np.abs(l2) < 1e-10, 1e-10, l2)
        ratio = l1 / safe_l2

        # Sign agreement: 1 if same direction, -1 if opposing
        sign_agree = np.sign(l1) * np.sign(l2)

        # Acceleration: LagT1 - older = LagT1 - (LagT2 - LagT1) = 2*LagT1 - LagT2
        accel = 2 * l1 - l2

        for signal, store in [(ratio, day_ics_ratio),
                               (older, day_ics_older),
                               (sign_agree, day_ics_sign),
                               (accel, day_ics_accel)]:
            s = signal.std()
            if s < 1e-10: continue
            z = (signal - signal.mean()) / s
            ic = per_day_ic(y_day, z)
            store.append(ic)

    def summarize(ics):
        arr = np.array([x for x in ics if not np.isnan(x)])
        if len(arr) == 0: return 0, 0, 0
        med = np.nanmedian(arr)
        icir = arr.mean() / (arr.std() + 1e-10)
        ppos = (arr > 0).mean()
        return med, abs(icir), ppos

    r_med, r_icir, r_ppos = summarize(day_ics_ratio)
    o_med, o_icir, o_ppos = summarize(day_ics_older)
    s_med, s_icir, s_ppos = summarize(day_ics_sign)
    a_med, a_icir, a_ppos = summarize(day_ics_accel)
    l1_icir = icir_map.get(lag1_col, 0)

    ratio_records.append({
        'base': base_col,
        'lag1_icir': l1_icir,
        'ratio_icir': r_icir, 'ratio_med': r_med,
        'older_icir': o_icir, 'older_med': o_med,
        'sign_icir':  s_icir, 'sign_med':  s_med,
        'accel_icir': a_icir, 'accel_med':  a_med,
    })

ratio_df = pd.DataFrame(ratio_records).sort_values('ratio_icir', ascending=False)

print(f"\n  Multi-lag signal comparison (top 15 by ratio ICIR):")
print(f"\n  {'Base':<40}  {'LagT1':>7}  {'Ratio':>7}  {'Older':>7}  {'Sign':>7}  {'Accel':>7}")
print(f"  {'-'*80}")
for _, row in ratio_df.head(15).iterrows():
    print(f"  {row['base']:<40}  {row['lag1_icir']:>7.3f}  "
          f"{row['ratio_icir']:>7.3f}  {row['older_icir']:>7.3f}  "
          f"{row['sign_icir']:>7.3f}  {row['accel_icir']:>7.3f}")

print(f"\n  ICIR comparison across all {len(ratio_df)} features:")
cols = ['lag1_icir', 'ratio_icir', 'older_icir', 'sign_icir', 'accel_icir']
labels = ['LagT1 (baseline)', 'LagT1/LagT2 ratio', 'Older momentum', 'Sign agreement', 'Acceleration']
for col, label in zip(cols, labels):
    print(f"  {label:<25}: mean={ratio_df[col].mean():.3f}  "
          f"max={ratio_df[col].max():.3f}  "
          f">1.0: {(ratio_df[col] > 1.0).sum()}/{len(ratio_df)}")

elapsed = (time.time() - t0) / 60
print(f"\n  [Hypothesis 3 complete — {elapsed:.1f}m]")


# ================================================================
# SUMMARY
# ================================================================
print("\n" + "=" * 70)
print("COMPLETE FINDINGS SUMMARY")
print("=" * 70)

print(f"""
  HYPOTHESIS 1 — PRICE FINGERPRINTING:
  → Results above show match rates and IC for matched pairs.
  → If match rate > 5% AND IC(matched) > +0.2: GAME CHANGER.
    Direct identity = direct return copy = very high IC.
  → If match rate near 0: assets don't share Price (different instruments).
  → If match rate high but IC low: Price collisions are coincidental.

  HYPOTHESIS 2 — RECONSTRUCTED RAW LAGS:
  → BASE - LagT1 = feature[t-T1] = raw order book state at prior time.
  → If reco_icir > 3.0: these are new gold features.
  → If reco_icir < lag1_icir: LagT1 differences carry more signal than levels.
  → Key question: ratio reco_icir / lag1_icir > 0.5 means worth including.

  HYPOTHESIS 3 — MULTI-LAG MOMENTUM:
  → LagT1/LagT2 ratio captures momentum acceleration.
  → Older momentum captures trend between T1 and T2 windows.
  → If any signal ICIR > lag1_icir: new signal found.
  → Sign agreement feature is purely binary but might be useful.

  Total elapsed: {(time.time()-t0)/60:.1f} min
""")
