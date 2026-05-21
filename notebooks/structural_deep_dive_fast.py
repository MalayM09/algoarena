# ================================================================
# STRUCTURAL DEEP DIVE (FAST VERSION) — Vectorized
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
t0 = time.time()

print("=" * 70)
print("STRUCTURAL DEEP DIVE (FAST) — Price Identity + Reco Lags + Momentum")
print("=" * 70)

print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)

train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days = set(train['day_id'].unique())
overlap    = sorted(train_days & set(test['day_id'].unique()))
new_days   = set(test['day_id'].unique()) - train_days

y_train  = train['TARGET'].values.astype(np.float64)
all_cols = set(train.columns) - {'ID', 'TARGET', 'day_id'}

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

print(f"Overlap: {len(overlap)} days  Future: {len(new_days)} days")


# ================================================================
# HYPOTHESIS 1: PRICE FINGERPRINTING (VECTORIZED)
# ================================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 1: PRICE FINGERPRINTING (vectorized merge)")
print("=" * 70)

price_feat  = 'Price'        if 'Price'        in all_cols else None
price_lag1  = 'Price_LagT1'  if 'Price_LagT1'  in all_cols else None
price_lag2  = 'Price_LagT2'  if 'Price_LagT2'  in all_cols else None

print(f"  Price: {price_feat}  Price_LagT1: {price_lag1}  Price_LagT2: {price_lag2}")

if price_feat and price_lag1:
    # Strategy: round Price and Price_LagT1 to n decimal places,
    # then merge liquid (train) and illiquid (test) on those rounded keys.
    # This is O(n log n) via hash join instead of O(n^2) per day.

    # Use all overlap days
    n_days_use = len(overlap)
    print(f"  Running over ALL {n_days_use} overlap days...")

    # Precompute absolute tolerance buckets (round to sig figs)
    # Try: exact (0 decimals tolerance), 2dp, 4dp, 6dp
    round_decimals = [8, 6, 4, 2]  # 8dp = near-exact, 2dp = loose

    results = {}
    for dp in round_decimals:
        total_illiq = 0
        total_match = 0
        all_ics     = []

        for day in overlap:
            tr = train[train['day_id'] == day].copy()
            te = test[test['day_id']   == day].copy()
            if len(tr) < 5 or len(te) < 5:
                continue

            y_liq = y_train[tr.index]
            tr = tr.reset_index(drop=True)
            tr['_y'] = y_liq

            # Round keys
            tr['_pk']  = tr[price_feat].round(dp)
            tr['_p1k'] = tr[price_lag1].round(dp)
            te['_pk']  = te[price_feat].round(dp)
            te['_p1k'] = te[price_lag1].round(dp)

            # Merge: for each illiquid asset find matching liquid assets
            merged = te[['_pk','_p1k']].merge(
                tr[['_pk','_p1k','_y']],
                on=['_pk','_p1k'], how='left'
            )
            # Average over multiple liquid matches
            pred = merged.groupby(merged.index)['_y'].mean().values

            matched_mask = ~np.isnan(pred)
            total_illiq += len(te)
            total_match += matched_mask.sum()

            if matched_mask.sum() >= 5:
                y_ill = y_train[te.index[matched_mask]]
                ic = per_day_ic(y_ill, pred[matched_mask])
                all_ics.append(ic)

        rate   = total_match / max(total_illiq, 1) * 100
        med_ic = np.nanmedian(all_ics) if all_ics else np.nan
        n_days_with_ic = len(all_ics)
        results[dp] = {'rate': rate, 'med_ic': med_ic, 'n_match': total_match,
                       'n_illiq': total_illiq, 'n_days': n_days_with_ic}

        elapsed = (time.time() - t0) / 60
        print(f"  Round {dp:>2}dp: match={rate:6.2f}%  ({total_match:>7}/{total_illiq})  "
              f"Med IC={med_ic:+.5f}  days_with_match={n_days_with_ic}  [{elapsed:.1f}m]")

    # Best result
    best_dp  = max(round_decimals, key=lambda d: results[d]['rate'])
    best_res = results[best_dp]
    print(f"\n  INTERPRETATION:")
    if best_res['rate'] > 10:
        print(f"  *** HIGH MATCH RATE ({best_res['rate']:.1f}%) — strong identity signal ***")
        print(f"  *** IC on matched pairs: {best_res['med_ic']:+.5f} ***")
        if best_res['med_ic'] > 0.2:
            print(f"  *** GAME CHANGER: direct return copy viable! ***")
        elif best_res['med_ic'] > 0.05:
            print(f"  *** USEFUL: matched pairs have meaningful IC ***")
        else:
            print(f"  *** Price collisions are coincidental — no identity signal ***")
    elif best_res['rate'] > 1:
        print(f"  MODERATE match rate ({best_res['rate']:.1f}%) — partial identity possible")
    else:
        print(f"  LOW match rate ({best_res['rate']:.1f}%) — assets likely different instruments")
        print(f"  Price does not uniquely identify assets across liquidity split")

    # Also test with Price_LagT2 as third key
    if price_lag2:
        print(f"\n  3-key fingerprint (Price + Price_LagT1 + Price_LagT2) at 6dp:")
        dp = 6
        total_illiq, total_match = 0, 0
        all_ics3 = []
        for day in overlap:
            tr = train[train['day_id'] == day].copy()
            te = test[test['day_id']   == day].copy()
            if len(tr) < 5 or len(te) < 5: continue
            y_liq = y_train[tr.index]
            tr = tr.reset_index(drop=True)
            tr['_y'] = y_liq
            for df in [tr, te]:
                df['_pk']  = df[price_feat].round(dp)
                df['_p1k'] = df[price_lag1].round(dp)
                df['_p2k'] = df[price_lag2].round(dp)
            merged = te[['_pk','_p1k','_p2k']].merge(
                tr[['_pk','_p1k','_p2k','_y']], on=['_pk','_p1k','_p2k'], how='left')
            pred = merged.groupby(merged.index)['_y'].mean().values
            matched_mask = ~np.isnan(pred)
            total_illiq += len(te)
            total_match += matched_mask.sum()
            if matched_mask.sum() >= 5:
                all_ics3.append(per_day_ic(y_train[te.index[matched_mask]],
                                           pred[matched_mask]))
        rate3   = total_match / max(total_illiq, 1) * 100
        med_ic3 = np.nanmedian(all_ics3) if all_ics3 else np.nan
        print(f"  3-key: match={rate3:.2f}%  ({total_match}/{total_illiq})  "
              f"Med IC={med_ic3:+.5f}")

elapsed = (time.time() - t0) / 60
print(f"\n  [Hypothesis 1 complete — {elapsed:.1f}m]")


# ================================================================
# HYPOTHESIS 2: RECONSTRUCTED RAW LAGS
# ================================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 2: RECONSTRUCTED RAW LAGS — BASE - LagT1 = feature[t-T1]")
print("=" * 70)

lag1_cols = [c for c in all_cols if c.endswith('_LagT1')]
base_of_lag1 = {}
for lc in lag1_cols:
    base = lc.replace('_LagT1', '')
    if base in all_cols and base in train.columns:
        base_of_lag1[lc] = base

print(f"  Base-LagT1 pairs found: {len(base_of_lag1)}")

# Compute IC of reconstructed lag vs LagT1 directly
# Use all training data, group by day
reco_records = []
for lag1_col, base_col in list(base_of_lag1.items()):
    day_ics_lag1 = []
    day_ics_reco = []
    day_ics_base = []

    for day, grp in train.groupby('day_id'):
        if len(grp) < 10: continue
        y_day = y_train[grp.index]
        n = len(grp)

        l1   = grp[lag1_col].fillna(0).values.astype(np.float64)
        base = grp[base_col].fillna(0).values.astype(np.float64)
        reco = base - l1  # = feature[t-T1]

        for sig, store in [(l1, day_ics_lag1), (reco, day_ics_reco), (base, day_ics_base)]:
            s = sig.std()
            if s < 1e-10: store.append(np.nan); continue
            z = (sig - sig.mean()) / s
            store.append(per_day_ic(y_day, z))

    def icir_of(ics):
        arr = np.array([x for x in ics if not np.isnan(x)])
        if len(arr) == 0: return 0.0, 0.0, 0.0
        return abs(arr.mean() / (arr.std() + 1e-10)), np.nanmedian(arr), (arr > 0).mean()

    l1_icir,   l1_med,   l1_ppos   = icir_of(day_ics_lag1)
    reco_icir, reco_med, reco_ppos = icir_of(day_ics_reco)
    base_icir, base_med, base_ppos = icir_of(day_ics_base)

    reco_records.append({
        'lag1_col': lag1_col, 'base_col': base_col,
        'lag1_icir': l1_icir,   'lag1_med': l1_med,
        'reco_icir': reco_icir, 'reco_med': reco_med, 'reco_ppos': reco_ppos,
        'base_icir': base_icir, 'base_med': base_med,
        'ratio': reco_icir / (l1_icir + 1e-6),
    })

reco_df = pd.DataFrame(reco_records).sort_values('reco_icir', ascending=False)

print(f"\n  Top 20 reconstructed raw lags by ICIR:")
print(f"\n  {'LagT1 col':<45}  {'Reco ICIR':>10}  {'LagT1 ICIR':>11}  {'Base ICIR':>10}  {'Ratio':>7}")
print(f"  {'-' * 90}")
for _, row in reco_df.head(20).iterrows():
    marker = ' *** NEW GOLD' if row['reco_icir'] > 3.0 else ''
    print(f"  {row['lag1_col']:<45}  {row['reco_icir']:>10.3f}  "
          f"{row['lag1_icir']:>11.3f}  {row['base_icir']:>10.3f}  "
          f"{row['ratio']:>7.3f}{marker}")

print(f"\n  Summary:")
print(f"  Reco ICIR > 3.0: {(reco_df['reco_icir'] > 3.0).sum():<4}  (gold threshold)")
print(f"  Reco ICIR > 1.0: {(reco_df['reco_icir'] > 1.0).sum():<4}")
print(f"  Reco ICIR > 0.5: {(reco_df['reco_icir'] > 0.5).sum():<4}")
print(f"  Mean reco ICIR:  {reco_df['reco_icir'].mean():.4f}")
print(f"  Mean LagT1 ICIR: {reco_df['lag1_icir'].mean():.4f}")
print(f"  Mean ratio (reco/lag1): {reco_df['ratio'].mean():.3f}")
print(f"  Max ratio: {reco_df['ratio'].max():.3f}  "
      f"(for {reco_df.loc[reco_df['ratio'].idxmax(), 'lag1_col']})")

elapsed = (time.time() - t0) / 60
print(f"\n  [Hypothesis 2 complete — {elapsed:.1f}m]")


# ================================================================
# HYPOTHESIS 3: MULTI-LAG MOMENTUM RATIOS
# ================================================================
print("\n" + "=" * 70)
print("HYPOTHESIS 3: MULTI-LAG MOMENTUM RATIOS")
print("=" * 70)

lag2_cols = [c for c in all_cols if c.endswith('_LagT2')]
lag2_of_base = {lc.replace('_LagT2', ''): lc for lc in lag2_cols
                if lc.replace('_LagT2', '') in all_cols}
lag1_of_base = {v: k for k, v in base_of_lag1.items()}

# Bases that have both LagT1 and LagT2
paired_bases = [b for b in lag1_of_base if b in lag2_of_base
                and lag1_of_base[b] in train.columns
                and lag2_of_base[b] in train.columns]

print(f"  Features with both LagT1 and LagT2: {len(paired_bases)}")

ratio_records = []
for base_col in paired_bases:
    lag1_col = lag1_of_base[base_col]
    lag2_col = lag2_of_base[base_col]

    day_store = {k: [] for k in ['lag1', 'older', 'accel', 'sign', 'ratio']}

    for day, grp in train.groupby('day_id'):
        if len(grp) < 10: continue
        y_day = y_train[grp.index]
        l1 = grp[lag1_col].fillna(0).values.astype(np.float64)
        l2 = grp[lag2_col].fillna(0).values.astype(np.float64)

        signals = {
            'lag1':  l1,
            'older': l2 - l1,                                    # change between T1 and T2
            'accel': 2 * l1 - l2,                                # acceleration: LagT1 - (LagT2-LagT1)
            'sign':  np.sign(l1) * np.sign(l2),                  # direction agreement
            'ratio': l1 / np.where(np.abs(l2) < 1e-10, 1e-10, l2),  # ratio
        }
        for k, sig in signals.items():
            s = sig.std()
            if s < 1e-10: day_store[k].append(np.nan); continue
            z = (sig - sig.mean()) / s
            day_store[k].append(per_day_ic(y_day, z))

    def get_icir(ics):
        arr = np.array([x for x in ics if not np.isnan(x)])
        if len(arr) == 0: return 0.0
        return abs(arr.mean() / (arr.std() + 1e-10))

    ratio_records.append({
        'base': base_col,
        'lag1_icir':  get_icir(day_store['lag1']),
        'older_icir': get_icir(day_store['older']),
        'accel_icir': get_icir(day_store['accel']),
        'sign_icir':  get_icir(day_store['sign']),
        'ratio_icir': get_icir(day_store['ratio']),
    })

ratio_df = pd.DataFrame(ratio_records)
# Best signal per row
ratio_df['best_derived'] = ratio_df[['older_icir','accel_icir','sign_icir','ratio_icir']].max(axis=1)
ratio_df['best_name']    = ratio_df[['older_icir','accel_icir','sign_icir','ratio_icir']].idxmax(axis=1)
ratio_df = ratio_df.sort_values('best_derived', ascending=False)

print(f"\n  Top 15 multi-lag signal combinations:")
print(f"\n  {'Base':<40}  {'LagT1':>7}  {'Older':>7}  {'Accel':>7}  {'Sign':>6}  {'Ratio':>7}  {'Best derived':>13}")
print(f"  {'-' * 100}")
for _, row in ratio_df.head(15).iterrows():
    print(f"  {row['base']:<40}  {row['lag1_icir']:>7.3f}  "
          f"{row['older_icir']:>7.3f}  {row['accel_icir']:>7.3f}  "
          f"{row['sign_icir']:>6.3f}  {row['ratio_icir']:>7.3f}  "
          f"{row['best_derived']:>8.3f} ({row['best_name'][:5]})")

print(f"\n  Mean ICIR across all {len(ratio_df)} features:")
for col, label in [('lag1_icir','LagT1 (baseline)'), ('older_icir','Older momentum'),
                   ('accel_icir','Acceleration'), ('sign_icir','Sign agree'),
                   ('ratio_icir','Ratio')]:
    print(f"  {label:<22}: mean={ratio_df[col].mean():.4f}  "
          f"max={ratio_df[col].max():.4f}  "
          f">LagT1: {(ratio_df[col] > ratio_df['lag1_icir']).sum()}/{len(ratio_df)}")

elapsed = (time.time() - t0) / 60
print(f"\n  [Hypothesis 3 complete — {elapsed:.1f}m]")


# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY & VERDICT")
print("=" * 70)
print(f"\n  Total elapsed: {elapsed:.1f} min")
print("""
  Three questions answered:

  1. PRICE FINGERPRINTING:
     → See match rates above. If match rate > 10% AND IC > 0.10:
       implement direct return copy for matched pairs → big LB jump.
     → If match rate near 0: assets ARE different instruments.

  2. RECONSTRUCTED RAW LAGS (BASE - LagT1):
     → If reco_icir > 3.0 for any feature: new gold features found.
     → If reco_icir << lag1_icir: differences (velocity) carry more
       signal than levels (position). LagT1 format is optimal.

  3. MULTI-LAG MOMENTUM:
     → If any derived ICIR > lag1_icir: new signal found.
     → If all derived < lag1_icir: current feature set is near-optimal
       for momentum decomposition. No new signal in combinations.
""")
