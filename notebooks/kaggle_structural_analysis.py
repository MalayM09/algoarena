# ================================================================
# KAGGLE STRUCTURAL DEEP DIVE
# Run this as a Kaggle Notebook (Script) or in a Kaggle session.
#
# SETUP INSTRUCTIONS:
#   1. Create a new Kaggle Notebook → paste this entire file
#   2. Set COMPETITION_SLUG below to your competition's dataset name
#      (the folder name you see under /kaggle/input/)
#   3. Run All → outputs are saved to /kaggle/working/
#   4. Copy-paste the printed output back here
#
# OUTPUT FILES (in /kaggle/working/):
#   - h1_price_fingerprint.csv   → match rates + IC per tolerance
#   - h2_reconstructed_lags.csv  → ICIR of BASE-LagT1 features
#   - h3_momentum_ratios.csv     → ICIR of multi-lag combinations
#   - summary.txt                → human-readable findings
# ================================================================

# ── CONFIGURE THIS ─────────────────────────────────────────────
COMPETITION_SLUG = "YOUR_COMPETITION_NAME_HERE"
# e.g. "jane-street-real-time-market-data-forecasting"
# Check /kaggle/input/ to see the exact folder name
# ───────────────────────────────────────────────────────────────

import os, time, warnings, gc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

# Paths
KAGGLE_INPUT = f'/kaggle/input/{COMPETITION_SLUG}'
OUT_DIR      = '/kaggle/working'

# Auto-detect parquet files
def find_file(folder, pattern):
    for f in os.listdir(folder):
        if pattern in f and f.endswith('.parquet'):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"No file matching '{pattern}' in {folder}")

TRAIN_PATH = find_file(KAGGLE_INPUT, 'train')
TEST_PATH  = find_file(KAGGLE_INPUT, 'test')

print("=" * 70)
print("KAGGLE STRUCTURAL DEEP DIVE")
print(f"Train: {TRAIN_PATH}")
print(f"Test:  {TEST_PATH}")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────
print("\n[1/6] Loading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)

# Convert to float32 to save memory
feat_cols = [c for c in train.columns if c not in ('ID', 'TARGET')]
for df in [train, test]:
    for c in feat_cols:
        if c in df.columns and df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)

# Day IDs
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days = set(train['day_id'].unique())
test_days  = set(test['day_id'].unique())
overlap    = sorted(train_days & test_days)
new_days   = test_days - train_days

y_train = train['TARGET'].values.astype(np.float64)
all_cols = set(train.columns) - {'ID', 'TARGET', 'day_id', 'SO3_T'}

print(f"  Train:   {len(train):,} rows | {train.shape[1]} cols")
print(f"  Test:    {len(test):,} rows  | {test.shape[1]} cols")
print(f"  Overlap: {len(overlap)} days | Future: {len(new_days)} days")
print(f"  Memory:  train={train.memory_usage(deep=True).sum()/1e9:.2f}GB  "
      f"test={test.memory_usage(deep=True).sum()/1e9:.2f}GB")

# ── Feature catalog ────────────────────────────────────────────
print("\n[2/6] Building feature catalog...")
lag1_cols = [c for c in all_cols if c.endswith('_LagT1') and c in train.columns]
lag2_cols = [c for c in all_cols if c.endswith('_LagT2') and c in train.columns]
lag3_cols = [c for c in all_cols if c.endswith('_LagT3') and c in train.columns]
base_cols  = [c for c in all_cols if not any(c.endswith(s) for s in ('_LagT1','_LagT2','_LagT3'))
              and c not in ('Price', 'SO3_T') and c in train.columns]

# Pairs: base ↔ LagT1
base_of_lag1 = {}
for lc in lag1_cols:
    base = lc.replace('_LagT1', '')
    if base in train.columns:
        base_of_lag1[lc] = base

lag2_of_base = {}
for lc in lag2_cols:
    base = lc.replace('_LagT2', '')
    if base in train.columns:
        lag2_of_base[base] = lc

lag1_of_base = {v: k for k, v in base_of_lag1.items()}

print(f"  LagT1: {len(lag1_cols)}  LagT2: {len(lag2_cols)}  LagT3: {len(lag3_cols)}")
print(f"  Base cols: {len(base_cols)}")
print(f"  Base-LagT1 pairs: {len(base_of_lag1)}")
print(f"  Bases with LagT1+LagT2: {sum(1 for b in lag1_of_base if b in lag2_of_base)}")

# Price features
price_feat = 'Price'       if 'Price'       in train.columns else None
price_lag1 = 'Price_LagT1' if 'Price_LagT1' in train.columns else None
price_lag2 = 'Price_LagT2' if 'Price_LagT2' in train.columns else None
print(f"  Price features: {price_feat}, {price_lag1}, {price_lag2}")

# ── Compute per-feature ICIR on training data ──────────────────
print("\n[3/6] Computing per-feature ICIR on ALL training days (this takes ~5-8 min)...")

def batch_icir(feature_list, max_feats=None):
    """Compute ICIR for a list of features. Memory-efficient, day-by-day."""
    if max_feats:
        feature_list = feature_list[:max_feats]
    records = {}
    for f in feature_list:
        records[f] = []

    for day, grp in train.groupby('day_id'):
        if len(grp) < 10: continue
        y_day = y_train[grp.index]
        for f in feature_list:
            vals = grp[f].fillna(0).values.astype(np.float64)
            s = vals.std()
            if s < 1e-10:
                records[f].append(np.nan)
                continue
            z = (vals - vals.mean()) / s
            ic = spearmanr(y_day, z)[0]
            records[f].append(ic)

    out = []
    for f, ics in records.items():
        arr = np.array([x for x in ics if not np.isnan(x)])
        if len(arr) < 10:
            out.append({'feature': f, 'mean_ic': 0, 'std_ic': 1, 'icir': 0,
                        'abs_icir': 0, 'med_ic': 0, 'ppos': 0.5, 'n_days': len(arr)})
            continue
        mean_ic  = arr.mean()
        std_ic   = arr.std()
        icir     = mean_ic / (std_ic + 1e-10)
        abs_icir = abs(icir)
        med_ic   = np.nanmedian(arr)
        ppos     = (arr > 0).mean()
        out.append({'feature': f, 'mean_ic': mean_ic, 'std_ic': std_ic,
                    'icir': icir, 'abs_icir': abs_icir, 'med_ic': med_ic,
                    'ppos': ppos, 'n_days': len(arr)})
    return pd.DataFrame(out).sort_values('abs_icir', ascending=False)

# Compute for lag1 features only (most predictive)
icir_lag1 = batch_icir(lag1_cols)
elapsed = (time.time() - t0) / 60
print(f"  LagT1 ICIR done — {elapsed:.1f}m")
print(f"  Top 5 LagT1 features:")
for _, row in icir_lag1.head(5).iterrows():
    print(f"    {row['feature']:<45}  ICIR={row['abs_icir']:.3f}  IC={row['mean_ic']:+.4f}")

# Gold features = ICIR≥3 and never flip sign
gold_mask = (icir_lag1['abs_icir'] >= 3) & \
            ((icir_lag1['ppos'] >= 0.99) | (icir_lag1['ppos'] <= 0.01))
gold_df = icir_lag1[gold_mask].sort_values('abs_icir', ascending=False)
top10   = gold_df['feature'].tolist()[:10]
ic_arr  = np.array([icir_lag1.set_index('feature')['mean_ic'].get(f, 0) for f in top10])
print(f"  Gold LagT1 features (ICIR≥3, never-flip): {len(gold_df)}")
print(f"  Top 10 used: {top10[:5]}...")

icir_map = icir_lag1.set_index('feature')['abs_icir'].to_dict()
ic_map   = icir_lag1.set_index('feature')['mean_ic'].to_dict()

gc.collect()

# ================================================================
# HYPOTHESIS 1: PRICE FINGERPRINTING
# ================================================================
print("\n" + "=" * 70)
print("[4/6] HYPOTHESIS 1: PRICE FINGERPRINTING")
print("  Can we match illiquid test assets to liquid train assets by Price?")
print("  LagT1 = feature[t] - feature[t-T1] → Price_LagT1 = recent price change")
print("  If Price AND Price_LagT1 match → same instrument → copy return directly")
print("=" * 70)

h1_records = []

if price_feat and price_lag1:
    # Try different rounding precisions (= tolerance buckets)
    round_dps = [8, 6, 4, 2, 1, 0]

    for dp in round_dps:
        total_illiq = 0
        total_match = 0
        all_ics     = []
        all_match_returns = []
        all_true_returns  = []

        for day in overlap:
            tr = train[train['day_id'] == day].copy()
            te = test[test['day_id']   == day].copy()
            if len(tr) < 5 or len(te) < 5: continue

            y_liq = y_train[tr.index]
            tr = tr.reset_index(drop=True)
            tr['_y'] = y_liq

            if dp > 0:
                tr['_pk']  = tr[price_feat].round(dp)
                tr['_p1k'] = tr[price_lag1].round(dp)
                te['_pk']  = te[price_feat].round(dp)
                te['_p1k'] = te[price_lag1].round(dp)
                merge_cols = ['_pk', '_p1k']
            else:
                # integer rounding
                tr['_pk']  = tr[price_feat].round(0).astype(int)
                tr['_p1k'] = tr[price_lag1].round(0).astype(int)
                te['_pk']  = te[price_feat].round(0).astype(int)
                te['_p1k'] = te[price_lag1].round(0).astype(int)
                merge_cols = ['_pk', '_p1k']

            # Vectorized merge
            te_reset = te.reset_index(drop=True)
            te_keys  = te_reset[merge_cols].copy()
            tr_keys  = tr[merge_cols + ['_y']].copy()

            merged = te_keys.merge(tr_keys, on=merge_cols, how='left')
            pred   = merged.groupby(merged.index)['_y'].mean().values

            matched_mask = ~np.isnan(pred)
            n_match = matched_mask.sum()
            total_illiq += len(te)
            total_match += n_match

            if n_match >= 5:
                y_ill = y_train[te.index[matched_mask]]
                p_ill = pred[matched_mask]
                ic = spearmanr(y_ill, p_ill)[0]
                all_ics.append(ic if not np.isnan(ic) else 0)
                all_match_returns.extend(p_ill.tolist())
                all_true_returns.extend(y_ill.tolist())

        match_rate = total_match / max(total_illiq, 1) * 100
        med_ic = np.nanmedian(all_ics) if all_ics else np.nan
        mean_ic = np.nanmean(all_ics) if all_ics else np.nan

        # Overall IC across all days
        if len(all_match_returns) > 10:
            overall_ic = spearmanr(all_true_returns, all_match_returns)[0]
        else:
            overall_ic = np.nan

        h1_records.append({
            'round_dp': dp,
            'match_rate_pct': match_rate,
            'n_matched': total_match,
            'n_illiq': total_illiq,
            'med_ic': med_ic,
            'mean_ic': mean_ic,
            'overall_ic': overall_ic,
            'n_days_with_match': len(all_ics),
        })

        elapsed = (time.time() - t0) / 60
        print(f"  dp={dp:>2}: match={match_rate:6.2f}%  ({total_match:>7}/{total_illiq})  "
              f"Med IC={med_ic:+.5f}  Overall IC={overall_ic:+.5f}  [{elapsed:.1f}m]")

    # 3-key fingerprint at dp=6
    if price_lag2:
        print(f"\n  3-key fingerprint (Price + Price_LagT1 + Price_LagT2) at dp=6:")
        total_illiq, total_match = 0, 0
        all_ics3, all_mr, all_tr = [], [], []
        for day in overlap:
            tr = train[train['day_id'] == day].copy()
            te = test[test['day_id']   == day].copy()
            if len(tr) < 5 or len(te) < 5: continue
            y_liq = y_train[tr.index]
            tr = tr.reset_index(drop=True)
            tr['_y'] = y_liq
            for df in [tr, te]:
                df['_pk']  = df[price_feat].round(6)
                df['_p1k'] = df[price_lag1].round(6)
                df['_p2k'] = df[price_lag2].round(6)
            merged = te.reset_index(drop=True)[['_pk','_p1k','_p2k']].merge(
                tr[['_pk','_p1k','_p2k','_y']], on=['_pk','_p1k','_p2k'], how='left')
            pred = merged.groupby(merged.index)['_y'].mean().values
            mm = ~np.isnan(pred)
            total_illiq += len(te); total_match += mm.sum()
            if mm.sum() >= 5:
                y_ill = y_train[te.index[mm]]
                all_ics3.append(spearmanr(y_ill, pred[mm])[0])
                all_mr.extend(pred[mm].tolist())
                all_tr.extend(y_ill.tolist())
        rate3 = total_match / max(total_illiq,1) * 100
        med3  = np.nanmedian(all_ics3) if all_ics3 else np.nan
        oic3  = spearmanr(all_tr, all_mr)[0] if len(all_tr) > 10 else np.nan
        h1_records.append({'round_dp': '3key_dp6', 'match_rate_pct': rate3,
                            'n_matched': total_match, 'n_illiq': total_illiq,
                            'med_ic': med3, 'mean_ic': np.nanmean(all_ics3),
                            'overall_ic': oic3, 'n_days_with_match': len(all_ics3)})
        print(f"  3-key dp=6: match={rate3:.2f}%  ({total_match}/{total_illiq})  "
              f"Med IC={med3:+.5f}  Overall IC={oic3:+.5f}")

else:
    print("  Price or Price_LagT1 not in dataset — skipping.")
    h1_records.append({'note': 'Price features not found'})

h1_df = pd.DataFrame(h1_records)
h1_df.to_csv(f'{OUT_DIR}/h1_price_fingerprint.csv', index=False)
print(f"\n  Saved: {OUT_DIR}/h1_price_fingerprint.csv")
elapsed = (time.time() - t0) / 60
print(f"  [H1 complete — {elapsed:.1f}m]")
gc.collect()


# ================================================================
# HYPOTHESIS 2: RECONSTRUCTED RAW LAGS (BASE - LagT1)
# ================================================================
print("\n" + "=" * 70)
print("[5/6] HYPOTHESIS 2: RECONSTRUCTED RAW LAGS")
print("  LagT1 = feature[t] - feature[t-T1]")
print("  BASE  = feature[t]")
print("  BASE - LagT1 = feature[t-T1]  ← raw lagged state NOT currently in features")
print("=" * 70)

h2_records = []

for lag1_col, base_col in base_of_lag1.items():
    day_ics_lag1 = []
    day_ics_reco = []
    day_ics_base = []

    for day, grp in train.groupby('day_id'):
        if len(grp) < 10: continue
        y_day = y_train[grp.index]

        l1   = grp[lag1_col].fillna(0).values.astype(np.float64)
        base = grp[base_col].fillna(0).values.astype(np.float64)
        reco = base - l1  # = feature[t-T1]

        for sig, store in [(l1, day_ics_lag1), (reco, day_ics_reco), (base, day_ics_base)]:
            s = sig.std()
            if s < 1e-10: store.append(np.nan); continue
            z = (sig - sig.mean()) / s
            store.append(spearmanr(y_day, z)[0])

    def icir_of(ics):
        arr = np.array([x for x in ics if not np.isnan(x)])
        if len(arr) < 10: return 0.0, 0.0, 0.5
        return abs(arr.mean() / (arr.std() + 1e-10)), np.nanmedian(arr), (arr > 0).mean()

    l1_icir,   l1_med,   l1_ppos   = icir_of(day_ics_lag1)
    reco_icir, reco_med, reco_ppos = icir_of(day_ics_reco)
    base_icir, base_med, base_ppos = icir_of(day_ics_base)

    h2_records.append({
        'lag1_col':   lag1_col,
        'base_col':   base_col,
        'reco_col':   f'{base_col}_raw_lag1',
        'lag1_icir':  l1_icir,   'lag1_med':  l1_med,  'lag1_ppos':  l1_ppos,
        'reco_icir':  reco_icir, 'reco_med':  reco_med,'reco_ppos':  reco_ppos,
        'base_icir':  base_icir, 'base_med':  base_med,'base_ppos':  base_ppos,
        'ratio_reco_vs_lag1': reco_icir / (l1_icir + 1e-6),
        'is_gold_lag1': l1_icir >= 3.0 and (l1_ppos >= 0.99 or l1_ppos <= 0.01),
        'is_gold_reco': reco_icir >= 3.0 and (reco_ppos >= 0.99 or reco_ppos <= 0.01),
    })

h2_df = pd.DataFrame(h2_records).sort_values('reco_icir', ascending=False)
h2_df.to_csv(f'{OUT_DIR}/h2_reconstructed_lags.csv', index=False)

print(f"\n  Top 20 reconstructed lags by ICIR:")
print(f"\n  {'lag1_col':<45}  {'Reco ICIR':>10}  {'LagT1 ICIR':>11}  {'Base ICIR':>10}  {'Ratio':>7}  {'Gold?':>6}")
print(f"  {'-' * 95}")
for _, row in h2_df.head(20).iterrows():
    gold = '  ***' if row['is_gold_reco'] else ''
    print(f"  {row['lag1_col']:<45}  {row['reco_icir']:>10.4f}  "
          f"{row['lag1_icir']:>11.4f}  {row['base_icir']:>10.4f}  "
          f"{row['ratio_reco_vs_lag1']:>7.3f}{gold}")

print(f"\n  Summary:")
print(f"  Total pairs tested:      {len(h2_df)}")
print(f"  Reco ICIR >= 3.0 (gold): {h2_df['is_gold_reco'].sum()}")
print(f"  Reco ICIR >= 1.0:        {(h2_df['reco_icir'] >= 1.0).sum()}")
print(f"  Reco ICIR >= 0.5:        {(h2_df['reco_icir'] >= 0.5).sum()}")
print(f"  Mean reco ICIR:          {h2_df['reco_icir'].mean():.4f}")
print(f"  Mean LagT1 ICIR:         {h2_df['lag1_icir'].mean():.4f}")
print(f"  Mean Base ICIR:          {h2_df['base_icir'].mean():.4f}")
print(f"  Reco beats LagT1:        {(h2_df['reco_icir'] > h2_df['lag1_icir']).sum()}/{len(h2_df)}")
print(f"  Reco beats Base:         {(h2_df['reco_icir'] > h2_df['base_icir']).sum()}/{len(h2_df)}")

elapsed = (time.time() - t0) / 60
print(f"  Saved: {OUT_DIR}/h2_reconstructed_lags.csv")
print(f"  [H2 complete — {elapsed:.1f}m]")
gc.collect()


# ================================================================
# HYPOTHESIS 3: MULTI-LAG MOMENTUM RATIOS
# ================================================================
print("\n" + "=" * 70)
print("[6/6] HYPOTHESIS 3: MULTI-LAG MOMENTUM RATIOS")
print("  LagT1 = velocity over window T1 (short)")
print("  LagT2 = velocity over window T2 (medium, T2 > T1)")
print("  LagT2 - LagT1 = momentum between T1 and T2 (older component)")
print("  2*LagT1 - LagT2 = acceleration (is short-term trend accelerating?)")
print("  sign(LagT1) * sign(LagT2) = momentum direction agreement")
print("=" * 70)

paired = [(b, lag1_of_base[b], lag2_of_base[b])
          for b in lag1_of_base
          if b in lag2_of_base
          and lag1_of_base[b] in train.columns
          and lag2_of_base[b] in train.columns]

print(f"  Features with both LagT1 and LagT2: {len(paired)}")

h3_records = []
for base_col, lag1_col, lag2_col in paired:
    stores = {k: [] for k in ['lag1','lag2','older','accel','sign','ratio','abs_ratio']}

    for day, grp in train.groupby('day_id'):
        if len(grp) < 10: continue
        y_day = y_train[grp.index]
        l1 = grp[lag1_col].fillna(0).values.astype(np.float64)
        l2 = grp[lag2_col].fillna(0).values.astype(np.float64)

        sigs = {
            'lag1':      l1,
            'lag2':      l2,
            'older':     l2 - l1,
            'accel':     2*l1 - l2,
            'sign':      np.sign(l1) * np.sign(l2),
            'ratio':     l1 / np.where(np.abs(l2) < 1e-10, 1e-10, l2),
            'abs_ratio': np.abs(l1) / (np.abs(l2) + 1e-10),
        }
        for k, sig in sigs.items():
            s = sig.std()
            if s < 1e-10: stores[k].append(np.nan); continue
            z = (sig - sig.mean()) / s
            stores[k].append(spearmanr(y_day, z)[0])

    def icir_of(ics):
        arr = np.array([x for x in ics if not np.isnan(x)])
        if len(arr) < 10: return 0.0
        return abs(arr.mean() / (arr.std() + 1e-10))

    row = {
        'base': base_col, 'lag1_col': lag1_col, 'lag2_col': lag2_col,
        **{f'{k}_icir': icir_of(v) for k, v in stores.items()}
    }
    row['best_derived'] = max(row['older_icir'], row['accel_icir'],
                               row['sign_icir'], row['ratio_icir'], row['abs_ratio_icir'])
    row['best_name'] = max(['older','accel','sign','ratio','abs_ratio'],
                            key=lambda x: row[f'{x}_icir'])
    h3_records.append(row)

h3_df = pd.DataFrame(h3_records).sort_values('best_derived', ascending=False)
h3_df.to_csv(f'{OUT_DIR}/h3_momentum_ratios.csv', index=False)

print(f"\n  Top 20 multi-lag signals:")
print(f"\n  {'base':<38}  {'lag1':>6}  {'lag2':>6}  {'older':>6}  {'accel':>6}  "
      f"{'sign':>6}  {'ratio':>6}  {'best':>8}")
print(f"  {'-' * 95}")
for _, row in h3_df.head(20).iterrows():
    print(f"  {row['base']:<38}  {row['lag1_icir']:>6.3f}  {row['lag2_icir']:>6.3f}  "
          f"{row['older_icir']:>6.3f}  {row['accel_icir']:>6.3f}  "
          f"{row['sign_icir']:>6.3f}  {row['ratio_icir']:>6.3f}  "
          f"{row['best_derived']:>6.3f}({row['best_name'][:3]})")

print(f"\n  Mean ICIR across {len(h3_df)} features:")
cols_labels = [('lag1_icir','LagT1'), ('lag2_icir','LagT2'), ('older_icir','Older'),
               ('accel_icir','Accel'), ('sign_icir','Sign'), ('ratio_icir','Ratio')]
for col, label in cols_labels:
    beats_lag1 = (h3_df[col] > h3_df['lag1_icir']).sum()
    print(f"  {label:<12}: mean={h3_df[col].mean():.4f}  "
          f"max={h3_df[col].max():.4f}  "
          f"beats LagT1: {beats_lag1}/{len(h3_df)}")

elapsed = (time.time() - t0) / 60
print(f"  Saved: {OUT_DIR}/h3_momentum_ratios.csv")
print(f"  [H3 complete — {elapsed:.1f}m]")
gc.collect()


# ================================================================
# WRITE SUMMARY FILE
# ================================================================
print("\n" + "=" * 70)
print("WRITING SUMMARY")
print("=" * 70)

lines = []
lines.append("=" * 70)
lines.append("STRUCTURAL DEEP DIVE — SUMMARY OF FINDINGS")
lines.append(f"Run completed: {time.strftime('%Y-%m-%d %H:%M')}")
lines.append(f"Total time: {(time.time()-t0)/60:.1f} min")
lines.append("=" * 70)

lines.append("\n=== HYPOTHESIS 1: PRICE FINGERPRINTING ===")
if len(h1_records) > 0 and 'match_rate_pct' in h1_df.columns:
    best = h1_df[h1_df['match_rate_pct'] == h1_df['match_rate_pct'].max()].iloc[0]
    lines.append(f"Best match rate: {best['match_rate_pct']:.2f}%  (at dp={best['round_dp']})")
    lines.append(f"Matched pairs: {best['n_matched']}/{best['n_illiq']}")
    lines.append(f"Median IC on matched: {best['med_ic']:+.5f}")
    lines.append(f"Overall IC on matched: {best['overall_ic']:+.5f}")
    if best['match_rate_pct'] > 10 and best['med_ic'] > 0.1:
        lines.append("VERDICT: GAME CHANGER — direct return copy viable!")
    elif best['match_rate_pct'] > 5:
        lines.append("VERDICT: USEFUL — partial identity matching possible")
    elif best['match_rate_pct'] < 0.5:
        lines.append("VERDICT: DEAD END — assets are distinct instruments")
    else:
        lines.append("VERDICT: WEAK — some matches but limited signal")

lines.append("\n=== HYPOTHESIS 2: RECONSTRUCTED RAW LAGS ===")
lines.append(f"Pairs tested: {len(h2_df)}")
lines.append(f"New gold features (ICIR>=3, never-flip): {h2_df['is_gold_reco'].sum()}")
lines.append(f"Mean reco ICIR: {h2_df['reco_icir'].mean():.4f}  vs  "
             f"Mean LagT1 ICIR: {h2_df['lag1_icir'].mean():.4f}")
lines.append(f"Reco beats LagT1: {(h2_df['reco_icir']>h2_df['lag1_icir']).sum()}/{len(h2_df)}")
if h2_df['is_gold_reco'].sum() > 0:
    lines.append("VERDICT: NEW GOLD FEATURES FOUND — add to Grinold engine!")
    lines.append("Top new features:")
    for _, r in h2_df[h2_df['is_gold_reco']].head(5).iterrows():
        lines.append(f"  {r['reco_col']}: ICIR={r['reco_icir']:.3f}")
elif h2_df['reco_icir'].mean() > h2_df['lag1_icir'].mean():
    lines.append("VERDICT: MARGINAL SIGNAL — reco lags slightly better than LagT1 on average")
else:
    lines.append("VERDICT: DEAD END — LagT1 differences carry more signal than levels")

lines.append("\n=== HYPOTHESIS 3: MULTI-LAG MOMENTUM ===")
lines.append(f"Pairs tested: {len(h3_df)}")
for col, label in [('older_icir','Older'), ('accel_icir','Accel'),
                    ('sign_icir','Sign'), ('ratio_icir','Ratio')]:
    beats = (h3_df[col] > h3_df['lag1_icir']).sum()
    lines.append(f"  {label:<10}: mean={h3_df[col].mean():.4f}  beats LagT1: {beats}/{len(h3_df)}")
best_derived_col = max(['older_icir','accel_icir','sign_icir','ratio_icir'],
                        key=lambda c: h3_df[c].mean())
if h3_df[best_derived_col].mean() > h3_df['lag1_icir'].mean() * 1.1:
    lines.append(f"VERDICT: NEW SIGNAL FOUND — {best_derived_col} beats LagT1!")
else:
    lines.append("VERDICT: DEAD END — LagT1 is already the optimal momentum representation")

lines.append("\n=== OVERALL CONCLUSION ===")
lines.append("See individual CSVs for full feature lists.")
lines.append(f"Output files in /kaggle/working/:")
lines.append("  h1_price_fingerprint.csv")
lines.append("  h2_reconstructed_lags.csv")
lines.append("  h3_momentum_ratios.csv")

summary_text = '\n'.join(lines)
print(summary_text)

with open(f'{OUT_DIR}/summary.txt', 'w') as f:
    f.write(summary_text)

print(f"\nSaved: {OUT_DIR}/summary.txt")
print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
print("DONE. Copy the output above and the 4 files from /kaggle/working/")
