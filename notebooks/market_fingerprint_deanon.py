# ================================================================
# DE-ANONYMIZATION PHASE 1: MARKET TIMELINE FINGERPRINTING
# ================================================================
# PURPOSE: Identify the real calendar dates behind the 428 anonymous
# training days by matching extreme market events (crashes, volatility
# spikes) to real-world data. If successful, enables Phase 2: matching
# illiquid test assets to liquid train assets by ticker symbol.
#
# What to look for in the output:
#   - COVID crash (March 2020): largest 1-day mean return drop (-5 to -10%)
#     plus sustained volatility spike
#   - 2022 rate hike: persistent downward drift
#   - 2018 Q4 selloff or 2020 March–April recovery
#
# Run on Kaggle for full data access. Output: market_fingerprint.csv
# ================================================================
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs')

import pyarrow.parquet as pq
schema = pq.read_schema(TRAIN_PATH)
all_cols = [f.name for f in schema]

# Load only what we need for the fingerprint
load_cols = ['ID', 'SO3_T', 'TARGET']
print("Loading minimal columns for fingerprint...")
t0 = time.time()
train = pd.read_parquet(TRAIN_PATH, columns=load_cols)
test  = pd.read_parquet(TEST_PATH,  columns=['ID', 'SO3_T'])
print(f"  Loaded in {time.time()-t0:.1f}s  |  Train: {len(train):,}  Test: {len(test):,}")

train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)

# Chronological sort of days
day_map = train[['day_id', 'SO3_T']].drop_duplicates().sort_values('SO3_T').reset_index(drop=True)
day_map['day_idx'] = day_map.index

# ── STEP 1: Daily macro statistics (the fingerprint) ───────────────
print("\nComputing daily macro statistics...")
records = []
for i, row in day_map.iterrows():
    day  = row['day_id']
    d    = train[train['day_id'] == day]
    te_d = test[test['day_id'] == day]
    records.append({
        'day_idx':          row['day_idx'],
        'day_id':           day,
        'SO3_T':            row['SO3_T'],
        'market_mean_ret':  d['TARGET'].mean(),
        'market_vol':       d['TARGET'].std(),
        'n_liquid':         len(d),
        'n_illiquid':       len(te_d),
        'n_total':          len(d) + len(te_d),
        'skewness':         float(pd.Series(d['TARGET']).skew()),
        'kurt':             float(pd.Series(d['TARGET']).kurtosis()),
        'pct_positive':     (d['TARGET'] > 0).mean(),
    })

macro = pd.DataFrame(records)
fp_path = os.path.join(OUT_DIR, 'market_fingerprint.csv')
macro.to_csv(fp_path, index=False)
print(f"  Saved: {fp_path}")

# ── STEP 2: Identify anchor events ───────────────────────────────
print("\n" + "=" * 65)
print("EXTREME EVENTS (anchors for calendar dating)")
print("=" * 65)

print("\nTop 5 most negative mean-return days (crash candidates):")
print(macro.nsmallest(5, 'market_mean_ret')[
    ['day_idx', 'SO3_T', 'market_mean_ret', 'market_vol', 'n_liquid']
].to_string(index=False))

print("\nTop 5 highest volatility days (panic candidates):")
print(macro.nlargest(5, 'market_vol')[
    ['day_idx', 'SO3_T', 'market_mean_ret', 'market_vol', 'n_liquid']
].to_string(index=False))

print("\nTop 5 most positive mean-return days (recovery/squeeze candidates):")
print(macro.nlargest(5, 'market_mean_ret')[
    ['day_idx', 'SO3_T', 'market_mean_ret', 'market_vol', 'n_liquid']
].to_string(index=False))

# Autocorrelation of daily mean return → AR(1) coefficient
ret_series = macro['market_mean_ret'].values
ac1 = spearmanr(ret_series[:-1], ret_series[1:])[0]
print(f"\n  Daily mean return AR(1) autocorr: {ac1:.4f}")
print(f"  (negative → mean-reverting market; positive → trending)")

# ── STEP 3: Rolling volatility clusters ───────────────────────────
print("\n" + "=" * 65)
print("ROLLING 21-DAY VOLATILITY (regime clusters)")
print("=" * 65)

macro['vol_21d'] = macro['market_vol'].rolling(21, center=True).mean()
macro['ret_21d'] = macro['market_mean_ret'].rolling(21, center=True).mean()

# Find volatility regimes
high_vol = macro[macro['vol_21d'] > macro['vol_21d'].quantile(0.75)]
print(f"\n  High-vol (75th pct) period spans day_idx:")
if len(high_vol) > 0:
    runs = []
    start = high_vol.iloc[0]['day_idx']
    prev = start
    for _, r in high_vol.iloc[1:].iterrows():
        if r['day_idx'] - prev > 5:
            runs.append((int(start), int(prev)))
            start = r['day_idx']
        prev = r['day_idx']
    runs.append((int(start), int(prev)))
    for lo, hi in runs:
        so3_lo = macro[macro['day_idx']==lo]['SO3_T'].values[0]
        so3_hi = macro[macro['day_idx']==hi]['SO3_T'].values[0]
        print(f"    days {lo:3d}–{hi:3d}  (SO3_T {so3_lo:.5f}–{so3_hi:.5f})")

# ── STEP 4: ASCII chart of normalized daily mean return ───────────
print("\n" + "=" * 65)
print("STEP 4: DAILY MEAN RETURN — ASCII OVERVIEW (every 5th day)")
print("=" * 65)
print("  Each row = one training day. |= zero line. █ = positive, ░ = negative")
print()
m = macro['market_mean_ret'].values
std_m = m.std() + 1e-10
for i in range(0, len(macro), max(1, len(macro)//50)):
    row = macro.iloc[i]
    norm = row['market_mean_ret'] / std_m
    bar_len = min(int(abs(norm) * 8), 20)
    bar = ('█' * bar_len) if norm >= 0 else ('░' * bar_len)
    sign = '+' if norm >= 0 else '-'
    so3 = row['SO3_T']
    print(f"  day {int(row['day_idx']):3d}  SO3={so3:.5f}  {sign}  {bar:<20}  {row['market_mean_ret']:+.5f}  vol={row['market_vol']:.5f}")

print("\n  NOTE: If a day shows ░░░░░░░░ (large negative bar), note its day_idx.")
print("  Match to known market events via Google: 'SP500 worst single-day drops'")
print("  COVID crash: ~March 12, 16, 18, 2020 (3 massive drops in a 10-day window)")
print("  2008 GFC: Oct 9, 15 2008; March 2009 bottom")
print("  2022 FED hikes: gradual downtrend, high vol")

print("\n  ── RECOMMENDED: Run on Kaggle, download market_fingerprint.csv,")
print("  then match to Yahoo Finance by comparing extreme events. ──")
