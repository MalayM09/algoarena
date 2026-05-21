# ================================================================
# ID SORT TOPOLOGY LEAK TEST — lean, no matplotlib, column-selective
# Loads ONLY: ID, SO3_T, TARGET (+ 1 lag feature for benchmark)
# Safe to run on 8GB RAM
# ================================================================
import os, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')

print("=" * 65)
print("ID SORT TOPOLOGY LEAK TEST (lean)")
print("=" * 65)

# ── Load only needed columns ───────────────────────────────────
print("\nLoading minimal columns...")

# First peek at column names to find one top lag feature
import pyarrow.parquet as pq
schema = pq.read_schema(TRAIN_PATH)
all_cols = [f.name for f in schema]
lag1_cols = [c for c in all_cols if c.endswith('_LagT1')]
top_lag = lag1_cols[0] if lag1_cols else None

load_cols_train = ['ID', 'SO3_T', 'TARGET'] + ([top_lag] if top_lag else [])
load_cols_test  = ['ID', 'SO3_T']

train = pd.read_parquet(TRAIN_PATH, columns=load_cols_train)
test  = pd.read_parquet(TEST_PATH,  columns=load_cols_test)
print(f"  Train: {len(train):,} rows   Test: {len(test):,} rows")
print(f"  Memory: train={train.memory_usage(deep=True).sum()/1e6:.0f}MB  "
      f"test={test.memory_usage(deep=True).sum()/1e6:.0f}MB")

train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)

overlap_days = sorted(set(train['day_id'].unique()) & set(test['day_id'].unique()))
print(f"  Overlap days: {len(overlap_days)}")

y_train = train['TARGET'].values.astype(np.float64)

# ── ID parser ──────────────────────────────────────────────────
def parse_id_int(id_series):
    extracted = id_series.astype(str).str.extract(r'(\d+)')
    return extracted[0].astype(np.int64).values

# ── STEP 1: Single-day namespace test ─────────────────────────
print("\n" + "=" * 65)
print("STEP 1: ID NAMESPACE — single day")
print("=" * 65)

sample_day = overlap_days[10]
tr_day = train[train['day_id'] == sample_day]
te_day = test[test['day_id']   == sample_day]

tr_ids = np.sort(parse_id_int(tr_day['ID']))
te_ids = np.sort(parse_id_int(te_day['ID']))

print(f"  Day: {sample_day}")
print(f"  Train IDs: min={tr_ids.min():>10,}  max={tr_ids.max():>10,}  n={len(tr_ids)}")
print(f"  Test  IDs: min={te_ids.min():>10,}  max={te_ids.max():>10,}  n={len(te_ids)}")
print(f"  Exact integer overlaps: {len(set(tr_ids) & set(te_ids))}")
interleave = not (tr_ids.max() < te_ids.min() or te_ids.max() < tr_ids.min())
print(f"  Train max < Test min?   {tr_ids.max() < te_ids.min()}  ← Case B: separate ranges")
print(f"  IDs interleave?         {interleave}  ← Case A: exploit possible")

# ── STEP 2: Full overlap — namespace check ─────────────────────
print("\n" + "=" * 65)
print("STEP 2: NAMESPACE TEST — all overlap days")
print("=" * 65)

tr_mins, tr_maxs, te_mins, te_maxs, exact_ovlp, interleaves = [], [], [], [], [], []

for day in overlap_days:
    tr = train[train['day_id'] == day]
    te = test[test['day_id']   == day]
    if len(tr) < 5 or len(te) < 5: continue
    trid = parse_id_int(tr['ID'])
    teid = parse_id_int(te['ID'])
    tr_mins.append(trid.min());  tr_maxs.append(trid.max())
    te_mins.append(teid.min());  te_maxs.append(teid.max())
    exact_ovlp.append(len(set(trid) & set(teid)))
    interleaves.append(not (trid.max() < teid.min() or teid.max() < trid.min()))

print(f"  Train ID range (all days): [{min(tr_mins):,}, {max(tr_maxs):,}]")
print(f"  Test  ID range (all days): [{min(te_mins):,}, {max(te_maxs):,}]")
print(f"  Exact ID overlaps: mean={np.mean(exact_ovlp):.1f}  "
      f"max={max(exact_ovlp)}  min={min(exact_ovlp)}")
print(f"  Days with exact overlap:  {sum(x>0 for x in exact_ovlp)}/{len(exact_ovlp)}")
print(f"  Days where ranges interleave: {sum(interleaves)}/{len(interleaves)}")

# ── STEP 3: ID-rank IC with TARGET ────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: IC OF ID INTEGER WITH TARGET (30 days)")
print("=" * 65)

id_ics, lag_ics = [], []

for day in overlap_days[:30]:
    tr = train[train['day_id'] == day].copy()
    if len(tr) < 10: continue
    tr['id_int'] = parse_id_int(tr['ID'])
    y = y_train[tr.index]

    ic_id = spearmanr(tr['id_int'].values, y)[0]
    id_ics.append(ic_id)

    if top_lag and top_lag in tr.columns:
        vals = tr[top_lag].fillna(0).values.astype(np.float64)
        if vals.std() > 1e-10:
            lag_ics.append(spearmanr(vals, y)[0])

arr_id = np.array([x for x in id_ics if not np.isnan(x)])
print(f"\n  ID integer Spearman IC with TARGET:")
print(f"  Mean IC:    {arr_id.mean():+.5f}")
print(f"  Median IC:  {np.nanmedian(arr_id):+.5f}")
print(f"  Std IC:     {arr_id.std():.5f}")
print(f"  ICIR:       {arr_id.mean()/arr_id.std():+.4f}")
print(f"  % positive: {(arr_id > 0).mean()*100:.1f}%")
print(f"  % negative: {(arr_id < 0).mean()*100:.1f}%")

if lag_ics:
    arr_lag = np.array([x for x in lag_ics if not np.isnan(x)])
    print(f"\n  Benchmark ({top_lag}) IC:")
    print(f"  Mean IC:    {arr_lag.mean():+.5f}")
    print(f"  ICIR:       {arr_lag.mean()/arr_lag.std():+.4f}")

# ── STEP 4: VERDICT ───────────────────────────────────────────
print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)

abs_mean_ic = abs(arr_id.mean())
stable = (arr_id > 0).mean() > 0.8 or (arr_id < 0).mean() > 0.8
n_interleave = sum(interleaves)

print(f"\n  ID mean |IC|:     {abs_mean_ic:.5f}")
print(f"  IC stable sign:   {stable}")
print(f"  ID ranges interleave on {n_interleave}/{len(interleaves)} days")

if n_interleave > 0 and abs_mean_ic > 0.01 and stable:
    print("\n  >>> GAME CHANGER: ID IS A LEAK + NAMESPACES INTERLEAVE <<<")
    print("  IDs identify assets across train/test → 1D kernel smoother viable")
    print("  ACTION: implement ID-proximity kernel for illiquid predictions")
elif n_interleave > 0 and abs_mean_ic > 0.01:
    print("\n  PARTIAL: IDs interleave BUT IC is unstable across days")
    print("  ID proximity sometimes useful but not reliably directional")
elif n_interleave == 0 and abs_mean_ic > 0.01:
    print("\n  PARTIAL: ID has IC with TARGET but ranges DO NOT interleave")
    print("  Liquid-illiquid IDs are in separate ranges → can't cross-match by ID")
    print("  The leak is real but not exploitable for test prediction")
    print("  ID rank could be added as a within-set feature")
elif abs_mean_ic < 0.003:
    print("\n  DEAD END: ID has near-zero IC — the topology chart shows VARIANCE")
    print("  patterns, not MEAN patterns. The rolling mean chart was misleading.")
    print("  IDs do not predict returns.")
else:
    print(f"\n  WEAK: IC={arr_id.mean():+.5f} — marginal signal in ID, not worth pursuing")

# ── STEP 5: ID topology ASCII plot ────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: ID TOPOLOGY (ASCII) — rolling mean by ID rank")
print("=" * 65)

tr = train[train['day_id'] == sample_day].copy()
tr['id_int'] = parse_id_int(tr['ID'])
tr['y'] = y_train[tr.index]
tr_s = tr.sort_values('id_int').reset_index(drop=True)
roll = tr_s['y'].rolling(50, center=True).mean().values

# Simple ASCII bar chart — 40 buckets
n_buckets = 40
bucket_means = []
step = len(roll) // n_buckets
for i in range(n_buckets):
    chunk = roll[i*step:(i+1)*step]
    valid = chunk[~np.isnan(chunk)]
    bucket_means.append(valid.mean() if len(valid) > 0 else 0)

bm = np.array(bucket_means)
bm_norm = (bm - bm.mean()) / (bm.std() + 1e-10)  # normalize

print(f"\n  Rolling mean of TARGET sorted by ID integer (day={sample_day})")
print(f"  Each bar = {step} assets   |  = 1 std unit")
print(f"  Range: [{bm.min():.5f}, {bm.max():.5f}]  std={bm.std():.5f}")
print()
for i, (val, norm) in enumerate(zip(bm, bm_norm)):
    bar_len = int(abs(norm) * 10)
    bar = ('█' * bar_len) if norm >= 0 else ('░' * bar_len)
    side = '+' if norm >= 0 else '-'
    id_lo = tr_s.iloc[i*step]['id_int'] if i*step < len(tr_s) else 0
    print(f"  ID~{id_lo:>8,}  {side}  {bar:<20}  {val:+.5f}")
print()

print("  If the bars show large alternating patterns (not random noise)")
print("  → ID encodes structural asset groupings")
print("  If bars look random around zero → no meaningful ID structure")
