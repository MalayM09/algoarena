# ================================================================
# ID SORT TOPOLOGY LEAK TEST — fully self-contained
# ================================================================
import re, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no display needed — saves to file
import matplotlib.pyplot as plt

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("ID SORT TOPOLOGY LEAK TEST")
print("=" * 65)

# ── Load ───────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
print(f"  Train: {train.shape}   Test: {test.shape}")

train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)

overlap_days = sorted(set(train['day_id'].unique()) & set(test['day_id'].unique()))
print(f"  Overlap days: {len(overlap_days)}")

y_train = train['TARGET'].values.astype(np.float64)

# ── ID parser ──────────────────────────────────────────────────
def parse_id_int(id_series):
    extracted = id_series.astype(str).str.extract(r'(\d+)')
    return extracted[0].astype(int).values

# ── STEP 1: Namespace test on ONE day ─────────────────────────
print("\n" + "=" * 65)
print("STEP 1: ID NAMESPACE TEST (single day)")
print("=" * 65)

sample_day = overlap_days[10]
tr_day = train[train['day_id'] == sample_day].copy()
te_day = test[test['day_id']   == sample_day].copy()

tr_ids = np.sort(parse_id_int(tr_day['ID']))
te_ids = np.sort(parse_id_int(te_day['ID']))

print(f"\n  Day: {sample_day}")
print(f"  Train IDs: min={tr_ids.min():>8}  max={tr_ids.max():>8}  n={len(tr_ids)}")
print(f"  Test  IDs: min={te_ids.min():>8}  max={te_ids.max():>8}  n={len(te_ids)}")

exact_overlap = len(set(tr_ids) & set(te_ids))
print(f"\n  Exact integer overlaps: {exact_overlap}")
print(f"  Train max < Test min?  {tr_ids.max() < te_ids.min()}  ← Case B: separate ranges")
print(f"  IDs interleave?        "
      f"{not (tr_ids.max() < te_ids.min() or te_ids.max() < tr_ids.min())}"
      f"  ← Case A: exploit possible")

# ── STEP 2: Namespace test across ALL overlap days ─────────────
print("\n" + "=" * 65)
print("STEP 2: NAMESPACE TEST ACROSS ALL OVERLAP DAYS")
print("=" * 65)

all_tr_min, all_tr_max = [], []
all_te_min, all_te_max = [], []
all_exact_overlap      = []

for day in overlap_days:
    tr = train[train['day_id'] == day]
    te = test[test['day_id']   == day]
    if len(tr) < 5 or len(te) < 5: continue
    trid = parse_id_int(tr['ID'])
    teid = parse_id_int(te['ID'])
    all_tr_min.append(trid.min()); all_tr_max.append(trid.max())
    all_te_min.append(teid.min()); all_te_max.append(teid.max())
    all_exact_overlap.append(len(set(trid) & set(teid)))

print(f"\n  Across {len(all_tr_min)} overlap days:")
print(f"  Train ID range:  [{min(all_tr_min)}, {max(all_tr_max)}]")
print(f"  Test  ID range:  [{min(all_te_min)}, {max(all_te_max)}]")
print(f"  Exact overlaps:  mean={np.mean(all_exact_overlap):.1f}  "
      f"max={max(all_exact_overlap)}  min={min(all_exact_overlap)}")
print(f"  Days with ANY exact overlap: "
      f"{sum(x > 0 for x in all_exact_overlap)}/{len(all_exact_overlap)}")

interleave_days = sum(
    1 for trmx, temn, temx, trmn in
    zip(all_tr_max, all_te_min, all_te_max, all_tr_min)
    if not (trmx < temn or temx < trmn)
)
print(f"  Days where ranges interleave: {interleave_days}/{len(all_tr_min)}")

# ── STEP 3: PLOT — single day ID distribution ──────────────────
print("\n" + "=" * 65)
print("STEP 3: PLOTS")
print("=" * 65)

tr_day_plot = train[train['day_id'] == sample_day].copy()
te_day_plot = test[test['day_id']   == sample_day].copy()
tr_day_plot['id_int'] = parse_id_int(tr_day_plot['ID'])
te_day_plot['id_int'] = parse_id_int(te_day_plot['ID'])
tr_day_plot['TARGET_val'] = y_train[tr_day_plot.index]

fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# -- Plot 1: ID distribution histogram
ax = axes[0]
ax.hist(tr_day_plot['id_int'], bins=80, alpha=0.6, color='cyan',
        label=f'Train/Liquid (n={len(tr_day_plot)})')
ax.hist(te_day_plot['id_int'], bins=80, alpha=0.6, color='orange',
        label=f'Test/Illiquid (n={len(te_day_plot)})')
ax.set_title(f'ID Integer Distribution — Day {sample_day}', fontsize=13)
ax.set_xlabel('ID (integer)')
ax.set_ylabel('Count')
ax.legend()
ax.grid(alpha=0.3)

# -- Plot 2: scatter of TARGET vs ID integer for liquid assets
ax = axes[1]
tr_sorted = tr_day_plot.sort_values('id_int')
ax.scatter(tr_sorted['id_int'], tr_sorted['TARGET_val'],
           s=1, c='cyan', alpha=0.2, label='Liquid TARGET')
# overlay rolling mean
rolling = tr_sorted.set_index('id_int')['TARGET_val'].rolling(50, center=True).mean()
ax.plot(rolling.index, rolling.values, color='yellow', linewidth=2,
        label='Rolling mean (w=50)')
ax.axhline(tr_sorted['TARGET_val'].mean(), color='red', linestyle='--',
           linewidth=1.5, label='Daily mean')
# mark where test IDs fall (y=0 line)
ax.scatter(te_day_plot['id_int'],
           [tr_sorted['TARGET_val'].mean()] * len(te_day_plot),
           s=8, c='orange', alpha=0.5, marker='|', label='Illiquid ID positions')
ax.set_title('TARGET vs ID Integer (liquid assets) — rolling mean reveals ID leak structure',
             fontsize=13)
ax.set_xlabel('ID (integer)')
ax.set_ylabel('TARGET')
ax.legend(markerscale=3)
ax.grid(alpha=0.3)

# -- Plot 3: rolling mean for 5 random days to check consistency
ax = axes[2]
colors = ['cyan', 'lime', 'magenta', 'yellow', 'white']
for i, day in enumerate(overlap_days[5:10]):
    tr = train[train['day_id'] == day].copy()
    tr['id_int'] = parse_id_int(tr['ID'])
    tr['y'] = y_train[tr.index]
    tr_s = tr.sort_values('id_int')
    roll = tr_s.set_index('id_int')['y'].rolling(50, center=True).mean()
    # normalize to mean=0 std=1 for comparison
    roll = (roll - roll.mean()) / (roll.std() + 1e-10)
    ax.plot(roll.index, roll.values, color=colors[i], alpha=0.7,
            linewidth=1.2, label=f'Day {i+6}')
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.set_title('Rolling Mean (normalized) across 5 different days — structure consistent?',
             fontsize=13)
ax.set_xlabel('ID (integer)')
ax.set_ylabel('Normalized rolling mean')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
path1 = os.path.join(OUT_DIR, 'id_leak_test.png')
plt.savefig(path1, dpi=130, bbox_inches='tight')
plt.close()
print(f"  Saved: {path1}")

# ── STEP 4: QUANTIFY THE LEAK STRENGTH ────────────────────────
print("\n" + "=" * 65)
print("STEP 4: QUANTIFY THE ID LEAK SIGNAL STRENGTH")
print("=" * 65)
print("  If ID predicts TARGET, ID-rank should have nonzero IC with TARGET")

day_ics_idrank = []
day_ics_grinold_proxy = []  # compare against a top lag feature

from scipy.stats import spearmanr

# top gold feature
top_lag = None
for c in train.columns:
    if c.endswith('_LagT1'):
        top_lag = c
        break

for day in overlap_days[:100]:  # sample 100 days
    tr = train[train['day_id'] == day].copy()
    if len(tr) < 10: continue
    tr['id_int'] = parse_id_int(tr['ID'])
    y = y_train[tr.index]

    # IC of ID rank vs TARGET
    ic_id = spearmanr(tr['id_int'], y)[0]
    day_ics_idrank.append(ic_id)

    # IC of top lag feature vs TARGET (benchmark)
    if top_lag and top_lag in tr.columns:
        vals = tr[top_lag].fillna(0).values.astype(np.float64)
        if vals.std() > 1e-10:
            day_ics_grinold_proxy.append(spearmanr(vals, y)[0])

arr_id = np.array([x for x in day_ics_idrank if not np.isnan(x)])
print(f"\n  ID integer IC with TARGET (100 days):")
print(f"  Mean IC:   {arr_id.mean():+.5f}")
print(f"  Median IC: {np.nanmedian(arr_id):+.5f}")
print(f"  Std IC:    {arr_id.std():.5f}")
print(f"  ICIR:      {arr_id.mean()/arr_id.std():+.4f}")
print(f"  % positive: {(arr_id > 0).mean()*100:.1f}%")
print(f"  % negative: {(arr_id < 0).mean()*100:.1f}%")

if len(day_ics_grinold_proxy) > 10:
    arr_g = np.array([x for x in day_ics_grinold_proxy if not np.isnan(x)])
    print(f"\n  Benchmark ({top_lag}) IC:")
    print(f"  Mean IC:   {arr_g.mean():+.5f}")
    print(f"  ICIR:      {arr_g.mean()/arr_g.std():+.4f}")

# Is the IC stable in sign?
if abs(arr_id.mean()) > 0.01:
    print(f"\n  VERDICT: ID integer has MEANINGFUL IC with TARGET ({arr_id.mean():+.4f})")
    print(f"  The ID sort IS a leak — ID rank predicts return direction.")
    if (arr_id > 0).mean() > 0.8 or (arr_id < 0).mean() > 0.8:
        print(f"  STABLE DIRECTION: ID rank always predicts in same direction → exploit viable!")
    else:
        print(f"  UNSTABLE: direction flips across days → weaker exploit")
elif abs(arr_id.mean()) > 0.003:
    print(f"\n  VERDICT: WEAK ID signal (IC={arr_id.mean():+.4f}) — marginal at best")
else:
    print(f"\n  VERDICT: ID has near-zero IC ({arr_id.mean():+.4f}) — not a predictive leak")
    print(f"  The wave pattern in the topology chart is NOT due to ID-return correlation")
    print(f"  (It may be due to local variance patterns, not mean shift)")

# ── STEP 5: IC distribution plot ──────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
ax.hist(arr_id, bins=30, color='cyan', edgecolor='black', alpha=0.8)
ax.axvline(arr_id.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean IC = {arr_id.mean():+.4f}')
ax.axvline(0, color='white', linestyle='-', linewidth=1)
ax.set_title('Distribution of ID-integer IC with TARGET across 100 days', fontsize=13)
ax.set_xlabel('Spearman IC')
ax.set_ylabel('Count')
ax.legend()
ax.grid(alpha=0.3)
path2 = os.path.join(OUT_DIR, 'id_ic_distribution.png')
plt.savefig(path2, dpi=130, bbox_inches='tight')
plt.close()
print(f"\n  IC distribution plot saved: {path2}")

print("\n" + "=" * 65)
print("DONE — Check outputs/id_leak_test.png and outputs/id_ic_distribution.png")
print("=" * 65)
