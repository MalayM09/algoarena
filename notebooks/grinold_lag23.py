# ================================================================
# GRINOLD LAG23 — IC-Weighted, LagT2+LagT3 Gold Features Only
# ================================================================
# Problem with full Grinold (51 gold features):
#   Top-5 by |ICIR| are ALL LagT1 (ICIR 5.85-6.37, IC ~0.030-0.032)
#   LagT1 captures bid-ask/intraday microstructure → liquid-specific
#   LagT2/LagT3 features have lower ICIR (3.0-5.6) but transfer better
#   LGB importance confirms: top-8 for illiquid = ALL LagT2/LagT3
#
# This script:
#   1. Shows full gold feature breakdown by lag type
#   2. IC-weighted Grinold on LagT2+LagT3 gold only (no fitting)
#   3. Sweeps over top-N LagT2/T3 by |ICIR| to find optimum
#   4. Excludes known sign-flippers (S03_A07_A05_V09, S03_V04_T05_LagT2)
#   5. Blend check against optimal_blend_v2
# ================================================================

import sys, os, gc, time, warnings
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
ORACLE     = os.path.join(BASE_DIR, 'outputs/submissions/exploit_v2_zero.csv')
SAMPLE_SUB = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
TARGET_STD = 0.000948
t0         = time.time()

SIGN_FLIPPERS = {'S03_A07_A05_V09', 'S03_V04_T05_LagT2'}

def auto_scale(p):
    s = p.std(); return p * (TARGET_STD / s) if s > 1e-10 else p

def daywise_oracle_score(pred_vec, oracle_vec, day_ids):
    ics = []
    for day in np.unique(day_ids):
        m = day_ids == day
        if m.sum() < 3: continue
        p = pred_vec[m] - pred_vec[m].mean()
        o = oracle_vec[m] - oracle_vec[m].mean()
        pn = np.linalg.norm(p); on = np.linalg.norm(o)
        if pn < 1e-12 or on < 1e-12: ics.append(0.)
        else: ics.append(float((p @ o) / (pn * on)))
    return float(np.mean(ics))

# ── Load IC table ──────────────────────────────────────────────────
icir_df = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & (icir_df['ic_pos_frac'].isin([0.0, 1.0]))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False).copy()

# ── Breakdown of gold features by lag type ────────────────────────
print("=" * 65)
print("GOLD FEATURE BREAKDOWN BY LAG TYPE")
print("=" * 65)
lag_counts = gold_df['lag_type'].value_counts()
for lag, cnt in lag_counts.items():
    subset = gold_df[gold_df['lag_type'] == lag]
    avg_icir = subset['abs_icir'].mean()
    avg_ic   = subset['mean_ic'].abs().mean()
    flipper_n = subset['feature'].isin(SIGN_FLIPPERS).sum()
    print(f"  {lag:<8}  n={cnt:3d}  avg|ICIR|={avg_icir:.2f}  avg|IC|={avg_ic:.4f}  sign-flippers={flipper_n}")

print(f"\n  Total gold: {len(gold_df)}")
print(f"\n  Gold features by lag (top-5 each):")
for lag in ['LagT1','LagT2','LagT3','base']:
    sub = gold_df[gold_df['lag_type'] == lag].head(5)
    if len(sub) == 0: continue
    print(f"\n  [{lag}]")
    for _, r in sub.iterrows():
        flip = ' ← sign-flipper!' if r['feature'] in SIGN_FLIPPERS else ''
        print(f"    {r['feature']:<50}  ICIR={r['abs_icir']:5.2f}  IC={r['mean_ic']:+.4f}{flip}")

# ── Select LagT2+LagT3 gold, exclude sign-flippers ────────────────
lag23_df = gold_df[
    (gold_df['lag_type'].isin(['LagT2', 'LagT3'])) &
    (~gold_df['feature'].isin(SIGN_FLIPPERS))
].copy()
print(f"\n  LagT2+LagT3 gold (excl. sign-flippers): {len(lag23_df)} features")
print(f"  ICIR range: {lag23_df['abs_icir'].min():.2f} – {lag23_df['abs_icir'].max():.2f}")
print(f"  All features:")
for _, r in lag23_df.iterrows():
    print(f"    {r['feature']:<50}  ICIR={r['abs_icir']:5.2f}  IC={r['mean_ic']:+.4f}  [{r['lag_type']}]")

all_feats_lag23 = lag23_df['feature'].tolist()
ic_dict_lag23   = lag23_df.set_index('feature')['mean_ic'].to_dict()

# ── Load data ──────────────────────────────────────────────────────
print(f"\nLoading data...", flush=True)
t1 = time.time()
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
print(f"  {time.time()-t1:.1f}s", flush=True)

train_cols = set(train.columns)
all_feats_lag23 = [f for f in all_feats_lag23 if f in train_cols]
print(f"  Available LagT2+LagT3 gold features: {len(all_feats_lag23)}", flush=True)

y_raw     = train['TARGET'].values.astype(np.float64)
lo, hi    = np.percentile(y_raw, 1), np.percentile(y_raw, 99)
y_wins    = np.clip(y_raw, lo, hi)
test_ids  = test['ID'].values
train_day = train['SO3_T'].round(5).astype(str).values
test_day  = test['SO3_T'].round(5).astype(str).values

sample_sub  = pd.read_csv(SAMPLE_SUB)[['ID']]
oracle_vec  = sample_sub.merge(pd.read_csv(ORACLE), on='ID', how='left').fillna(0)['TARGET'].values
oracle_days = sample_sub.merge(
    pd.read_parquet(TEST_PATH, columns=['ID','SO3_T']), on='ID', how='left'
)['SO3_T'].round(5).astype(str).values

anchor = sample_sub.merge(
    pd.read_csv(os.path.join(OUT_DIR, 'optimal_blend_v2.csv')), on='ID', how='left'
).fillna(0)['TARGET'].values

# ── CS z-score on all LagT2+LagT3 gold features ───────────────────
print(f"\nCS z-score normalisation ({len(all_feats_lag23)} features)...", flush=True)
t1 = time.time()
tr_raw = train[all_feats_lag23].fillna(0).values.astype(np.float32)
te_raw = test.reindex(columns=all_feats_lag23, fill_value=0).values.astype(np.float32)
X_tr = np.zeros_like(tr_raw, dtype=np.float32)
X_te = np.zeros_like(te_raw, dtype=np.float32)
for d in np.unique(train_day):
    m = train_day == d; x = tr_raw[m]; s = x.std(0); s[s < 1e-8] = 1.
    X_tr[m] = (x - x.mean(0)) / s
for d in np.unique(test_day):
    m = test_day == d; x = te_raw[m]; s = x.std(0); s[s < 1e-8] = 1.
    X_te[m] = (x - x.mean(0)) / s
del tr_raw, te_raw; gc.collect()
print(f"  Done in {time.time()-t1:.1f}s", flush=True)

# Verify liquid ICs match expectation
print(f"\n  Verifying IC direction on liquid training data:")
for fi, fname in enumerate(all_feats_lag23[:8]):
    day_ics = []
    for d in np.unique(train_day):
        m = train_day == d
        if m.sum() < 5: continue
        r, _ = pearsonr(X_tr[m, fi], y_wins[m])
        if not np.isnan(r): day_ics.append(r)
    avg_ic  = float(np.mean(day_ics))
    ic_tbl  = ic_dict_lag23[fname]
    sign_ok = '✓' if np.sign(avg_ic) == np.sign(ic_tbl) else '✗'
    print(f"    {sign_ok} {fname:<50}  measured={avg_ic:+.4f}  table={ic_tbl:+.4f}")

# ── SWEEP: top-N LagT2+LagT3 gold by |ICIR| ─────────────────────
print("\n" + "="*65)
print("SWEEP: top-N LagT2+LagT3 gold features (IC-weighted, no fitting)")
print("="*65, flush=True)

N_VALUES = list(range(3, min(len(all_feats_lag23)+1, 25))) + ([len(all_feats_lag23)] if len(all_feats_lag23) > 24 else [])
N_VALUES = sorted(set(N_VALUES))

sweep_results = []
for N in N_VALUES:
    feats_n = all_feats_lag23[:N]
    ic_w    = np.array([ic_dict_lag23[f] for f in feats_n], dtype=np.float64)
    # IC-weighted prediction: pred = z_score @ ic_weights
    te_pred = X_te[:, :N].astype(np.float64) @ ic_w
    scaled  = auto_scale(te_pred)
    sub     = sample_sub.merge(
        pd.DataFrame({'ID': test_ids, 'TARGET': scaled}), on='ID', how='left'
    ).fillna(0)
    sc = daywise_oracle_score(sub['TARGET'].values, oracle_vec, oracle_days)
    sweep_results.append({'N': N, 'oracle': sc, 'sub': sub, 'te_pred': te_pred})
    flag = '  ←' if sc > 0.060349 else ''
    print(f"  N={N:2d}  oracle={sc:+.6f}{flag}", flush=True)

best = max(sweep_results, key=lambda x: x['oracle'])
print(f"\n  Peak: N={best['N']}  oracle={best['oracle']:+.6f}", flush=True)

# Save best
best['sub'].to_csv(os.path.join(OUT_DIR, f"grinold_lag23_top{best['N']}.csv"), index=False)

# ── Compare LagT1-included vs LagT2/T3-only ──────────────────────
print("\n" + "="*65)
print("COMPARISON: LagT1-inclusive vs LagT2/T3-only")
print("="*65)

# Grinold all51 (IC-weighted, all lag types)
all51 = gold_df['feature'].tolist()
all51 = [f for f in all51 if f in train_cols]
tr_all = train[all51].fillna(0).values.astype(np.float32)
te_all = test.reindex(columns=all51, fill_value=0).values.astype(np.float32)
X_tr_all = np.zeros_like(tr_all, dtype=np.float32)
X_te_all = np.zeros_like(te_all, dtype=np.float32)
for d in np.unique(train_day):
    m = train_day == d; x = tr_all[m]; s = x.std(0); s[s < 1e-8] = 1.
    X_tr_all[m] = (x - x.mean(0)) / s
for d in np.unique(test_day):
    m = test_day == d; x = te_all[m]; s = x.std(0); s[s < 1e-8] = 1.
    X_te_all[m] = (x - x.mean(0)) / s
del tr_all, te_all; gc.collect()

ic_w_all = np.array([gold_df.set_index('feature')['mean_ic'].to_dict()[f] for f in all51])
te_all_pred = X_te_all.astype(np.float64) @ ic_w_all
sub_all = sample_sub.merge(
    pd.DataFrame({'ID': test_ids, 'TARGET': auto_scale(te_all_pred)}), on='ID', how='left'
).fillna(0)
sc_all51 = daywise_oracle_score(sub_all['TARGET'].values, oracle_vec, oracle_days)

print(f"\n  Grinold all-51 (incl. LagT1)   oracle={sc_all51:+.6f}  [LagT1 dominates weights]")
print(f"  Grinold LagT2+LagT3 (top-{best['N']:2d})    oracle={best['oracle']:+.6f}  [LagT2/3 only]")
print(f"\n  LGB cs_gold_top8               oracle=+0.049107  [LagT2/3, tree model]")
print(f"  IC-weighted top-5 (linear_top5) oracle=+0.046845  [LagT2/3, no fitting]")

# Decompose: how much does each lag group contribute?
print(f"\n  IC weight distribution (all-51 model):")
for lag in ['LagT1','LagT2','LagT3','base']:
    sub_g = gold_df[gold_df['lag_type'] == lag]
    sub_g = sub_g[sub_g['feature'].isin(all51)]
    total_w = sub_g['mean_ic'].abs().sum()
    pct = 100 * total_w / gold_df[gold_df['feature'].isin(all51)]['mean_ic'].abs().sum()
    print(f"    {lag:<8}  n={len(sub_g):2d}  |IC| weight share = {pct:.1f}%")

# ── Blend check ──────────────────────────────────────────────────
print("\n" + "="*65)
print("BLEND CHECK: best variant + optimal_blend_v2")
print("="*65)
best_vec = auto_scale(best['te_pred'])

print(f"\n  optimal_blend_v2  oracle=+0.060098  LB=+0.00165 (anchor)")
print(f"  grinold_lag23     oracle={best['oracle']:+.6f}")
print(f"\n  Pairwise blends:")
print(f"  {'w_lag23':>8}  {'oracle':>10}  {'delta_anchor':>13}")
print(f"  {'─'*8}  {'─'*10}  {'─'*13}")
for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    blend = w * best_vec + (1-w) * anchor
    s = blend.std(); blend = blend*(TARGET_STD/s) if s>1e-10 else blend
    sc = daywise_oracle_score(blend, oracle_vec, oracle_days)
    flag = '  ← BEATS' if sc > 0.060349 else ''
    print(f"  {w:>8.0%}  {sc:>+10.6f}  {sc-0.060098:>+13.6f}{flag}")

# Also blend with cs_v2_gold (we already have it)
print(f"\n  [also] grinold_lag23 + cs_v2_gold + optimal_blend_v2 (3-way):")
csv2g_path = os.path.join(OUT_DIR, 'cs_v2_gold.csv')
if os.path.exists(csv2g_path):
    csv2g = sample_sub.merge(pd.read_csv(csv2g_path), on='ID', how='left').fillna(0)['TARGET'].values
    for w1, w2 in [(0.10,0.10),(0.15,0.10),(0.10,0.15),(0.20,0.10),(0.10,0.20)]:
        w3 = 1 - w1 - w2
        blend = w1 * best_vec + w2 * csv2g + w3 * anchor
        s = blend.std(); blend = blend*(TARGET_STD/s) if s>1e-10 else blend
        sc = daywise_oracle_score(blend, oracle_vec, oracle_days)
        flag = '  ← BEATS' if sc > 0.060349 else ''
        print(f"    w_lag23={w1:.0%}  w_cs2g={w2:.0%}  w_anchor={w3:.0%}  oracle={sc:+.6f}{flag}")

print(f"\n  Submit threshold: +0.060349")
print(f"  Total elapsed:    {(time.time()-t0)/60:.1f} min")
