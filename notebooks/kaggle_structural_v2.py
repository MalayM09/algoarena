# ================================================================
# KAGGLE STRUCTURAL DEEP DIVE — v2 (all merge bugs fixed)
#
# ROOT BUG FIX: pandas merge() resets the index on the output
# DataFrame. Using merged.index to groupby is WRONG when there
# are multiple matches (extra rows = wrong size). All merges now
# use an explicit _row_id key and reindex after groupby.
#
# NEW in v2:
#   - Correct merge via _row_id everywhere
#   - H1 has working within-train IC validation
#   - H4: LagT3 momentum signals (3-horizon multi-lag)
#   - Clean summary auto-verdict
#
# SETUP:
#   1. Paste into Kaggle Notebook
#   2. Set COMPETITION_SLUG to your competition folder name
#   3. Run All — outputs go to /kaggle/working/
# ================================================================

COMPETITION_SLUG = "competitions/short-horizon-return-prediction-challenge-by-i-rage/"

import os, time, warnings, gc
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

KAGGLE_INPUT = f'/kaggle/input/{COMPETITION_SLUG}'
OUT_DIR      = '/kaggle/working'

def find_file(folder, pattern):
    for f in os.listdir(folder):
        if pattern in f and f.endswith('.parquet'):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"No parquet matching '{pattern}' in {folder}")

TRAIN_PATH = find_file(KAGGLE_INPUT, 'train')
TEST_PATH  = find_file(KAGGLE_INPUT, 'test')

print("=" * 70)
print("KAGGLE STRUCTURAL DEEP DIVE v2")
print(f"  Train: {TRAIN_PATH}")
print(f"  Test:  {TEST_PATH}")
print("=" * 70)

# ── Load ───────────────────────────────────────────────────────
print("\n[1/6] Loading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)

# float64 → float32 to save memory
feat_cols = [c for c in train.columns if c not in ('ID','TARGET')]
for df in [train, test]:
    for c in feat_cols:
        if c in df.columns and df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)

train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
overlap  = sorted(set(train['day_id'].unique()) & set(test['day_id'].unique()))
new_days = set(test['day_id'].unique()) - set(train['day_id'].unique())

# y_train positional — matches train row 0..N-1 exactly (reset_index above)
y_train  = train['TARGET'].values.astype(np.float64)
all_cols = set(train.columns) - {'ID','TARGET','day_id','SO3_T'}

print(f"  Train:   {len(train):,} rows | Test: {len(test):,} rows")
print(f"  Overlap: {len(overlap)} days | Future: {len(new_days)} days")
print(f"  Memory:  train={train.memory_usage(deep=True).sum()/1e9:.2f}GB  "
      f"test={test.memory_usage(deep=True).sum()/1e9:.2f}GB")

# ── Feature catalog ────────────────────────────────────────────
print("\n[2/6] Building feature catalog...")
lag1_cols = [c for c in all_cols if c.endswith('_LagT1') and c in train.columns]
lag2_cols = [c for c in all_cols if c.endswith('_LagT2') and c in train.columns]
lag3_cols = [c for c in all_cols if c.endswith('_LagT3') and c in train.columns]

base_of_lag1 = {lc: lc.replace('_LagT1','')
                for lc in lag1_cols if lc.replace('_LagT1','') in train.columns}
lag2_of_base = {lc.replace('_LagT2',''):lc
                for lc in lag2_cols if lc.replace('_LagT2','') in train.columns}
lag3_of_base = {lc.replace('_LagT3',''):lc
                for lc in lag3_cols if lc.replace('_LagT3','') in train.columns}
lag1_of_base = {v: k for k, v in base_of_lag1.items()}

price_feat = 'Price'        if 'Price'        in train.columns else None
price_lag1 = 'Price_LagT1'  if 'Price_LagT1'  in train.columns else None
price_lag2 = 'Price_LagT2'  if 'Price_LagT2'  in train.columns else None

print(f"  LagT1:{len(lag1_cols)}  LagT2:{len(lag2_cols)}  LagT3:{len(lag3_cols)}")
print(f"  Base-LagT1 pairs: {len(base_of_lag1)}")
print(f"  Bases with LagT1+LagT2: {sum(1 for b in lag1_of_base if b in lag2_of_base)}")
print(f"  Bases with all 3 lags:  "
      f"{sum(1 for b in lag1_of_base if b in lag2_of_base and b in lag3_of_base)}")
print(f"  Price features: {price_feat}, {price_lag1}, {price_lag2}")

# ── Helper: safe merge using _row_id (THE KEY FIX) ────────────
def safe_merge_predict(left_df, right_df, key_cols, target_col='_y'):
    """
    Left-join left_df onto right_df on key_cols.
    Handles multiple matches by averaging. Returns array of len(left_df).
    Uses explicit _row_id to avoid pandas merge index reset bug.
    """
    left = left_df[key_cols].copy()
    left['_row_id'] = np.arange(len(left_df), dtype=np.int32)

    merged = left.merge(right_df[key_cols + [target_col]], on=key_cols, how='left')
    # groupby _row_id → average multiple matches → reindex to ensure full length
    pred = (merged.groupby('_row_id')[target_col]
                  .mean()
                  .reindex(np.arange(len(left_df)))
                  .values)
    return pred   # NaN where no match

# ── Per-feature ICIR ───────────────────────────────────────────
print("\n[3/6] Computing LagT1 ICIR on ALL training days (~5-8 min)...")

def batch_icir(feature_list):
    records = {f: [] for f in feature_list}
    for day, grp in train.groupby('day_id'):
        if len(grp) < 10: continue
        y_day = y_train[grp.index.values]   # positional index is safe
        for f in feature_list:
            vals = grp[f].fillna(0).values.astype(np.float64)
            s = vals.std()
            if s < 1e-10: records[f].append(np.nan); continue
            records[f].append(spearmanr(y_day, (vals-vals.mean())/s)[0])
    out = []
    for f, ics in records.items():
        arr = np.array([x for x in ics if not np.isnan(x)])
        if len(arr) < 10:
            out.append({'feature':f,'mean_ic':0,'std_ic':1,'icir':0,
                        'abs_icir':0,'med_ic':0,'ppos':0.5,'n_days':len(arr)})
            continue
        m, s = arr.mean(), arr.std()
        out.append({'feature':f,'mean_ic':m,'std_ic':s,'icir':m/(s+1e-10),
                    'abs_icir':abs(m/(s+1e-10)),'med_ic':np.nanmedian(arr),
                    'ppos':(arr>0).mean(),'n_days':len(arr)})
    return pd.DataFrame(out).sort_values('abs_icir', ascending=False)

icir_lag1 = batch_icir(lag1_cols)
elapsed = (time.time()-t0)/60
print(f"  Done — {elapsed:.1f}m")
for _, r in icir_lag1.head(5).iterrows():
    print(f"    {r['feature']:<45}  ICIR={r['abs_icir']:.3f}  IC={r['mean_ic']:+.4f}")

gold_mask = ((icir_lag1['abs_icir'] >= 3) &
             ((icir_lag1['ppos'] >= 0.99) | (icir_lag1['ppos'] <= 0.01)))
gold_df = icir_lag1[gold_mask].sort_values('abs_icir', ascending=False)
top10   = gold_df['feature'].tolist()[:10]
ic_map  = icir_lag1.set_index('feature')['mean_ic'].to_dict()
print(f"  Gold features (ICIR>=3, never-flip): {len(gold_df)}")
print(f"  Top 10: {top10[:3]}...")
gc.collect()


# ================================================================
# HYPOTHESIS 1: PRICE FINGERPRINTING
# ================================================================
print("\n" + "="*70)
print("[4/6] HYPOTHESIS 1: PRICE FINGERPRINTING")
print("  If Price AND Price_LagT1 match between liquid and illiquid asset")
print("  on the same day → they may be the same instrument → copy return")
print("="*70)

h1_records = []

if price_feat and price_lag1:
    round_dps = [8, 6, 4, 2, 1, 0]

    # ── A: Match rate train→test (no IC because test has no TARGET) ───
    print("\n  A. Match rate: train (liquid) → test (illiquid) by price key")
    for dp in round_dps:
        total_illiq, total_match = 0, 0
        for day in overlap:
            tr_pos = np.where(train['day_id'] == day)[0]
            te_pos = np.where(test['day_id']  == day)[0]
            if len(tr_pos) < 5 or len(te_pos) < 5: continue

            tr_sub = train.iloc[tr_pos][[ price_feat, price_lag1]].copy()
            te_sub = test.iloc[te_pos][[  price_feat, price_lag1]].copy()
            tr_sub['_y'] = y_train[tr_pos]

            if dp > 0:
                for df in [tr_sub, te_sub]:
                    df['_pk']  = df[price_feat].round(dp)
                    df['_p1k'] = df[price_lag1].round(dp)
            else:
                for df in [tr_sub, te_sub]:
                    df['_pk']  = df[price_feat].round(0).astype(int)
                    df['_p1k'] = df[price_lag1].round(0).astype(int)

            pred = safe_merge_predict(te_sub, tr_sub, ['_pk','_p1k'])
            total_illiq += len(te_pos)
            total_match += int((~np.isnan(pred)).sum())

        rate = total_match / max(total_illiq, 1) * 100
        h1_records.append({'section':'A_train_test','round_dp':dp,
                            'match_rate_pct':rate,'n_matched':total_match,
                            'n_total':total_illiq})
        elapsed = (time.time()-t0)/60
        print(f"    dp={dp:>2}: match={rate:6.2f}%  ({total_match:>7}/{total_illiq})  "
              f"[{elapsed:.1f}m]")

    # ── B: Within-train IC validation (liquid half → illiquid half) ───
    # Split each training day 50/50 → use liquid half to predict illiquid half
    # This gives a TRUE IC since we know both sides' returns
    print(f"\n  B. Within-train IC: split each overlap day 50/50, "
          f"predict illiquid half from liquid half (50 days)")
    rng = np.random.default_rng(42)
    ic_val_rows = []

    for day in overlap[:50]:
        tr_pos = np.where(train['day_id'] == day)[0]
        if len(tr_pos) < 20: continue

        perm    = rng.permutation(len(tr_pos))
        liq_pos = tr_pos[perm[:len(perm)//2]]
        ill_pos = tr_pos[perm[len(perm)//2:]]

        liq_sub = train.iloc[liq_pos][[price_feat, price_lag1]].copy()
        ill_sub = train.iloc[ill_pos][[price_feat, price_lag1]].copy()
        y_liq   = y_train[liq_pos]
        y_ill   = y_train[ill_pos]   # ground truth for illiquid half

        liq_sub['_y'] = y_liq

        for dp in [6]:
            for df in [liq_sub, ill_sub]:
                df['_pk']  = df[price_feat].round(dp)
                df['_p1k'] = df[price_lag1].round(dp)

            # safe_merge_predict guarantees len(pred) == len(ill_pos)
            pred = safe_merge_predict(ill_sub, liq_sub, ['_pk','_p1k'])
            mm   = ~np.isnan(pred)

            if mm.sum() >= 5:
                # y_ill is already len(ill_pos); mm is same length — safe
                ic = spearmanr(y_ill[mm], pred[mm])[0]
                if not np.isnan(ic):
                    ic_val_rows.append({'day':day,'dp':dp,
                                        'n_matched':int(mm.sum()),
                                        'n_total':len(ill_pos),'ic':ic})

    if ic_val_rows:
        ic_val_df = pd.DataFrame(ic_val_rows)
        match_pct = ic_val_df['n_matched'].sum() / ic_val_df['n_total'].sum() * 100
        med_ic    = float(ic_val_df['ic'].median())
        mean_ic   = float(ic_val_df['ic'].mean())
        print(f"    Within-train match rate (dp=6): {match_pct:.2f}%")
        print(f"    Median IC of matched pairs:     {med_ic:+.5f}")
        print(f"    Mean   IC of matched pairs:     {mean_ic:+.5f}")
        if med_ic > 0.10:
            print(f"    *** VERDICT: GAME CHANGER — price fingerprint has strong IC ***")
        elif med_ic > 0.02:
            print(f"    VERDICT: USEFUL — price fingerprint has real but weak IC")
        else:
            print(f"    VERDICT: DEAD END — IC too low, price coincidences are noise")
        ic_val_df.to_csv(f'{OUT_DIR}/h1_within_train_ic.csv', index=False)
        print(f"    Saved: h1_within_train_ic.csv")
    else:
        print(f"    No within-train matches at dp=6 — assets are distinct instruments")

    # ── C: 3-key fingerprint (Price + LagT1 + LagT2) ─────────────
    if price_lag2:
        print(f"\n  C. 3-key fingerprint (Price + Price_LagT1 + Price_LagT2) dp=6:")
        total_illiq, total_match = 0, 0
        for day in overlap:
            tr_pos = np.where(train['day_id'] == day)[0]
            te_pos = np.where(test['day_id']  == day)[0]
            if len(tr_pos) < 5 or len(te_pos) < 5: continue
            tr_sub = train.iloc[tr_pos][[price_feat,price_lag1,price_lag2]].copy()
            te_sub = test.iloc[te_pos][[  price_feat,price_lag1,price_lag2]].copy()
            tr_sub['_y'] = y_train[tr_pos]
            for df in [tr_sub, te_sub]:
                df['_pk']  = df[price_feat].round(6)
                df['_p1k'] = df[price_lag1].round(6)
                df['_p2k'] = df[price_lag2].round(6)
            pred = safe_merge_predict(te_sub, tr_sub, ['_pk','_p1k','_p2k'])
            total_illiq += len(te_pos)
            total_match += int((~np.isnan(pred)).sum())
        rate3 = total_match / max(total_illiq,1) * 100
        h1_records.append({'section':'C_3key','round_dp':'3key_dp6',
                            'match_rate_pct':rate3,'n_matched':total_match,
                            'n_total':total_illiq})
        print(f"    match={rate3:.2f}%  ({total_match}/{total_illiq})")

else:
    print("  Price or Price_LagT1 not found — skipping H1")
    h1_records.append({'note':'Price features not found'})

h1_df = pd.DataFrame(h1_records)
h1_df.to_csv(f'{OUT_DIR}/h1_price_fingerprint.csv', index=False)
elapsed = (time.time()-t0)/60
print(f"\n  Saved: h1_price_fingerprint.csv  [{elapsed:.1f}m]")
gc.collect()


# ================================================================
# HYPOTHESIS 2: RECONSTRUCTED RAW LAGS (BASE - LagT1)
# ================================================================
print("\n" + "="*70)
print("[5/6] HYPOTHESIS 2: RECONSTRUCTED RAW LAGS")
print("  BASE - LagT1 = feature[t-T1]  ← raw lagged state not in dataset")
print("="*70)

def icir_of(ics):
    arr = np.array([x for x in ics if not np.isnan(x)])
    if len(arr) < 10: return 0.0, 0.0, 0.5
    m = arr.mean(); s = arr.std()
    return abs(m/(s+1e-10)), np.nanmedian(arr), (arr>0).mean()

h2_records = []
for lag1_col, base_col in base_of_lag1.items():
    stores = {'lag1':[], 'reco':[], 'base':[]}
    for day, grp in train.groupby('day_id'):
        if len(grp) < 10: continue
        y_day = y_train[grp.index.values]
        l1   = grp[lag1_col].fillna(0).values.astype(np.float64)
        base = grp[base_col].fillna(0).values.astype(np.float64)
        reco = base - l1
        for sig, key in [(l1,'lag1'),(reco,'reco'),(base,'base')]:
            s = sig.std()
            if s < 1e-10: stores[key].append(np.nan); continue
            stores[key].append(spearmanr(y_day,(sig-sig.mean())/s)[0])

    l1_icir,   l1_med,   l1_pp  = icir_of(stores['lag1'])
    rc_icir,   rc_med,   rc_pp  = icir_of(stores['reco'])
    ba_icir,   ba_med,   ba_pp  = icir_of(stores['base'])

    h2_records.append({
        'lag1_col': lag1_col, 'base_col': base_col,
        'reco_col': f'{base_col}_raw_lag1',
        'lag1_icir': l1_icir, 'lag1_med': l1_med, 'lag1_ppos': l1_pp,
        'reco_icir': rc_icir, 'reco_med': rc_med, 'reco_ppos': rc_pp,
        'base_icir': ba_icir, 'base_med': ba_med, 'base_ppos': ba_pp,
        'ratio_reco_vs_lag1': rc_icir / (l1_icir + 1e-6),
        'is_gold_lag1': l1_icir >= 3.0 and (l1_pp >= 0.99 or l1_pp <= 0.01),
        'is_gold_reco': rc_icir >= 3.0 and (rc_pp >= 0.99 or rc_pp <= 0.01),
    })

h2_df = pd.DataFrame(h2_records).sort_values('reco_icir', ascending=False)
h2_df.to_csv(f'{OUT_DIR}/h2_reconstructed_lags.csv', index=False)

print(f"\n  Top 20 reconstructed lags by ICIR:")
print(f"  {'lag1_col':<45}  {'Reco':>8}  {'LagT1':>8}  {'Base':>8}  {'Ratio':>7}  Gold?")
print(f"  {'-'*88}")
for _, r in h2_df.head(20).iterrows():
    g = ' ***' if r['is_gold_reco'] else ''
    print(f"  {r['lag1_col']:<45}  {r['reco_icir']:>8.4f}  {r['lag1_icir']:>8.4f}"
          f"  {r['base_icir']:>8.4f}  {r['ratio_reco_vs_lag1']:>7.3f}{g}")

print(f"\n  Summary ({len(h2_df)} pairs):")
print(f"  Gold reco (ICIR>=3, never-flip): {h2_df['is_gold_reco'].sum()}")
print(f"  Reco ICIR >= 1.0: {(h2_df['reco_icir']>=1.0).sum()}")
print(f"  Mean reco ICIR:   {h2_df['reco_icir'].mean():.4f}"
      f"  vs LagT1: {h2_df['lag1_icir'].mean():.4f}"
      f"  vs Base: {h2_df['base_icir'].mean():.4f}")
print(f"  Reco > LagT1: {(h2_df['reco_icir']>h2_df['lag1_icir']).sum()}/{len(h2_df)}")
elapsed = (time.time()-t0)/60
print(f"  Saved: h2_reconstructed_lags.csv  [{elapsed:.1f}m]")
gc.collect()


# ================================================================
# HYPOTHESIS 3: MULTI-LAG MOMENTUM (LagT1 + LagT2)
# ================================================================
print("\n" + "="*70)
print("[6/6] HYPOTHESIS 3: MULTI-LAG MOMENTUM RATIOS")
print("  older  = LagT2 - LagT1  (change between T1 and T2 windows)")
print("  accel  = 2*LagT1 - LagT2  (is momentum accelerating?)")
print("  sign   = sign(LagT1) * sign(LagT2)  (direction agreement)")
print("  ratio  = LagT1 / LagT2  (relative magnitude)")
print("="*70)

def icir_scalar(ics):
    arr = np.array([x for x in ics if not np.isnan(x)])
    if len(arr) < 10: return 0.0
    return abs(arr.mean() / (arr.std() + 1e-10))

paired = [(b, lag1_of_base[b], lag2_of_base[b])
          for b in lag1_of_base
          if b in lag2_of_base
          and lag1_of_base[b] in train.columns
          and lag2_of_base[b] in train.columns]

h3_records = []
for base_col, lag1_col, lag2_col in paired:
    stores = {k:[] for k in ['lag1','lag2','older','accel','sign','ratio','abs_ratio']}
    for day, grp in train.groupby('day_id'):
        if len(grp) < 10: continue
        y_day = y_train[grp.index.values]
        l1 = grp[lag1_col].fillna(0).values.astype(np.float64)
        l2 = grp[lag2_col].fillna(0).values.astype(np.float64)
        sigs = {
            'lag1': l1, 'lag2': l2,
            'older': l2 - l1,
            'accel': 2*l1 - l2,
            'sign':  np.sign(l1) * np.sign(l2),
            'ratio': l1 / np.where(np.abs(l2)<1e-10, 1e-10, l2),
            'abs_ratio': np.abs(l1) / (np.abs(l2)+1e-10),
        }
        for k, sig in sigs.items():
            s = sig.std()
            if s < 1e-10: stores[k].append(np.nan); continue
            stores[k].append(spearmanr(y_day,(sig-sig.mean())/s)[0])

    row = {'base':base_col,'lag1_col':lag1_col,'lag2_col':lag2_col,
           **{f'{k}_icir':icir_scalar(v) for k,v in stores.items()}}
    row['best_derived'] = max(row['older_icir'],row['accel_icir'],
                               row['sign_icir'],row['ratio_icir'],row['abs_ratio_icir'])
    row['best_name'] = max(['older','accel','sign','ratio','abs_ratio'],
                            key=lambda x: row[f'{x}_icir'])
    h3_records.append(row)

h3_df = pd.DataFrame(h3_records).sort_values('best_derived', ascending=False)
h3_df.to_csv(f'{OUT_DIR}/h3_momentum_ratios.csv', index=False)

print(f"\n  Top 20 multi-lag signals:")
print(f"  {'base':<38}  {'L1':>6}  {'L2':>6}  {'old':>6}  {'acc':>6}  "
      f"{'sgn':>6}  {'rat':>6}  {'best':>8}")
print(f"  {'-'*88}")
for _, r in h3_df.head(20).iterrows():
    print(f"  {r['base']:<38}  {r['lag1_icir']:>6.3f}  {r['lag2_icir']:>6.3f}  "
          f"{r['older_icir']:>6.3f}  {r['accel_icir']:>6.3f}  "
          f"{r['sign_icir']:>6.3f}  {r['ratio_icir']:>6.3f}  "
          f"{r['best_derived']:>6.3f}({r['best_name'][:3]})")

print(f"\n  Mean ICIR across {len(h3_df)} features:")
for col, label in [('lag1_icir','LagT1'),('lag2_icir','LagT2'),('older_icir','Older'),
                   ('accel_icir','Accel'),('sign_icir','Sign'),('ratio_icir','Ratio')]:
    beats = (h3_df[col] > h3_df['lag1_icir']).sum()
    print(f"  {label:<12}: mean={h3_df[col].mean():.4f}  "
          f"max={h3_df[col].max():.4f}  beats LagT1: {beats}/{len(h3_df)}")

# ── BONUS: 3-horizon signals (LagT1 + LagT2 + LagT3) ─────────
triple = [(b, lag1_of_base[b], lag2_of_base[b], lag3_of_base[b])
          for b in lag1_of_base
          if b in lag2_of_base and b in lag3_of_base
          and all(c in train.columns for c in
                  [lag1_of_base[b], lag2_of_base[b], lag3_of_base[b]])]

if triple:
    print(f"\n  BONUS: 3-horizon signals ({len(triple)} features with LagT1+T2+T3):")
    h3b_records = []
    for base_col, l1c, l2c, l3c in triple:
        stores3 = {k:[] for k in ['l1','l2','l3','short_v_long','momentum_score','triple_sign']}
        for day, grp in train.groupby('day_id'):
            if len(grp) < 10: continue
            y_day = y_train[grp.index.values]
            l1 = grp[l1c].fillna(0).values.astype(np.float64)
            l2 = grp[l2c].fillna(0).values.astype(np.float64)
            l3 = grp[l3c].fillna(0).values.astype(np.float64)
            sigs3 = {
                'l1': l1, 'l2': l2, 'l3': l3,
                # short-term minus long-term momentum
                'short_v_long': l1 - l3,
                # weighted combo: 3*l1 - 2*l2 (emphasises recent)
                'momentum_score': 3*l1 - 2*l2,
                # all 3 horizons agree on direction
                'triple_sign': np.sign(l1) * np.sign(l2) * np.sign(l3),
            }
            for k, sig in sigs3.items():
                s = sig.std()
                if s < 1e-10: stores3[k].append(np.nan); continue
                stores3[k].append(spearmanr(y_day,(sig-sig.mean())/s)[0])

        row3 = {'base':base_col,
                **{f'{k}_icir':icir_scalar(v) for k,v in stores3.items()}}
        row3['best3'] = max(row3['short_v_long_icir'],row3['momentum_score_icir'],
                            row3['triple_sign_icir'])
        h3b_records.append(row3)

    h3b_df = pd.DataFrame(h3b_records).sort_values('best3', ascending=False)
    h3b_df.to_csv(f'{OUT_DIR}/h3b_triple_lag.csv', index=False)

    print(f"  {'base':<38}  {'L1':>6}  {'svl':>7}  {'mom':>7}  {'3sgn':>7}  {'best':>7}")
    print(f"  {'-'*78}")
    for _, r in h3b_df.head(10).iterrows():
        print(f"  {r['base']:<38}  {r['l1_icir']:>6.3f}  "
              f"{r['short_v_long_icir']:>7.3f}  {r['momentum_score_icir']:>7.3f}  "
              f"{r['triple_sign_icir']:>7.3f}  {r['best3']:>7.3f}")

    print(f"\n  Mean ICIRs:")
    for col, label in [('l1_icir','LagT1'),('short_v_long_icir','Short v Long'),
                       ('momentum_score_icir','Mom Score'),('triple_sign_icir','Triple Sign')]:
        beats = (h3b_df[col] > h3b_df['l1_icir']).sum()
        print(f"  {label:<18}: mean={h3b_df[col].mean():.4f}  "
              f"max={h3b_df[col].max():.4f}  beats LagT1: {beats}/{len(h3b_df)}")
    print(f"  Saved: h3b_triple_lag.csv")

elapsed = (time.time()-t0)/60
print(f"\n  Saved: h3_momentum_ratios.csv  [{elapsed:.1f}m]")
gc.collect()


# ================================================================
# WRITE SUMMARY
# ================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

lines = ["="*70,
         "STRUCTURAL DEEP DIVE v2 — SUMMARY",
         f"Run: {time.strftime('%Y-%m-%d %H:%M')}  |  "
         f"Total time: {(time.time()-t0)/60:.1f} min",
         "="*70]

# H1
lines.append("\n=== H1: PRICE FINGERPRINTING ===")
if 'match_rate_pct' in h1_df.columns:
    h1_num = h1_df[h1_df['match_rate_pct'].notna()].copy()
    if len(h1_num):
        best = h1_num.loc[h1_num['match_rate_pct'].astype(float).idxmax()]
        lines.append(f"Best match rate: {float(best['match_rate_pct']):.2f}%  dp={best['round_dp']}")
        lines.append(f"Matched: {best['n_matched']}/{best['n_total']}")
        rate = float(best['match_rate_pct'])
        if rate > 10:   lines.append("VERDICT: HIGH MATCH — price identity viable")
        elif rate > 1:  lines.append("VERDICT: PARTIAL MATCH — some shared instruments")
        else:           lines.append("VERDICT: DEAD END — distinct instruments, no cross-match")
    if 'ic_val_rows' in dir() and ic_val_rows:
        lines.append(f"Within-train IC at dp=6: median={float(ic_val_df['ic'].median()):+.5f}")
else:
    lines.append("Price features not found — skipped")

# H2
lines.append("\n=== H2: RECONSTRUCTED RAW LAGS ===")
lines.append(f"Pairs: {len(h2_df)}  |  Gold reco: {h2_df['is_gold_reco'].sum()}")
lines.append(f"Mean reco ICIR: {h2_df['reco_icir'].mean():.4f}  "
             f"vs LagT1: {h2_df['lag1_icir'].mean():.4f}")
if h2_df['is_gold_reco'].sum() > 0:
    lines.append("VERDICT: NEW GOLD FEATURES — add to Grinold!")
    for _, r in h2_df[h2_df['is_gold_reco']].head(5).iterrows():
        lines.append(f"  {r['reco_col']}: ICIR={r['reco_icir']:.3f}")
elif h2_df['reco_icir'].mean() > h2_df['lag1_icir'].mean():
    lines.append("VERDICT: MARGINAL — reco slightly better on average")
else:
    lines.append("VERDICT: DEAD END — differences beat levels")

# H3
lines.append("\n=== H3: MULTI-LAG MOMENTUM ===")
lines.append(f"Pairs: {len(h3_df)}")
for col, label in [('older_icir','Older'),('accel_icir','Accel'),
                   ('sign_icir','Sign'),('ratio_icir','Ratio')]:
    beats = (h3_df[col] > h3_df['lag1_icir']).sum()
    lines.append(f"  {label:<10}: mean={h3_df[col].mean():.4f}  "
                 f"beats LagT1: {beats}/{len(h3_df)}")
best_h3 = max(['older_icir','accel_icir','sign_icir','ratio_icir'],
               key=lambda c: h3_df[c].mean())
if h3_df[best_h3].mean() > h3_df['lag1_icir'].mean() * 1.1:
    lines.append(f"VERDICT: NEW SIGNAL — {best_h3} beats LagT1!")
else:
    lines.append("VERDICT: LagT1 already optimal")

lines += ["", "Files saved in /kaggle/working/:",
          "  h1_price_fingerprint.csv", "  h1_within_train_ic.csv (if matches found)",
          "  h2_reconstructed_lags.csv", "  h3_momentum_ratios.csv",
          "  h3b_triple_lag.csv (if LagT3 pairs exist)"]

summary_text = '\n'.join(lines)
print(summary_text)
with open(f'{OUT_DIR}/summary.txt','w') as f:
    f.write(summary_text)

print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
print("DONE — paste printed output + CSV files back for analysis.")
