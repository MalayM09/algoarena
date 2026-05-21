# ================================================================
# OFI (Order Flow Imbalance) + MARKET INTERCEPT ENGINE
# ================================================================
# OFI = base_feature - LagT1_feature  (change in order book state)
#
# Literature finding (Kolm et al. 2023, Cont et al. 2023):
#   OFI features predict short-term returns 65-75% better than
#   raw LOB snapshots because they capture the FLOW of information
#   into the order book, not just the current state.
#
# Strategy:
#   1. Compute 111 OFI features (base - LagT1 for each pair)
#   2. Run IC/ICIR analysis to find "gold OFI" features
#   3. Compare PI OOF: snapshot Grinold vs OFI Grinold
#   4. Combine best signal with market intercept
#   5. Generate submission files
#
# Baseline for comparison:
#   market_intercept_only        (std=0.003077, pure market beta)
#   market_intercept_top10_p005  (std=0.003237, market + snapshot Grinold)
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
t0 = time.time()

print("=" * 70)
print("OFI (Order Flow Imbalance) + MARKET INTERCEPT ENGINE")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)

cols = set(train.columns) - {'ID', 'TARGET'}

# ── Identify OFI pairs ─────────────────────────────────────────────────────
lag1_feats = {c for c in cols if c.endswith('_LagT1')}
base_feats  = {c for c in cols if not any(c.endswith(f'_Lag{t}') for t in ['T1','T2','T3'])}
pairs = sorted([(b, b+'_LagT1') for b in base_feats if b+'_LagT1' in lag1_feats])
print(f"OFI pairs available: {len(pairs)}")

# ── Day IDs ────────────────────────────────────────────────────────────────
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days = set(train['day_id'].unique())
overlap    = train_days & set(test['day_id'].unique())
new_days   = set(test['day_id'].unique()) - train_days
y_train    = train['TARGET'].values.astype(np.float64)
test_ids   = test['ID'].values

print(f"Train days: {len(train_days)}  Overlap: {len(overlap)}  Future: {len(new_days)}")

# ── BookShape proxy ────────────────────────────────────────────────────────
b_near = [c for c in cols if 'Lag' not in c and any(f'_B0{i}' in c for i in range(5))]
b_far  = [c for c in cols if 'Lag' not in c and any(f'_B{i}' in c for i in ['06','07','08','09','10'])]
train['bookshape'] = (train[b_near].fillna(0).sum(1) - train[b_far].fillna(0).sum(1)).values

# ── Helpers ────────────────────────────────────────────────────────────────
def zscore_cols(X, clip=5.0):
    m = X.mean(0); s = X.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip)

def rank_ic(y_true, y_pred):
    if len(y_true) < 5: return np.nan
    r, _ = spearmanr(y_true, y_pred)
    return r

# ================================================================
# STEP 1: Compute OFI features and run ICIR analysis
# ================================================================
print("\n" + "=" * 70)
print("STEP 1: OFI Feature ICIR Analysis (428 days × 111 features)")
print("=" * 70)

ofi_names = [f'OFI_{b}' for b, _ in pairs]
base_cols  = [b for b, _ in pairs]
lag1_cols  = [l for _, l in pairs]

# Compute per-day IC for each OFI feature
print("  Computing IC for each OFI feature across all training days...")
ofi_ic_by_day = {name: [] for name in ofi_names}
ofi_ic_pos    = {name: 0  for name in ofi_names}
ofi_ic_neg    = {name: 0  for name in ofi_names}
ofi_n_days    = {name: 0  for name in ofi_names}

for day, grp in train.groupby('day_id'):
    y_day = y_train[grp.index]
    if len(y_day) < 20: continue

    X_base = grp[base_cols].fillna(0).values.astype(np.float64)
    X_lag1 = grp[lag1_cols].fillna(0).values.astype(np.float64)
    X_ofi  = X_base - X_lag1   # raw OFI (NOT z-scored yet)
    X_ofi_z = zscore_cols(X_ofi)  # per-day z-score

    for j, name in enumerate(ofi_names):
        ic = rank_ic(y_day, X_ofi_z[:, j])
        if np.isnan(ic): continue
        ofi_ic_by_day[name].append(ic)
        ofi_n_days[name]   += 1
        if ic > 0: ofi_ic_pos[name] += 1
        else:       ofi_ic_neg[name] += 1

# Summarise
ofi_summary = []
for name in ofi_names:
    ics = np.array(ofi_ic_by_day[name])
    if len(ics) < 10: continue
    mean_ic = ics.mean()
    std_ic  = ics.std() + 1e-8
    icir    = mean_ic / std_ic
    pos_frac = ofi_ic_pos[name] / max(ofi_n_days[name], 1)
    ofi_summary.append({
        'feature' : name,
        'mean_ic' : mean_ic,
        'std_ic'  : std_ic,
        'icir'    : icir,
        'abs_icir': abs(icir),
        'ic_pos_frac': pos_frac,
        'n_days'  : ofi_n_days[name],
    })

ofi_df = pd.DataFrame(ofi_summary).sort_values('abs_icir', ascending=False)
print(f"\n  OFI feature ICIR summary ({len(ofi_df)} features):")
print(f"  {'Feature':<55}  {'ICIR':>8}  {'mean_IC':>10}  {'%pos':>8}")
print(f"  {'-'*85}")
for _, row in ofi_df.head(20).iterrows():
    print(f"  {row['feature']:<55}  {row['icir']:+8.3f}  {row['mean_ic']:+10.5f}  {row['ic_pos_frac']*100:7.1f}%")

# Gold OFI: abs_icir >= 1.5 AND ic_pos_frac either <= 0.3 or >= 0.7 (consistent direction)
gold_ofi_df = ofi_df[
    (ofi_df['abs_icir'] >= 1.5) &
    ((ofi_df['ic_pos_frac'] <= 0.3) | (ofi_df['ic_pos_frac'] >= 0.7))
].copy()

print(f"\n  Gold OFI features (|ICIR|>=1.5, consistent direction): {len(gold_ofi_df)}")
if len(gold_ofi_df) == 0:
    print("  WARNING: No gold OFI features found. Relaxing threshold to |ICIR|>=0.5...")
    gold_ofi_df = ofi_df[ofi_df['abs_icir'] >= 0.5].head(20).copy()
    print(f"  Using top {len(gold_ofi_df)} OFI features by |ICIR|.")

print(f"\n  Top gold OFI features:")
for _, row in gold_ofi_df.head(10).iterrows():
    print(f"    {row['feature']:<55}  ICIR={row['icir']:+.3f}  %pos={row['ic_pos_frac']*100:.0f}%")

gold_ofi_names  = gold_ofi_df['feature'].tolist()
gold_ofi_ic     = gold_ofi_df.set_index('feature')['mean_ic'].to_dict()
gold_ofi_icw    = np.array([gold_ofi_ic[n] for n in gold_ofi_names])

# Map OFI feature name → (base_col, lag1_col)
ofi_name_to_cols = {f'OFI_{b}': (b, b+'_LagT1') for b, _ in pairs}


# ================================================================
# STEP 2: Compare snapshot Grinold vs OFI Grinold in PI OOF
# ================================================================
print("\n" + "=" * 70)
print("STEP 2: PI OOF — Snapshot vs OFI vs Combined vs + Market Intercept")
print("=" * 70)

# Load gold snapshot features (top10)
icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False).copy()
snap_feats = [f for f in gold_df['feature'].tolist() if f in cols][:10]
snap_ic    = np.array([gold_df.set_index('feature')['mean_ic'][f] for f in snap_feats])

# Daily liquid mean return
daily_liquid_mean = train.groupby('day_id')['TARGET'].mean()

SCALE = 0.005  # confirmed optimal cross-sectional scale

results = {
    'snapshot_only': {'ics':[], 'r2s':[]},
    'ofi_only':      {'ics':[], 'r2s':[]},
    'combined_snap_ofi': {'ics':[], 'r2s':[]},
    'snapshot_intercept': {'ics':[], 'r2s':[]},
    'ofi_intercept':      {'ics':[], 'r2s':[]},
    'combined_intercept': {'ics':[], 'r2s':[]},
}

for day, grp in train.groupby('day_id'):
    n = len(grp)
    if n < 20: continue

    y_day  = y_train[grp.index]
    bs_day = grp['bookshape'].values
    bs_med = np.median(bs_day)

    liq_mask  = bs_day >= bs_med
    illiq_mask = bs_day < bs_med
    if illiq_mask.sum() < 5 or liq_mask.sum() < 5: continue

    y_illiq  = y_day[illiq_mask]
    market_intercept = y_day[liq_mask].mean()

    # Snapshot signal
    X_snap   = grp[snap_feats].fillna(0).values.astype(np.float64)
    X_snap_z = zscore_cols(X_snap)
    pred_snap = X_snap_z @ snap_ic
    pred_snap -= pred_snap.mean()

    # OFI signal
    if len(gold_ofi_names) > 0:
        ofi_cols_base = [ofi_name_to_cols[n][0] for n in gold_ofi_names]
        ofi_cols_lag1 = [ofi_name_to_cols[n][1] for n in gold_ofi_names]
        X_base = grp[ofi_cols_base].fillna(0).values.astype(np.float64)
        X_lag1 = grp[ofi_cols_lag1].fillna(0).values.astype(np.float64)
        X_ofi  = X_base - X_lag1
        X_ofi_z = zscore_cols(X_ofi)
        pred_ofi = X_ofi_z @ gold_ofi_icw
        pred_ofi -= pred_ofi.mean()
    else:
        pred_ofi = np.zeros(n)

    # Combined: normalise each signal, then sum
    snap_std = pred_snap.std() + 1e-8
    ofi_std  = pred_ofi.std()  + 1e-8
    pred_combined = pred_snap / snap_std + pred_ofi / ofi_std
    pred_combined -= pred_combined.mean()

    for key, pred_cs in [('snapshot_only', pred_snap),
                          ('ofi_only',      pred_ofi),
                          ('combined_snap_ofi', pred_combined)]:
        ic = rank_ic(y_illiq, pred_cs[illiq_mask])
        r2 = r2_score(y_illiq, pred_cs[illiq_mask] * SCALE)
        results[key]['ics'].append(ic)
        results[key]['r2s'].append(r2)

    for key, pred_cs in [('snapshot_intercept', pred_snap),
                          ('ofi_intercept',      pred_ofi),
                          ('combined_intercept', pred_combined)]:
        pred_with = pred_cs[illiq_mask] * SCALE + market_intercept
        ic = rank_ic(y_illiq, pred_with)   # IC unaffected by intercept
        r2 = r2_score(y_illiq, pred_with)
        results[key]['ics'].append(ic)
        results[key]['r2s'].append(r2)

print(f"\n  {'Model':<30}  {'Med IC':>10}  {'%pos IC':>9}  {'Med R²':>12}  {'Mean R²':>12}  {'%R²>0':>8}")
print(f"  {'-'*88}")
for key, res in results.items():
    ics = np.array([x for x in res['ics'] if not np.isnan(x)])
    r2s = np.array([x for x in res['r2s'] if not np.isnan(x)])
    print(f"  {key:<30}  {np.median(ics):+10.5f}  {(ics>0).mean()*100:8.1f}%  "
          f"{np.median(r2s):+12.6f}  {r2s.mean():+12.6f}  {(r2s>0).mean()*100:7.1f}%")

# Identify best model
best_key = max(results.keys(), key=lambda k: np.nanmedian(results[k]['ics']))
print(f"\n  Best model by median IC: {best_key}")

# Improvement of OFI vs snapshot
snap_ic_med = np.median([x for x in results['snapshot_only']['ics'] if not np.isnan(x)])
ofi_ic_med  = np.median([x for x in results['ofi_only']['ics']      if not np.isnan(x)])
comb_ic_med = np.median([x for x in results['combined_snap_ofi']['ics'] if not np.isnan(x)])

print(f"\n  Signal quality comparison:")
print(f"    Snapshot top10        : {snap_ic_med:+.5f}")
print(f"    OFI gold              : {ofi_ic_med:+.5f}  ({(ofi_ic_med/snap_ic_med - 1)*100:+.1f}% vs snapshot)")
print(f"    Combined (snap + OFI) : {comb_ic_med:+.5f}  ({(comb_ic_med/snap_ic_med - 1)*100:+.1f}% vs snapshot)")


# ================================================================
# STEP 3: OFI ICIR for top features in detail
# ================================================================
print("\n" + "=" * 70)
print("STEP 3: OFI vs Snapshot Feature Quality Comparison")
print("=" * 70)

snap_icirs = gold_df[gold_df['feature'].isin(snap_feats)][['feature','abs_icir','mean_ic','ic_pos_frac']].copy()
print(f"\n  Top10 SNAPSHOT features used (ICIR):")
for _, row in snap_icirs.iterrows():
    print(f"    {row['feature']:<50}  ICIR={row['abs_icir']:.2f}")

print(f"\n  Top OFI features (ICIR):")
for _, row in gold_ofi_df.head(10).iterrows():
    print(f"    {row['feature']:<50}  ICIR={row['icir']:+.2f}  %pos={row['ic_pos_frac']*100:.0f}%")

print(f"\n  Snapshot top10 mean |ICIR|: {snap_icirs['abs_icir'].mean():.2f}")
print(f"  Gold OFI mean |ICIR|      : {gold_ofi_df['abs_icir'].mean():.2f}")


# ================================================================
# STEP 4: Generate test predictions
# ================================================================
print("\n" + "=" * 70)
print("STEP 4: Generate Test Predictions")
print("=" * 70)

# Rebuild daily liquid mean (for intercept)
daily_liquid_mean = train.groupby('day_id')['TARGET'].mean()

# Future day intercept (Ridge on daily-aggregated features — same as market_intercept_grinold.py)
print("  Fitting future-day market return model...")
daily_feat_rows, daily_targets, day_order = [], [], []
for day, grp in train.groupby('day_id'):
    X_day    = grp[snap_feats].fillna(0).values.astype(np.float64)
    X_day_z  = zscore_cols(X_day)
    daily_feat_rows.append(X_day_z.mean(0))
    daily_targets.append(daily_liquid_mean[day])
    day_order.append(day)

X_daily   = np.array(daily_feat_rows)
y_daily   = np.array(daily_targets)
scaler_d  = StandardScaler()
X_daily_s = scaler_d.fit_transform(X_daily)
ridge_d   = Ridge(alpha=1.0)
ridge_d.fit(X_daily_s, y_daily)

future_intercepts = {}
for day, grp_te in test.groupby('day_id'):
    if day in new_days:
        X_te   = grp_te[snap_feats].fillna(0).values.astype(np.float64)
        X_te_z = zscore_cols(X_te)
        feat   = X_te_z.mean(0).reshape(1, -1)
        future_intercepts[day] = ridge_d.predict(scaler_d.transform(feat))[0]

# Build test predictions
te_snap    = np.zeros(len(test))
te_ofi     = np.zeros(len(test))
te_combined = np.zeros(len(test))
te_intercepts = np.zeros(len(test))

test_b_near = [c for c in set(test.columns) - {'ID'} if 'Lag' not in c and any(f'_B0{i}' in c for i in range(5))]
test_b_far  = [c for c in set(test.columns) - {'ID'} if 'Lag' not in c and any(f'_B{i}' in c for i in ['06','07','08','09','10'])]
test['bookshape'] = (test[test_b_near].fillna(0).sum(1) - test[test_b_far].fillna(0).sum(1)).values

for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index

    # Snapshot
    X_s   = grp_te[snap_feats].fillna(0).values.astype(np.float64)
    X_s_z = zscore_cols(X_s)
    ps    = X_s_z @ snap_ic;  ps -= ps.mean()
    te_snap[te_idx] = ps

    # OFI
    if len(gold_ofi_names) > 0:
        ofi_cb = [ofi_name_to_cols[n][0] for n in gold_ofi_names]
        ofi_cl = [ofi_name_to_cols[n][1] for n in gold_ofi_names]
        X_b = grp_te[ofi_cb].fillna(0).values.astype(np.float64)
        X_l = grp_te[ofi_cl].fillna(0).values.astype(np.float64)
        X_o = X_b - X_l
        X_oz = zscore_cols(X_o)
        po   = X_oz @ gold_ofi_icw;  po -= po.mean()
        te_ofi[te_idx] = po
    else:
        te_ofi[te_idx] = 0.0

    # Combined
    ss = ps.std() + 1e-8
    os_ = te_ofi[te_idx].std() + 1e-8
    pc = ps / ss + te_ofi[te_idx] / os_
    pc -= pc.mean()
    te_combined[te_idx] = pc

    # Intercept
    if day in overlap:
        te_intercepts[te_idx] = daily_liquid_mean[day]
    else:
        te_intercepts[te_idx] = future_intercepts.get(day, 0.0)

print(f"  Predictions computed for {len(test):,} test rows")
print(f"\n  Signal std comparison (raw, before scale):")
print(f"    Snapshot cs : {te_snap.std():.4f}")
print(f"    OFI cs      : {te_ofi.std():.4f}")
print(f"    Combined cs : {te_combined.std():.4f}")
print(f"    Intercept   : {te_intercepts.std():.6f}")


# ================================================================
# STEP 5: Save submissions
# ================================================================
print("\n" + "=" * 70)
print("STEP 5: Saving Submissions")
print("=" * 70)

sample_sub = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/sample_submission.csv'))[['ID']]

def save_sub(preds, name):
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': preds})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    path = os.path.join(OUT_DIR, f'{name}.csv')
    sub.to_csv(path, index=False)
    t = sub['TARGET']
    print(f"  {name:<55}  std={t.std():.6f}  mean={t.mean():+.8f}")
    return path

print()
# OFI only (no intercept) at p005 scale
save_sub(te_ofi * SCALE, 'ofi_only_p005')

# OFI + market intercept
save_sub(te_intercepts + te_ofi * SCALE, 'market_intercept_ofi_p005')

# Combined snapshot + OFI + market intercept
save_sub(te_intercepts + te_combined * SCALE, 'market_intercept_combined_p005')

# Snapshot + intercept at p005 (same as market_intercept_grinold.py, sanity check)
save_sub(te_intercepts + te_snap * SCALE, 'market_intercept_snap_p005_verify')


# ================================================================
# STEP 6: Full comparison analysis vs existing CSVs
# ================================================================
print("\n" + "=" * 70)
print("STEP 6: Full Comparison vs Existing Submissions")
print("=" * 70)

old_subs = {
    'market_intercept_only':       'market_intercept_only.csv',
    'market_intercept_snap_p005':  'market_intercept_top10_p005.csv',
    'grinold_allday_p005':         'grinold_allday_top10_probe_005.csv',
}

new_subs = {
    'ofi_only_p005':               'ofi_only_p005.csv',
    'market_intercept_ofi_p005':   'market_intercept_ofi_p005.csv',
    'market_intercept_comb_p005':  'market_intercept_combined_p005.csv',
}

all_preds = {}
for label, fname in {**old_subs, **new_subs}.items():
    path = os.path.join(OUT_DIR, fname)
    if os.path.exists(path):
        df_ = pd.read_csv(path).set_index('ID')['TARGET']
        all_preds[label] = df_

# Stats table
print(f"\n  {'Submission':<35}  {'std':>10}  {'mean':>12}  {'%pos':>8}  {'LB_score':>12}")
print(f"  {'-'*82}")
known_lb = {
    'grinold_allday_p005':        '+0.00096',
    'market_intercept_only':      'TBD',
    'market_intercept_snap_p005': 'TBD',
}
for label, pred in all_preds.items():
    lb = known_lb.get(label, 'NEW')
    print(f"  {label:<35}  {pred.std():10.6f}  {pred.mean():+12.8f}  {(pred>0).mean()*100:7.1f}%  {lb:>12}")

# Rank IC cross-correlation matrix
print(f"\n  Spearman rank correlation between submissions:")
labels = list(all_preds.keys())
common = all_preds[labels[0]].index
for l in labels[1:]:
    common = common.intersection(all_preds[l].index)

print(f"  {'':35}", end='')
for l in labels:
    print(f"  {l[:12]:>12}", end='')
print()
for l1 in labels:
    print(f"  {l1:<35}", end='')
    for l2 in labels:
        v1 = all_preds[l1].loc[common].values
        v2 = all_preds[l2].loc[common].values
        ic, _ = spearmanr(v1, v2)
        print(f"  {ic:+12.4f}", end='')
    print()

# Sign agreement with best known (grinold_allday_p005)
if 'grinold_allday_p005' in all_preds:
    ref = all_preds['grinold_allday_p005'].loc[common].values
    print(f"\n  Sign agreement vs grinold_allday_p005 (our best, LB=+0.00096):")
    for label in labels:
        v = all_preds[label].loc[common].values
        agree = (np.sign(ref) == np.sign(v)).mean()
        print(f"    {label:<35}  {agree*100:.1f}%")


# ================================================================
# STEP 7: Submission priority guide
# ================================================================
print("\n" + "=" * 70)
print("STEP 7: SUBMISSION PRIORITY GUIDE")
print("=" * 70)

snap_r2_med  = np.median([x for x in results['snapshot_intercept']['r2s'] if not np.isnan(x)])
ofi_r2_med   = np.median([x for x in results['ofi_intercept']['r2s']      if not np.isnan(x)])
comb_r2_med  = np.median([x for x in results['combined_intercept']['r2s'] if not np.isnan(x)])
snap_ic_f    = np.median([x for x in results['snapshot_intercept']['ics'] if not np.isnan(x)])
ofi_ic_f     = np.median([x for x in results['ofi_intercept']['ics']      if not np.isnan(x)])
comb_ic_f    = np.median([x for x in results['combined_intercept']['ics'] if not np.isnan(x)])

print(f"""
PI OOF Summary (with market intercept):
  Model                     Med IC        Med R²
  snapshot + intercept  :  {snap_ic_f:+.5f}    {snap_r2_med:+.6f}
  OFI + intercept       :  {ofi_ic_f:+.5f}    {ofi_r2_med:+.6f}
  combined + intercept  :  {comb_ic_f:+.5f}    {comb_r2_med:+.6f}

Expected LB scores (market factor ~0.0123 + cross-sectional IC²):
  market_intercept_only              : ~0.012  (pure market beta)
  market_intercept_snap_p005         : ~0.013  (market + snapshot alpha)
  market_intercept_ofi_p005          : ~{0.012 + ofi_ic_f**2:.4f}  (market + OFI alpha)
  market_intercept_combined_p005     : ~{0.012 + comb_ic_f**2:.4f}  (market + combined alpha)

Submission order (tomorrow):
  1. market_intercept_only              → isolates pure market beta contribution
  2. market_intercept_snap_p005         → baseline with market fix (expected ~0.013)
  3. market_intercept_ofi_p005          → does OFI improve over snapshot?
  4. market_intercept_combined_p005     → best of both signals

Note: #1 and #2 are already saved from market_intercept_grinold.py
      #3 and #4 are new from this script.

Total elapsed: {(time.time()-t0)/60:.1f} min
""")
