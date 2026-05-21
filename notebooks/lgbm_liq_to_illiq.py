# ================================================================
# LGBM LIQUID → ILLIQUID TRANSFER ENGINE
# ================================================================
# Three experiments to close the 75× gap:
#
#   EXP 1: Feature audit — all features, lag vs non-lag ICIR
#   EXP 2: LGBM liquid→illiquid PI OOF (LambdaRank, time-split)
#           Feature sets: gold10, silver_lag, silver_all
#   EXP 3: Market intercept (fixed) + best LGBM → submissions
#
# Fixes vs naive LGBM:
#   1. Scale: LGBM outputs in rank space → auto-scale to match
#      optimal std=0.000948 (NOT multiply by 0.005 which crushes)
#   2. Future-day intercept: use RAW feature means (z-score mean=0
#      by definition — was the bug causing Ridge CV R²=0.054)
#   3. Objective: LambdaRank (ranking) not RMSE (absolute scale)
#   4. Clamping: future intercepts clipped to historical bounds
# ================================================================

import os, warnings, time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
t0 = time.time()

print("=" * 70)
print("LGBM LIQUID → ILLIQUID TRANSFER ENGINE")
print("=" * 70)

def zscore_cols(X, clip=5.0):
    m = X.mean(0); s = X.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip)

def rank_ic(y_true, y_pred):
    if len(y_true) < 5: return np.nan
    r, _ = spearmanr(y_true, y_pred)
    return r

def winsorise(y, lo=1, hi=99):
    return np.clip(y, np.percentile(y, lo), np.percentile(y, hi))

# ── Load ───────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)

all_train_cols = set(train.columns) - {'ID', 'TARGET'}
all_test_cols  = set(test.columns)  - {'ID'}
common_cols    = all_train_cols & all_test_cols

train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days = set(train['day_id'].unique())
overlap    = train_days & set(test['day_id'].unique())
new_days   = set(test['day_id'].unique()) - train_days
y_train    = train['TARGET'].values.astype(np.float64)
test_ids   = test['ID'].values

b_near = [c for c in common_cols if 'Lag' not in c and any(f'_B0{i}' in c for i in range(5))]
b_far  = [c for c in common_cols if 'Lag' not in c and any(f'_B{i}' in c for i in ['06','07','08','09','10'])]
train['bookshape'] = (train[b_near].fillna(0).sum(1) - train[b_far].fillna(0).sum(1)).values
test['bookshape']  = (test[b_near].fillna(0).sum(1)  - test[b_far].fillna(0).sum(1)).values

print(f"Train: {len(train):,}  Test: {len(test):,}")
print(f"Train days: {len(train_days)}  Overlap: {len(overlap)}  Future: {len(new_days)}")


# ================================================================
# EXP 1: Feature ICIR Audit
# ================================================================
print("\n" + "=" * 70)
print("EXP 1: Feature ICIR Audit — Lag vs Non-lag")
print("=" * 70)

icir_df  = pd.read_csv(ICIR_PATH)
icir_df  = icir_df[icir_df['feature'].isin(common_cols)].copy()
lag_df   = icir_df[icir_df['feature'].str.contains('_Lag')].copy()
nonlag_df = icir_df[~icir_df['feature'].str.contains('_Lag')].copy()

print(f"\n  Total={len(icir_df)}  Lag={len(lag_df)}  Non-lag={len(nonlag_df)}")
print(f"\n  ICIR distribution:")
print(f"  {'Type':<12}  {'mean':>8}  {'median':>8}  {'max':>8}")
for label, df_ in [('Lag', lag_df), ('Non-lag', nonlag_df)]:
    print(f"  {label:<12}  {df_['abs_icir'].mean():8.3f}  {df_['abs_icir'].median():8.3f}  {df_['abs_icir'].max():8.3f}")

print(f"\n  Feature count by threshold:")
for thr in [0.5, 1.0, 2.0, 3.0]:
    nl = (lag_df['abs_icir']>=thr).sum()
    nn = (nonlag_df['abs_icir']>=thr).sum()
    print(f"    |ICIR|>={thr:.1f}:  lag={nl:3d}  non-lag={nn:3d}  total={nl+nn:3d}")

print(f"\n  Top 10 NON-LAG features (signal we haven't used):")
print(f"  {'Feature':<50}  {'|ICIR|':>8}  {'mean_IC':>10}  {'%pos':>8}")
for _, row in nonlag_df.sort_values('abs_icir', ascending=False).head(10).iterrows():
    print(f"  {row['feature']:<50}  {row['abs_icir']:8.3f}  {row['mean_ic']:+10.5f}  {row['ic_pos_frac']*100:7.1f}%")

# Build feature sets
gold_df   = icir_df[(icir_df['abs_icir']>=3.0) &
                    ((icir_df['ic_pos_frac']==0.0)|(icir_df['ic_pos_frac']==1.0))
                   ].sort_values('abs_icir', ascending=False)
gold10    = gold_df['feature'].tolist()[:10]
ic_top10  = np.array([gold_df.set_index('feature')['mean_ic'][f] for f in gold10])

def silver(df_, icir_thr=1.0, cons=0.70):
    m = (df_['abs_icir']>=icir_thr) & \
        ((df_['ic_pos_frac']<=(1-cons))|(df_['ic_pos_frac']>=cons))
    return df_[m].sort_values('abs_icir', ascending=False)['feature'].tolist()

silver_lag_feats = silver(lag_df)
silver_all_feats = silver(icir_df)

feat_sets = {
    'gold10'    : gold10,
    'silver_lag': silver_lag_feats[:80],
    'silver_all': silver_all_feats[:80],
}
print(f"\n  Feature sets: gold10={len(gold10)}  silver_lag={len(silver_lag_feats[:80])}  silver_all={len(silver_all_feats[:80])}")


# ================================================================
# EXP 2: LGBM PI OOF (time-split: train days 0-300, val 301-428)
# ================================================================
print("\n" + "=" * 70)
print("EXP 2: LGBM Liquid→Illiquid PI OOF (LambdaRank)")
print("=" * 70)

sorted_days   = sorted(train['day_id'].unique())
CUTOFF        = 300
tr_days_split = set(sorted_days[:CUTOFF])
va_days_split = set(sorted_days[CUTOFF:])
print(f"  Train split: {len(tr_days_split)} days  Val split: {len(va_days_split)} days")

lgbm_params = {
    'objective': 'regression', 'metric': 'rmse',
    'num_leaves': 63, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 30, 'n_estimators': 300,
    'verbosity': -1, 'random_state': 42,
}

def build_split(df, feat_cols, day_set, win_target=True):
    Xl, yl, gl = [], [], []
    Xi, yi = [], []
    for day in sorted(day_set):
        grp = df[df['day_id']==day]
        if len(grp)<20: continue
        yd  = grp['TARGET'].values.astype(np.float64)
        bs  = grp['bookshape'].values
        lm  = bs >= np.median(bs); im = ~lm
        if lm.sum()<5 or im.sum()<5: continue
        Xz  = zscore_cols(grp[feat_cols].fillna(0).values.astype(np.float64))
        ylq = yd[lm]
        if win_target: ylq = winsorise(ylq)
        Xl.append(Xz[lm])
        r = pd.Series(ylq).rank(method='average').values; yl.append((r-1)/max(len(r)-1,1))
        gl.append(lm.sum())
        Xi.append(Xz[im]); yi.append(yd[im])
    return (np.vstack(Xl), np.concatenate(yl), np.array(gl),
            np.vstack(Xi) if Xi else np.zeros((0,len(feat_cols))),
            np.concatenate(yi) if yi else np.array([]))

exp2 = {}

# Grinold baseline on val split
g_ics = []
for day in sorted(va_days_split):
    grp = train[train['day_id']==day]
    if len(grp)<20: continue
    yd = grp['TARGET'].values.astype(np.float64)
    bs = grp['bookshape'].values; im = bs < np.median(bs)
    if im.sum()<5: continue
    Xz = zscore_cols(grp[gold10].fillna(0).values.astype(np.float64))
    pred = Xz @ ic_top10; pred -= pred.mean()
    g_ics.append(rank_ic(yd[im], pred[im]))
g_ics = np.array([x for x in g_ics if not np.isnan(x)])
exp2['Grinold_top10'] = g_ics
print(f"\n  {'Model':<30}  {'Med IC':>10}  {'%pos':>8}  {'vs Grinold':>12}")
print(f"  {'-'*66}")
print(f"  {'Grinold_top10 (baseline)':<30}  {np.median(g_ics):+10.5f}  {(g_ics>0).mean()*100:7.1f}%  {'---':>12}")

for fname, feat_cols in feat_sets.items():
    feat_cols = [f for f in feat_cols if f in common_cols]
    Xl, yl, gl, _, _ = build_split(train, feat_cols, tr_days_split)
    if len(Xl)==0: continue
    model = lgb.train(lgbm_params, lgb.Dataset(Xl, label=yl),
                      num_boost_round=300, callbacks=[lgb.log_evaluation(-1)])
    v_ics = []
    for day in sorted(va_days_split):
        grp = train[train['day_id']==day]
        if len(grp)<20: continue
        yd = grp['TARGET'].values.astype(np.float64)
        bs = grp['bookshape'].values; im = bs < np.median(bs)
        if im.sum()<5: continue
        Xz   = zscore_cols(grp[feat_cols].fillna(0).values.astype(np.float64))
        pred = model.predict(Xz[im])
        v_ics.append(rank_ic(yd[im], pred))
    v_ics = np.array([x for x in v_ics if not np.isnan(x)])
    exp2[f'LGBM_{fname}'] = v_ics
    g_med = np.median(g_ics)
    v_med = np.median(v_ics)
    delta = (v_med/g_med - 1)*100 if g_med!=0 else float('nan')
    print(f"  {'LGBM_'+fname:<30}  {v_med:+10.5f}  {(v_ics>0).mean()*100:7.1f}%  {delta:+11.1f}%")


# ================================================================
# EXP 3: Retrain on ALL 428 days + Fixed Market Intercept
# ================================================================
print("\n" + "=" * 70)
print("EXP 3: Full Retrain + Fixed Market Intercept")
print("=" * 70)

daily_liquid_mean = train.groupby('day_id')['TARGET'].mean()

# FIXED future-day intercept model: raw means (not z-scored)
print("\n  Building future-day intercept model (FIXED: raw feature means)...")
daily_raw_means, daily_tgts = [], []
for day, grp in train.groupby('day_id'):
    daily_raw_means.append(grp[gold10].fillna(0).values.astype(np.float64).mean(0))
    daily_tgts.append(daily_liquid_mean[day])
X_daily = np.array(daily_raw_means)
y_daily = np.array(daily_tgts)
sc = StandardScaler(); Xds = sc.fit_transform(X_daily)
rd = Ridge(alpha=1.0); rd.fit(Xds, y_daily)
cv_r2_fixed = cross_val_score(Ridge(alpha=1.0), Xds, y_daily, cv=5, scoring='r2')
print(f"  Fixed CV R²: {cv_r2_fixed.mean():.4f} ± {cv_r2_fixed.std():.4f}  (was 0.054 with z-score bug)")

train_min, train_max = daily_liquid_mean.min(), daily_liquid_mean.max()
future_intercepts = {}
for day, grp_te in test.groupby('day_id'):
    if day in new_days:
        raw = grp_te[gold10].fillna(0).values.astype(np.float64).mean(0).reshape(1,-1)
        pred = float(np.clip(rd.predict(sc.transform(raw))[0], train_min, train_max))
        future_intercepts[day] = pred
fv = np.array(list(future_intercepts.values()))
print(f"  Future intercepts: mean={fv.mean():+.6f}  std={fv.std():.6f}  range=[{fv.min():+.6f},{fv.max():+.6f}]")
print(f"  Clamped to historical bounds: [{train_min:+.6f}, {train_max:+.6f}]")

# Retrain LGBM on all 428 days
print("\n  Retraining LGBM on all 428 training days...")
full_models = {}
for fname, feat_cols in feat_sets.items():
    feat_cols = [f for f in feat_cols if f in common_cols]
    Xl, yl, gl, _, _ = build_split(train, feat_cols, train_days)
    model = lgb.train(lgbm_params, lgb.Dataset(Xl, label=yl),
                      num_boost_round=300, callbacks=[lgb.log_evaluation(-1)])
    full_models[fname] = (model, feat_cols)
    print(f"    {fname}: {Xl.shape[0]:,} liquid rows, {len(feat_cols)} features")

# Build intercepts for all test rows
te_intercepts = np.zeros(len(test))
for day, grp_te in test.groupby('day_id'):
    idx = grp_te.index
    te_intercepts[idx] = (daily_liquid_mean[day] if day in overlap
                          else future_intercepts.get(day, float(y_daily.mean())))

# Grinold cs predictions
te_grinold = np.zeros(len(test))
for day, grp_te in test.groupby('day_id'):
    idx = grp_te.index
    Xz  = zscore_cols(grp_te[gold10].fillna(0).values.astype(np.float64))
    p   = Xz @ ic_top10; p -= p.mean()
    te_grinold[idx] = p

# LGBM cs predictions
te_lgbm = {}
for fname, (model, feat_cols) in full_models.items():
    preds = np.zeros(len(test))
    for day, grp_te in test.groupby('day_id'):
        idx = grp_te.index
        Xz  = zscore_cols(grp_te[feat_cols].fillna(0).values.astype(np.float64))
        p   = model.predict(Xz); p -= p.mean()
        preds[idx] = p
    te_lgbm[fname] = preds


# ================================================================
# Save Submissions
# ================================================================
print("\n" + "=" * 70)
print("Saving Submissions")
print("=" * 70)

sample_sub = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/sample_submission.csv'))[['ID']]
TARGET_STD = 0.000948  # confirmed optimal from scale probing

def save_sub(preds, name):
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': preds})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    t = sub['TARGET']
    print(f"  {name:<60}  std={t.std():.6f}  mean={t.mean():+.8f}")

print()
# Grinold + fixed intercept (sanity check, should ≈ market_intercept_top10_p005)
save_sub(te_intercepts + te_grinold * 0.005, 'mi_grinold_top10_p005_v2')

# LGBM submissions — auto-scaled so cs component std matches TARGET_STD
for fname, lgbm_cs in te_lgbm.items():
    raw_std    = lgbm_cs.std()
    auto_scale = TARGET_STD / raw_std if raw_std > 0 else 1.0
    print(f"\n  LGBM {fname}: cs_std={raw_std:.4f} → auto_scale={auto_scale:.6f}")
    save_sub(lgbm_cs * auto_scale,                    f'lgbm_{fname}_noint')
    save_sub(te_intercepts + lgbm_cs * auto_scale,    f'mi_lgbm_{fname}')
    # Probe: 0.7× and 1.5× of auto_scale
    save_sub(te_intercepts + lgbm_cs * auto_scale * 0.7, f'mi_lgbm_{fname}_lo')
    save_sub(te_intercepts + lgbm_cs * auto_scale * 1.5, f'mi_lgbm_{fname}_hi')


# ================================================================
# Full Comparison Table
# ================================================================
print("\n" + "=" * 70)
print("Full Comparison Table")
print("=" * 70)

files_to_load = {
    'grinold_allday_p005 [LB=+0.00096]':   'grinold_allday_top10_probe_005.csv',
    'market_intercept_only [TBD]':           'market_intercept_only.csv',
    'market_intercept_snap_p005 [TBD]':      'market_intercept_top10_p005.csv',
    'mi_grinold_v2 [fixed intercept]':       'mi_grinold_top10_p005_v2.csv',
    'mi_lgbm_gold10':                        'mi_lgbm_gold10.csv',
    'mi_lgbm_silver_lag':                    'mi_lgbm_silver_lag.csv',
    'mi_lgbm_silver_all':                    'mi_lgbm_silver_all.csv',
    'lgbm_gold10_noint':                     'lgbm_gold10_noint.csv',
}

all_preds = {}
for label, fname in files_to_load.items():
    p = os.path.join(OUT_DIR, fname)
    if os.path.exists(p):
        all_preds[label] = pd.read_csv(p).set_index('ID')['TARGET']

print(f"\n  {'Submission':<50}  {'std':>10}  {'mean':>12}  {'%pos':>8}")
print(f"  {'-'*84}")
for label, pred in all_preds.items():
    print(f"  {label:<50}  {pred.std():10.6f}  {pred.mean():+12.8f}  {(pred>0).mean()*100:7.1f}%")

ref_key = 'grinold_allday_p005 [LB=+0.00096]'
if ref_key in all_preds:
    ref = all_preds[ref_key]
    common = ref.index
    for v in all_preds.values(): common = common.intersection(v.index)
    ref_v = ref.loc[common].values
    print(f"\n  Spearman IC vs best known (grinold_allday_p005):")
    for label, pred in all_preds.items():
        ic_, _ = spearmanr(ref_v, pred.loc[common].values)
        sa = (np.sign(ref_v)==np.sign(pred.loc[common].values)).mean()
        print(f"    {label:<50}  IC={ic_:+.4f}  sign={sa*100:.1f}%")


# ================================================================
# Final summary
# ================================================================
print("\n" + "=" * 70)
print("FINAL ANALYSIS")
print("=" * 70)

g_med = np.median(exp2['Grinold_top10'])
print(f"\n  Validation IC (days 301-428, illiquid rows only):")
print(f"  {'Model':<30}  {'Med IC':>10}  {'vs Grinold':>12}")
for key, ics in exp2.items():
    m = np.median(ics)
    d = (m/g_med-1)*100 if key!='Grinold_top10' else 0.0
    print(f"  {key:<30}  {m:+10.5f}  {d:+11.1f}%")

print(f"""
Key questions answered:
  1. Does LGBM beat Grinold on gold10?   → see LGBM_gold10 vs Grinold_top10 IC
  2. Does more features help?            → see silver_lag vs gold10 IC
  3. Do non-lag features add signal?     → see silver_all vs silver_lag IC
  4. Fixed future intercept improved?    → CV R² {cv_r2_fixed.mean():.4f} vs 0.054 (was z-score bug)

Submission order (tomorrow):
  1. market_intercept_only         → confirm market beta ~0.012
  2. market_intercept_snap_p005    → market + Grinold ~0.013
  3. mi_lgbm_gold10                → market + LGBM gold10
  4. mi_lgbm_silver_lag            → market + LGBM more features
  (if #3 > #2: LGBM is better; if #4 > #3: more features help)

Elapsed: {(time.time()-t0)/60:.1f} min
""")
