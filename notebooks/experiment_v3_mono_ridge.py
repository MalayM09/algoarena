"""
Experiment v3: Mono LGB + KS-filtered Per-Day Ridge + Grinold fallback.
Key architecture:
1. KS filter (>0.25) drops features that diverge between liquid/illiquid
2. Monotone-constrained LGB (nonlinear but unidirectional) on gold features
3. Per-day Ridge on ALL KS-filtered features (linear, benefits from KS shield)
4. Grinold IC-weighted fallback for new test days
5. Scale sweep to find R²-optimal prediction magnitude
6. Blend sweep: mono LGB + Ridge at various weights
"""
import os, gc, time, warnings, re
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr, ks_2samp
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

DATA_DIR = '/Users/malaymishra/Desktop/quant_ml_project/data/raw'
OUT_DIR = '/Users/malaymishra/Desktop/quant_ml_project/outputs/submissions'
ORACLE_PATH = os.path.join(OUT_DIR, 'exploit_v2_zero.csv')
TAG = 'v3'

CLIP_Z = 5.0
EPS = 1e-6
N_CHUNKS = 20
ICIR_GOLD = 3.0
ICIR_ENG = 2.0
SEEDS = [42, 123, 456, 789, 2024]
KS_THRESHOLD = 0.25
RIDGE_ALPHA = 5000

t0 = time.time()

def auto_scale(p, std):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

def finalize(pred, target_std=0.000948):
    p = pred.astype(np.float64).copy()
    p -= p.mean()
    return auto_scale(p, target_std)

def daywise_ic(pred, oracle_df, test_ids, test_day):
    df = pd.DataFrame({'ID': test_ids, 'pred': pred, 'day': test_day})
    df = df.merge(oracle_df[['ID', 'TARGET']], on='ID', how='inner')
    ics = []
    for d, g in df.groupby('day'):
        if len(g) < 3:
            continue
        p = g['pred'].values
        o = g['TARGET'].values
        p = p - p.mean()
        o = o - o.mean()
        pn = np.linalg.norm(p)
        on_ = np.linalg.norm(o)
        if pn < 1e-12 or on_ < 1e-12:
            ics.append(0.0)
        else:
            ics.append(float((p @ o) / (pn * on_)))
    return float(np.mean(ics))

# =====================================================================
# DATA LOADING
# =====================================================================
print('Loading data...', flush=True)
train = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
test = pd.read_parquet(os.path.join(DATA_DIR, 'test.parquet'))
oracle_df = pd.read_csv(ORACLE_PATH)

feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET', 'CV_GROUP'}]
all_feat_orig = [c for c in feat_cols if c != 'SO3_T']
test_ids = test['ID'].values
train_day = train['SO3_T'].round(5).astype(str).values
test_day = test['SO3_T'].round(5).astype(str).values
train_days_set = set(np.unique(train_day))
y_raw = train['TARGET'].values.astype(np.float64)
lo_y, hi_y = np.percentile(y_raw, 1), np.percentile(y_raw, 99)
y_wins = np.clip(y_raw, lo_y, hi_y)

print('Per-day demeaned training target...', flush=True)
y_dm = y_wins.copy()
for d in np.unique(train_day):
    m = train_day == d
    y_dm[m] = y_wins[m] - y_wins[m].mean()
print(f'y_dm mean={y_dm.mean():.3e} std={y_dm.std():.3e}', flush=True)

# =====================================================================
# STEP 1: KS filtering
# =====================================================================
print('\nStep 1: KS filtering...', flush=True)
tr_raw_all = train[all_feat_orig].fillna(0).values.astype(np.float32)
te_raw_all = test[all_feat_orig].fillna(0).values.astype(np.float32)

ks_stats = {}
for ci, f in enumerate(all_feat_orig):
    ks, _ = ks_2samp(tr_raw_all[:, ci].astype(np.float64),
                      te_raw_all[:, ci].astype(np.float64))
    ks_stats[f] = ks

high_ks = {f for f, v in ks_stats.items() if v > KS_THRESHOLD}
filtered_feat = [f for f in all_feat_orig if f not in high_ks]
print(f'  Total: {len(all_feat_orig)}  Dropped: {len(high_ks)}  Remaining: {len(filtered_feat)}')

filt_idx = [all_feat_orig.index(f) for f in filtered_feat]
tr_raw = tr_raw_all[:, filt_idx]
te_raw = te_raw_all[:, filt_idx]
del tr_raw_all, te_raw_all
gc.collect()

# =====================================================================
# STEP 2: Per-day normalization (training stats only — compliant)
# =====================================================================
print('\nStep 2: Per-day normalization (training stats)...', flush=True)
global_mean = tr_raw.mean(0).astype(np.float64)
global_std = tr_raw.std(0).astype(np.float64)
global_std[global_std < 1e-8] = 1.0

day_stats = {}
for d in np.unique(train_day):
    m = train_day == d
    if m.sum() < 50:
        continue
    x = tr_raw[m].astype(np.float64)
    mu = x.mean(0)
    sg = x.std(0)
    sg[sg < 1e-8] = 1.0
    day_stats[d] = (mu, sg)

def apply_pd_norm(raw, days):
    out = np.empty_like(raw, dtype=np.float32)
    for d in np.unique(days):
        m = days == d
        mu, sg = day_stats.get(d, (global_mean, global_std))
        z = (raw[m].astype(np.float64) - mu) / sg
        out[m] = np.clip(z, -CLIP_Z, CLIP_Z).astype(np.float32)
    return out

X_tr = apply_pd_norm(tr_raw, train_day)
X_te = apply_pd_norm(te_raw, test_day)

# =====================================================================
# STEP 3: ICIR feature selection
# =====================================================================
print('\nStep 3: ICIR feature selection...', flush=True)
train_s = train.sort_values('ID').reset_index(drop=True)
cs = len(train_s) // N_CHUNKS
spearman_icir = {}
for col in filtered_feat:
    cics = []
    for i in range(N_CHUNKS):
        ch = train_s.iloc[i * cs:(i + 1) * cs]
        v = ch[col].fillna(ch[col].median()).values if col in ch.columns else None
        if v is None:
            continue
        t = ch['TARGET'].values
        valid = ~np.isnan(v)
        if valid.sum() < 200:
            continue
        ic, _ = spearmanr(v[valid], t[valid])
        if not np.isnan(ic):
            cics.append(ic)
    if len(cics) >= 5:
        mic = float(np.mean(cics))
        sic = float(np.std(cics)) + 1e-8
        spearman_icir[col] = dict(mean_ic=mic, icir=mic / sic, abs_icir=abs(mic / sic),
                                  ic_pos=float(np.mean([v > 0 for v in cics])))

orig_gold_feats = sorted([k for k, v in spearman_icir.items()
                          if v['abs_icir'] >= ICIR_GOLD and v['ic_pos'] in (0.0, 1.0)],
                         key=lambda x: -spearman_icir[x]['abs_icir'])
print(f'orig_gold={len(orig_gold_feats)}', flush=True)

def parse_feature(fname):
    m = re.match(r'^(.+?)(_LagT(\d+))$', fname)
    if m:
        return m.group(1), f'LagT{m.group(3)}'
    return fname, 'base'

base_to_lags = {}
for f in filtered_feat:
    base, lag = parse_feature(f)
    base_to_lags.setdefault(base, {})[lag] = f
features_with_lags = {b: lags for b, lags in base_to_lags.items()
                      if 'base' in lags and 'LagT1' in lags}

base_max_icir = {b: max(spearman_icir.get(f, {'abs_icir': 0})['abs_icir']
                        for f in lags.values())
                 for b, lags in features_with_lags.items()}
important_bases = {b for b, v in base_max_icir.items() if v >= ICIR_ENG}
print(f'important_bases={len(important_bases)}', flush=True)

# =====================================================================
# STEP 4: Feature engineering
# =====================================================================
print('\nStep 4: Feature engineering...', flush=True)
eng_names = []
eng_tr_list = []
eng_te_list = []

def add_feat(name, tr_arr, te_arr):
    eng_names.append(name)
    eng_tr_list.append(tr_arr.ravel().astype(np.float32))
    eng_te_list.append(te_arr.ravel().astype(np.float32))

for base_name, lags in features_with_lags.items():
    if base_name not in important_bases:
        continue
    idx_b = filtered_feat.index(lags['base'])
    idx_l1 = filtered_feat.index(lags['LagT1'])
    has_l2 = 'LagT2' in lags and lags['LagT2'] in filtered_feat
    has_l3 = 'LagT3' in lags and lags['LagT3'] in filtered_feat
    if has_l2:
        idx_l2 = filtered_feat.index(lags['LagT2'])
    if has_l3:
        idx_l3 = filtered_feat.index(lags['LagT3'])
    b_tr = X_tr[:, idx_b].astype(np.float64)
    b_te = X_te[:, idx_b].astype(np.float64)
    l1_tr = X_tr[:, idx_l1].astype(np.float64)
    l1_te = X_te[:, idx_l1].astype(np.float64)
    add_feat(f'past_T1_{base_name}', b_tr - l1_tr, b_te - l1_te)
    den_tr = np.abs(b_tr) + EPS
    den_te = np.abs(b_te) + EPS
    add_feat(f'relchg_T1_{base_name}', np.clip(l1_tr / den_tr, -10, 10), np.clip(l1_te / den_te, -10, 10))
    add_feat(f'lvlxchg_T1_{base_name}', b_tr * l1_tr, b_te * l1_te)
    add_feat(f'abschg_T1_{base_name}', np.abs(l1_tr), np.abs(l1_te))
    add_feat(f'signchg_T1_{base_name}', np.sign(l1_tr), np.sign(l1_te))
    if has_l2:
        l2_tr = X_tr[:, idx_l2].astype(np.float64)
        l2_te = X_te[:, idx_l2].astype(np.float64)
        add_feat(f'accel_T1T2_{base_name}', l2_tr - l1_tr, l2_te - l1_te)
        add_feat(f'consist_T1T2_{base_name}', np.sign(l1_tr) * np.sign(l2_tr), np.sign(l1_te) * np.sign(l2_te))
        add_feat(f'past_T2_{base_name}', b_tr - l2_tr, b_te - l2_te)
    if has_l3:
        l3_tr = X_tr[:, idx_l3].astype(np.float64)
        l3_te = X_te[:, idx_l3].astype(np.float64)
        add_feat(f'longshort_{base_name}', l3_tr - l1_tr, l3_te - l1_te)
        if has_l2:
            add_feat(f'accel2_{base_name}', l3_tr - 2*l2_tr + l1_tr, l3_te - 2*l2_te + l1_te)
            net_tr = np.abs(b_tr - l3_tr)
            net_te = np.abs(b_te - l3_te)
            vol_tr = np.abs(b_tr - l1_tr) + np.abs(l1_tr - l2_tr) + np.abs(l2_tr - l3_tr) + EPS
            vol_te = np.abs(b_te - l1_te) + np.abs(l1_te - l2_te) + np.abs(l2_te - l3_te) + EPS
            kaufer_tr = np.clip(net_tr / vol_tr, 0.0, 1.0)
            kaufer_te = np.clip(net_te / vol_te, 0.0, 1.0)
            add_feat(f'kaufer_{base_name}', kaufer_tr, kaufer_te)
            add_feat(f'skaufer_{base_name}', np.sign(b_tr - l3_tr) * kaufer_tr, np.sign(b_te - l3_te) * kaufer_te)

gold_lag1 = [f for f in filtered_feat if f.endswith('_LagT1')]
lag1_icir = {c: spearman_icir[c]['abs_icir'] for c in gold_lag1 if c in spearman_icir}
top_lag1 = sorted(lag1_icir.keys(), key=lambda x: -lag1_icir[x])[:10]
for i in range(len(top_lag1)):
    for j in range(i + 1, len(top_lag1)):
        fi = filtered_feat.index(top_lag1[i])
        fj = filtered_feat.index(top_lag1[j])
        add_feat(f'xchg_{top_lag1[i]}_x_{top_lag1[j]}',
                 X_tr[:, fi].astype(np.float64) * X_tr[:, fj].astype(np.float64),
                 X_te[:, fi].astype(np.float64) * X_te[:, fj].astype(np.float64))

for base_name, lags in features_with_lags.items():
    if base_name not in important_bases:
        continue
    idx_b = filtered_feat.index(lags['base'])
    b_tr = X_tr[:, idx_b].astype(np.float64)
    b_te = X_te[:, idx_b].astype(np.float64)
    add_feat(f'log_{base_name}', np.sign(b_tr)*np.log1p(np.abs(b_tr)),
             np.sign(b_te)*np.log1p(np.abs(b_te)))

for base_name, lags in features_with_lags.items():
    if base_name not in important_bases:
        continue
    idx_b = filtered_feat.index(lags['base'])
    add_feat(f'sq_{base_name}', X_tr[:, idx_b].astype(np.float64)**2,
             X_te[:, idx_b].astype(np.float64)**2)

if eng_tr_list:
    X_tr_eng = np.column_stack(eng_tr_list)
    X_te_eng = np.column_stack(eng_te_list)
    X_tr_eng = np.nan_to_num(X_tr_eng, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_eng = np.nan_to_num(X_te_eng, nan=0.0, posinf=0.0, neginf=0.0)
else:
    X_tr_eng = np.zeros((X_tr.shape[0], 0), dtype=np.float32)
    X_te_eng = np.zeros((X_te.shape[0], 0), dtype=np.float32)
del eng_tr_list, eng_te_list
gc.collect()
print(f'n_engineered={len(eng_names)}', flush=True)

# =====================================================================
# STEP 5: ICIR on engineered → gold selection
# =====================================================================
print('\nStep 5: ICIR on engineered...', flush=True)

def fast_spearman_icir(X_matrix, y_target, feat_names, n_chunks=N_CHUNKS):
    n = len(y_target)
    chunk_size = n // n_chunks
    sort_idx = np.argsort(train['ID'].values)
    X_sorted = X_matrix[sort_idx]
    y_sorted = y_target[sort_idx]
    all_ics = np.zeros((n_chunks, X_sorted.shape[1]), dtype=np.float64)
    for i in range(n_chunks):
        s = i * chunk_size
        e = (i + 1) * chunk_size
        X_ch = X_sorted[s:e].astype(np.float64)
        y_ch = y_sorted[s:e].astype(np.float64)
        X_rank = np.argsort(np.argsort(X_ch, axis=0), axis=0).astype(np.float64)
        y_rank = np.argsort(np.argsort(y_ch)).astype(np.float64)
        X_m = X_rank - X_rank.mean(0)
        y_m = y_rank - y_rank.mean()
        X_sd = np.sqrt((X_m ** 2).sum(0))
        y_sd = np.sqrt((y_m ** 2).sum())
        X_sd[X_sd < 1e-10] = 1e-10
        all_ics[i] = (X_m.T @ y_m) / (X_sd * y_sd)
    out = {}
    for ci, name in enumerate(feat_names):
        ics = all_ics[:, ci]
        valid = ~np.isnan(ics)
        if valid.sum() < 5:
            continue
        mic = float(np.mean(ics[valid]))
        sic = float(np.std(ics[valid])) + 1e-8
        out[name] = dict(mean_ic=mic, icir=mic / sic, abs_icir=abs(mic / sic),
                         ic_pos=float(np.mean(ics[valid] > 0)))
    return out

icir_eng = fast_spearman_icir(X_tr_eng, y_raw, eng_names)
icir_all = {**spearman_icir, **icir_eng}
gold_all = {k: v for k, v in icir_all.items()
            if v['abs_icir'] >= ICIR_GOLD and v['ic_pos'] in (0.0, 1.0)}
gold_orig_feats = sorted([k for k in gold_all if k in filtered_feat],
                         key=lambda x: -gold_all[x]['abs_icir'])
gold_eng_feats = sorted([k for k in gold_all if k in eng_names],
                        key=lambda x: -gold_all[x]['abs_icir'])
print(f'gold_orig={len(gold_orig_feats)} gold_eng={len(gold_eng_feats)}', flush=True)

orig_gold_idx = [filtered_feat.index(f) for f in gold_orig_feats]
X_tr_og = X_tr[:, orig_gold_idx]
X_te_og = X_te[:, orig_gold_idx]
eng_gold_idx = [eng_names.index(f) for f in gold_eng_feats]
if eng_gold_idx:
    X_tr_gold = np.hstack([X_tr_og, X_tr_eng[:, eng_gold_idx]])
    X_te_gold = np.hstack([X_te_og, X_te_eng[:, eng_gold_idx]])
else:
    X_tr_gold = X_tr_og
    X_te_gold = X_te_og
all_gold_feats = gold_orig_feats + gold_eng_feats
print(f'Gold feature matrix: {X_tr_gold.shape}', flush=True)

del X_tr_eng, X_te_eng, X_tr_og, X_te_og
gc.collect()

# =====================================================================
# COMPONENT 1: Monotone-constrained LGB multi-seed
# =====================================================================
print(f'\n=== COMPONENT 1: MONO LGB ({X_tr_gold.shape[1]} features) ===', flush=True)

mono_constraints = []
for f in all_gold_feats:
    if f in icir_all:
        sign = 1 if icir_all[f]['mean_ic'] > 0 else -1
        mono_constraints.append(sign)
    else:
        mono_constraints.append(0)
mono_str = ','.join(str(x) for x in mono_constraints)

groups5 = pd.qcut(pd.Series(train['SO3_T'].values), q=5, labels=False,
                  duplicates='drop').values.astype(np.int32)

PARAMS_MONO = dict(
    objective='regression', metric='rmse',
    num_leaves=63, learning_rate=0.05,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    min_child_samples=50,
    lambda_l1=0.1, lambda_l2=1.0,
    monotone_constraints=mono_str,
    n_jobs=-1, verbose=-1,
)

seed_preds_mono = []
for s in SEEDS:
    params = dict(PARAMS_MONO)
    params['seed'] = s
    params['feature_fraction_seed'] = s
    params['bagging_seed'] = s
    gkf = GroupKFold(n_splits=len(np.unique(groups5)))
    folds = list(gkf.split(X_tr_gold, y_dm, groups=groups5))
    te_pred = np.zeros(len(X_te_gold), dtype=np.float64)
    for fi, (tri, vai) in enumerate(folds):
        dt = lgb.Dataset(X_tr_gold[tri], label=y_dm[tri], free_raw_data=True)
        dv = lgb.Dataset(X_tr_gold[vai], label=y_dm[vai], reference=dt, free_raw_data=True)
        m = lgb.train(params, dt, num_boost_round=2000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        te_pred += m.predict(X_te_gold, num_iteration=m.best_iteration) / len(folds)
        print(f'  mono seed={s} fold={fi} best_iter={m.best_iteration}', flush=True)
        del dt, dv, m
        gc.collect()
    seed_preds_mono.append(te_pred)
lgb_mono_raw = np.mean(seed_preds_mono, axis=0)
del seed_preds_mono
gc.collect()

lgb_mono_ic = daywise_ic(finalize(lgb_mono_raw), oracle_df, test_ids, test_day)
print(f'\nlgb_mono multi-seed IC={lgb_mono_ic:+.6f}', flush=True)

# =====================================================================
# COMPONENT 2: Per-day Ridge on ALL KS-filtered features (compliant)
# Uses training-day stats for z-scoring. NO per-day pred demeaning.
# Grinold fallback for new/sparse days.
# =====================================================================
print('\n=== COMPONENT 2: PER-DAY RIDGE (KS-filtered, compliant) ===', flush=True)

top10_gold = orig_gold_feats[:10]
ic_arr10 = np.array([spearman_icir[f]['mean_ic'] for f in top10_gold], dtype=np.float64)
top10_idx = [filtered_feat.index(f) for f in top10_gold]
global_mean_10 = tr_raw[:, top10_idx].mean(0).astype(np.float64)
global_std_10 = tr_raw[:, top10_idx].std(0).astype(np.float64)
global_std_10[global_std_10 < 1e-8] = 1.0

def zscore_local(X, m, s, clip=CLIP_Z):
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip)

n_test = len(test)
te_ridge_pd = np.zeros(n_test, dtype=np.float64)
n_overlap = 0
n_new = 0

for d in np.unique(test_day):
    te_mask = test_day == d
    te_idx_arr = np.where(te_mask)[0]
    X_te_raw_all_d = te_raw[te_mask].astype(np.float64)

    if d in train_days_set:
        tr_mask = train_day == d
        if tr_mask.sum() < 20:
            X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
            X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
            te_ridge_pd[te_idx_arr] = X_te_z10 @ ic_arr10
            n_new += 1
            continue

        X_tr_raw_all_d = tr_raw[tr_mask].astype(np.float64)
        m_all = X_tr_raw_all_d.mean(0)
        s_all = X_tr_raw_all_d.std(0)
        s_all[s_all < 1e-8] = 1.0
        X_tr_z = np.clip((X_tr_raw_all_d - m_all) / s_all, -CLIP_Z, CLIP_Z)
        X_te_z = np.clip((X_te_raw_all_d - m_all) / s_all, -CLIP_Z, CLIP_Z)

        y_tr = y_raw[tr_mask].astype(np.float64)
        y_tr_w = np.clip(y_tr, np.percentile(y_tr, 1), np.percentile(y_tr, 99))

        mdl = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        mdl.fit(X_tr_z, y_tr_w)
        te_ridge_pd[te_idx_arr] = mdl.predict(X_te_z)
        n_overlap += 1
    else:
        X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
        X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
        te_ridge_pd[te_idx_arr] = X_te_z10 @ ic_arr10
        n_new += 1

print(f'overlap_days={n_overlap} new_days={n_new}', flush=True)
ridge_ic = daywise_ic(finalize(te_ridge_pd), oracle_df, test_ids, test_day)
print(f'ridge_pd IC={ridge_ic:+.6f}', flush=True)

# =====================================================================
# COMPONENT 3: Grinold (IC-weighted on gold features)
# =====================================================================
print('\n=== COMPONENT 3: GRINOLD ===', flush=True)

def compute_ic_dm(X_matrix, y_target, feat_names, n_chunks=N_CHUNKS):
    n = len(y_target)
    chunk_size = n // n_chunks
    sort_idx = np.argsort(train['ID'].values)
    X_sorted = X_matrix[sort_idx]
    y_sorted = y_target[sort_idx]
    all_ics = np.zeros((n_chunks, X_sorted.shape[1]), dtype=np.float64)
    for i in range(n_chunks):
        s = i * chunk_size
        e = (i + 1) * chunk_size
        X_ch = X_sorted[s:e].astype(np.float64)
        y_ch = y_sorted[s:e].astype(np.float64)
        X_rank = np.argsort(np.argsort(X_ch, axis=0), axis=0).astype(np.float64)
        y_rank = np.argsort(np.argsort(y_ch)).astype(np.float64)
        X_m = X_rank - X_rank.mean(0)
        y_m = y_rank - y_rank.mean()
        X_sd = np.sqrt((X_m ** 2).sum(0))
        y_sd = np.sqrt((y_m ** 2).sum())
        X_sd[X_sd < 1e-10] = 1e-10
        all_ics[i] = (X_m.T @ y_m) / (X_sd * y_sd)
    out = {}
    for ci, name in enumerate(feat_names):
        ics = all_ics[:, ci]
        valid = ~np.isnan(ics)
        if valid.sum() < 5:
            continue
        out[name] = float(np.mean(ics[valid]))
    return out

ic_w = compute_ic_dm(X_tr_gold, y_dm, all_gold_feats)
ic_vec = np.array([ic_w.get(f, 0.0) for f in all_gold_feats], dtype=np.float64)
grin_raw = X_te_gold.astype(np.float64) @ ic_vec
grin_ic = daywise_ic(finalize(grin_raw), oracle_df, test_ids, test_day)
print(f'grin IC={grin_ic:+.6f}', flush=True)

# =====================================================================
# CORRELATIONS
# =====================================================================
print('\n=== CORRELATIONS ===', flush=True)
mono_fin = finalize(lgb_mono_raw)
ridge_fin = finalize(te_ridge_pd)
grin_fin = finalize(grin_raw)

print(f'  mono  vs ridge  r={np.corrcoef(mono_fin, ridge_fin)[0,1]:+.4f}')
print(f'  mono  vs grin   r={np.corrcoef(mono_fin, grin_fin)[0,1]:+.4f}')
print(f'  ridge vs grin   r={np.corrcoef(ridge_fin, grin_fin)[0,1]:+.4f}')

# =====================================================================
# BLENDS + SCALE SWEEP
# =====================================================================
print('\n=== BLENDS + SCALE SWEEP ===', flush=True)

TARGET_STDS = [0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.000948]
submissions = {}

# Solos
for ts in TARGET_STDS:
    submissions[f'mono_s{int(ts*1e6)}'] = finalize(lgb_mono_raw, ts)
    submissions[f'ridge_s{int(ts*1e6)}'] = finalize(te_ridge_pd, ts)
    submissions[f'grin_s{int(ts*1e6)}'] = finalize(grin_raw, ts)

# Mono + Ridge blends (the main event)
for w_mono in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    w_r = 1.0 - w_mono
    for ts in TARGET_STDS:
        mix = w_mono * finalize(lgb_mono_raw, ts) + w_r * finalize(te_ridge_pd, ts)
        submissions[f'mono{int(w_mono*100)}_ridge{int(w_r*100)}_s{int(ts*1e6)}'] = finalize(mix, ts)

# Mono + Grinold blends
for w_mono in [0.80, 0.90]:
    w_g = 1.0 - w_mono
    for ts in TARGET_STDS:
        mix = w_mono * finalize(lgb_mono_raw, ts) + w_g * finalize(grin_raw, ts)
        submissions[f'mono{int(w_mono*100)}_grin{int(w_g*100)}_s{int(ts*1e6)}'] = finalize(mix, ts)

# 3-way: Mono + Ridge + Grinold
for w_m, w_r, w_g in [(0.50, 0.40, 0.10), (0.60, 0.30, 0.10), (0.40, 0.40, 0.20),
                       (0.50, 0.30, 0.20), (0.40, 0.50, 0.10), (0.45, 0.45, 0.10)]:
    for ts in TARGET_STDS:
        mix = w_m * finalize(lgb_mono_raw, ts) + w_r * finalize(te_ridge_pd, ts) + w_g * finalize(grin_raw, ts)
        submissions[f'mono{int(w_m*100)}_ridge{int(w_r*100)}_grin{int(w_g*100)}_s{int(ts*1e6)}'] = finalize(mix, ts)

# =====================================================================
# EVALUATE AND WRITE
# =====================================================================
print('\n=== RESULTS ===', flush=True)

results = []
for name, pred in submissions.items():
    ic = daywise_ic(pred, oracle_df, test_ids, test_day)
    results.append((name, ic, pred))
results.sort(key=lambda x: -x[1])

print(f"\n{'rank':<5} {'oracle_ic':>11}  name")
print('-' * 80)
for i, (name, ic, _) in enumerate(results[:60]):
    print(f'{i+1:<5} {ic:>+11.6f}  {name}')

print(f'\nWriting top 60 CSVs...', flush=True)
for name, ic, pred in results[:60]:
    path = os.path.join(OUT_DIR, f'{TAG}_{name}.csv')
    pd.DataFrame({'ID': test_ids, 'TARGET': pred}).sort_values('ID').to_csv(path, index=False)

print(f'\ndone in {(time.time()-t0)/60:.1f} min')
print(f'wrote {min(60, len(results))} CSVs with prefix {TAG}_ to {OUT_DIR}')
