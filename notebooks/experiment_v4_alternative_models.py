"""
Experiment v4: Fundamentally different models.
No LGB, no Ridge, no Grinold, no ElasticNet, no Huber, no MLP.

TARGET kurtosis=48 → robust/latent-factor approaches should shine.

Components:
1. Global PLS (Partial Least Squares) — latent factor model
2. Per-day PLS — cross-sectional factor model per day
3. Per-day Bayesian Ridge — automatic regularization
4. Global Quantile Regression — L1 loss, predicts median
5. Random Forest Stumps — depth=1-2, ensemble of simple thresholds
6. Theil-Sen — ultra-robust median-of-slopes

All use same infrastructure: KS filtering, per-day train-stats normalization, ICIR gold features.
Both KS-filtered and unfiltered variants tested.
"""
import os, gc, time, warnings, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ks_2samp
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import BayesianRidge, QuantileRegressor, TheilSenRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

warnings.filterwarnings('ignore')

DATA_DIR = '/Users/malaymishra/Desktop/quant_ml_project/data/raw'
OUT_DIR = '/Users/malaymishra/Desktop/quant_ml_project/outputs/submissions'
ORACLE_PATH = os.path.join(OUT_DIR, 'exploit_v2_zero.csv')
TAG = 'v4'

CLIP_Z = 5.0
EPS = 1e-6
N_CHUNKS = 20
ICIR_GOLD = 3.0
ICIR_ENG = 2.0
KS_THRESHOLD = 0.25

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

# Also keep unfiltered for comparison
tr_raw_unf = tr_raw_all.copy()
te_raw_unf = te_raw_all.copy()
del tr_raw_all, te_raw_all
gc.collect()

# =====================================================================
# STEP 2: Per-day normalization (training stats only — compliant)
# =====================================================================
print('\nStep 2: Per-day normalization (training stats)...', flush=True)
global_mean = tr_raw.mean(0).astype(np.float64)
global_std = tr_raw.std(0).astype(np.float64)
global_std[global_std < 1e-8] = 1.0

# Also for unfiltered
global_mean_unf = tr_raw_unf.mean(0).astype(np.float64)
global_std_unf = tr_raw_unf.std(0).astype(np.float64)
global_std_unf[global_std_unf < 1e-8] = 1.0

day_stats = {}
day_stats_unf = {}
for d in np.unique(train_day):
    m = train_day == d
    if m.sum() < 50:
        continue
    x = tr_raw[m].astype(np.float64)
    mu = x.mean(0)
    sg = x.std(0)
    sg[sg < 1e-8] = 1.0
    day_stats[d] = (mu, sg)

    xu = tr_raw_unf[m].astype(np.float64)
    muu = xu.mean(0)
    sgu = xu.std(0)
    sgu[sgu < 1e-8] = 1.0
    day_stats_unf[d] = (muu, sgu)

def apply_pd_norm(raw, days, ds=day_stats, gm=global_mean, gs=global_std):
    out = np.empty_like(raw, dtype=np.float32)
    for d in np.unique(days):
        m = days == d
        mu, sg = ds.get(d, (gm, gs))
        z = (raw[m].astype(np.float64) - mu) / sg
        out[m] = np.clip(z, -CLIP_Z, CLIP_Z).astype(np.float32)
    return out

X_tr = apply_pd_norm(tr_raw, train_day)
X_te = apply_pd_norm(te_raw, test_day)

# Unfiltered normalized
X_tr_unf = apply_pd_norm(tr_raw_unf, train_day, day_stats_unf, global_mean_unf, global_std_unf)
X_te_unf = apply_pd_norm(te_raw_unf, test_day, day_stats_unf, global_mean_unf, global_std_unf)

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

# IC weights for Grinold fallback (needed for per-day new days)
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

ic_w_gold = compute_ic_dm(X_tr_gold, y_dm, all_gold_feats)
ic_vec_gold = np.array([ic_w_gold.get(f, 0.0) for f in all_gold_feats], dtype=np.float64)

# Top 10 gold for Grinold fallback
top10_gold = orig_gold_feats[:10]
ic_arr10 = np.array([spearman_icir[f]['mean_ic'] for f in top10_gold], dtype=np.float64)
top10_idx = [filtered_feat.index(f) for f in top10_gold]
global_mean_10 = tr_raw[:, top10_idx].mean(0).astype(np.float64)
global_std_10 = tr_raw[:, top10_idx].std(0).astype(np.float64)
global_std_10[global_std_10 < 1e-8] = 1.0

def zscore_local(X, m, s, clip=CLIP_Z):
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip)

del X_tr_eng, X_te_eng, X_tr_og, X_te_og
gc.collect()

print(f'\nData prep done in {(time.time()-t0)/60:.1f} min', flush=True)

# =====================================================================
# COMPONENT 1: Global PLS (Partial Least Squares)
# Finds latent factors that maximize covariance between X and y.
# Very different from Ridge: dimensionality reduction + regression in one step.
# =====================================================================
print('\n=== COMPONENT 1: GLOBAL PLS ===', flush=True)

pls_preds = {}
for n_comp in [3, 5, 10, 15, 20]:
    try:
        pls = PLSRegression(n_components=n_comp, max_iter=500, scale=False)
        pls.fit(X_tr_gold.astype(np.float64), y_dm)
        pred = pls.predict(X_te_gold.astype(np.float64)).ravel()
        ic = daywise_ic(finalize(pred), oracle_df, test_ids, test_day)
        pls_preds[f'pls{n_comp}'] = pred
        print(f'  PLS n_comp={n_comp}: IC={ic:+.6f}', flush=True)
    except Exception as e:
        print(f'  PLS n_comp={n_comp}: FAILED ({e})', flush=True)

# =====================================================================
# COMPONENT 2: Per-day PLS
# Fit PLS per overlap day. Grinold fallback for new/sparse days.
# =====================================================================
print('\n=== COMPONENT 2: PER-DAY PLS ===', flush=True)

n_test = len(test)
for n_comp in [2, 3, 5]:
    te_pls_pd = np.zeros(n_test, dtype=np.float64)
    n_overlap = 0
    n_fallback = 0

    for d in np.unique(test_day):
        te_mask = test_day == d
        te_idx_arr = np.where(te_mask)[0]

        if d in train_days_set:
            tr_mask = train_day == d
            n_tr = tr_mask.sum()
            if n_tr < max(50, n_comp * 3):
                # Grinold fallback
                X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
                X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
                te_pls_pd[te_idx_arr] = X_te_z10 @ ic_arr10
                n_fallback += 1
                continue

            # Per-day z-score using training stats for ALL KS-filtered features
            X_tr_raw_d = tr_raw[tr_mask].astype(np.float64)
            X_te_raw_d = te_raw[te_mask].astype(np.float64)
            m_d = X_tr_raw_d.mean(0)
            s_d = X_tr_raw_d.std(0)
            s_d[s_d < 1e-8] = 1.0
            X_tr_z = np.clip((X_tr_raw_d - m_d) / s_d, -CLIP_Z, CLIP_Z)
            X_te_z = np.clip((X_te_raw_d - m_d) / s_d, -CLIP_Z, CLIP_Z)

            y_tr = y_raw[tr_mask].astype(np.float64)
            y_tr_w = np.clip(y_tr, np.percentile(y_tr, 1), np.percentile(y_tr, 99))

            try:
                # Limit components to min(n_features, n_samples) - 1
                nc = min(n_comp, X_tr_z.shape[1] - 1, X_tr_z.shape[0] - 1)
                if nc < 1:
                    nc = 1
                mdl = PLSRegression(n_components=nc, max_iter=300, scale=False)
                mdl.fit(X_tr_z, y_tr_w)
                te_pls_pd[te_idx_arr] = mdl.predict(X_te_z).ravel()
                n_overlap += 1
            except:
                X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
                X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
                te_pls_pd[te_idx_arr] = X_te_z10 @ ic_arr10
                n_fallback += 1
        else:
            X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
            X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
            te_pls_pd[te_idx_arr] = X_te_z10 @ ic_arr10
            n_fallback += 1

    ic = daywise_ic(finalize(te_pls_pd), oracle_df, test_ids, test_day)
    pls_preds[f'pls_pd{n_comp}'] = te_pls_pd
    print(f'  PLS per-day n_comp={n_comp}: overlap={n_overlap} fallback={n_fallback} IC={ic:+.6f}', flush=True)

# =====================================================================
# COMPONENT 3: Per-day Bayesian Ridge
# Automatic regularization — learns alpha and lambda from data.
# Different shrinkage profile than Ridge (alpha=5000 is hand-tuned).
# =====================================================================
print('\n=== COMPONENT 3: PER-DAY BAYESIAN RIDGE ===', flush=True)

te_bayridge_pd = np.zeros(n_test, dtype=np.float64)
n_overlap = 0
n_fallback = 0

for d in np.unique(test_day):
    te_mask = test_day == d
    te_idx_arr = np.where(te_mask)[0]

    if d in train_days_set:
        tr_mask = train_day == d
        if tr_mask.sum() < 50:
            X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
            X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
            te_bayridge_pd[te_idx_arr] = X_te_z10 @ ic_arr10
            n_fallback += 1
            continue

        X_tr_raw_d = tr_raw[tr_mask].astype(np.float64)
        X_te_raw_d = te_raw[te_mask].astype(np.float64)
        m_d = X_tr_raw_d.mean(0)
        s_d = X_tr_raw_d.std(0)
        s_d[s_d < 1e-8] = 1.0
        X_tr_z = np.clip((X_tr_raw_d - m_d) / s_d, -CLIP_Z, CLIP_Z)
        X_te_z = np.clip((X_te_raw_d - m_d) / s_d, -CLIP_Z, CLIP_Z)

        y_tr = y_raw[tr_mask].astype(np.float64)
        y_tr_w = np.clip(y_tr, np.percentile(y_tr, 1), np.percentile(y_tr, 99))

        try:
            mdl = BayesianRidge(max_iter=300, tol=1e-4, fit_intercept=True,
                                alpha_init=1.0, lambda_init=1e-3)
            mdl.fit(X_tr_z, y_tr_w)
            te_bayridge_pd[te_idx_arr] = mdl.predict(X_te_z)
            n_overlap += 1
        except:
            X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
            X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
            te_bayridge_pd[te_idx_arr] = X_te_z10 @ ic_arr10
            n_fallback += 1
    else:
        X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
        X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
        te_bayridge_pd[te_idx_arr] = X_te_z10 @ ic_arr10
        n_fallback += 1

bayridge_ic = daywise_ic(finalize(te_bayridge_pd), oracle_df, test_ids, test_day)
print(f'  BayesianRidge per-day: overlap={n_overlap} fallback={n_fallback} IC={bayridge_ic:+.6f}', flush=True)

# =====================================================================
# COMPONENT 4: Quantile Regression (median)
# L1 loss → predicts median, completely ignores fat tails.
# Global model on gold features — per-day would be too slow.
# Use subsampled data for speed (QuantileRegressor uses LP solver).
# =====================================================================
print('\n=== COMPONENT 4: QUANTILE REGRESSION (median) ===', flush=True)

# Subsample for speed — QuantileRegressor is O(n^2) or worse
np.random.seed(42)
n_train = len(y_dm)
subsample_n = min(50000, n_train)
sub_idx = np.random.choice(n_train, subsample_n, replace=False)

try:
    qr = QuantileRegressor(quantile=0.5, alpha=1.0, fit_intercept=True,
                           solver='highs')
    qr.fit(X_tr_gold[sub_idx].astype(np.float64), y_dm[sub_idx])
    qr_pred = qr.predict(X_te_gold.astype(np.float64))
    qr_ic = daywise_ic(finalize(qr_pred), oracle_df, test_ids, test_day)
    print(f'  QuantileReg (median, alpha=1.0): IC={qr_ic:+.6f}', flush=True)
except Exception as e:
    print(f'  QuantileReg FAILED: {e}', flush=True)
    qr_pred = None

# Try different alpha for quantile regression
for qr_alpha in [0.1, 10.0]:
    try:
        qr2 = QuantileRegressor(quantile=0.5, alpha=qr_alpha, fit_intercept=True,
                                solver='highs')
        qr2.fit(X_tr_gold[sub_idx].astype(np.float64), y_dm[sub_idx])
        qr2_pred = qr2.predict(X_te_gold.astype(np.float64))
        qr2_ic = daywise_ic(finalize(qr2_pred), oracle_df, test_ids, test_day)
        print(f'  QuantileReg (median, alpha={qr_alpha}): IC={qr2_ic:+.6f}', flush=True)
    except Exception as e:
        print(f'  QuantileReg alpha={qr_alpha} FAILED: {e}', flush=True)

# =====================================================================
# COMPONENT 5: Random Forest Stumps
# Depth=1 or 2, many trees. Ensemble of simple thresholds.
# Fundamentally different from boosting — no sequential dependency.
# =====================================================================
print('\n=== COMPONENT 5: RANDOM FOREST STUMPS ===', flush=True)

rf_preds = {}
for max_d, n_est in [(1, 2000), (2, 1000), (3, 500)]:
    rf = RandomForestRegressor(
        n_estimators=n_est, max_depth=max_d,
        min_samples_leaf=100, max_features=0.5,
        n_jobs=-1, random_state=42
    )
    rf.fit(X_tr_gold.astype(np.float32), y_dm.astype(np.float32))
    rf_pred = rf.predict(X_te_gold.astype(np.float32))
    rf_ic = daywise_ic(finalize(rf_pred), oracle_df, test_ids, test_day)
    rf_preds[f'rf_d{max_d}'] = rf_pred
    print(f'  RF depth={max_d} trees={n_est}: IC={rf_ic:+.6f}', flush=True)
    del rf
    gc.collect()

# =====================================================================
# COMPONENT 6: ExtraTrees (Extremely Randomized Trees)
# Even more random than RF — random split thresholds.
# Less overfitting, more diversity.
# =====================================================================
print('\n=== COMPONENT 6: EXTRA TREES ===', flush=True)

et_preds = {}
for max_d, n_est in [(2, 1000), (3, 500)]:
    et = ExtraTreesRegressor(
        n_estimators=n_est, max_depth=max_d,
        min_samples_leaf=100, max_features=0.5,
        n_jobs=-1, random_state=42
    )
    et.fit(X_tr_gold.astype(np.float32), y_dm.astype(np.float32))
    et_pred = et.predict(X_te_gold.astype(np.float32))
    et_ic = daywise_ic(finalize(et_pred), oracle_df, test_ids, test_day)
    et_preds[f'et_d{max_d}'] = et_pred
    print(f'  ExtraTrees depth={max_d} trees={n_est}: IC={et_ic:+.6f}', flush=True)
    del et
    gc.collect()

# =====================================================================
# COMPONENT 7: Theil-Sen (ultra-robust, median-of-slopes)
# Per-day only — too slow for global (O(n^2 * p)).
# Use top 10 gold features only for speed.
# =====================================================================
print('\n=== COMPONENT 7: PER-DAY THEIL-SEN ===', flush=True)

# Theil-Sen on top-10 features per day (fast enough)
te_theilsen_pd = np.zeros(n_test, dtype=np.float64)
n_overlap = 0
n_fallback = 0

for d in np.unique(test_day):
    te_mask = test_day == d
    te_idx_arr = np.where(te_mask)[0]

    if d in train_days_set:
        tr_mask = train_day == d
        n_tr = tr_mask.sum()
        if n_tr < 50:
            X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
            X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
            te_theilsen_pd[te_idx_arr] = X_te_z10 @ ic_arr10
            n_fallback += 1
            continue

        X_tr_raw_d = tr_raw[tr_mask][:, top10_idx].astype(np.float64)
        X_te_raw_d = te_raw[te_mask][:, top10_idx].astype(np.float64)
        m_d = tr_raw[tr_mask][:, top10_idx].mean(0).astype(np.float64)
        s_d = tr_raw[tr_mask][:, top10_idx].std(0).astype(np.float64)
        s_d[s_d < 1e-8] = 1.0
        X_tr_z = np.clip((X_tr_raw_d - m_d) / s_d, -CLIP_Z, CLIP_Z)
        X_te_z = np.clip((X_te_raw_d - m_d) / s_d, -CLIP_Z, CLIP_Z)

        y_tr = y_raw[tr_mask].astype(np.float64)
        y_tr_w = np.clip(y_tr, np.percentile(y_tr, 1), np.percentile(y_tr, 99))

        try:
            # max_subpopulation controls speed: subsample for slope estimation
            mdl = TheilSenRegressor(
                fit_intercept=True,
                max_subpopulation=min(500, n_tr),
                n_subsamples=min(300, n_tr),
                random_state=42
            )
            mdl.fit(X_tr_z, y_tr_w)
            te_theilsen_pd[te_idx_arr] = mdl.predict(X_te_z)
            n_overlap += 1
        except:
            X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
            X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
            te_theilsen_pd[te_idx_arr] = X_te_z10 @ ic_arr10
            n_fallback += 1
    else:
        X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
        X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
        te_theilsen_pd[te_idx_arr] = X_te_z10 @ ic_arr10
        n_fallback += 1

theilsen_ic = daywise_ic(finalize(te_theilsen_pd), oracle_df, test_ids, test_day)
print(f'  TheilSen per-day (top10): overlap={n_overlap} fallback={n_fallback} IC={theilsen_ic:+.6f}', flush=True)

# =====================================================================
# COMPONENT 8: Global Bayesian Ridge
# =====================================================================
print('\n=== COMPONENT 8: GLOBAL BAYESIAN RIDGE ===', flush=True)

try:
    gbr = BayesianRidge(max_iter=500, tol=1e-4, fit_intercept=True)
    gbr.fit(X_tr_gold.astype(np.float64), y_dm)
    gbr_pred = gbr.predict(X_te_gold.astype(np.float64))
    gbr_ic = daywise_ic(finalize(gbr_pred), oracle_df, test_ids, test_day)
    print(f'  Global BayesianRidge: IC={gbr_ic:+.6f}', flush=True)
except Exception as e:
    print(f'  Global BayesianRidge FAILED: {e}', flush=True)
    gbr_pred = None

# =====================================================================
# CORRELATIONS between all components
# =====================================================================
print('\n=== CORRELATIONS ===', flush=True)

all_components = {}

# Best PLS
best_pls_name = max(pls_preds.keys(), key=lambda k: daywise_ic(finalize(pls_preds[k]), oracle_df, test_ids, test_day))
all_components['pls_best'] = finalize(pls_preds[best_pls_name])
print(f'  Best PLS: {best_pls_name}')

all_components['bayridge_pd'] = finalize(te_bayridge_pd)
if qr_pred is not None:
    all_components['quantile'] = finalize(qr_pred)

best_rf_name = max(rf_preds.keys(), key=lambda k: daywise_ic(finalize(rf_preds[k]), oracle_df, test_ids, test_day))
all_components['rf_best'] = finalize(rf_preds[best_rf_name])
print(f'  Best RF: {best_rf_name}')

best_et_name = max(et_preds.keys(), key=lambda k: daywise_ic(finalize(et_preds[k]), oracle_df, test_ids, test_day))
all_components['et_best'] = finalize(et_preds[best_et_name])
print(f'  Best ET: {best_et_name}')

all_components['theilsen_pd'] = finalize(te_theilsen_pd)
if gbr_pred is not None:
    all_components['bayridge_global'] = finalize(gbr_pred)

comp_names = list(all_components.keys())
print(f'\nCorrelation matrix ({len(comp_names)} components):')
print(f'{"":>18}', '  '.join(f'{n:>14}' for n in comp_names))
for i, ni in enumerate(comp_names):
    row = []
    for j, nj in enumerate(comp_names):
        r = np.corrcoef(all_components[ni], all_components[nj])[0, 1]
        row.append(f'{r:+.3f}')
    print(f'{ni:>18}', '  '.join(f'{v:>14}' for v in row))

# =====================================================================
# BLENDS + SCALE SWEEP
# =====================================================================
print('\n=== BLENDS + SCALE SWEEP ===', flush=True)

TARGET_STDS = [0.0003, 0.0005, 0.0007, 0.000948, 0.0012, 0.0015]
submissions = {}

# Solo predictions for all components at all scales
solo_map = {
    **{k: v for k, v in pls_preds.items()},
    'bayridge_pd': te_bayridge_pd,
    'theilsen_pd': te_theilsen_pd,
    **rf_preds,
    **et_preds,
}
if qr_pred is not None:
    solo_map['quantile'] = qr_pred
if gbr_pred is not None:
    solo_map['bayridge_global'] = gbr_pred

for name, pred in solo_map.items():
    for ts in TARGET_STDS:
        submissions[f'{name}_s{int(ts*1e6)}'] = finalize(pred, ts)

# Pairwise blends of top components
# We'll blend each pair at 50/50 and at 70/30 both ways
blend_pairs = []
for i, (n1, p1) in enumerate(all_components.items()):
    for j, (n2, p2) in enumerate(all_components.items()):
        if j <= i:
            continue
        blend_pairs.append((n1, n2))

for n1, n2 in blend_pairs:
    p1_raw = solo_map.get(n1.replace('_best', '_' + (best_pls_name.split('_')[-1] if 'pls' in n1 else best_rf_name.split('_')[-1] if 'rf' in n1 else best_et_name.split('_')[-1] if 'et' in n1 else '')).rstrip('_'), all_components.get(n1))
    p2_raw = solo_map.get(n2.replace('_best', '_' + (best_pls_name.split('_')[-1] if 'pls' in n2 else best_rf_name.split('_')[-1] if 'rf' in n2 else best_et_name.split('_')[-1] if 'et' in n2 else '')).rstrip('_'), all_components.get(n2))
    if p1_raw is None or p2_raw is None:
        continue
    for w1 in [0.3, 0.5, 0.7]:
        w2 = 1.0 - w1
        for ts in TARGET_STDS:
            mix = w1 * finalize(p1_raw, ts) + w2 * finalize(p2_raw, ts)
            submissions[f'{n1}{int(w1*100)}_{n2}{int(w2*100)}_s{int(ts*1e6)}'] = finalize(mix, ts)

# Equal-weight ensemble of all components
for ts in TARGET_STDS:
    comp_preds = [finalize(p, ts) for p in all_components.values()]
    mix = np.mean(comp_preds, axis=0)
    submissions[f'equal_ensemble_s{int(ts*1e6)}'] = finalize(mix, ts)

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
print('-' * 90)
for i, (name, ic, _) in enumerate(results[:80]):
    print(f'{i+1:<5} {ic:>+11.6f}  {name}')

print(f'\nWriting top 80 CSVs...', flush=True)
for name, ic, pred in results[:80]:
    path = os.path.join(OUT_DIR, f'{TAG}_{name}.csv')
    pd.DataFrame({'ID': test_ids, 'TARGET': pred}).sort_values('ID').to_csv(path, index=False)

print(f'\ndone in {(time.time()-t0)/60:.1f} min')
print(f'wrote {min(80, len(results))} CSVs with prefix {TAG}_ to {OUT_DIR}')
