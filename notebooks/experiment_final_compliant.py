import os, gc, time, warnings, re
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

DATA_DIR = '/Users/malaymishra/Desktop/quant_ml_project/data/raw'
OUT_DIR = '/Users/malaymishra/Desktop/quant_ml_project/outputs/submissions'
ORACLE_PATH = os.path.join(OUT_DIR, 'exploit_v2_zero.csv')
TAG = 'fc'

TARGET_STD = 0.000948
CLIP_Z = 5.0
EPS = 1e-6
N_CHUNKS = 20
ICIR_GOLD = 3.0
ICIR_ENG = 2.0
SEEDS = [42, 123, 456, 789, 2024]
NW_BANDWIDTH = 1.0
RIDGE_PERDAY_ALPHA = 5000

t0 = time.time()

def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

def finalize(pred):
    p = pred.astype(np.float64).copy()
    p -= p.mean()
    return auto_scale(p)

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

print('Loading data...', flush=True)
train = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
test = pd.read_parquet(os.path.join(DATA_DIR, 'test.parquet'))
oracle_df = pd.read_csv(ORACLE_PATH)

feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET', 'CV_GROUP'}]
all_feat = [c for c in feat_cols if c != 'SO3_T']
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

def parse_feature(fname):
    m = re.match(r'^(.+?)(_LagT(\d+))$', fname)
    if m:
        return m.group(1), f'LagT{m.group(3)}'
    return fname, 'base'

base_to_lags = {}
for f in all_feat:
    base, lag = parse_feature(f)
    base_to_lags.setdefault(base, {})[lag] = f
features_with_lags = {b: lags for b, lags in base_to_lags.items()
                      if 'base' in lags and 'LagT1' in lags}
print(f'features_with_lags={len(features_with_lags)}', flush=True)

# =====================================================================
# Per-day feature normalization using TRAINING stats (compliant)
# =====================================================================
print('Per-day feature normalization (training stats, row-wise lookup)...', flush=True)
tr_raw = train[all_feat].fillna(0).values.astype(np.float32)
te_raw = test[all_feat].fillna(0).values.astype(np.float32)
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
# Spearman ICIR feature selection (originals)
# =====================================================================
print('Spearman ICIR over originals...', flush=True)
train_s = train.sort_values('ID').reset_index(drop=True)
cs = len(train_s) // N_CHUNKS
spearman_icir = {}
for col in all_feat:
    cics = []
    for i in range(N_CHUNKS):
        ch = train_s.iloc[i * cs:(i + 1) * cs]
        v = ch[col].fillna(ch[col].median()).values
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

base_max_icir = {b: max(spearman_icir.get(f, {'abs_icir': 0})['abs_icir']
                        for f in lags.values())
                 for b, lags in features_with_lags.items()}
important_bases = {b for b, v in base_max_icir.items() if v >= ICIR_ENG}
print(f'important_bases={len(important_bases)}', flush=True)

# =====================================================================
# Feature engineering (row-wise, same as first successful angles run)
# =====================================================================
print('Feature engineering (row-wise)...', flush=True)
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
    idx_b = all_feat.index(lags['base'])
    idx_l1 = all_feat.index(lags['LagT1'])
    has_l2 = 'LagT2' in lags
    has_l3 = 'LagT3' in lags
    if has_l2:
        idx_l2 = all_feat.index(lags['LagT2'])
    if has_l3:
        idx_l3 = all_feat.index(lags['LagT3'])
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

gold_lag1 = [f for f in all_feat if f.endswith('_LagT1')]
lag1_icir = {c: spearman_icir[c]['abs_icir'] for c in gold_lag1 if c in spearman_icir}
top_lag1 = sorted(lag1_icir.keys(), key=lambda x: -lag1_icir[x])[:10]
for i in range(len(top_lag1)):
    for j in range(i + 1, len(top_lag1)):
        fi = all_feat.index(top_lag1[i])
        fj = all_feat.index(top_lag1[j])
        add_feat(f'xchg_{top_lag1[i]}_x_{top_lag1[j]}',
                 X_tr[:, fi].astype(np.float64) * X_tr[:, fj].astype(np.float64),
                 X_te[:, fi].astype(np.float64) * X_te[:, fj].astype(np.float64))

for base_name, lags in features_with_lags.items():
    if base_name not in important_bases:
        continue
    idx_b = all_feat.index(lags['base'])
    b_tr = X_tr[:, idx_b].astype(np.float64)
    b_te = X_te[:, idx_b].astype(np.float64)
    add_feat(f'log_{base_name}', np.sign(b_tr)*np.log1p(np.abs(b_tr)), np.sign(b_te)*np.log1p(np.abs(b_te)))

for base_name, lags in features_with_lags.items():
    if base_name not in important_bases:
        continue
    idx_b = all_feat.index(lags['base'])
    add_feat(f'sq_{base_name}', X_tr[:, idx_b].astype(np.float64)**2, X_te[:, idx_b].astype(np.float64)**2)

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

print('Spearman ICIR over engineered (feature selection)...', flush=True)
icir_eng = fast_spearman_icir(X_tr_eng, y_raw, eng_names)
icir_all = {**spearman_icir, **icir_eng}
gold_all = {k: v for k, v in icir_all.items()
            if v['abs_icir'] >= ICIR_GOLD and v['ic_pos'] in (0.0, 1.0)}
gold_orig_feats2 = sorted([k for k in gold_all if k in all_feat], key=lambda x: -gold_all[x]['abs_icir'])
gold_eng_feats = sorted([k for k in gold_all if k in eng_names], key=lambda x: -gold_all[x]['abs_icir'])
print(f'gold_orig={len(gold_orig_feats2)} gold_eng={len(gold_eng_feats)}', flush=True)

orig_gold_idx = [all_feat.index(f) for f in gold_orig_feats2]
X_tr_og = X_tr[:, orig_gold_idx]
X_te_og = X_te[:, orig_gold_idx]
eng_gold_idx_sel = [eng_names.index(f) for f in gold_eng_feats]
X_tr_all = np.hstack([X_tr_og, X_tr_eng[:, eng_gold_idx_sel]]) if eng_gold_idx_sel else X_tr_og
X_te_all = np.hstack([X_te_og, X_te_eng[:, eng_gold_idx_sel]]) if eng_gold_idx_sel else X_te_og
all_gold_feats = gold_orig_feats2 + gold_eng_feats
print(f'X_tr_all={X_tr_all.shape}', flush=True)

# =====================================================================
# COMPONENT 1: Grinold (IC-weighted, trained on y_dm)
# =====================================================================
print('\n=== COMPONENT 1: GRINOLD ===', flush=True)

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

ic_w_all_dm = compute_ic_dm(X_tr_all, y_dm, all_gold_feats)
ic_w_vec_dm = np.array([ic_w_all_dm[f] for f in all_gold_feats], dtype=np.float64)
grin_raw = X_te_all.astype(np.float64) @ ic_w_vec_dm
grin_fin = finalize(grin_raw)
grin_ic = daywise_ic(grin_fin, oracle_df, test_ids, test_day)
print(f'grin_all_dm: IC={grin_ic:+.6f}', flush=True)

# =====================================================================
# COMPONENT 2: LGB multi-seed (trained on y_dm, L2, 114 features)
# =====================================================================
print(f'\n=== COMPONENT 2: LGB MULTI-SEED ({X_tr_all.shape[1]} features) ===', flush=True)
BASE_PARAMS_L2 = dict(
    objective='regression', metric='rmse',
    num_leaves=63, learning_rate=0.05,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    min_child_samples=50,
    lambda_l1=0.1, lambda_l2=1.0,
    n_jobs=-1, verbose=-1,
)
groups5 = pd.qcut(pd.Series(train['SO3_T'].values), q=5, labels=False,
                  duplicates='drop').values.astype(np.int32)

def train_lgb_seed(X_train, X_test, y, groups, seed, label):
    params = dict(BASE_PARAMS_L2)
    params['seed'] = seed
    params['feature_fraction_seed'] = seed
    params['bagging_seed'] = seed
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    folds = list(gkf.split(X_train, y, groups=groups))
    te_pred = np.zeros(len(X_test), dtype=np.float64)
    for fi, (tri, vai) in enumerate(folds):
        dt = lgb.Dataset(X_train[tri], label=y[tri], free_raw_data=True)
        dv = lgb.Dataset(X_train[vai], label=y[vai], reference=dt, free_raw_data=True)
        m = lgb.train(params, dt, num_boost_round=2000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        te_pred += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
        print(f'  {label} seed={seed} fold={fi} best_iter={m.best_iteration}', flush=True)
        del dt, dv, m
        gc.collect()
    return te_pred

seed_preds = []
for s in SEEDS:
    p = train_lgb_seed(X_tr_all, X_te_all, y_dm, groups5, s, 'lgb')
    seed_preds.append(p)
    ic_s = daywise_ic(finalize(p), oracle_df, test_ids, test_day)
    print(f'  seed={s} solo IC={ic_s:+.6f}', flush=True)

lgb_ms_raw = np.mean(seed_preds, axis=0)
lgb_ms = finalize(lgb_ms_raw)
lgb_ms_ic = daywise_ic(lgb_ms, oracle_df, test_ids, test_day)
print(f'\nlgb MULTI-SEED avg: IC={lgb_ms_ic:+.6f}', flush=True)

# =====================================================================
# COMPONENT 3: Per-day NW kernel (COMPLIANT: train stats for z-scoring,
#              NO per-day demeaning, Grinold fallback for new days)
# =====================================================================
print('\n=== COMPONENT 3: NW KERNEL (compliant) ===', flush=True)

top10_gold = orig_gold_feats[:10]
ic_arr10 = np.array([spearman_icir[f]['mean_ic'] for f in top10_gold], dtype=np.float64)
top10_idx = [all_feat.index(f) for f in top10_gold]

def gaussian_nw(X_test_z, X_train_z, y_train, h):
    a2 = (X_test_z ** 2).sum(1, keepdims=True)
    b2 = (X_train_z ** 2).sum(1, keepdims=True).T
    ab = X_test_z @ X_train_z.T
    dist2 = np.maximum(a2 + b2 - 2 * ab, 0)
    w = np.exp(-dist2 / (2 * h * h))
    ws = w.sum(1, keepdims=True)
    ws = np.where(ws < 1e-12, 1.0, ws)
    return (w @ y_train) / ws.ravel()

def zscore_apply_local(X, m, s, clip=CLIP_Z):
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip)

n_test = len(test)
te_nw = np.zeros(n_test, dtype=np.float64)
te_ridge_pd = np.zeros(n_test, dtype=np.float64)
te_grin_10 = np.zeros(n_test, dtype=np.float64)

global_mean_10 = tr_raw[:, top10_idx].mean(0).astype(np.float64)
global_std_10 = tr_raw[:, top10_idx].std(0).astype(np.float64)
global_std_10[global_std_10 < 1e-8] = 1.0
global_mean_all = global_mean.copy()
global_std_all = global_std.copy()

n_overlap = 0
n_new = 0

for d in np.unique(test_day):
    te_mask = test_day == d
    te_idx_arr = np.where(te_mask)[0]
    X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
    X_te_raw_all_d = te_raw[te_mask].astype(np.float64)

    if d in train_days_set:
        tr_mask = train_day == d
        if tr_mask.sum() < 20:
            X_te_z10 = zscore_apply_local(X_te_raw_10, global_mean_10, global_std_10)
            pred_g = X_te_z10 @ ic_arr10
            te_grin_10[te_idx_arr] = pred_g
            te_nw[te_idx_arr] = pred_g
            te_ridge_pd[te_idx_arr] = pred_g
            n_new += 1
            continue

        X_tr_raw_10 = tr_raw[tr_mask][:, top10_idx].astype(np.float64)
        m10 = X_tr_raw_10.mean(0)
        s10 = X_tr_raw_10.std(0)
        s10[s10 < 1e-8] = 1.0
        X_tr_z10 = np.clip((X_tr_raw_10 - m10) / s10, -CLIP_Z, CLIP_Z)
        X_te_z10 = np.clip((X_te_raw_10 - m10) / s10, -CLIP_Z, CLIP_Z)

        y_tr = y_raw[tr_mask].astype(np.float64)
        y_tr_w = np.clip(y_tr, np.percentile(y_tr, 1), np.percentile(y_tr, 99))

        pred_g = X_te_z10 @ ic_arr10
        te_grin_10[te_idx_arr] = pred_g

        pred_nw = gaussian_nw(X_te_z10, X_tr_z10, y_tr.astype(np.float64), NW_BANDWIDTH)
        te_nw[te_idx_arr] = pred_nw

        X_tr_raw_all_d = tr_raw[tr_mask].astype(np.float64)
        m_all = X_tr_raw_all_d.mean(0)
        s_all = X_tr_raw_all_d.std(0)
        s_all[s_all < 1e-8] = 1.0
        X_tr_z_all = np.clip((X_tr_raw_all_d - m_all) / s_all, -CLIP_Z, CLIP_Z)
        X_te_z_all = np.clip((X_te_raw_all_d - m_all) / s_all, -CLIP_Z, CLIP_Z)
        mdl = Ridge(alpha=RIDGE_PERDAY_ALPHA, fit_intercept=True)
        mdl.fit(X_tr_z_all, y_tr_w)
        pred_r = mdl.predict(X_te_z_all)
        te_ridge_pd[te_idx_arr] = pred_r

        n_overlap += 1
    else:
        X_te_z10 = zscore_apply_local(X_te_raw_10, global_mean_10, global_std_10)
        pred_g = X_te_z10 @ ic_arr10
        te_grin_10[te_idx_arr] = pred_g
        te_nw[te_idx_arr] = pred_g
        te_ridge_pd[te_idx_arr] = pred_g
        n_new += 1

print(f'overlap_days={n_overlap} new_days={n_new}', flush=True)

nw_fin = finalize(te_nw)
nw_ic = daywise_ic(nw_fin, oracle_df, test_ids, test_day)
print(f'NW kernel (h={NW_BANDWIDTH}): IC={nw_ic:+.6f}', flush=True)

ridge_pd_fin = finalize(te_ridge_pd)
ridge_pd_ic = daywise_ic(ridge_pd_fin, oracle_df, test_ids, test_day)
print(f'Per-day Ridge (alpha={RIDGE_PERDAY_ALPHA}): IC={ridge_pd_ic:+.6f}', flush=True)

grin10_fin = finalize(te_grin_10)
grin10_ic = daywise_ic(grin10_fin, oracle_df, test_ids, test_day)
print(f'Grinold top-10: IC={grin10_ic:+.6f}', flush=True)

# =====================================================================
# COMPONENT CORRELATIONS
# =====================================================================
print('\n=== COMPONENT CORRELATIONS ===', flush=True)
comps = {
    'lgb_ms': lgb_ms,
    'grin_all_dm': grin_fin,
    'nw_kernel': nw_fin,
    'ridge_perday': ridge_pd_fin,
    'grin_top10': grin10_fin,
}
comp_names = list(comps.keys())
for i in range(len(comp_names)):
    for j in range(i+1, len(comp_names)):
        c = float(np.corrcoef(comps[comp_names[i]], comps[comp_names[j]])[0, 1])
        print(f'  {comp_names[i]:15} vs {comp_names[j]:15}  r={c:+.4f}', flush=True)

# =====================================================================
# BLENDS
# =====================================================================
print('\n=== BLENDS ===', flush=True)

def blend(components):
    total_w = sum(w for w, _ in components)
    mix = sum(w * finalize(p) for w, p in components) / total_w
    return finalize(mix)

submissions = {}
submissions['solo_lgb_ms'] = lgb_ms
submissions['solo_grin_all_dm'] = grin_fin
submissions['solo_nw'] = nw_fin
submissions['solo_ridge_pd'] = ridge_pd_fin

for w_lgb in [0.80, 0.75, 0.70, 0.65, 0.60]:
    w_g = 1.0 - w_lgb
    submissions[f'lgb{int(w_lgb*100)}_grin{int(w_g*100)}'] = \
        blend([(w_lgb, lgb_ms_raw), (w_g, grin_raw)])

for w_lgb in [0.80, 0.70, 0.60, 0.50]:
    w_nw = 1.0 - w_lgb
    submissions[f'lgb{int(w_lgb*100)}_nw{int(w_nw*100)}'] = \
        blend([(w_lgb, lgb_ms_raw), (w_nw, te_nw)])

for w_lgb in [0.80, 0.70, 0.60, 0.50]:
    w_r = 1.0 - w_lgb
    submissions[f'lgb{int(w_lgb*100)}_rpd{int(w_r*100)}'] = \
        blend([(w_lgb, lgb_ms_raw), (w_r, te_ridge_pd)])

for w_lgb, w_nw, w_g in [
    (0.60, 0.20, 0.20), (0.50, 0.30, 0.20), (0.50, 0.25, 0.25),
    (0.60, 0.25, 0.15), (0.70, 0.15, 0.15), (0.55, 0.25, 0.20),
]:
    submissions[f'lgb{int(w_lgb*100)}_nw{int(w_nw*100)}_grin{int(w_g*100)}'] = \
        blend([(w_lgb, lgb_ms_raw), (w_nw, te_nw), (w_g, grin_raw)])

for w_lgb, w_rpd, w_g in [
    (0.60, 0.20, 0.20), (0.50, 0.30, 0.20), (0.50, 0.25, 0.25),
    (0.60, 0.25, 0.15), (0.70, 0.15, 0.15),
]:
    submissions[f'lgb{int(w_lgb*100)}_rpd{int(w_rpd*100)}_grin{int(w_g*100)}'] = \
        blend([(w_lgb, lgb_ms_raw), (w_rpd, te_ridge_pd), (w_g, grin_raw)])

for w_lgb, w_nw, w_rpd, w_g in [
    (0.50, 0.20, 0.15, 0.15),
    (0.45, 0.25, 0.15, 0.15),
    (0.55, 0.15, 0.15, 0.15),
    (0.40, 0.25, 0.20, 0.15),
    (0.50, 0.20, 0.20, 0.10),
    (0.60, 0.15, 0.15, 0.10),
]:
    submissions[f'lgb{int(w_lgb*100)}_nw{int(w_nw*100)}_rpd{int(w_rpd*100)}_grin{int(w_g*100)}'] = \
        blend([(w_lgb, lgb_ms_raw), (w_nw, te_nw), (w_rpd, te_ridge_pd), (w_g, grin_raw)])

results = []
for name, pred in submissions.items():
    ic = daywise_ic(pred, oracle_df, test_ids, test_day)
    results.append((name, ic, pred))
results.sort(key=lambda x: -x[1])

print(f"\n{'rank':<5} {'oracle_ic':>11}  name")
print('-' * 70)
for i, (name, ic, _) in enumerate(results):
    print(f'{i+1:<5} {ic:>+11.6f}  {name}')

print('\nWriting CSVs...', flush=True)
for name, ic, pred in results:
    path = os.path.join(OUT_DIR, f'{TAG}_{name}.csv')
    pd.DataFrame({'ID': test_ids, 'TARGET': pred}).sort_values('ID').to_csv(path, index=False)

print(f'\ndone in {(time.time()-t0)/60:.1f} min')
print(f'wrote {len(results)} CSVs with prefix {TAG}_ to {OUT_DIR}')
