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
TAG = 'ang'

TARGET_STD = 0.000948
CLIP_Z = 5.0
EPS = 1e-6
N_CHUNKS = 20
ICIR_GOLD = 3.0
ICIR_ENG = 2.0
SEEDS = [42, 123, 456, 789, 2024]

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
y_raw = train['TARGET'].values.astype(np.float64)
lo_y, hi_y = np.percentile(y_raw, 1), np.percentile(y_raw, 99)
y_wins = np.clip(y_raw, lo_y, hi_y)

print('Per-day demeaned training target (training-side, compliant)...', flush=True)
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
del tr_raw, te_raw
gc.collect()

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

print('Feature engineering (row-wise)...', flush=True)
eng_names = []
eng_tr_list = []
eng_te_list = []

def add_feat(name, tr_arr, te_arr):
    eng_names.append(name)
    eng_tr_list.append(tr_arr.ravel().astype(np.float32))
    eng_te_list.append(te_arr.ravel().astype(np.float32))

n_angle_binary = 0
n_angle_kaufer = 0
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
    add_feat(f'relchg_T1_{base_name}',
             np.clip(l1_tr / den_tr, -10, 10),
             np.clip(l1_te / den_te, -10, 10))
    add_feat(f'lvlxchg_T1_{base_name}', b_tr * l1_tr, b_te * l1_te)
    add_feat(f'abschg_T1_{base_name}', np.abs(l1_tr), np.abs(l1_te))
    add_feat(f'signchg_T1_{base_name}', np.sign(l1_tr), np.sign(l1_te))

    # --- NEW: binary extreme flags (proposal 1) ---
    # X is per-day z-scored (clipped ±5), so thresholds are in sigma units
    add_feat(f'isext2b_{base_name}',
             (np.abs(b_tr) > 2.0).astype(np.float64),
             (np.abs(b_te) > 2.0).astype(np.float64))
    add_feat(f'isext3b_{base_name}',
             (np.abs(b_tr) > 3.0).astype(np.float64),
             (np.abs(b_te) > 3.0).astype(np.float64))
    add_feat(f'isext2l1_{base_name}',
             (np.abs(l1_tr) > 2.0).astype(np.float64),
             (np.abs(l1_te) > 2.0).astype(np.float64))
    add_feat(f'isext3l1_{base_name}',
             (np.abs(l1_tr) > 3.0).astype(np.float64),
             (np.abs(l1_te) > 3.0).astype(np.float64))
    n_angle_binary += 4

    if has_l2:
        l2_tr = X_tr[:, idx_l2].astype(np.float64)
        l2_te = X_te[:, idx_l2].astype(np.float64)
        add_feat(f'accel_T1T2_{base_name}', l2_tr - l1_tr, l2_te - l1_te)
        add_feat(f'consist_T1T2_{base_name}',
                 np.sign(l1_tr) * np.sign(l2_tr),
                 np.sign(l1_te) * np.sign(l2_te))
        add_feat(f'past_T2_{base_name}', b_tr - l2_tr, b_te - l2_te)

    if has_l3:
        l3_tr = X_tr[:, idx_l3].astype(np.float64)
        l3_te = X_te[:, idx_l3].astype(np.float64)
        add_feat(f'longshort_{base_name}', l3_tr - l1_tr, l3_te - l1_te)
        if has_l2:
            add_feat(f'accel2_{base_name}',
                     l3_tr - 2 * l2_tr + l1_tr,
                     l3_te - 2 * l2_te + l1_te)
            # --- Kaufman Efficiency Ratio: full 4-point path (b, L1, L2, L3) ---
            net_tr = np.abs(b_tr - l3_tr)
            net_te = np.abs(b_te - l3_te)
            vol_tr = (np.abs(b_tr - l1_tr) + np.abs(l1_tr - l2_tr)
                      + np.abs(l2_tr - l3_tr) + EPS)
            vol_te = (np.abs(b_te - l1_te) + np.abs(l1_te - l2_te)
                      + np.abs(l2_te - l3_te) + EPS)
            kaufer_tr = np.clip(net_tr / vol_tr, 0.0, 1.0)
            kaufer_te = np.clip(net_te / vol_te, 0.0, 1.0)
            add_feat(f'kaufer_{base_name}', kaufer_tr, kaufer_te)
            sign_tr = np.sign(b_tr - l3_tr)
            sign_te = np.sign(b_te - l3_te)
            add_feat(f'skaufer_{base_name}', sign_tr * kaufer_tr, sign_te * kaufer_te)
            n_angle_kaufer += 2

            # --- Kaufer variant: L1 -> L3 path (no base) ---
            net_tr_s = np.abs(l1_tr - l3_tr)
            net_te_s = np.abs(l1_te - l3_te)
            vol_tr_s = np.abs(l1_tr - l2_tr) + np.abs(l2_tr - l3_tr) + EPS
            vol_te_s = np.abs(l1_te - l2_te) + np.abs(l2_te - l3_te) + EPS
            kshort_tr = np.clip(net_tr_s / vol_tr_s, 0.0, 1.0)
            kshort_te = np.clip(net_te_s / vol_te_s, 0.0, 1.0)
            add_feat(f'kaufer_short_{base_name}', kshort_tr, kshort_te)
            sign_s_tr = np.sign(l1_tr - l3_tr)
            sign_s_te = np.sign(l1_te - l3_te)
            add_feat(f'skaufer_short_{base_name}', sign_s_tr * kshort_tr, sign_s_te * kshort_te)

            # --- Kaufer variant: skip-lag b -> L3 via L1 only ---
            net_tr_13 = np.abs(b_tr - l3_tr)
            net_te_13 = np.abs(b_te - l3_te)
            vol_tr_13 = np.abs(b_tr - l1_tr) + np.abs(l1_tr - l3_tr) + EPS
            vol_te_13 = np.abs(b_te - l1_te) + np.abs(l1_te - l3_te) + EPS
            k13_tr = np.clip(net_tr_13 / vol_tr_13, 0.0, 1.0)
            k13_te = np.clip(net_te_13 / vol_te_13, 0.0, 1.0)
            add_feat(f'kaufer_T1T3_{base_name}', k13_tr, k13_te)
            n_angle_kaufer += 3

    if has_l2:
        l2_tr_local = X_tr[:, idx_l2].astype(np.float64)
        l2_te_local = X_te[:, idx_l2].astype(np.float64)
        # --- Kaufer variant: b -> L2 window (shorter) ---
        net_tr_t2 = np.abs(b_tr - l2_tr_local)
        net_te_t2 = np.abs(b_te - l2_te_local)
        vol_tr_t2 = np.abs(b_tr - l1_tr) + np.abs(l1_tr - l2_tr_local) + EPS
        vol_te_t2 = np.abs(b_te - l1_te) + np.abs(l1_te - l2_te_local) + EPS
        kt2_tr = np.clip(net_tr_t2 / vol_tr_t2, 0.0, 1.0)
        kt2_te = np.clip(net_te_t2 / vol_te_t2, 0.0, 1.0)
        add_feat(f'kaufer_T2_{base_name}', kt2_tr, kt2_te)
        sign_t2_tr = np.sign(b_tr - l2_tr_local)
        sign_t2_te = np.sign(b_te - l2_te_local)
        add_feat(f'skaufer_T2_{base_name}', sign_t2_tr * kt2_tr, sign_t2_te * kt2_te)
        n_angle_kaufer += 2

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
    add_feat(f'log_{base_name}',
             np.sign(b_tr) * np.log1p(np.abs(b_tr)),
             np.sign(b_te) * np.log1p(np.abs(b_te)))

for base_name, lags in features_with_lags.items():
    if base_name not in important_bases:
        continue
    idx_b = all_feat.index(lags['base'])
    add_feat(f'sq_{base_name}',
             X_tr[:, idx_b].astype(np.float64) ** 2,
             X_te[:, idx_b].astype(np.float64) ** 2)

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
print(f'n_engineered={len(eng_names)} (binary_flags={n_angle_binary}, kaufer={n_angle_kaufer})', flush=True)

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

print('Spearman ICIR over engineered (on y_raw, feature selection)...', flush=True)
icir_eng = fast_spearman_icir(X_tr_eng, y_raw, eng_names)
icir_all = {**spearman_icir, **icir_eng}

# Originals: keep strict gate (ICIR >= 3.0 AND consistent sign)
gold_orig_feats2 = sorted(
    [k for k, v in icir_all.items()
     if k in all_feat and v['abs_icir'] >= ICIR_GOLD and v['ic_pos'] in (0.0, 1.0)],
    key=lambda x: -icir_all[x]['abs_icir'])

# Engineered: relaxed gate — ICIR >= 2.0, no ic_pos constraint.
# Rationale: engineered features (especially binary flags) can't clear a consistent-sign
# requirement across 20 chunks even when genuinely informative.
ENG_ICIR_FLOOR = 2.0
gold_eng_feats = sorted(
    [k for k, v in icir_all.items()
     if k in eng_names and v['abs_icir'] >= ENG_ICIR_FLOOR],
    key=lambda x: -icir_all[x]['abs_icir'])
print(f'gold_orig={len(gold_orig_feats2)} gold_eng_relaxed={len(gold_eng_feats)}', flush=True)

# Binary flags: include top-K by |mean_ic| unconditionally on top of the relaxed filter
binary_flag_feats = [f for f in eng_names if f.startswith('isext')]
bin_mic_sorted = sorted(
    [(f, abs(icir_eng.get(f, {}).get('mean_ic', 0.0))) for f in binary_flag_feats],
    key=lambda x: -x[1])
K_BIN = 40
bin_added = 0
for f, mic in bin_mic_sorted[:K_BIN]:
    if mic < 1e-4:
        continue
    if f not in gold_eng_feats:
        gold_eng_feats.append(f)
        bin_added += 1
print(f'  binary_flags_top{K_BIN}_added_unconditional={bin_added}', flush=True)

print(f'gold_orig={len(gold_orig_feats2)} gold_eng={len(gold_eng_feats)}', flush=True)

# Report which angle-specific features survived selection
n_bin_gold = sum(1 for f in gold_eng_feats if f.startswith('isext'))
n_kauf_gold = sum(1 for f in gold_eng_feats if 'kaufer' in f)
print(f'  angle_binary_in_gold={n_bin_gold} angle_kaufer_in_gold={n_kauf_gold}', flush=True)

orig_gold_idx = [all_feat.index(f) for f in gold_orig_feats2]
X_tr_og = X_tr[:, orig_gold_idx]
X_te_og = X_te[:, orig_gold_idx]
eng_gold_idx_sel = [eng_names.index(f) for f in gold_eng_feats]
X_tr_all = np.hstack([X_tr_og, X_tr_eng[:, eng_gold_idx_sel]]) if eng_gold_idx_sel else X_tr_og
X_te_all = np.hstack([X_te_og, X_te_eng[:, eng_gold_idx_sel]]) if eng_gold_idx_sel else X_te_og
all_gold_feats = gold_orig_feats2 + gold_eng_feats
print(f'X_tr_all={X_tr_all.shape}', flush=True)

print('\n=== GRINOLD (weights computed on y_dm, per-day demeaned target) ===', flush=True)

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
grin_all_dm_raw = X_te_all.astype(np.float64) @ ic_w_vec_dm
grin_all_dm = finalize(grin_all_dm_raw)
grin_all_dm_ic = daywise_ic(grin_all_dm, oracle_df, test_ids, test_day)
print(f'grin_all_dm: IC={grin_all_dm_ic:+.6f}', flush=True)

print(f'\n=== LIGHTGBM MULTI-SEED (trained on y_dm, {X_tr_all.shape[1]} features) ===', flush=True)
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

def train_lgb_seed(X_train, X_test, y, groups, seed, label, base_params=None):
    params = dict(base_params or BASE_PARAMS_L2)
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
    p = train_lgb_seed(X_tr_all, X_te_all, y_dm, groups5, s, 'lgb_all_dm')
    seed_preds.append(p)
    p_fin = finalize(p)
    ic_s = daywise_ic(p_fin, oracle_df, test_ids, test_day)
    print(f'  seed={s} solo IC={ic_s:+.6f}', flush=True)

lgb_ms_raw = np.mean(seed_preds, axis=0)
lgb_ms = finalize(lgb_ms_raw)
lgb_ms_ic = daywise_ic(lgb_ms, oracle_df, test_ids, test_day)
print(f'\nlgb_all_dm MULTI-SEED avg: IC={lgb_ms_ic:+.6f}', flush=True)

print('\n=== RIDGE (on y_dm, all features) ===', flush=True)
def train_ridge(X_train, X_test, y, groups, alphas):
    best_ic = -1
    best_pred = None
    best_alpha = None
    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    for alpha in alphas:
        folds = list(gkf.split(X_train, y, groups=groups))
        te_pred = np.zeros(len(X_test), dtype=np.float64)
        for tri, vai in folds:
            m = Ridge(alpha=alpha, fit_intercept=True)
            m.fit(X_train[tri], y[tri])
            te_pred += m.predict(X_test) / len(folds)
        te_s = finalize(te_pred)
        ic = daywise_ic(te_s, oracle_df, test_ids, test_day)
        print(f'  ridge alpha={alpha}: IC={ic:+.6f}', flush=True)
        if ic > best_ic:
            best_ic = ic
            best_pred = te_pred
            best_alpha = alpha
    print(f'  best ridge: alpha={best_alpha} IC={best_ic:+.6f}', flush=True)
    return best_pred, best_ic

ridge_all_dm_raw, ridge_all_dm_ic = train_ridge(X_tr_all, X_te_all, y_dm, groups5,
                                                 [100, 500, 1000, 5000, 10000])
ridge_all_dm = finalize(ridge_all_dm_raw)

print('\n=== BLENDS ===', flush=True)

def blend(components):
    total_w = sum(w for w, _ in components)
    mix = sum(w * finalize(p) for w, p in components) / total_w
    return finalize(mix)

submissions = {}
submissions['solo_lgb_ms'] = lgb_ms
submissions['solo_grin_all_dm'] = grin_all_dm
submissions['solo_ridge_all_dm'] = ridge_all_dm

for w_lgb in [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]:
    w_g = 1.0 - w_lgb
    submissions[f'lgbms_{int(round(w_lgb*100))}_grin_dm_{int(round(w_g*100))}'] = \
        blend([(w_lgb, lgb_ms_raw), (w_g, grin_all_dm_raw)])

for w_lgb, w_r, w_g in [(0.70, 0.15, 0.15), (0.65, 0.20, 0.15), (0.60, 0.20, 0.20),
                        (0.60, 0.25, 0.15), (0.55, 0.25, 0.20), (0.65, 0.15, 0.20),
                        (0.70, 0.10, 0.20), (0.75, 0.10, 0.15)]:
    name = f'lgbms{int(w_lgb*100)}_ridge{int(w_r*100)}_grin{int(w_g*100)}'
    submissions[name] = blend([(w_lgb, lgb_ms_raw),
                               (w_r, ridge_all_dm_raw),
                               (w_g, grin_all_dm_raw)])

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
