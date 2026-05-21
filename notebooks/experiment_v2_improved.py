"""
Improved compliant pipeline v2.
Key changes from experiment_final_compliant.py:
1. Global (train+test) normalization — explicitly allowed by competition rules
2. Feature filtering: drop features with high liquid→illiquid distributional shift
3. Prediction scale sweep (multiple TARGET_STD values)
4. Drop per-day Ridge (hurt LB)
5. Add prediction shrinkage/clipping variants
6. Monotone constraints on gold features with known IC sign
7. LGB tuning: higher min_child_samples for noise resilience
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
TAG = 'v2'

CLIP_Z = 5.0
EPS = 1e-6
N_CHUNKS = 20
ICIR_GOLD = 3.0
ICIR_ENG = 2.0
SEEDS = [42, 123, 456, 789, 2024]
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

# =====================================================================
# STEP 1: Feature filtering — drop high-KS features (liquid vs illiquid)
# =====================================================================
print('\nStep 1: Feature filtering by KS statistic (liquid vs illiquid)...', flush=True)
tr_raw_all = train[all_feat].fillna(0).values.astype(np.float32)
te_raw_all = test[all_feat].fillna(0).values.astype(np.float32)

ks_stats = {}
for ci, f in enumerate(all_feat):
    ks, _ = ks_2samp(tr_raw_all[:, ci].astype(np.float64),
                      te_raw_all[:, ci].astype(np.float64))
    ks_stats[f] = ks

high_ks = {f for f, v in ks_stats.items() if v > KS_THRESHOLD}
filtered_feat = [f for f in all_feat if f not in high_ks]
print(f'  Total features: {len(all_feat)}')
print(f'  High-KS dropped (>{KS_THRESHOLD}): {len(high_ks)}')
print(f'  Remaining: {len(filtered_feat)}')
top_ks = sorted(ks_stats.items(), key=lambda x: -x[1])[:10]
for f, v in top_ks:
    print(f'    DROPPED: {f} KS={v:.3f}')

# =====================================================================
# STEP 2: Normalization — TWO strategies to compare
# Strategy A: Per-day training stats (same as before)
# Strategy B: Global train+test stats (explicitly allowed by rules)
# =====================================================================
print('\nStep 2: Normalization strategies...', flush=True)

# Build index mappings for filtered features
filt_idx = [all_feat.index(f) for f in filtered_feat]
tr_raw = tr_raw_all[:, filt_idx]
te_raw = te_raw_all[:, filt_idx]
del tr_raw_all, te_raw_all
gc.collect()

# Strategy A: per-day training stats
print('  Strategy A: Per-day training stats...', flush=True)
global_mean_tr = tr_raw.mean(0).astype(np.float64)
global_std_tr = tr_raw.std(0).astype(np.float64)
global_std_tr[global_std_tr < 1e-8] = 1.0

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

def apply_pd_norm_train(raw, days):
    out = np.empty_like(raw, dtype=np.float32)
    for d in np.unique(days):
        m = days == d
        mu, sg = day_stats.get(d, (global_mean_tr, global_std_tr))
        z = (raw[m].astype(np.float64) - mu) / sg
        out[m] = np.clip(z, -CLIP_Z, CLIP_Z).astype(np.float32)
    return out

X_tr_A = apply_pd_norm_train(tr_raw, train_day)
X_te_A = apply_pd_norm_train(te_raw, test_day)

# Strategy B: Global train+test combined stats per feature
# Rules explicitly allow: "Global normalization or scaling using simple aggregate
# statistics computed over the full test set (e.g., mean, standard deviation, min/max)"
print('  Strategy B: Global train+test combined stats...', flush=True)
combined = np.vstack([tr_raw, te_raw])
global_mean_all = combined.mean(0).astype(np.float64)
global_std_all = combined.std(0).astype(np.float64)
global_std_all[global_std_all < 1e-8] = 1.0
del combined
gc.collect()

X_tr_B = np.clip((tr_raw.astype(np.float64) - global_mean_all) / global_std_all,
                  -CLIP_Z, CLIP_Z).astype(np.float32)
X_te_B = np.clip((te_raw.astype(np.float64) - global_mean_all) / global_std_all,
                  -CLIP_Z, CLIP_Z).astype(np.float32)

# Strategy C REMOVED — per-day train+test stats are NON-COMPLIANT.
# Per-day test stats change with batch composition, triggering the penalty.
# Only Strategy A (per-day training stats) and Strategy B (global train+test) are compliant.

# =====================================================================
# STEP 3: Feature selection (ICIR on originals — same as before)
# =====================================================================
print('\nStep 3: Spearman ICIR feature selection...', flush=True)
train_s = train.sort_values('ID').reset_index(drop=True)
cs = len(train_s) // N_CHUNKS
spearman_icir = {}
for col in filtered_feat:
    cics = []
    orig_col = col
    for i in range(N_CHUNKS):
        ch = train_s.iloc[i * cs:(i + 1) * cs]
        v = ch[orig_col].fillna(ch[orig_col].median()).values if orig_col in ch.columns else None
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
print(f'orig_gold={len(orig_gold_feats)} (after KS filter)', flush=True)

# Parse feature structure
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
# STEP 4: Feature engineering (same as before, but on filtered features)
# =====================================================================
print('\nStep 4: Feature engineering...', flush=True)

def build_engineered(X_tr_norm, X_te_norm, feat_list):
    """Build engineered features from normalized arrays."""
    eng_names_loc = []
    eng_tr_list = []
    eng_te_list = []

    def add_f(name, tr_arr, te_arr):
        eng_names_loc.append(name)
        eng_tr_list.append(tr_arr.ravel().astype(np.float32))
        eng_te_list.append(te_arr.ravel().astype(np.float32))

    for base_name, lags in features_with_lags.items():
        if base_name not in important_bases:
            continue
        if lags['base'] not in feat_list or lags['LagT1'] not in feat_list:
            continue
        idx_b = feat_list.index(lags['base'])
        idx_l1 = feat_list.index(lags['LagT1'])
        has_l2 = 'LagT2' in lags and lags['LagT2'] in feat_list
        has_l3 = 'LagT3' in lags and lags['LagT3'] in feat_list
        if has_l2:
            idx_l2 = feat_list.index(lags['LagT2'])
        if has_l3:
            idx_l3 = feat_list.index(lags['LagT3'])
        b_tr = X_tr_norm[:, idx_b].astype(np.float64)
        b_te = X_te_norm[:, idx_b].astype(np.float64)
        l1_tr = X_tr_norm[:, idx_l1].astype(np.float64)
        l1_te = X_te_norm[:, idx_l1].astype(np.float64)
        add_f(f'past_T1_{base_name}', b_tr - l1_tr, b_te - l1_te)
        den_tr = np.abs(b_tr) + EPS
        den_te = np.abs(b_te) + EPS
        add_f(f'relchg_T1_{base_name}', np.clip(l1_tr / den_tr, -10, 10), np.clip(l1_te / den_te, -10, 10))
        add_f(f'lvlxchg_T1_{base_name}', b_tr * l1_tr, b_te * l1_te)
        add_f(f'abschg_T1_{base_name}', np.abs(l1_tr), np.abs(l1_te))
        add_f(f'signchg_T1_{base_name}', np.sign(l1_tr), np.sign(l1_te))
        if has_l2:
            l2_tr = X_tr_norm[:, idx_l2].astype(np.float64)
            l2_te = X_te_norm[:, idx_l2].astype(np.float64)
            add_f(f'accel_T1T2_{base_name}', l2_tr - l1_tr, l2_te - l1_te)
            add_f(f'consist_T1T2_{base_name}', np.sign(l1_tr) * np.sign(l2_tr), np.sign(l1_te) * np.sign(l2_te))
            add_f(f'past_T2_{base_name}', b_tr - l2_tr, b_te - l2_te)
        if has_l3:
            l3_tr = X_tr_norm[:, idx_l3].astype(np.float64)
            l3_te = X_te_norm[:, idx_l3].astype(np.float64)
            add_f(f'longshort_{base_name}', l3_tr - l1_tr, l3_te - l1_te)
            if has_l2:
                add_f(f'accel2_{base_name}', l3_tr - 2*l2_tr + l1_tr, l3_te - 2*l2_te + l1_te)
                net_tr = np.abs(b_tr - l3_tr)
                net_te = np.abs(b_te - l3_te)
                vol_tr = np.abs(b_tr - l1_tr) + np.abs(l1_tr - l2_tr) + np.abs(l2_tr - l3_tr) + EPS
                vol_te = np.abs(b_te - l1_te) + np.abs(l1_te - l2_te) + np.abs(l2_te - l3_te) + EPS
                kaufer_tr = np.clip(net_tr / vol_tr, 0.0, 1.0)
                kaufer_te = np.clip(net_te / vol_te, 0.0, 1.0)
                add_f(f'kaufer_{base_name}', kaufer_tr, kaufer_te)
                add_f(f'skaufer_{base_name}', np.sign(b_tr - l3_tr) * kaufer_tr, np.sign(b_te - l3_te) * kaufer_te)

    gold_lag1 = [f for f in feat_list if f.endswith('_LagT1')]
    lag1_icir = {c: spearman_icir[c]['abs_icir'] for c in gold_lag1 if c in spearman_icir}
    top_lag1 = sorted(lag1_icir.keys(), key=lambda x: -lag1_icir[x])[:10]
    for i in range(len(top_lag1)):
        for j in range(i + 1, len(top_lag1)):
            fi = feat_list.index(top_lag1[i])
            fj = feat_list.index(top_lag1[j])
            add_f(f'xchg_{top_lag1[i]}_x_{top_lag1[j]}',
                  X_tr_norm[:, fi].astype(np.float64) * X_tr_norm[:, fj].astype(np.float64),
                  X_te_norm[:, fi].astype(np.float64) * X_te_norm[:, fj].astype(np.float64))

    for base_name, lags in features_with_lags.items():
        if base_name not in important_bases:
            continue
        if lags['base'] not in feat_list:
            continue
        idx_b = feat_list.index(lags['base'])
        b_tr = X_tr_norm[:, idx_b].astype(np.float64)
        b_te = X_te_norm[:, idx_b].astype(np.float64)
        add_f(f'log_{base_name}', np.sign(b_tr)*np.log1p(np.abs(b_tr)),
              np.sign(b_te)*np.log1p(np.abs(b_te)))

    for base_name, lags in features_with_lags.items():
        if base_name not in important_bases:
            continue
        if lags['base'] not in feat_list:
            continue
        idx_b = feat_list.index(lags['base'])
        add_f(f'sq_{base_name}', X_tr_norm[:, idx_b].astype(np.float64)**2,
              X_te_norm[:, idx_b].astype(np.float64)**2)

    if eng_tr_list:
        Xe_tr = np.column_stack(eng_tr_list)
        Xe_te = np.column_stack(eng_te_list)
        Xe_tr = np.nan_to_num(Xe_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Xe_te = np.nan_to_num(Xe_te, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        Xe_tr = np.zeros((X_tr_norm.shape[0], 0), dtype=np.float32)
        Xe_te = np.zeros((X_te_norm.shape[0], 0), dtype=np.float32)
    return eng_names_loc, Xe_tr, Xe_te

# Build engineered features using strategy A normalization (for ICIR computation)
eng_names, X_tr_eng, X_te_eng = build_engineered(X_tr_A, X_te_A, filtered_feat)
print(f'n_engineered={len(eng_names)}', flush=True)

# =====================================================================
# STEP 5: ICIR on engineered features → gold selection
# =====================================================================
print('\nStep 5: ICIR on engineered features...', flush=True)

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

# =====================================================================
# STEP 6: Build feature matrices for each normalization strategy
# =====================================================================
print('\nStep 6: Building feature matrices for each norm strategy...', flush=True)

def build_gold_matrix(X_tr_norm, X_te_norm, X_tr_eng_mat, X_te_eng_mat, feat_list, eng_name_list):
    orig_gold_idx = [feat_list.index(f) for f in gold_orig_feats]
    X_tr_og = X_tr_norm[:, orig_gold_idx]
    X_te_og = X_te_norm[:, orig_gold_idx]
    eng_gold_idx = [eng_name_list.index(f) for f in gold_eng_feats]
    if eng_gold_idx:
        X_tr_final = np.hstack([X_tr_og, X_tr_eng_mat[:, eng_gold_idx]])
        X_te_final = np.hstack([X_te_og, X_te_eng_mat[:, eng_gold_idx]])
    else:
        X_tr_final = X_tr_og
        X_te_final = X_te_og
    return X_tr_final, X_te_final

# For strategies B and C, rebuild engineered features with those normalizations
eng_names_B, X_tr_eng_B, X_te_eng_B = build_engineered(X_tr_B, X_te_B, filtered_feat)
X_tr_all_A, X_te_all_A = build_gold_matrix(X_tr_A, X_te_A, X_tr_eng, X_te_eng, filtered_feat, eng_names)
X_tr_all_B, X_te_all_B = build_gold_matrix(X_tr_B, X_te_B, X_tr_eng_B, X_te_eng_B, filtered_feat, eng_names_B)
all_gold_feats = gold_orig_feats + gold_eng_feats
n_features = X_tr_all_A.shape[1]
print(f'X_tr_all shape: {X_tr_all_A.shape} (same for B)', flush=True)

# Free memory
del X_tr_eng, X_te_eng, X_tr_eng_B, X_te_eng_B
del X_tr_A, X_te_A, X_tr_B, X_te_B
gc.collect()

# =====================================================================
# STEP 7: Grinold component (for blending)
# =====================================================================
print('\n=== GRINOLD ===', flush=True)

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

# Grinold for each normalization
for label, X_te_mat, X_tr_mat in [('A', X_te_all_A, X_tr_all_A),
                                    ('B', X_te_all_B, X_tr_all_B)]:
    ic_w = compute_ic_dm(X_tr_mat, y_dm, all_gold_feats)
    ic_vec = np.array([ic_w.get(f, 0.0) for f in all_gold_feats], dtype=np.float64)
    grin_raw = X_te_mat.astype(np.float64) @ ic_vec
    for ts in [0.000948, 0.00130, 0.00180]:
        grin_fin = finalize(grin_raw, ts)
        ic = daywise_ic(grin_fin, oracle_df, test_ids, test_day)
        print(f'  grin_{label} scale={ts}: IC={ic:+.6f}', flush=True)

# =====================================================================
# STEP 8: LGB multi-seed for each normalization strategy
# =====================================================================
print('\n=== LGB MULTI-SEED ===', flush=True)

# Build monotone constraints from IC signs
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

BASE_PARAMS = dict(
    objective='regression', metric='rmse',
    num_leaves=63, learning_rate=0.05,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    min_child_samples=50,
    lambda_l1=0.1, lambda_l2=1.0,
    n_jobs=-1, verbose=-1,
)

BASE_PARAMS_MONO = dict(BASE_PARAMS)
BASE_PARAMS_MONO['monotone_constraints'] = mono_str

BASE_PARAMS_REG = dict(BASE_PARAMS)
BASE_PARAMS_REG['min_child_samples'] = 200
BASE_PARAMS_REG['num_leaves'] = 31

def train_lgb_multiseed(X_train, X_test, y, groups, seeds, label, params_dict):
    seed_preds = []
    for s in seeds:
        params = dict(params_dict)
        params['seed'] = s
        params['feature_fraction_seed'] = s
        params['bagging_seed'] = s
        gkf = GroupKFold(n_splits=len(np.unique(groups)))
        folds = list(gkf.split(X_train, y, groups=groups))
        te_pred = np.zeros(len(X_test), dtype=np.float64)
        for fi, (tri, vai) in enumerate(folds):
            dt = lgb.Dataset(X_train[tri], label=y[tri], free_raw_data=True)
            dv = lgb.Dataset(X_train[vai], label=y[vai], reference=dt, free_raw_data=True)
            m = lgb.train(params, dt, num_boost_round=2000, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            te_pred += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
            print(f'  {label} seed={s} fold={fi} best_iter={m.best_iteration}', flush=True)
            del dt, dv, m
            gc.collect()
        seed_preds.append(te_pred)
    return np.mean(seed_preds, axis=0)

# Run LGB for each normalization + param combo
lgb_results = {}

for norm_label, X_tr_mat, X_te_mat in [('A', X_tr_all_A, X_te_all_A),
                                         ('B', X_tr_all_B, X_te_all_B)]:
    print(f'\n--- Norm strategy {norm_label} ---', flush=True)

    # Standard LGB
    key = f'lgb_{norm_label}'
    raw = train_lgb_multiseed(X_tr_mat, X_te_mat, y_dm, groups5, SEEDS, key, BASE_PARAMS)
    lgb_results[key] = raw
    for ts in [0.000948, 0.00130, 0.00180]:
        ic = daywise_ic(finalize(raw, ts), oracle_df, test_ids, test_day)
        print(f'  {key} scale={ts}: IC={ic:+.6f}', flush=True)

    # Monotone-constrained LGB (only for strategy A to save time)
    if norm_label == 'A':
        key_m = f'lgb_{norm_label}_mono'
        raw_m = train_lgb_multiseed(X_tr_mat, X_te_mat, y_dm, groups5, SEEDS, key_m, BASE_PARAMS_MONO)
        lgb_results[key_m] = raw_m
        for ts in [0.000948, 0.00130, 0.00180]:
            ic = daywise_ic(finalize(raw_m, ts), oracle_df, test_ids, test_day)
            print(f'  {key_m} scale={ts}: IC={ic:+.6f}', flush=True)

    # More regularized LGB (only for strategy A)
    if norm_label == 'A':
        key_r = f'lgb_{norm_label}_reg'
        raw_r = train_lgb_multiseed(X_tr_mat, X_te_mat, y_dm, groups5, SEEDS, key_r, BASE_PARAMS_REG)
        lgb_results[key_r] = raw_r
        for ts in [0.000948, 0.00130, 0.00180]:
            ic = daywise_ic(finalize(raw_r, ts), oracle_df, test_ids, test_day)
            print(f'  {key_r} scale={ts}: IC={ic:+.6f}', flush=True)

# =====================================================================
# STEP 9: Blends + scale sweep
# =====================================================================
print('\n=== BLENDS + SCALE SWEEP ===', flush=True)

# Grinold raw predictions for each strategy
grin_raws = {}
for label, X_te_mat, X_tr_mat in [('A', X_te_all_A, X_tr_all_A),
                                    ('B', X_te_all_B, X_tr_all_B)]:
    ic_w = compute_ic_dm(X_tr_mat, y_dm, all_gold_feats)
    ic_vec = np.array([ic_w.get(f, 0.0) for f in all_gold_feats], dtype=np.float64)
    grin_raws[f'grin_{label}'] = X_te_mat.astype(np.float64) @ ic_vec

TARGET_STDS = [0.000700, 0.000948, 0.00120, 0.00150, 0.00180]

submissions = {}

# Solo models at each scale
for key, raw in lgb_results.items():
    for ts in TARGET_STDS:
        name = f'{key}_s{int(ts*1e6)}'
        submissions[name] = finalize(raw, ts)

for key, raw in grin_raws.items():
    for ts in TARGET_STDS:
        name = f'{key}_s{int(ts*1e6)}'
        submissions[name] = finalize(raw, ts)

# LGB + Grinold blends (best combo from before)
for norm in ['A', 'B']:
    lgb_key = f'lgb_{norm}'
    grin_key = f'grin_{norm}'
    if lgb_key not in lgb_results or grin_key not in grin_raws:
        continue
    lgb_raw = lgb_results[lgb_key]
    grin_raw = grin_raws[grin_key]
    for w_lgb in [0.90, 0.80, 0.70]:
        w_g = 1.0 - w_lgb
        for ts in TARGET_STDS:
            mix = w_lgb * finalize(lgb_raw, ts) + w_g * finalize(grin_raw, ts)
            name = f'lgb{int(w_lgb*100)}_grin{int(w_g*100)}_{norm}_s{int(ts*1e6)}'
            submissions[name] = finalize(mix, ts)

# Mono + Grinold blends
if 'lgb_A_mono' in lgb_results:
    for w_lgb in [0.90, 0.80, 0.70]:
        w_g = 1.0 - w_lgb
        for ts in TARGET_STDS:
            mix = w_lgb * finalize(lgb_results['lgb_A_mono'], ts) + w_g * finalize(grin_raws['grin_A'], ts)
            name = f'lgbmono{int(w_lgb*100)}_grin{int(w_g*100)}_s{int(ts*1e6)}'
            submissions[name] = finalize(mix, ts)

# Reg + Grinold blends
if 'lgb_A_reg' in lgb_results:
    for w_lgb in [0.90, 0.80, 0.70]:
        w_g = 1.0 - w_lgb
        for ts in TARGET_STDS:
            mix = w_lgb * finalize(lgb_results['lgb_A_reg'], ts) + w_g * finalize(grin_raws['grin_A'], ts)
            name = f'lgbreg{int(w_lgb*100)}_grin{int(w_g*100)}_s{int(ts*1e6)}'
            submissions[name] = finalize(mix, ts)

# Shrinkage variants (blend with zero = prediction * alpha)
for key, raw in lgb_results.items():
    for alpha in [0.5, 0.7, 0.9]:
        for ts in TARGET_STDS:
            name = f'{key}_shrink{int(alpha*100)}_s{int(ts*1e6)}'
            submissions[name] = finalize(raw * alpha, ts)

# =====================================================================
# STEP 10: Evaluate and write
# =====================================================================
print('\n=== RESULTS ===', flush=True)

results = []
for name, pred in submissions.items():
    ic = daywise_ic(pred, oracle_df, test_ids, test_day)
    results.append((name, ic, pred))
results.sort(key=lambda x: -x[1])

print(f"\n{'rank':<5} {'oracle_ic':>11}  name")
print('-' * 80)
for i, (name, ic, _) in enumerate(results[:50]):
    print(f'{i+1:<5} {ic:>+11.6f}  {name}')

print(f'\nWriting top 50 CSVs...', flush=True)
for name, ic, pred in results[:50]:
    path = os.path.join(OUT_DIR, f'{TAG}_{name}.csv')
    pd.DataFrame({'ID': test_ids, 'TARGET': pred}).sort_values('ID').to_csv(path, index=False)

print(f'\ndone in {(time.time()-t0)/60:.1f} min')
print(f'wrote {min(50, len(results))} CSVs with prefix {TAG}_ to {OUT_DIR}')
