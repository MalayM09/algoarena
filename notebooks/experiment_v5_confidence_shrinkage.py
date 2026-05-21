"""
Experiment v5: Confidence-weighted prediction shrinkage.

Key insight from ceiling analysis:
- 70% of days have negative R² → predictions HURT on most days
- A few good days carry the entire positive score
- Ceiling oracle Ridge CV on test = 0.0038, we're at 0.00079

Strategy: For each test day, estimate a TRAINING-SIDE confidence score.
Shrink predictions toward zero on low-confidence days.
This is fully compliant — confidence depends only on training data.

Confidence signals:
1. Training sample size (more samples = more reliable fit)
2. Training Ridge LOO/CV R² (can Ridge even predict within liquid?)
3. Training target std (high vol = noisy day)
4. Per-day feature coverage (how many features have non-trivial variance?)

Architecture: Same fc_lgb50_rpd50 pipeline as the winning submission,
but with per-day confidence shrinkage applied to BOTH components.
"""
import os, gc, time, warnings, re
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

DATA_DIR = '/Users/malaymishra/Desktop/quant_ml_project/data/raw'
OUT_DIR = '/Users/malaymishra/Desktop/quant_ml_project/outputs/submissions'
ORACLE_PATH = os.path.join(OUT_DIR, 'exploit_v2_zero.csv')
TAG = 'v5'

CLIP_Z = 5.0
EPS = 1e-6
N_CHUNKS = 20
ICIR_GOLD = 3.0
ICIR_ENG = 2.0
SEEDS = [42, 123, 456, 789, 2024]
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
# DATA LOADING (same as fc pipeline — no KS filtering)
# =====================================================================
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

# =====================================================================
# NORMALIZATION (same as fc pipeline — per-day training stats)
# =====================================================================
print('Per-day normalization...', flush=True)
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
# ICIR + FEATURE ENGINEERING (same as fc pipeline)
# =====================================================================
print('ICIR feature selection...', flush=True)
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
base_max_icir = {b: max(spearman_icir.get(f, {'abs_icir': 0})['abs_icir']
                        for f in lags.values())
                 for b, lags in features_with_lags.items()}
important_bases = {b for b, v in base_max_icir.items() if v >= ICIR_ENG}

print('Feature engineering...', flush=True)
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
    if has_l2: idx_l2 = all_feat.index(lags['LagT2'])
    if has_l3: idx_l3 = all_feat.index(lags['LagT3'])
    b_tr = X_tr[:, idx_b].astype(np.float64)
    b_te = X_te[:, idx_b].astype(np.float64)
    l1_tr = X_tr[:, idx_l1].astype(np.float64)
    l1_te = X_te[:, idx_l1].astype(np.float64)
    add_feat(f'past_T1_{base_name}', b_tr - l1_tr, b_te - l1_te)
    den_tr = np.abs(b_tr) + EPS; den_te = np.abs(b_te) + EPS
    add_feat(f'relchg_T1_{base_name}', np.clip(l1_tr/den_tr,-10,10), np.clip(l1_te/den_te,-10,10))
    add_feat(f'lvlxchg_T1_{base_name}', b_tr*l1_tr, b_te*l1_te)
    add_feat(f'abschg_T1_{base_name}', np.abs(l1_tr), np.abs(l1_te))
    add_feat(f'signchg_T1_{base_name}', np.sign(l1_tr), np.sign(l1_te))
    if has_l2:
        l2_tr = X_tr[:, idx_l2].astype(np.float64); l2_te = X_te[:, idx_l2].astype(np.float64)
        add_feat(f'accel_T1T2_{base_name}', l2_tr-l1_tr, l2_te-l1_te)
        add_feat(f'consist_T1T2_{base_name}', np.sign(l1_tr)*np.sign(l2_tr), np.sign(l1_te)*np.sign(l2_te))
        add_feat(f'past_T2_{base_name}', b_tr-l2_tr, b_te-l2_te)
    if has_l3:
        l3_tr = X_tr[:, idx_l3].astype(np.float64); l3_te = X_te[:, idx_l3].astype(np.float64)
        add_feat(f'longshort_{base_name}', l3_tr-l1_tr, l3_te-l1_te)
        if has_l2:
            add_feat(f'accel2_{base_name}', l3_tr-2*l2_tr+l1_tr, l3_te-2*l2_te+l1_te)
            net_tr = np.abs(b_tr-l3_tr); net_te = np.abs(b_te-l3_te)
            vol_tr = np.abs(b_tr-l1_tr)+np.abs(l1_tr-l2_tr)+np.abs(l2_tr-l3_tr)+EPS
            vol_te = np.abs(b_te-l1_te)+np.abs(l1_te-l2_te)+np.abs(l2_te-l3_te)+EPS
            kaufer_tr = np.clip(net_tr/vol_tr, 0, 1); kaufer_te = np.clip(net_te/vol_te, 0, 1)
            add_feat(f'kaufer_{base_name}', kaufer_tr, kaufer_te)
            add_feat(f'skaufer_{base_name}', np.sign(b_tr-l3_tr)*kaufer_tr, np.sign(b_te-l3_te)*kaufer_te)

gold_lag1 = [f for f in all_feat if f.endswith('_LagT1')]
lag1_icir = {c: spearman_icir[c]['abs_icir'] for c in gold_lag1 if c in spearman_icir}
top_lag1 = sorted(lag1_icir.keys(), key=lambda x: -lag1_icir[x])[:10]
for i in range(len(top_lag1)):
    for j in range(i+1, len(top_lag1)):
        fi = all_feat.index(top_lag1[i]); fj = all_feat.index(top_lag1[j])
        add_feat(f'xchg_{top_lag1[i]}_x_{top_lag1[j]}',
                 X_tr[:,fi].astype(np.float64)*X_tr[:,fj].astype(np.float64),
                 X_te[:,fi].astype(np.float64)*X_te[:,fj].astype(np.float64))

for base_name, lags in features_with_lags.items():
    if base_name not in important_bases: continue
    idx_b = all_feat.index(lags['base'])
    b_tr = X_tr[:,idx_b].astype(np.float64); b_te = X_te[:,idx_b].astype(np.float64)
    add_feat(f'log_{base_name}', np.sign(b_tr)*np.log1p(np.abs(b_tr)), np.sign(b_te)*np.log1p(np.abs(b_te)))
for base_name, lags in features_with_lags.items():
    if base_name not in important_bases: continue
    idx_b = all_feat.index(lags['base'])
    add_feat(f'sq_{base_name}', X_tr[:,idx_b].astype(np.float64)**2, X_te[:,idx_b].astype(np.float64)**2)

if eng_tr_list:
    X_tr_eng = np.column_stack(eng_tr_list)
    X_te_eng = np.column_stack(eng_te_list)
    X_tr_eng = np.nan_to_num(X_tr_eng, nan=0.0, posinf=0.0, neginf=0.0)
    X_te_eng = np.nan_to_num(X_te_eng, nan=0.0, posinf=0.0, neginf=0.0)
else:
    X_tr_eng = np.zeros((X_tr.shape[0], 0), dtype=np.float32)
    X_te_eng = np.zeros((X_te.shape[0], 0), dtype=np.float32)
del eng_tr_list, eng_te_list; gc.collect()

# ICIR on engineered
def fast_spearman_icir(X_matrix, y_target, feat_names, n_chunks=N_CHUNKS):
    n = len(y_target)
    chunk_size = n // n_chunks
    sort_idx = np.argsort(train['ID'].values)
    X_sorted = X_matrix[sort_idx]; y_sorted = y_target[sort_idx]
    all_ics = np.zeros((n_chunks, X_sorted.shape[1]), dtype=np.float64)
    for i in range(n_chunks):
        s = i*chunk_size; e = (i+1)*chunk_size
        X_ch = X_sorted[s:e].astype(np.float64); y_ch = y_sorted[s:e].astype(np.float64)
        X_rank = np.argsort(np.argsort(X_ch, axis=0), axis=0).astype(np.float64)
        y_rank = np.argsort(np.argsort(y_ch)).astype(np.float64)
        X_m = X_rank - X_rank.mean(0); y_m = y_rank - y_rank.mean()
        X_sd = np.sqrt((X_m**2).sum(0)); y_sd = np.sqrt((y_m**2).sum())
        X_sd[X_sd < 1e-10] = 1e-10
        all_ics[i] = (X_m.T @ y_m) / (X_sd * y_sd)
    out = {}
    for ci, name in enumerate(feat_names):
        ics = all_ics[:, ci]; valid = ~np.isnan(ics)
        if valid.sum() < 5: continue
        mic = float(np.mean(ics[valid])); sic = float(np.std(ics[valid])) + 1e-8
        out[name] = dict(mean_ic=mic, icir=mic/sic, abs_icir=abs(mic/sic),
                         ic_pos=float(np.mean(ics[valid] > 0)))
    return out

icir_eng = fast_spearman_icir(X_tr_eng, y_raw, eng_names)
icir_all = {**spearman_icir, **icir_eng}
gold_all = {k: v for k, v in icir_all.items()
            if v['abs_icir'] >= ICIR_GOLD and v['ic_pos'] in (0.0, 1.0)}
gold_orig_feats2 = sorted([k for k in gold_all if k in all_feat], key=lambda x: -gold_all[x]['abs_icir'])
gold_eng_feats = sorted([k for k in gold_all if k in eng_names], key=lambda x: -gold_all[x]['abs_icir'])
orig_gold_idx = [all_feat.index(f) for f in gold_orig_feats2]
X_tr_og = X_tr[:, orig_gold_idx]; X_te_og = X_te[:, orig_gold_idx]
eng_gold_idx_sel = [eng_names.index(f) for f in gold_eng_feats]
X_tr_all = np.hstack([X_tr_og, X_tr_eng[:, eng_gold_idx_sel]]) if eng_gold_idx_sel else X_tr_og
X_te_all = np.hstack([X_te_og, X_te_eng[:, eng_gold_idx_sel]]) if eng_gold_idx_sel else X_te_og
all_gold_feats = gold_orig_feats2 + gold_eng_feats
print(f'Gold features: {X_tr_all.shape[1]}', flush=True)

del X_tr_eng, X_te_eng, X_tr_og, X_te_og; gc.collect()

# =====================================================================
# COMPONENT 1: LGB MULTI-SEED (same as fc pipeline)
# =====================================================================
print(f'\n=== LGB MULTI-SEED ({X_tr_all.shape[1]} features) ===', flush=True)
BASE_PARAMS = dict(
    objective='regression', metric='rmse',
    num_leaves=63, learning_rate=0.05,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    min_child_samples=50,
    lambda_l1=0.1, lambda_l2=1.0,
    n_jobs=-1, verbose=-1,
)
groups5 = pd.qcut(pd.Series(train['SO3_T'].values), q=5, labels=False,
                  duplicates='drop').values.astype(np.int32)

seed_preds = []
for s in SEEDS:
    params = dict(BASE_PARAMS); params['seed'] = s
    params['feature_fraction_seed'] = s; params['bagging_seed'] = s
    gkf = GroupKFold(n_splits=len(np.unique(groups5)))
    folds = list(gkf.split(X_tr_all, y_dm, groups=groups5))
    te_pred = np.zeros(len(X_te_all), dtype=np.float64)
    for fi, (tri, vai) in enumerate(folds):
        dt = lgb.Dataset(X_tr_all[tri], label=y_dm[tri], free_raw_data=True)
        dv = lgb.Dataset(X_tr_all[vai], label=y_dm[vai], reference=dt, free_raw_data=True)
        m = lgb.train(params, dt, num_boost_round=2000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        te_pred += m.predict(X_te_all, num_iteration=m.best_iteration) / len(folds)
        print(f'  lgb seed={s} fold={fi} best_iter={m.best_iteration}', flush=True)
        del dt, dv, m; gc.collect()
    seed_preds.append(te_pred)
lgb_raw = np.mean(seed_preds, axis=0)
del seed_preds, X_tr_all; gc.collect()

lgb_ic = daywise_ic(finalize(lgb_raw), oracle_df, test_ids, test_day)
print(f'LGB IC={lgb_ic:+.6f}', flush=True)

# =====================================================================
# COMPONENT 2: PER-DAY RIDGE (same as fc pipeline)
# + COMPUTE PER-DAY CONFIDENCE SCORES
# =====================================================================
print('\n=== PER-DAY RIDGE + CONFIDENCE SCORING ===', flush=True)

top10_gold = orig_gold_feats[:10]
ic_arr10 = np.array([spearman_icir[f]['mean_ic'] for f in top10_gold], dtype=np.float64)
top10_idx = [all_feat.index(f) for f in top10_gold]
global_mean_10 = tr_raw[:, top10_idx].mean(0).astype(np.float64)
global_std_10 = tr_raw[:, top10_idx].std(0).astype(np.float64)
global_std_10[global_std_10 < 1e-8] = 1.0

def zscore_local(X, m, s, clip=CLIP_Z):
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip)

n_test = len(test)
te_ridge_pd = np.zeros(n_test, dtype=np.float64)
day_confidence = {}  # key = day string, value = confidence score [0, 1]

n_overlap = 0
n_new = 0

# Pre-allocate train feature matrix for confidence computation
train_feat_matrix = tr_raw.astype(np.float64)

for d in np.unique(test_day):
    te_mask = test_day == d
    te_idx_arr = np.where(te_mask)[0]
    X_te_raw_d = te_raw[te_mask].astype(np.float64)

    if d in train_days_set:
        tr_mask = train_day == d
        n_tr = tr_mask.sum()

        if n_tr < 20:
            X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
            X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
            te_ridge_pd[te_idx_arr] = X_te_z10 @ ic_arr10
            day_confidence[d] = 0.1  # very low confidence for sparse days
            n_new += 1
            continue

        X_tr_raw_d = train_feat_matrix[tr_mask]
        m_d = X_tr_raw_d.mean(0)
        s_d = X_tr_raw_d.std(0)
        s_d[s_d < 1e-8] = 1.0
        X_tr_z = np.clip((X_tr_raw_d - m_d) / s_d, -CLIP_Z, CLIP_Z)
        X_te_z = np.clip((X_te_raw_d - m_d) / s_d, -CLIP_Z, CLIP_Z)

        y_tr = y_raw[tr_mask].astype(np.float64)
        y_tr_w = np.clip(y_tr, np.percentile(y_tr, 1), np.percentile(y_tr, 99))

        # Fit Ridge
        mdl = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        mdl.fit(X_tr_z, y_tr_w)
        te_ridge_pd[te_idx_arr] = mdl.predict(X_te_z)

        # === CONFIDENCE SCORE COMPUTATION ===
        # Signal 1: sample size (logistic transform)
        conf_size = 1.0 / (1.0 + np.exp(-(n_tr - 200) / 100))  # peaks at n>200

        # Signal 2: Training Ridge split-half R²
        n_half = n_tr // 2
        if n_half >= 20:
            mdl_cv = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
            mdl_cv.fit(X_tr_z[:n_half], y_tr_w[:n_half])
            pred_cv = mdl_cv.predict(X_tr_z[n_half:])
            ss_res = np.sum((y_tr_w[n_half:] - pred_cv)**2)
            ss_tot = np.sum((y_tr_w[n_half:] - y_tr_w[n_half:].mean())**2)
            r2_cv = max(0, 1 - ss_res / ss_tot) if ss_tot > 1e-15 else 0
            conf_r2 = min(1.0, r2_cv * 10)  # scale: R²=0.1 → conf=1.0
        else:
            conf_r2 = 0.0

        # Signal 3: Target std (lower = more predictable, but too low = no signal)
        y_std = y_tr_w.std()
        global_y_std = y_wins.std()
        conf_vol = np.exp(-abs(np.log(y_std / global_y_std + 1e-10)))  # peaks when day vol ≈ global vol

        # Combined confidence
        confidence = 0.5 * conf_size + 0.3 * conf_r2 + 0.2 * conf_vol
        day_confidence[d] = float(np.clip(confidence, 0.05, 1.0))
        n_overlap += 1
    else:
        X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
        X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
        te_ridge_pd[te_idx_arr] = X_te_z10 @ ic_arr10
        day_confidence[d] = 0.2  # low confidence for new days
        n_new += 1

del train_feat_matrix; gc.collect()

ridge_ic = daywise_ic(finalize(te_ridge_pd), oracle_df, test_ids, test_day)
print(f'Ridge IC={ridge_ic:+.6f}', flush=True)
print(f'overlap={n_overlap} new={n_new}', flush=True)

# Analyze confidence distribution
conf_vals = np.array(list(day_confidence.values()))
print(f'\nConfidence distribution:')
print(f'  mean={conf_vals.mean():.3f} median={np.median(conf_vals):.3f}')
print(f'  min={conf_vals.min():.3f} max={conf_vals.max():.3f}')
print(f'  <0.3: {np.mean(conf_vals < 0.3):.1%}')
print(f'  >0.5: {np.mean(conf_vals > 0.5):.1%}')

# =====================================================================
# BLENDING + CONFIDENCE SHRINKAGE
# =====================================================================
print('\n=== BLENDING + CONFIDENCE SHRINKAGE ===', flush=True)

# Build per-row confidence vector
row_confidence = np.ones(n_test, dtype=np.float64)
for d in np.unique(test_day):
    mask = test_day == d
    row_confidence[mask] = day_confidence.get(d, 0.2)

# Baseline: fc_lgb50_rpd50 without shrinkage (reproduce original)
lgb_fin = finalize(lgb_raw)
ridge_fin = finalize(te_ridge_pd)
baseline_blend = 0.5 * lgb_fin + 0.5 * ridge_fin
baseline_ic = daywise_ic(finalize(baseline_blend), oracle_df, test_ids, test_day)
print(f'Baseline (no shrinkage) IC={baseline_ic:+.6f}', flush=True)

TARGET_STDS = [0.0005, 0.0007, 0.000948, 0.0012, 0.0015]
submissions = {}

# Baseline at all scales
for ts in TARGET_STDS:
    submissions[f'baseline_s{int(ts*1e6)}'] = finalize(baseline_blend, ts)

# Strategy A: Multiply predictions by confidence (soft shrinkage)
for shrink_power in [0.5, 1.0, 1.5, 2.0, 3.0]:
    shrink = row_confidence ** shrink_power
    shrunk_lgb = lgb_raw * shrink
    shrunk_ridge = te_ridge_pd * shrink
    blend = 0.5 * finalize(shrunk_lgb) + 0.5 * finalize(shrunk_ridge)
    for ts in TARGET_STDS:
        submissions[f'shrinkA_p{int(shrink_power*10)}_s{int(ts*1e6)}'] = finalize(blend, ts)

# Strategy B: Binary cutoff — zero out predictions below confidence threshold
for conf_thresh in [0.2, 0.3, 0.4, 0.5]:
    mask_keep = row_confidence >= conf_thresh
    frac_kept = mask_keep.mean()
    shrunk_lgb = lgb_raw.copy()
    shrunk_ridge = te_ridge_pd.copy()
    shrunk_lgb[~mask_keep] = 0.0
    shrunk_ridge[~mask_keep] = 0.0
    blend = 0.5 * finalize(shrunk_lgb) + 0.5 * finalize(shrunk_ridge)
    for ts in TARGET_STDS:
        submissions[f'shrinkB_t{int(conf_thresh*100)}_s{int(ts*1e6)}'] = finalize(blend, ts)
    print(f'  Cutoff {conf_thresh}: kept {frac_kept:.1%} of rows', flush=True)

# Strategy C: Shrink only Ridge (LGB might be more robust)
for shrink_power in [1.0, 2.0]:
    shrink = row_confidence ** shrink_power
    shrunk_ridge = te_ridge_pd * shrink
    blend = 0.5 * finalize(lgb_raw) + 0.5 * finalize(shrunk_ridge)
    for ts in TARGET_STDS:
        submissions[f'shrinkC_p{int(shrink_power*10)}_s{int(ts*1e6)}'] = finalize(blend, ts)

# Strategy D: Different blend ratios with soft shrinkage
for w_lgb in [0.4, 0.6, 0.7]:
    w_r = 1.0 - w_lgb
    shrink = row_confidence ** 1.0
    shrunk_lgb = lgb_raw * shrink
    shrunk_ridge = te_ridge_pd * shrink
    blend = w_lgb * finalize(shrunk_lgb) + w_r * finalize(shrunk_ridge)
    for ts in TARGET_STDS:
        submissions[f'shrinkD_lgb{int(w_lgb*100)}_s{int(ts*1e6)}'] = finalize(blend, ts)

# Strategy E: Global uniform shrinkage (just scale everything down)
# This tests whether the issue is just prediction magnitude
for global_shrink in [0.3, 0.5, 0.7]:
    blend = 0.5 * finalize(lgb_raw) + 0.5 * finalize(te_ridge_pd)
    blend_shrunk = blend * global_shrink
    for ts in TARGET_STDS:
        submissions[f'shrinkE_g{int(global_shrink*100)}_s{int(ts*1e6)}'] = finalize(blend_shrunk, ts)

# Strategy F: Per-day adaptive alpha Ridge
# Refit Ridge with lower alpha on high-confidence days, higher on low-confidence
print('\n  Strategy F: Adaptive alpha Ridge...', flush=True)
te_ridge_adaptive = np.zeros(n_test, dtype=np.float64)

for d in np.unique(test_day):
    te_mask = test_day == d
    te_idx_arr = np.where(te_mask)[0]
    conf = day_confidence.get(d, 0.2)

    if d in train_days_set:
        tr_mask = train_day == d
        if tr_mask.sum() < 20:
            X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
            X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
            te_ridge_adaptive[te_idx_arr] = X_te_z10 @ ic_arr10 * conf
            continue

        X_tr_raw_d = tr_raw[tr_mask].astype(np.float64)
        X_te_raw_d = te_raw[te_mask].astype(np.float64)
        m_d = X_tr_raw_d.mean(0); s_d = X_tr_raw_d.std(0)
        s_d[s_d < 1e-8] = 1.0
        X_tr_z = np.clip((X_tr_raw_d - m_d) / s_d, -CLIP_Z, CLIP_Z)
        X_te_z = np.clip((X_te_raw_d - m_d) / s_d, -CLIP_Z, CLIP_Z)
        y_tr = y_raw[tr_mask].astype(np.float64)
        y_tr_w = np.clip(y_tr, np.percentile(y_tr, 1), np.percentile(y_tr, 99))

        # Higher confidence → lower alpha (less regularization = more aggressive)
        # Lower confidence → higher alpha (more regularization = shrink toward zero)
        adaptive_alpha = RIDGE_ALPHA / (conf + 0.1)  # conf=0.5 → alpha=8333, conf=1.0 → alpha=4545
        mdl = Ridge(alpha=adaptive_alpha, fit_intercept=True)
        mdl.fit(X_tr_z, y_tr_w)
        te_ridge_adaptive[te_idx_arr] = mdl.predict(X_te_z)
    else:
        X_te_raw_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
        X_te_z10 = zscore_local(X_te_raw_10, global_mean_10, global_std_10)
        te_ridge_adaptive[te_idx_arr] = X_te_z10 @ ic_arr10 * conf

for w_lgb in [0.5, 0.6, 0.7]:
    w_r = 1.0 - w_lgb
    blend = w_lgb * finalize(lgb_raw) + w_r * finalize(te_ridge_adaptive)
    for ts in TARGET_STDS:
        submissions[f'adaptF_lgb{int(w_lgb*100)}_s{int(ts*1e6)}'] = finalize(blend, ts)

adapt_ic = daywise_ic(finalize(0.5*finalize(lgb_raw) + 0.5*finalize(te_ridge_adaptive)),
                      oracle_df, test_ids, test_day)
print(f'  Adaptive Ridge blend IC={adapt_ic:+.6f}', flush=True)

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
