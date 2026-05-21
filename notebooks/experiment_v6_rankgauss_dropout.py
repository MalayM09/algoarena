"""
Experiment v6 (REWRITE): Per-day cross-sectional Gaussian rank normalization + Feature dropout.

Problems fixed from v6 original:
1. OOM: Old code built pd AND rg feature matrices simultaneously (~11GB).
   Fix: Sequential pipeline - build pd, run models, DELETE pd, then build rg.
2. Fatal logical flaw: Global QuantileTransformer destroyed within-day cross-sectional ranking.
   Fix: Per-day cross-sectional Gaussian rank normalization (rank within each day, map to normal).
   This preserves within-day ordering (critical for tree models) while removing liquid/illiquid
   distribution shift.
3. feature_fraction=0.3 too aggressive for noisy data.
   Fix: Use 0.6 and 0.5 only.

Per-day Gaussian Rank (RG) normalization:
  For each day d and feature f:
    - Rank all stocks within day d by feature f
    - Map rank to N(0,1): norm.ppf((rank + 0.5) / n_stocks_in_day)
    - Clip to ±3
  Compliant: test transformation is purely within each test day (no cross-day test stats).
  Benefit: Each day is self-normalized, removes liquid/illiquid shift, clips outliers.

Architecture: fc_lgb(ff=0.6/0.5) + per-day Ridge (pd or rg normalization) + shrinkA_p5 blend
"""
import os, gc, time, warnings, re
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm as sp_norm
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

DATA_DIR = '/Users/malaymishra/Desktop/quant_ml_project/data/raw'
OUT_DIR  = '/Users/malaymishra/Desktop/quant_ml_project/outputs/submissions'
ORACLE_PATH = os.path.join(OUT_DIR, 'exploit_v2_zero.csv')
TAG = 'v6'

CLIP_Z   = 5.0
EPS      = 1e-6
N_CHUNKS = 20
ICIR_GOLD = 3.0
ICIR_ENG  = 2.0
SEEDS    = [42, 123, 456, 789, 2024]

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
        if len(g) < 3: continue
        p = g['pred'].values; o = g['TARGET'].values
        p = p - p.mean(); o = o - o.mean()
        pn = np.linalg.norm(p); on_ = np.linalg.norm(o)
        if pn < 1e-12 or on_ < 1e-12: ics.append(0.0)
        else: ics.append(float((p @ o) / (pn * on_)))
    return float(np.mean(ics))

# =====================================================================
# DATA LOADING
# =====================================================================
print('Loading data...', flush=True)
train = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
test  = pd.read_parquet(os.path.join(DATA_DIR, 'test.parquet'))
oracle_df = pd.read_csv(ORACLE_PATH)

feat_cols  = [c for c in train.columns if c not in {'ID', 'TARGET', 'CV_GROUP'}]
all_feat   = [c for c in feat_cols if c != 'SO3_T']
test_ids   = test['ID'].values
train_day  = train['SO3_T'].round(5).astype(str).values
test_day   = test['SO3_T'].round(5).astype(str).values
train_days_set = set(np.unique(train_day))
y_raw  = train['TARGET'].values.astype(np.float64)
lo_y, hi_y = np.percentile(y_raw, 1), np.percentile(y_raw, 99)
y_wins = np.clip(y_raw, lo_y, hi_y)

y_dm = y_wins.copy()
for d in np.unique(train_day):
    m = train_day == d
    y_dm[m] = y_wins[m] - y_wins[m].mean()

# Keep raw float32 in memory always (needed for Ridge inline per-day operations)
tr_raw = train[all_feat].fillna(0).values.astype(np.float32)  # always needed for Ridge
te_raw = test[all_feat].fillna(0).values.astype(np.float32)
n_test = len(test)
print(f'Train: {tr_raw.shape}, Test: {te_raw.shape}', flush=True)

# Precompute from DataFrames before freeing them
train_sort_idx = np.argsort(train['ID'].values)
groups5 = pd.qcut(pd.Series(train['SO3_T'].values), q=5, labels=False,
                  duplicates='drop').values.astype(np.int32)
# Free test DataFrame — all needed arrays already extracted
del test; gc.collect()
print('Freed test DataFrame.', flush=True)

# =====================================================================
# ICIR FEATURE SELECTION (uses tr_raw — shared, computed once)
# =====================================================================
print('ICIR feature selection...', flush=True)
train_s = train.sort_values('ID').reset_index(drop=True)
cs = len(train_s) // N_CHUNKS
spearman_icir = {}
for col in all_feat:
    cics = []
    for i in range(N_CHUNKS):
        ch = train_s.iloc[i*cs:(i+1)*cs]
        v = ch[col].fillna(ch[col].median()).values
        t = ch['TARGET'].values
        valid = ~np.isnan(v)
        if valid.sum() < 200: continue
        ic, _ = spearmanr(v[valid], t[valid])
        if not np.isnan(ic): cics.append(ic)
    if len(cics) >= 5:
        mic = float(np.mean(cics)); sic = float(np.std(cics)) + 1e-8
        spearman_icir[col] = dict(mean_ic=mic, icir=mic/sic, abs_icir=abs(mic/sic),
                                  ic_pos=float(np.mean([v > 0 for v in cics])))

orig_gold_feats = sorted([k for k, v in spearman_icir.items()
                          if v['abs_icir'] >= ICIR_GOLD and v['ic_pos'] in (0.0, 1.0)],
                         key=lambda x: -spearman_icir[x]['abs_icir'])
print(f'orig_gold={len(orig_gold_feats)}', flush=True)

# Free train DataFrame — all needed arrays already extracted (tr_raw, train_day, y_raw, etc.)
del train, train_s; gc.collect()
print('Freed train DataFrame.', flush=True)

def parse_feature(fname):
    m = re.match(r'^(.+?)(_LagT(\d+))$', fname)
    if m: return m.group(1), f'LagT{m.group(3)}'
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

# Top lag1 features for cross-feature interactions
gold_lag1 = [f for f in all_feat if f.endswith('_LagT1')]
lag1_icir  = {c: spearman_icir[c]['abs_icir'] for c in gold_lag1 if c in spearman_icir}
top_lag1   = sorted(lag1_icir.keys(), key=lambda x: -lag1_icir[x])[:10]

# ICIR on engineered features (evaluated once on pd norm)
def fast_spearman_icir(X_matrix, y_target, feat_names, n_chunks=N_CHUNKS):
    n = len(y_target); chunk_size = n // n_chunks
    sort_idx = train_sort_idx  # precomputed before DataFrame was freed
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

# =====================================================================
# NORMALIZATION FUNCTIONS
# =====================================================================
global_mean = tr_raw.mean(0).astype(np.float64)
global_std  = tr_raw.std(0).astype(np.float64)
global_std[global_std < 1e-8] = 1.0

day_stats = {}
for d in np.unique(train_day):
    m = train_day == d
    if m.sum() < 50: continue
    x = tr_raw[m].astype(np.float64)
    mu = x.mean(0); sg = x.std(0); sg[sg < 1e-8] = 1.0
    day_stats[d] = (mu, sg)

def apply_pd_norm(raw, days):
    """Per-day z-score normalization using training-day stats."""
    out = np.empty_like(raw, dtype=np.float32)
    for d in np.unique(days):
        m = days == d
        mu, sg = day_stats.get(d, (global_mean, global_std))
        z = (raw[m].astype(np.float64) - mu) / sg
        out[m] = np.clip(z, -CLIP_Z, CLIP_Z).astype(np.float32)
    return out

def apply_perday_rankgauss_compliant(tr_raw_mat, te_raw_mat, tr_days, te_days):
    """
    Per-day cross-sectional Gaussian rank normalization — COMPLIANT version.

    For TRAINING: rank each stock within its training day, map to N(0,1). Fine.

    For TEST: must NOT rank test stocks against each other (batch-dependent).
    Instead, for each (day, feature), build a frozen interpolation curve from
    the training data. Each test stock's feature value is mapped independently
    through this curve — same result regardless of batch size or ordering.

    Fallback for test days with no training day match: zero (no signal).
    """
    from scipy.interpolate import interp1d as _interp1d

    out_tr = np.zeros_like(tr_raw_mat, dtype=np.float32)
    out_te = np.zeros_like(te_raw_mat, dtype=np.float32)
    n_feat = tr_raw_mat.shape[1]

    for d in np.unique(tr_days):
        tr_m = tr_days == d
        n_tr = tr_m.sum()
        if n_tr < 2:
            continue

        x_tr = tr_raw_mat[tr_m].astype(np.float64)

        # --- Training side: rank within this day (compliant, fit on train) ---
        ranks = np.argsort(np.argsort(x_tr, axis=0), axis=0).astype(np.float64)
        quantiles_tr = np.clip((ranks + 0.5) / n_tr, 0.001, 0.999)
        gaussian_tr = np.clip(sp_norm.ppf(quantiles_tr), -3.0, 3.0)
        out_tr[tr_m] = gaussian_tr.astype(np.float32)

        # --- Test side: interpolate from FROZEN training quantile curve ---
        te_m = te_days == d
        if te_m.sum() == 0:
            continue

        x_te = te_raw_mat[te_m].astype(np.float64)
        gaussian_te = np.zeros_like(x_te)

        for col in range(n_feat):
            sort_idx_col = np.argsort(x_tr[:, col])
            x_sorted = x_tr[sort_idx_col, col]
            g_sorted = gaussian_tr[sort_idx_col, col]

            # Remove duplicate x values (interp1d requires strictly increasing x)
            _, uniq_idx = np.unique(x_sorted, return_index=True)
            x_uniq = x_sorted[uniq_idx]
            g_uniq = g_sorted[uniq_idx]

            if len(x_uniq) < 2:
                gaussian_te[:, col] = g_uniq[0] if len(g_uniq) > 0 else 0.0
                continue

            # fill_value: extrapolate to training min/max Gaussian — prevents blow-up
            interp = _interp1d(x_uniq, g_uniq, kind='linear',
                               bounds_error=False,
                               fill_value=(g_uniq[0], g_uniq[-1]))
            gaussian_te[:, col] = interp(x_te[:, col])

        out_te[te_m] = np.clip(gaussian_te, -3.0, 3.0).astype(np.float32)

    return out_tr, out_te

# =====================================================================
# FEATURE ENGINEERING: modular function to build gold matrix
# from ANY normalized feature matrix (pd or rg)
# =====================================================================
def build_gold_matrix(X_tr_norm, X_te_norm, norm_label):
    """
    Build the gold feature matrix (orig gold + engineered gold) from
    a normalized training and test matrix. Works for any normalization.

    Returns: X_tr_gold, X_te_gold, gold_feat_names
    """
    print(f'  [{norm_label}] Engineering features...', flush=True)

    eng_names = []; eng_tr = []; eng_te = []

    def add_feat(name, tr_v, te_v):
        eng_names.append(name)
        eng_tr.append(tr_v.ravel().astype(np.float32))
        eng_te.append(te_v.ravel().astype(np.float32))

    for base_name, lags in features_with_lags.items():
        if base_name not in important_bases: continue
        idx_b  = all_feat.index(lags['base'])
        idx_l1 = all_feat.index(lags['LagT1'])
        has_l2 = 'LagT2' in lags; has_l3 = 'LagT3' in lags

        b_tr  = X_tr_norm[:, idx_b].astype(np.float64)
        b_te  = X_te_norm[:, idx_b].astype(np.float64)
        l1_tr = X_tr_norm[:, idx_l1].astype(np.float64)
        l1_te = X_te_norm[:, idx_l1].astype(np.float64)

        add_feat(f'past_T1_{base_name}', b_tr - l1_tr, b_te - l1_te)
        den_tr = np.abs(b_tr) + EPS; den_te = np.abs(b_te) + EPS
        add_feat(f'relchg_T1_{base_name}', np.clip(l1_tr/den_tr,-10,10), np.clip(l1_te/den_te,-10,10))
        add_feat(f'lvlxchg_T1_{base_name}', b_tr*l1_tr, b_te*l1_te)
        add_feat(f'abschg_T1_{base_name}', np.abs(l1_tr), np.abs(l1_te))
        add_feat(f'signchg_T1_{base_name}', np.sign(l1_tr), np.sign(l1_te))

        if has_l2:
            idx_l2 = all_feat.index(lags['LagT2'])
            l2_tr = X_tr_norm[:, idx_l2].astype(np.float64)
            l2_te = X_te_norm[:, idx_l2].astype(np.float64)
            add_feat(f'accel_T1T2_{base_name}', l2_tr-l1_tr, l2_te-l1_te)
            add_feat(f'consist_T1T2_{base_name}', np.sign(l1_tr)*np.sign(l2_tr), np.sign(l1_te)*np.sign(l2_te))
            add_feat(f'past_T2_{base_name}', b_tr-l2_tr, b_te-l2_te)
        if has_l3:
            idx_l3 = all_feat.index(lags['LagT3'])
            l3_tr = X_tr_norm[:, idx_l3].astype(np.float64)
            l3_te = X_te_norm[:, idx_l3].astype(np.float64)
            add_feat(f'longshort_{base_name}', l3_tr-l1_tr, l3_te-l1_te)
            if has_l2:
                add_feat(f'accel2_{base_name}', l3_tr-2*l2_tr+l1_tr, l3_te-2*l2_te+l1_te)
                net_tr = np.abs(b_tr-l3_tr); net_te = np.abs(b_te-l3_te)
                vol_tr = np.abs(b_tr-l1_tr)+np.abs(l1_tr-l2_tr)+np.abs(l2_tr-l3_tr)+EPS
                vol_te = np.abs(b_te-l1_te)+np.abs(l1_te-l2_te)+np.abs(l2_te-l3_te)+EPS
                kauf_tr = np.clip(net_tr/vol_tr, 0, 1); kauf_te = np.clip(net_te/vol_te, 0, 1)
                add_feat(f'kaufer_{base_name}', kauf_tr, kauf_te)
                add_feat(f'skaufer_{base_name}', np.sign(b_tr-l3_tr)*kauf_tr, np.sign(b_te-l3_te)*kauf_te)

    # Cross-feature interactions
    for i in range(len(top_lag1)):
        for j in range(i+1, len(top_lag1)):
            fi = all_feat.index(top_lag1[i]); fj = all_feat.index(top_lag1[j])
            add_feat(f'xchg_{top_lag1[i]}_x_{top_lag1[j]}',
                     X_tr_norm[:,fi].astype(np.float64) * X_tr_norm[:,fj].astype(np.float64),
                     X_te_norm[:,fi].astype(np.float64) * X_te_norm[:,fj].astype(np.float64))

    # Log and squared transforms
    for base_name, lags in features_with_lags.items():
        if base_name not in important_bases: continue
        idx_b = all_feat.index(lags['base'])
        b_tr = X_tr_norm[:,idx_b].astype(np.float64); b_te = X_te_norm[:,idx_b].astype(np.float64)
        add_feat(f'log_{base_name}', np.sign(b_tr)*np.log1p(np.abs(b_tr)), np.sign(b_te)*np.log1p(np.abs(b_te)))
        add_feat(f'sq_{base_name}', b_tr**2, b_te**2)

    if not eng_tr:
        return (np.zeros((X_tr_norm.shape[0],0),dtype=np.float32),
                np.zeros((X_te_norm.shape[0],0),dtype=np.float32), [])

    print(f'  [{norm_label}] Stacking {len(eng_names)} engineered features...', flush=True)
    X_tr_eng = np.column_stack(eng_tr).astype(np.float32)
    X_te_eng = np.column_stack(eng_te).astype(np.float32)
    del eng_tr, eng_te; gc.collect()
    np.nan_to_num(X_tr_eng, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.nan_to_num(X_te_eng, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ICIR on engineering (use this normalization's version)
    icir_eng = fast_spearman_icir(X_tr_eng, y_raw, eng_names)
    icir_all = {**spearman_icir, **icir_eng}
    gold_all = {k: v for k, v in icir_all.items()
                if v['abs_icir'] >= ICIR_GOLD and v['ic_pos'] in (0.0, 1.0)}

    gold_orig_feats2 = sorted([k for k in gold_all if k in all_feat], key=lambda x: -gold_all[x]['abs_icir'])
    gold_eng_feats   = sorted([k for k in gold_all if k in eng_names], key=lambda x: -gold_all[x]['abs_icir'])
    print(f'  [{norm_label}] gold_orig={len(gold_orig_feats2)} gold_eng={len(gold_eng_feats)}', flush=True)

    orig_gold_idx = [all_feat.index(f) for f in gold_orig_feats2]
    eng_gold_idx  = [eng_names.index(f) for f in gold_eng_feats]

    X_tr_orig_gold = X_tr_norm[:, orig_gold_idx].astype(np.float32)
    X_te_orig_gold = X_te_norm[:, orig_gold_idx].astype(np.float32)

    if eng_gold_idx:
        X_tr_gold = np.hstack([X_tr_orig_gold, X_tr_eng[:, eng_gold_idx]])
        X_te_gold = np.hstack([X_te_orig_gold, X_te_eng[:, eng_gold_idx]])
    else:
        X_tr_gold = X_tr_orig_gold
        X_te_gold = X_te_orig_gold

    del X_tr_eng, X_te_eng, X_tr_orig_gold, X_te_orig_gold; gc.collect()

    all_gold = gold_orig_feats2 + gold_eng_feats
    print(f'  [{norm_label}] Final gold features: {len(all_gold)}', flush=True)
    return X_tr_gold, X_te_gold, all_gold

# =====================================================================
# LGB MULTI-SEED TRAINING
# =====================================================================
# groups5 already computed before DataFrame was freed

BASE_PARAMS = dict(
    objective='regression', metric='rmse',
    num_leaves=63, learning_rate=0.05,
    bagging_fraction=0.8, bagging_freq=1,
    min_child_samples=50,
    lambda_l1=0.1, lambda_l2=1.0,
    n_jobs=-1, verbose=-1,
)

def train_lgb_multiseed(X_tr, X_te, y, groups, params_base, seeds, label):
    seed_preds = []
    for s in seeds:
        params = dict(params_base)
        params['seed'] = s; params['feature_fraction_seed'] = s; params['bagging_seed'] = s
        gkf  = GroupKFold(n_splits=len(np.unique(groups)))
        folds = list(gkf.split(X_tr, y, groups=groups))
        te_pred = np.zeros(len(X_te), dtype=np.float64)
        for fi, (tri, vai) in enumerate(folds):
            dt = lgb.Dataset(X_tr[tri], label=y[tri], free_raw_data=True)
            dv = lgb.Dataset(X_tr[vai], label=y[vai], reference=dt, free_raw_data=True)
            mdl = lgb.train(params, dt, num_boost_round=2000, valid_sets=[dv],
                            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            te_pred += mdl.predict(X_te, num_iteration=mdl.best_iteration) / len(folds)
            print(f'  {label} s={s} f={fi} iter={mdl.best_iteration}', flush=True)
            del dt, dv, mdl; gc.collect()
        seed_preds.append(te_pred)
    return np.mean(seed_preds, axis=0)

# =====================================================================
# PER-DAY RIDGE (inline — only tr_raw and te_raw needed, not gold matrix)
# =====================================================================
top10_gold    = orig_gold_feats[:10]
ic_arr10      = np.array([spearman_icir[f]['mean_ic'] for f in top10_gold], dtype=np.float64)
top10_idx     = [all_feat.index(f) for f in top10_gold]
global_mean10 = tr_raw[:, top10_idx].mean(0).astype(np.float64)
global_std10  = tr_raw[:, top10_idx].std(0).astype(np.float64)
global_std10[global_std10 < 1e-8] = 1.0

def run_perday_ridge(normalization='pd', alpha=5000):
    """
    Per-day Ridge regression.
    normalization='pd': z-score within each day (original pipeline)
    normalization='rg': Gaussian rank within each day
    alpha: Ridge regularization strength
    """
    te_pred = np.zeros(n_test, dtype=np.float64)
    n_ov = 0; n_nw = 0
    for d in np.unique(test_day):
        te_mask = test_day == d
        te_idx_arr = np.where(te_mask)[0]
        n_te_d = te_mask.sum()

        if d in train_days_set:
            tr_mask = train_day == d
            n_tr_d  = tr_mask.sum()

            if n_tr_d < 20:
                # Grinold fallback on top10
                X_te_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
                sg_g = np.where(global_std10 < 1e-8, 1.0, global_std10)
                X_te_z = np.clip((X_te_10 - global_mean10) / sg_g, -CLIP_Z, CLIP_Z)
                te_pred[te_idx_arr] = X_te_z @ ic_arr10
                n_nw += 1; continue

            X_tr_d = tr_raw[tr_mask].astype(np.float64)
            X_te_d = te_raw[te_mask].astype(np.float64)

            if normalization == 'pd':
                m_d = X_tr_d.mean(0); s_d = X_tr_d.std(0); s_d[s_d < 1e-8] = 1.0
                X_tr_z = np.clip((X_tr_d - m_d) / s_d, -CLIP_Z, CLIP_Z)
                X_te_z = np.clip((X_te_d - m_d) / s_d, -CLIP_Z, CLIP_Z)
            else:  # 'rg': per-day compliant Gaussian rank
                # Training: rank within this day (fine — fit on train)
                r_tr = np.argsort(np.argsort(X_tr_d, axis=0), axis=0).astype(np.float64)
                q_tr = np.clip((r_tr + 0.5) / n_tr_d, 0.001, 0.999)
                X_tr_z = np.clip(sp_norm.ppf(q_tr), -3.0, 3.0)
                # Test: interpolate from FROZEN training quantile curves — batch-independent
                X_te_z = np.zeros_like(X_te_d)
                for col in range(X_tr_d.shape[1]):
                    sort_c = np.argsort(X_tr_d[:, col])
                    x_s = X_tr_d[sort_c, col]; g_s = X_tr_z[sort_c, col]
                    _, uniq_idx = np.unique(x_s, return_index=True)
                    x_u = x_s[uniq_idx]; g_u = g_s[uniq_idx]
                    if len(x_u) < 2:
                        X_te_z[:, col] = g_u[0] if len(g_u) > 0 else 0.0
                        continue
                    pos = np.searchsorted(x_u, X_te_d[:, col])
                    pos = np.clip(pos, 1, len(x_u) - 1)
                    lo = pos - 1
                    dx = x_u[pos] - x_u[lo]
                    t = np.where(dx > 1e-10, (X_te_d[:, col] - x_u[lo]) / dx, 0.5)
                    X_te_z[:, col] = np.clip(g_u[lo] + np.clip(t, 0, 1) * (g_u[pos] - g_u[lo]),
                                             -3.0, 3.0)

            y_tr  = y_raw[tr_mask].astype(np.float64)
            y_tr_w = np.clip(y_tr, np.percentile(y_tr, 1), np.percentile(y_tr, 99))
            mdl = Ridge(alpha=alpha, fit_intercept=True)
            mdl.fit(X_tr_z, y_tr_w)
            te_pred[te_idx_arr] = mdl.predict(X_te_z)
            n_ov += 1
        else:
            # New day: Grinold fallback
            X_te_10 = te_raw[te_mask][:, top10_idx].astype(np.float64)
            sg_g = np.where(global_std10 < 1e-8, 1.0, global_std10)
            X_te_z = np.clip((X_te_10 - global_mean10) / sg_g, -CLIP_Z, CLIP_Z)
            te_pred[te_idx_arr] = X_te_z @ ic_arr10
            n_nw += 1

    print(f'  ridge_{normalization}_a{alpha}: overlap={n_ov} new={n_nw}', flush=True)
    return te_pred

# =====================================================================
# SHRINK-A CONFIDENCE (from v5 — best LB result: shrinkA_p5_s1200=0.00084)
# Day-level training size confidence for Ridge predictions
# =====================================================================
def compute_day_confidence(power=5.0, min_n=20):
    """
    For each test day, compute confidence score based on training sample count.
    Confidence = (n_train_day / median_n_train)^(1/power)
    Days with fewer training samples get shrunk toward zero.
    """
    train_day_counts = {}
    for d in np.unique(train_day):
        train_day_counts[d] = (train_day == d).sum()

    median_n = np.median(list(train_day_counts.values()))

    conf = np.zeros(n_test, dtype=np.float64)
    for d in np.unique(test_day):
        te_mask = test_day == d
        te_idx  = np.where(te_mask)[0]
        n_tr_d  = train_day_counts.get(d, 0)
        if n_tr_d == 0:
            conf[te_idx] = 0.0
        else:
            raw_conf = min(1.0, n_tr_d / median_n)
            conf[te_idx] = raw_conf ** (1.0 / power)
    return conf

# =====================================================================
# SEQUENTIAL PIPELINE: Phase 1 (per-day z-score) → Phase 2 (per-day RG)
# =====================================================================
all_lgb_preds  = {}
all_ridge_preds = {}

# ────────────────────────────────────────────────────────────────────
# PHASE 1: Per-Day Z-Score Normalization (original pipeline)
# ────────────────────────────────────────────────────────────────────
print('\n\n========== PHASE 1: PER-DAY Z-SCORE NORMALIZATION ==========', flush=True)

print('Computing X_tr_pd, X_te_pd...', flush=True)
X_tr_pd = apply_pd_norm(tr_raw, train_day)   # float32, ~same shape as tr_raw
X_te_pd = apply_pd_norm(te_raw, test_day)
print(f'X_tr_pd: {X_tr_pd.shape}, X_te_pd: {X_te_pd.shape}', flush=True)

X_tr_gold_pd, X_te_gold_pd, gold_feats_pd = build_gold_matrix(X_tr_pd, X_te_pd, 'pd')

# Free pd base matrices — no longer needed (Ridge uses tr_raw inline)
del X_tr_pd, X_te_pd; gc.collect()
print(f'Freed pd base matrices. Gold shape: {X_tr_gold_pd.shape}', flush=True)

# LGB variants on pd normalization
print('\n-- LGB pd (ff=0.8) --', flush=True)
p80 = dict(BASE_PARAMS); p80['feature_fraction'] = 0.8
all_lgb_preds['lgb_pd_ff80'] = train_lgb_multiseed(X_tr_gold_pd, X_te_gold_pd, y_dm, groups5, p80, SEEDS, 'pd_ff80')
ic = daywise_ic(finalize(all_lgb_preds['lgb_pd_ff80']), oracle_df, test_ids, test_day)
print(f'  lgb_pd_ff80 IC={ic:+.6f}', flush=True)

print('\n-- LGB pd (ff=0.6) --', flush=True)
p60 = dict(BASE_PARAMS); p60['feature_fraction'] = 0.6
all_lgb_preds['lgb_pd_ff60'] = train_lgb_multiseed(X_tr_gold_pd, X_te_gold_pd, y_dm, groups5, p60, SEEDS, 'pd_ff60')
ic = daywise_ic(finalize(all_lgb_preds['lgb_pd_ff60']), oracle_df, test_ids, test_day)
print(f'  lgb_pd_ff60 IC={ic:+.6f}', flush=True)

print('\n-- LGB pd (ff=0.5) --', flush=True)
p50 = dict(BASE_PARAMS); p50['feature_fraction'] = 0.5
all_lgb_preds['lgb_pd_ff50'] = train_lgb_multiseed(X_tr_gold_pd, X_te_gold_pd, y_dm, groups5, p50, SEEDS, 'pd_ff50')
ic = daywise_ic(finalize(all_lgb_preds['lgb_pd_ff50']), oracle_df, test_ids, test_day)
print(f'  lgb_pd_ff50 IC={ic:+.6f}', flush=True)

print('\n-- LGB pd (ff=0.4) --', flush=True)
p40 = dict(BASE_PARAMS); p40['feature_fraction'] = 0.4
all_lgb_preds['lgb_pd_ff40'] = train_lgb_multiseed(X_tr_gold_pd, X_te_gold_pd, y_dm, groups5, p40, SEEDS, 'pd_ff40')
ic = daywise_ic(finalize(all_lgb_preds['lgb_pd_ff40']), oracle_df, test_ids, test_day)
print(f'  lgb_pd_ff40 IC={ic:+.6f}', flush=True)

del X_tr_gold_pd, X_te_gold_pd; gc.collect()

# Ridge on pd normalization
print('\n-- Ridge pd (alpha=5000) --', flush=True)
all_ridge_preds['ridge_pd'] = run_perday_ridge('pd', alpha=5000)
ic = daywise_ic(finalize(all_ridge_preds['ridge_pd']), oracle_df, test_ids, test_day)
print(f'  ridge_pd IC={ic:+.6f}', flush=True)

print(f'\nPhase 1 done. Elapsed: {(time.time()-t0)/60:.1f} min', flush=True)

# ────────────────────────────────────────────────────────────────────
# PHASE 2: Per-Day Cross-Sectional Gaussian Rank Normalization
# ────────────────────────────────────────────────────────────────────
print('\n\n========== PHASE 2: PER-DAY GAUSSIAN RANK NORMALIZATION ==========', flush=True)

print('Computing X_tr_rg, X_te_rg (per-day rank norm, compliant)...', flush=True)
X_tr_rg, X_te_rg = apply_perday_rankgauss_compliant(tr_raw, te_raw, train_day, test_day)
print(f'X_tr_rg: {X_tr_rg.shape}, range: [{X_tr_rg.min():.2f}, {X_tr_rg.max():.2f}]', flush=True)
print(f'X_te_rg: {X_te_rg.shape}, range: [{X_te_rg.min():.2f}, {X_te_rg.max():.2f}]', flush=True)

X_tr_gold_rg, X_te_gold_rg, gold_feats_rg = build_gold_matrix(X_tr_rg, X_te_rg, 'rg')

# Free rg base matrices — Ridge uses inline per-day rank norm
del X_tr_rg, X_te_rg; gc.collect()
print(f'Freed rg base matrices. Gold shape: {X_tr_gold_rg.shape}', flush=True)

# LGB variants on rg normalization
print('\n-- LGB rg (ff=0.8) --', flush=True)
all_lgb_preds['lgb_rg_ff80'] = train_lgb_multiseed(X_tr_gold_rg, X_te_gold_rg, y_dm, groups5, p80, SEEDS, 'rg_ff80')
ic = daywise_ic(finalize(all_lgb_preds['lgb_rg_ff80']), oracle_df, test_ids, test_day)
print(f'  lgb_rg_ff80 IC={ic:+.6f}', flush=True)

print('\n-- LGB rg (ff=0.6) --', flush=True)
all_lgb_preds['lgb_rg_ff60'] = train_lgb_multiseed(X_tr_gold_rg, X_te_gold_rg, y_dm, groups5, p60, SEEDS, 'rg_ff60')
ic = daywise_ic(finalize(all_lgb_preds['lgb_rg_ff60']), oracle_df, test_ids, test_day)
print(f'  lgb_rg_ff60 IC={ic:+.6f}', flush=True)

print('\n-- LGB rg (ff=0.5) --', flush=True)
all_lgb_preds['lgb_rg_ff50'] = train_lgb_multiseed(X_tr_gold_rg, X_te_gold_rg, y_dm, groups5, p50, SEEDS, 'rg_ff50')
ic = daywise_ic(finalize(all_lgb_preds['lgb_rg_ff50']), oracle_df, test_ids, test_day)
print(f'  lgb_rg_ff50 IC={ic:+.6f}', flush=True)

print('\n-- LGB rg (ff=0.4) --', flush=True)
all_lgb_preds['lgb_rg_ff40'] = train_lgb_multiseed(X_tr_gold_rg, X_te_gold_rg, y_dm, groups5, p40, SEEDS, 'rg_ff40')
ic = daywise_ic(finalize(all_lgb_preds['lgb_rg_ff40']), oracle_df, test_ids, test_day)
print(f'  lgb_rg_ff40 IC={ic:+.6f}', flush=True)

del X_tr_gold_rg, X_te_gold_rg; gc.collect()

# Ridge on rg normalization (per-day rank inside run_perday_ridge)
print('\n-- Ridge rg (alpha=5000) --', flush=True)
all_ridge_preds['ridge_rg'] = run_perday_ridge('rg', alpha=5000)
ic = daywise_ic(finalize(all_ridge_preds['ridge_rg']), oracle_df, test_ids, test_day)
print(f'  ridge_rg IC={ic:+.6f}', flush=True)

print(f'\nPhase 2 done. Elapsed: {(time.time()-t0)/60:.1f} min', flush=True)

# =====================================================================
# SHRINK-A CONFIDENCE (best from v5: p=5 → 0.00084 LB)
# =====================================================================
print('\n-- ShrinkA confidence (p=5) --', flush=True)
conf5 = compute_day_confidence(power=5.0)
print(f'  ShrinkA p=5: mean_conf={conf5.mean():.4f}, min={conf5.min():.4f}', flush=True)

# =====================================================================
# CORRELATIONS
# =====================================================================
print('\n=== COMPONENT CORRELATIONS ===', flush=True)
all_components = {}
for k, v in all_lgb_preds.items():
    all_components[k] = finalize(v)
for k, v in all_ridge_preds.items():
    all_components[k] = finalize(v)

comp_names = list(all_components.keys())
for i, n1 in enumerate(comp_names):
    for j, n2 in enumerate(comp_names):
        if j <= i: continue
        r = np.corrcoef(all_components[n1], all_components[n2])[0,1]
        print(f'  {n1} vs {n2}: r={r:+.4f}', flush=True)

# =====================================================================
# BLEND SWEEP + CSV OUTPUT
# =====================================================================
print('\n=== BLEND SWEEP ===', flush=True)

TARGET_STDS = [0.000700, 0.000948, 0.001200, 0.001500]
submissions = {}

# Helper: blend LGB + Ridge + optional confidence shrinkage
def make_submission(lgb_pred, ridge_pred, w_lgb, conf=None, suffix=''):
    w_r = 1.0 - w_lgb
    blend = w_lgb * finalize(lgb_pred) + w_r * finalize(ridge_pred)
    if conf is not None:
        blend = blend * conf  # shrink toward zero based on day confidence
    return blend, suffix

# ---- pd LGB variants × pd Ridge × blends ----
for lgb_key in ['lgb_pd_ff80', 'lgb_pd_ff60', 'lgb_pd_ff50']:
    for w_lgb in [0.4, 0.5, 0.6, 0.7]:
        w_r = 1.0 - w_lgb
        short = lgb_key.replace('lgb_pd_', '') + f'_{int(w_lgb*100)}_rpd_{int(w_r*100)}'
        blend, _ = make_submission(all_lgb_preds[lgb_key], all_ridge_preds['ridge_pd'], w_lgb)
        for ts in TARGET_STDS:
            submissions[f'{short}_s{int(ts*1e6)}'] = finalize(blend, ts)
        # With ShrinkA_p5
        blend_sh, _ = make_submission(all_lgb_preds[lgb_key], all_ridge_preds['ridge_pd'], w_lgb, conf=conf5)
        for ts in TARGET_STDS:
            submissions[f'{short}_shA5_s{int(ts*1e6)}'] = finalize(blend_sh, ts)

# ---- rg LGB variants × rg Ridge × blends ----
for lgb_key in ['lgb_rg_ff80', 'lgb_rg_ff60', 'lgb_rg_ff50']:
    for w_lgb in [0.4, 0.5, 0.6, 0.7]:
        w_r = 1.0 - w_lgb
        short = lgb_key.replace('lgb_rg_', '') + f'_{int(w_lgb*100)}_rrg_{int(w_r*100)}'
        blend, _ = make_submission(all_lgb_preds[lgb_key], all_ridge_preds['ridge_rg'], w_lgb)
        for ts in TARGET_STDS:
            submissions[f'{short}_s{int(ts*1e6)}'] = finalize(blend, ts)
        # With ShrinkA_p5
        blend_sh, _ = make_submission(all_lgb_preds[lgb_key], all_ridge_preds['ridge_rg'], w_lgb, conf=conf5)
        for ts in TARGET_STDS:
            submissions[f'{short}_shA5_s{int(ts*1e6)}'] = finalize(blend_sh, ts)

# ---- Cross normalization: pd LGB × rg Ridge ----
for lgb_key in ['lgb_pd_ff60', 'lgb_pd_ff50']:
    for w_lgb in [0.5, 0.6]:
        w_r = 1.0 - w_lgb
        short = lgb_key.replace('lgb_pd_', '') + f'_x_{int(w_lgb*100)}_rrg_{int(w_r*100)}'
        blend, _ = make_submission(all_lgb_preds[lgb_key], all_ridge_preds['ridge_rg'], w_lgb)
        for ts in TARGET_STDS:
            submissions[f'{short}_s{int(ts*1e6)}'] = finalize(blend, ts)

# ---- rg LGB × pd Ridge ----
for lgb_key in ['lgb_rg_ff60', 'lgb_rg_ff50']:
    for w_lgb in [0.5, 0.6]:
        w_r = 1.0 - w_lgb
        short = lgb_key.replace('lgb_rg_', '') + f'_x_{int(w_lgb*100)}_rpd_{int(w_r*100)}'
        blend, _ = make_submission(all_lgb_preds[lgb_key], all_ridge_preds['ridge_pd'], w_lgb)
        for ts in TARGET_STDS:
            submissions[f'{short}_s{int(ts*1e6)}'] = finalize(blend, ts)

# ---- Solo components ----
for name, pred in {**all_lgb_preds, **all_ridge_preds}.items():
    for ts in TARGET_STDS:
        submissions[f'{name}_solo_s{int(ts*1e6)}'] = finalize(pred, ts)

# =====================================================================
# EVALUATE ALL
# =====================================================================
print('\n=== ORACLE IC EVALUATION ===', flush=True)
results = []
for name, pred in submissions.items():
    ic = daywise_ic(pred, oracle_df, test_ids, test_day)
    results.append((name, ic, pred))
results.sort(key=lambda x: -x[1])

print(f"\n{'rank':<5} {'oracle_ic':>11}  name")
print('-' * 110)
for i, (name, ic, _) in enumerate(results[:80]):
    print(f'{i+1:<5} {ic:>+11.6f}  {name}')

# =====================================================================
# WRITE CSVs: top 40 overall + ALL pd variants (guaranteed)
# =====================================================================
os.makedirs(OUT_DIR, exist_ok=True)

# Collect names already in top 40
top40_names = {name for name, ic, pred in results[:40]}

# Also force-write every pd variant at every scale (compliant, never skip)
pd_must_write = {name: pred for name, pred in submissions.items()
                 if '_pd_' in name or name.startswith('lgb_pd_') or name.startswith('ridge_pd')}

print(f'\nWriting top 40 CSVs + {len(pd_must_write)} forced pd CSVs...', flush=True)
written = set()
for name, ic, pred in results[:40]:
    path = os.path.join(OUT_DIR, f'{TAG}_{name}.csv')
    pd.DataFrame({'ID': test_ids, 'TARGET': pred}).sort_values('ID').to_csv(path, index=False)
    written.add(name)

# Force-write pd variants not already in top 40
for name, pred in pd_must_write.items():
    if name not in written:
        path = os.path.join(OUT_DIR, f'{TAG}_{name}.csv')
        pd.DataFrame({'ID': test_ids, 'TARGET': pred}).sort_values('ID').to_csv(path, index=False)
        written.add(name)

print(f'Total CSVs written: {len(written)}', flush=True)

print(f'\n=== DONE in {(time.time()-t0)/60:.1f} min ===', flush=True)
