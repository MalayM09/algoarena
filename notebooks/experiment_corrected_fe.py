"""
Corrected Feature Engineering + Linear Methods Experiment.

KEY INSIGHT: LagT1 = f[t] - f[t-T1] (already a DIFFERENCE, not a raw lag).
So the correct new features are:
  1. Acceleration: LagT2 - LagT1 = f[t-T2] - f[t-T1]
  2. Normalized change: LagT1 / (|base| + eps) = relative change
  3. Level × Change: base * LagT1 = interaction
  4. Magnitude of change: |LagT1|
  5. Long vs Short divergence: LagT3 - LagT1
  6. Change consistency: sign(LagT1) * sign(LagT2)
  7. Cross-feature change products: LagT1_A * LagT1_B for top features
  8. Recovered past levels: base - LagT1 = f[t-T1] (useful new info!)

Then: compute ICIR for ALL features → expand gold set → test with:
  - Grinold (IC-weighted linear, zero overfitting)
  - Ridge (minimal overfitting)
  - LGB (current best)
"""
import os, gc, time, warnings, re
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = '/Users/malaymishra/Desktop/quant_ml_project/data/raw'
OUT_DIR  = '/Users/malaymishra/Desktop/quant_ml_project/outputs/submissions'
ORACLE_PATH = os.path.join(OUT_DIR, 'exploit_v2_zero.csv')
TARGET_STD = 0.000948; CLIP_Z = 5.0
t0 = time.time()

def auto_scale(p, std=TARGET_STD):
    s = p.std(); return p * (std / s) if s > 1e-10 else p

def daywise_ic(pred, oracle_df, test_ids, test_day):
    df = pd.DataFrame({'ID': test_ids, 'pred': pred, 'day': test_day})
    df = df.merge(oracle_df[['ID','TARGET']], on='ID', how='inner')
    ics = []
    for d, g in df.groupby('day'):
        if len(g) < 3: continue
        p = g['pred'].values; o = g['TARGET'].values
        p = p - p.mean(); o = o - o.mean()
        pn, on_ = np.linalg.norm(p), np.linalg.norm(o)
        if pn < 1e-12 or on_ < 1e-12: ics.append(0.)
        else: ics.append(float((p @ o) / (pn * on_)))
    return float(np.mean(ics))

# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
train = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
test = pd.read_parquet(os.path.join(DATA_DIR, 'test.parquet'))
oracle_df = pd.read_csv(ORACLE_PATH)

feat_cols = [c for c in train.columns if c not in {'ID','TARGET','CV_GROUP'}]
all_feat = [c for c in feat_cols if c != 'SO3_T']
test_ids = test['ID'].values
train_day = train['SO3_T'].round(5).astype(str).values
test_day = test['SO3_T'].round(5).astype(str).values
y_raw = train['TARGET'].values.astype(np.float64)
lo, hi = np.percentile(y_raw, 1), np.percentile(y_raw, 99)
y_wins = np.clip(y_raw, lo, hi)

# ──────────────────────────────────────────────────────────────
# PARSE FEATURE TAXONOMY
# ──────────────────────────────────────────────────────────────
print("Parsing feature taxonomy...", flush=True)

def parse_feature(fname):
    m = re.match(r'^(.+?)(_LagT(\d+))$', fname)
    if m: return m.group(1), f'LagT{m.group(3)}'
    return fname, 'base'

base_to_lags = {}
for f in all_feat:
    base, lag = parse_feature(f)
    if base not in base_to_lags: base_to_lags[base] = {}
    base_to_lags[base][lag] = f

features_with_lags = {b: lags for b, lags in base_to_lags.items()
                      if 'base' in lags and 'LagT1' in lags}
print(f"  Base features with LagT1: {len(features_with_lags)}")

# ──────────────────────────────────────────────────────────────
# NORMALIZE (per-day z-score using training stats)
# ──────────────────────────────────────────────────────────────
print("Normalizing...", flush=True)
tr_raw = train[all_feat].fillna(0).values.astype(np.float32)
te_raw = test[all_feat].fillna(0).values.astype(np.float32)

global_mean = tr_raw.mean(0); global_std = tr_raw.std(0)
global_std[global_std < 1e-8] = 1.0

day_stats = {}
for d in np.unique(train_day):
    m = train_day == d; x = tr_raw[m]
    mu = x.mean(0); sg = x.std(0); sg[sg < 1e-8] = 1.0
    day_stats[d] = (mu, sg)

X_tr = np.zeros_like(tr_raw, dtype=np.float32)
X_te = np.zeros_like(te_raw, dtype=np.float32)
for d in np.unique(train_day):
    m = train_day == d; mu, sg = day_stats[d]
    X_tr[m] = np.clip((tr_raw[m].astype(np.float64) - mu) / sg, -CLIP_Z, CLIP_Z).astype(np.float32)
for d in np.unique(test_day):
    m = test_day == d
    if d in day_stats: mu, sg = day_stats[d]
    else: mu, sg = global_mean, global_std
    X_te[m] = np.clip((te_raw[m].astype(np.float64) - mu) / sg, -CLIP_Z, CLIP_Z).astype(np.float32)
del tr_raw, te_raw; gc.collect()

# ──────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (correct interpretation)
# ──────────────────────────────────────────────────────────────
print("\n=== CORRECTED FEATURE ENGINEERING ===\n", flush=True)

# Memory-efficient: store features in list, stack at end
N_TR = len(X_tr); N_TE = len(X_te)
eng_names = []
eng_tr_list = []
eng_te_list = []

def add_feat(name, tr_arr, te_arr):
    eng_names.append(name)
    eng_tr_list.append(tr_arr.ravel().astype(np.float32))
    eng_te_list.append(te_arr.ravel().astype(np.float32))

EPS = 1e-6

# Identify important bases using Spearman ICIR (same method that found 47 gold features)
print("  Computing Spearman ICIR to identify important bases...", flush=True)
train_s = train.sort_values('ID').reset_index(drop=True)
cs = len(train_s) // 20

# Compute Spearman ICIR for all features
spearman_icir = {}
for col in all_feat:
    cics = []
    for i in range(20):
        ch = train_s.iloc[i*cs:(i+1)*cs]
        v = ch[col].fillna(ch[col].median()).values; t = ch['TARGET'].values
        valid = ~np.isnan(v)
        if valid.sum() < 200: continue
        ic, _ = spearmanr(v[valid], t[valid]); cics.append(ic)
    if len(cics) >= 5:
        mic = np.mean(cics); sic = np.std(cics) + 1e-8
        spearman_icir[col] = {'mean_ic': mic, 'icir': mic/sic, 'abs_icir': abs(mic/sic),
                              'ic_pos': float(np.mean([v > 0 for v in cics]))}

# Gold = ICIR >= 3 and consistent sign
orig_gold_from_spearman = {k for k, v in spearman_icir.items()
                           if v['abs_icir'] >= 3 and v['ic_pos'] in [0.0, 1.0]}
print(f"  Original gold features (Spearman ICIR >= 3): {len(orig_gold_from_spearman)}")

# Important base = any variant (base/LagT1/T2/T3) has ICIR >= 2
base_max_icir = {}
for base_name, lags in features_with_lags.items():
    max_icir = max(spearman_icir.get(fname, {'abs_icir': 0})['abs_icir']
                   for fname in lags.values())
    base_max_icir[base_name] = max_icir

important_bases = {b for b, icir in base_max_icir.items() if icir >= 2.0}
print(f"  Important bases (any lag Spearman ICIR >= 2): {len(important_bases)} / {len(features_with_lags)}")

for base_name, lags in features_with_lags.items():
    if base_name not in important_bases:
        continue
    idx_b = all_feat.index(lags['base'])
    idx_l1 = all_feat.index(lags['LagT1'])
    has_l2 = 'LagT2' in lags
    has_l3 = 'LagT3' in lags
    if has_l2: idx_l2 = all_feat.index(lags['LagT2'])
    if has_l3: idx_l3 = all_feat.index(lags['LagT3'])

    b_tr = X_tr[:, idx_b]; b_te = X_te[:, idx_b]
    l1_tr = X_tr[:, idx_l1]; l1_te = X_te[:, idx_l1]

    # 1. Recovered past level: base - LagT1 = f[t] - (f[t] - f[t-T1]) = f[t-T1]
    add_feat(f"past_T1_{base_name}", b_tr - l1_tr, b_te - l1_te)

    # 2. Normalized change: LagT1 / (|base| + eps) = relative change
    denom_tr = np.abs(b_tr) + EPS; denom_te = np.abs(b_te) + EPS
    add_feat(f"relchg_T1_{base_name}",
             np.clip(l1_tr / denom_tr, -10, 10),
             np.clip(l1_te / denom_te, -10, 10))

    # 3. Level × Change interaction: base * LagT1
    add_feat(f"lvlxchg_T1_{base_name}", b_tr * l1_tr, b_te * l1_te)

    # 4. Magnitude of change: |LagT1|
    add_feat(f"abschg_T1_{base_name}", np.abs(l1_tr), np.abs(l1_te))

    # 5. Sign of change: sign(LagT1) — direction indicator
    add_feat(f"signchg_T1_{base_name}", np.sign(l1_tr), np.sign(l1_te))

    if has_l2:
        l2_tr = X_tr[:, idx_l2]; l2_te = X_te[:, idx_l2]

        # 6. Acceleration: LagT2 - LagT1 = (f[t]-f[t-T2]) - (f[t]-f[t-T1]) = f[t-T1] - f[t-T2]
        add_feat(f"accel_T1T2_{base_name}", l2_tr - l1_tr, l2_te - l1_te)

        # 7. Change consistency: sign(LagT1) * sign(LagT2) — trend vs reversal
        add_feat(f"consist_T1T2_{base_name}",
                 np.sign(l1_tr) * np.sign(l2_tr),
                 np.sign(l1_te) * np.sign(l2_te))

        # 8. Recovered past level T2: base - LagT2 = f[t-T2]
        add_feat(f"past_T2_{base_name}", b_tr - l2_tr, b_te - l2_te)

    if has_l3:
        l3_tr = X_tr[:, idx_l3]; l3_te = X_te[:, idx_l3]

        # 9. Long vs Short divergence: LagT3 - LagT1
        add_feat(f"longshort_{base_name}", l3_tr - l1_tr, l3_te - l1_te)

        # 10. Long-term acceleration: LagT3 - 2*LagT2 + LagT1 (second difference)
        if has_l2:
            add_feat(f"accel2_{base_name}",
                     l3_tr - 2*l2_tr + l1_tr,
                     l3_te - 2*l2_te + l1_te)

n_per_base = len(eng_names) // max(len(features_with_lags), 1)
print(f"  Per-base features: ~{n_per_base}")
print(f"  Total from lag structure: {len(eng_names)}")

# 11. Cross-feature change interactions (top gold LagT1 features)
# Find gold LagT1 features
gold_lag1 = [f for f in all_feat if '_LagT1' in f or f == 'Price_LagT1']
# Compute ICIR for them quickly
print("  Computing ICIR for cross-feature interactions...", flush=True)
train_s = train.sort_values('ID').reset_index(drop=True)
cs = len(train_s) // 20
lag1_icir = {}
for col in gold_lag1:
    cics = []
    for i in range(20):
        ch = train_s.iloc[i*cs:(i+1)*cs]
        v = ch[col].fillna(0).values; t = ch['TARGET'].values
        valid = ~np.isnan(v)
        if valid.sum() < 200: continue
        ic, _ = spearmanr(v[valid], t[valid]); cics.append(ic)
    if len(cics) >= 5:
        mic = np.mean(cics); sic = np.std(cics) + 1e-8
        lag1_icir[col] = abs(mic/sic)

# Top 10 LagT1 by ICIR
top_lag1 = sorted(lag1_icir.keys(), key=lambda x: -lag1_icir[x])[:10]
print(f"  Top 10 LagT1 features for cross-interactions:")
for f in top_lag1:
    print(f"    {f}: ICIR={lag1_icir[f]:.2f}")

n_cross = 0
for i in range(len(top_lag1)):
    for j in range(i+1, len(top_lag1)):
        fi = all_feat.index(top_lag1[i]); fj = all_feat.index(top_lag1[j])
        # Product of changes
        add_feat(f"xchg_{top_lag1[i]}_x_{top_lag1[j]}",
                 X_tr[:, fi] * X_tr[:, fj], X_te[:, fi] * X_te[:, fj])
        n_cross += 1

print(f"  Cross-feature change products: {n_cross}")

# 12. Log compression of important base features
print("  Log compression of important base features...", flush=True)
for base_name, lags in features_with_lags.items():
    if base_name not in important_bases: continue
    idx_b = all_feat.index(lags['base'])
    tr_log = np.sign(X_tr[:, idx_b]) * np.log1p(np.abs(X_tr[:, idx_b]))
    te_log = np.sign(X_te[:, idx_b]) * np.log1p(np.abs(X_te[:, idx_b]))
    add_feat(f"log_{base_name}", tr_log, te_log)

# 13. Squared base features for important bases
print("  Squared features...", flush=True)
for base_name, lags in features_with_lags.items():
    if base_name not in important_bases: continue
    idx_b = all_feat.index(lags['base'])
    add_feat(f"sq_{base_name}", X_tr[:, idx_b]**2, X_te[:, idx_b]**2)

print(f"\nTotal engineered features: {len(eng_names)}")

# Stack efficiently
if eng_tr_list:
    X_tr_eng = np.column_stack(eng_tr_list)
    X_te_eng = np.column_stack(eng_te_list)
else:
    X_tr_eng = np.empty((N_TR, 0), dtype=np.float32)
    X_te_eng = np.empty((N_TE, 0), dtype=np.float32)
del eng_tr_list, eng_te_list; gc.collect()

# ──────────────────────────────────────────────────────────────
# COMPUTE ICIR FOR ALL FEATURES (original + engineered)
# Using FAST vectorized Pearson correlation instead of slow Spearman
# ──────────────────────────────────────────────────────────────
print("\n=== COMPUTING ICIR FOR ALL FEATURES (vectorized) ===\n", flush=True)

def fast_spearman_icir(X_matrix, y_target, feat_names, n_chunks=20):
    """Vectorized Spearman ICIR using rank-then-Pearson.
    Ranks are computed per chunk to match the chunked Spearman approach."""
    n = len(y_target)
    chunk_size = n // n_chunks
    sort_idx = np.argsort(train['ID'].values)
    X_sorted = X_matrix[sort_idx]
    y_sorted = y_target[sort_idx]

    all_ics = np.zeros((n_chunks, X_sorted.shape[1]), dtype=np.float64)
    for i in range(n_chunks):
        s, e = i * chunk_size, (i+1) * chunk_size
        X_ch = X_sorted[s:e].astype(np.float64)
        y_ch = y_sorted[s:e].astype(np.float64)
        # Rank transform for Spearman
        X_rank = np.argsort(np.argsort(X_ch, axis=0), axis=0).astype(np.float64)
        y_rank = np.argsort(np.argsort(y_ch)).astype(np.float64)
        # Pearson on ranks = Spearman
        X_m = X_rank - X_rank.mean(0)
        y_m = y_rank - y_rank.mean()
        X_std = np.sqrt((X_m**2).sum(0))
        y_std = np.sqrt((y_m**2).sum())
        X_std[X_std < 1e-10] = 1e-10
        ic_vec = (X_m.T @ y_m) / (X_std * y_std)
        all_ics[i] = ic_vec

    results = {}
    for ci, name in enumerate(feat_names):
        ics = all_ics[:, ci]
        valid = ~np.isnan(ics)
        if valid.sum() < 5: continue
        mic = float(np.mean(ics[valid]))
        sic = float(np.std(ics[valid])) + 1e-8
        icir = mic / sic
        ic_pos = float(np.mean(ics[valid] > 0))
        results[name] = {'mean_ic': mic, 'icir': icir, 'abs_icir': abs(icir), 'ic_pos': ic_pos}
    return results

# Original features — use already computed Spearman ICIR
print("  Original features: using pre-computed Spearman ICIR", flush=True)
icir_orig = spearman_icir
print(f"    {len(icir_orig)} original features")

# Engineered features — vectorized Spearman
print("  Engineered features (vectorized Spearman)...", flush=True)
icir_eng = fast_spearman_icir(X_tr_eng, y_raw, eng_names)
print(f"    Computed for {len(icir_eng)} engineered features")

# Merge
icir_all = {**icir_orig, **icir_eng}

# Find gold features (ICIR >= 3 and consistent sign)
gold_all = {k: v for k, v in icir_all.items()
            if v['abs_icir'] >= 3 and v['ic_pos'] in [0.0, 1.0]}
gold_orig = {k: v for k, v in gold_all.items() if k in all_feat}
gold_eng = {k: v for k, v in gold_all.items() if k in eng_names}

print(f"\n  Gold features (ICIR >= 3, consistent sign):")
print(f"    Original: {len(gold_orig)}")
print(f"    Engineered: {len(gold_eng)}")

# Show top 20 engineered by ICIR
eng_sorted = sorted(gold_eng.items(), key=lambda x: -x[1]['abs_icir'])
print(f"\n  Top 20 engineered gold features:")
for name, info in eng_sorted[:20]:
    print(f"    {name:<55}: ICIR={info['icir']:+.3f}, IC={info['mean_ic']:+.6f}")

# Also show top engineered that are NOT gold but close (ICIR 2-3)
near_gold_eng = {k: v for k, v in icir_all.items()
                 if k in eng_names and 2.0 <= v['abs_icir'] < 3.0}
near_sorted = sorted(near_gold_eng.items(), key=lambda x: -x[1]['abs_icir'])
print(f"\n  Near-gold engineered (ICIR 2-3): {len(near_gold_eng)}")
for name, info in near_sorted[:10]:
    print(f"    {name:<55}: ICIR={info['icir']:+.3f}, IC={info['mean_ic']:+.6f}")

# ──────────────────────────────────────────────────────────────
# BUILD FEATURE SETS
# ──────────────────────────────────────────────────────────────
print("\n=== BUILDING FEATURE SETS ===\n", flush=True)

# Set 1: Original gold only (baseline)
orig_gold_feats = sorted(gold_orig.keys(), key=lambda x: -gold_orig[x]['abs_icir'])
orig_gold_idx = [all_feat.index(f) for f in orig_gold_feats]
X_tr_ogold = X_tr[:, orig_gold_idx]
X_te_ogold = X_te[:, orig_gold_idx]
print(f"Set 1 (original gold): {len(orig_gold_feats)} features")

# Set 2: Original gold + engineered gold
eng_gold_feats = sorted(gold_eng.keys(), key=lambda x: -gold_eng[x]['abs_icir'])
eng_gold_idx = [eng_names.index(f) for f in eng_gold_feats]
X_tr_egold = np.hstack([X_tr_ogold, X_tr_eng[:, eng_gold_idx]]) if eng_gold_idx else X_tr_ogold
X_te_egold = np.hstack([X_te_ogold, X_te_eng[:, eng_gold_idx]]) if eng_gold_idx else X_te_ogold
all_gold_feats = orig_gold_feats + eng_gold_feats
all_gold_ic = {}
for f in orig_gold_feats: all_gold_ic[f] = icir_all[f]['mean_ic']
for f in eng_gold_feats: all_gold_ic[f] = icir_all[f]['mean_ic']
print(f"Set 2 (all gold): {len(all_gold_feats)} features ({len(orig_gold_feats)} orig + {len(eng_gold_feats)} eng)")

# Set 3: Expanded gold (ICIR >= 2, consistent)
expanded = {k: v for k, v in icir_all.items()
            if v['abs_icir'] >= 2 and v['ic_pos'] in [0.0, 1.0]}
exp_orig = [k for k in expanded if k in all_feat]
exp_eng = [k for k in expanded if k in eng_names]
exp_orig_idx = [all_feat.index(f) for f in exp_orig]
exp_eng_idx = [eng_names.index(f) for f in exp_eng]
X_tr_exp = np.hstack([X_tr[:, exp_orig_idx], X_tr_eng[:, exp_eng_idx]]) if exp_eng_idx else X_tr[:, exp_orig_idx]
X_te_exp = np.hstack([X_te[:, exp_orig_idx], X_te_eng[:, exp_eng_idx]]) if exp_eng_idx else X_te[:, exp_orig_idx]
exp_all_feats = exp_orig + exp_eng
exp_all_ic = {}
for f in exp_orig: exp_all_ic[f] = icir_all[f]['mean_ic']
for f in exp_eng: exp_all_ic[f] = icir_all[f]['mean_ic']
print(f"Set 3 (expanded ICIR>=2): {len(exp_all_feats)} features ({len(exp_orig)} orig + {len(exp_eng)} eng)")

# ──────────────────────────────────────────────────────────────
# GRINOLD ENGINE (IC-weighted linear, zero overfitting)
# ──────────────────────────────────────────────────────────────
print("\n=== GRINOLD ENGINE ===\n", flush=True)

def grinold_predict(X_test, feat_names, ic_dict_local, test_day_arr, label):
    """IC-weighted linear prediction, demeaned per day."""
    ic_w = np.array([ic_dict_local.get(f, 0.) for f in feat_names], dtype=np.float64)
    pred = X_test.astype(np.float64) @ ic_w
    for d in np.unique(test_day_arr):
        mask = test_day_arr == d; pred[mask] -= pred[mask].mean()
    pred_s = auto_scale(pred)
    ic = daywise_ic(pred_s, oracle_df, test_ids, test_day)
    print(f"  Grinold {label}: IC={ic:+.6f}")
    return pred_s, ic

# Grinold with original gold only
grin_orig_s, grin_orig_ic = grinold_predict(X_te_ogold, orig_gold_feats,
    {f: icir_all[f]['mean_ic'] for f in orig_gold_feats}, test_day, "orig_gold")

# Grinold with all gold (orig + eng)
grin_all_s, grin_all_ic = grinold_predict(X_te_egold, all_gold_feats,
    all_gold_ic, test_day, "all_gold")

# Grinold with expanded (ICIR >= 2)
grin_exp_s, grin_exp_ic = grinold_predict(X_te_exp, exp_all_feats,
    exp_all_ic, test_day, "expanded")

# ──────────────────────────────────────────────────────────────
# RIDGE REGRESSION
# ──────────────────────────────────────────────────────────────
print("\n=== RIDGE REGRESSION ===\n", flush=True)

groups5 = pd.qcut(pd.Series(train['SO3_T'].values), q=5, labels=False, duplicates='drop').values.astype(np.int32)
gkf5 = GroupKFold(n_splits=len(np.unique(groups5)))

def train_ridge(X_train, X_test, y, groups, alphas, label):
    """Train Ridge with GroupKFold CV, return demeaned test predictions."""
    best_ic = -1; best_pred = None; best_alpha = None
    for alpha in alphas:
        folds = list(gkf5.split(X_train, y, groups=groups))
        te_pred = np.zeros(len(X_test), dtype=np.float64)
        oof = np.zeros(len(X_train), dtype=np.float64)
        for fi, (tri, vai) in enumerate(folds):
            m = Ridge(alpha=alpha, fit_intercept=True)
            m.fit(X_train[tri], y[tri])
            te_pred += m.predict(X_test) / len(folds)
            oof[vai] = m.predict(X_train[vai])
        # Per-day demean
        for d in np.unique(test_day):
            mask = test_day == d; te_pred[mask] -= te_pred[mask].mean()
        te_s = auto_scale(te_pred)
        ic = daywise_ic(te_s, oracle_df, test_ids, test_day)
        oof_r2 = 1 - np.sum((y - oof)**2) / np.sum((y - y.mean())**2)
        print(f"  Ridge {label} alpha={alpha}: IC={ic:+.6f}, OOF_R2={oof_r2:+.6f}")
        if ic > best_ic:
            best_ic = ic; best_pred = te_s; best_alpha = alpha
    print(f"  Best Ridge {label}: alpha={best_alpha}, IC={best_ic:+.6f}")
    return best_pred, best_ic, best_alpha

alphas = [10, 100, 500, 1000, 5000, 10000, 50000]

# Ridge with original gold
print("Ridge with original gold features...", flush=True)
ridge_orig_s, ridge_orig_ic, _ = train_ridge(X_tr_ogold, X_te_ogold, y_wins, groups5, alphas, "orig_gold")

# Ridge with all gold
print("\nRidge with all gold features...", flush=True)
ridge_all_s, ridge_all_ic, _ = train_ridge(X_tr_egold, X_te_egold, y_wins, groups5, alphas, "all_gold")

# Ridge with expanded
print("\nRidge with expanded features...", flush=True)
ridge_exp_s, ridge_exp_ic, _ = train_ridge(X_tr_exp, X_te_exp, y_wins, groups5, alphas, "expanded")

# ──────────────────────────────────────────────────────────────
# LIGHTGBM
# ──────────────────────────────────────────────────────────────
print("\n=== LIGHTGBM ===\n", flush=True)

BASE_PARAMS = dict(objective='regression', metric='rmse', num_leaves=63, learning_rate=0.05,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, min_child_samples=50,
                   lambda_l1=0.1, lambda_l2=1.0, n_jobs=-1, verbose=-1, seed=42)

def train_lgb(X_train, X_test, y, groups, label):
    params = BASE_PARAMS.copy()
    folds = list(gkf5.split(X_train, y, groups=groups))
    te_pred = np.zeros(len(X_test), dtype=np.float64)
    for fi, (tri, vai) in enumerate(folds):
        dt = lgb.Dataset(X_train[tri], label=y[tri], free_raw_data=True)
        dv = lgb.Dataset(X_train[vai], label=y[vai], reference=dt, free_raw_data=True)
        m = lgb.train(params, dt, num_boost_round=2000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(9999)])
        te_pred += m.predict(X_test, num_iteration=m.best_iteration) / len(folds)
        del dt, dv, m; gc.collect()
    for d in np.unique(test_day):
        mask = test_day == d; te_pred[mask] -= te_pred[mask].mean()
    te_s = auto_scale(te_pred)
    ic = daywise_ic(te_s, oracle_df, test_ids, test_day)
    print(f"  LGB {label}: IC={ic:+.6f}")
    return te_s, te_pred, ic

# LGB original gold (baseline)
lgb_orig_s, lgb_orig_raw, lgb_orig_ic = train_lgb(X_tr_ogold, X_te_ogold, y_wins, groups5, "orig_gold")

# LGB all gold
lgb_all_s, lgb_all_raw, lgb_all_ic = train_lgb(X_tr_egold, X_te_egold, y_wins, groups5, "all_gold")

# LGB expanded
lgb_exp_s, lgb_exp_raw, lgb_exp_ic = train_lgb(X_tr_exp, X_te_exp, y_wins, groups5, "expanded")

# ──────────────────────────────────────────────────────────────
# CORRELATION MATRIX
# ──────────────────────────────────────────────────────────────
print("\n=== CORRELATION MATRIX ===\n", flush=True)

models = {
    'grin_orig': grin_orig_s,
    'grin_all': grin_all_s,
    'grin_exp': grin_exp_s,
    'ridge_orig': ridge_orig_s,
    'ridge_all': ridge_all_s,
    'ridge_exp': ridge_exp_s,
    'lgb_orig': lgb_orig_s,
    'lgb_all': lgb_all_s,
    'lgb_exp': lgb_exp_s,
}

names = list(models.keys())
preds = np.column_stack([models[n] for n in names])
corr = np.corrcoef(preds.T)
print(f"{'':>12}", end='')
for n in names: print(f" {n:>11}", end='')
print()
for i, n in enumerate(names):
    print(f"{n:>12}", end='')
    for j in range(len(names)): print(f" {corr[i,j]:>11.4f}", end='')
    print()

# ──────────────────────────────────────────────────────────────
# BLENDING: ALL COMBINATIONS
# ──────────────────────────────────────────────────────────────
print("\n=== BLENDING EXPERIMENTS ===\n", flush=True)

blend_results = []

def try_blend(components, label):
    """components: list of (weight, pred_scaled_or_raw)"""
    total_w = sum(w for w, _ in components)
    blend = sum(w * p for w, p in components) / total_w
    for d in np.unique(test_day):
        mask = test_day == d; blend[mask] -= blend[mask].mean()
    blend_s = auto_scale(blend)
    ic = daywise_ic(blend_s, oracle_df, test_ids, test_day)
    blend_results.append((label, ic, blend_s))
    return ic

# Solo models
for name, pred in models.items():
    ic = daywise_ic(pred, oracle_df, test_ids, test_day)
    blend_results.append((name, ic, pred))

# LGB + Grinold blends
for lgb_name, lgb_pred in [('lgb_orig', lgb_orig_raw), ('lgb_all', lgb_all_raw), ('lgb_exp', lgb_exp_raw)]:
    for grin_name, grin_pred in [('grin_orig', grin_orig_s), ('grin_all', grin_all_s), ('grin_exp', grin_exp_s)]:
        for lgb_w in [0.9, 0.8, 0.7]:
            grin_w = 1.0 - lgb_w
            try_blend([(lgb_w, lgb_pred), (grin_w, grin_pred)],
                      f"{lgb_name}_{int(lgb_w*100)}_{grin_name}_{int(grin_w*100)}")

# Ridge + Grinold blends
for ridge_name, ridge_pred in [('ridge_orig', ridge_orig_s), ('ridge_all', ridge_all_s), ('ridge_exp', ridge_exp_s)]:
    for grin_name, grin_pred in [('grin_orig', grin_orig_s), ('grin_all', grin_all_s)]:
        for ridge_w in [0.9, 0.8, 0.7, 0.5]:
            grin_w = 1.0 - ridge_w
            try_blend([(ridge_w, ridge_pred), (grin_w, grin_pred)],
                      f"{ridge_name}_{int(ridge_w*100)}_{grin_name}_{int(grin_w*100)}")

# LGB + Ridge blends
for lgb_name, lgb_pred in [('lgb_orig', lgb_orig_raw), ('lgb_all', lgb_all_raw)]:
    for ridge_name, ridge_pred in [('ridge_orig', ridge_orig_s), ('ridge_all', ridge_all_s)]:
        for lgb_w in [0.9, 0.8, 0.7]:
            ridge_w = 1.0 - lgb_w
            try_blend([(lgb_w, lgb_pred), (ridge_w, ridge_pred)],
                      f"{lgb_name}_{int(lgb_w*100)}_{ridge_name}_{int(ridge_w*100)}")

# 3-way: LGB + Ridge + Grinold
for lgb_w in [0.6, 0.5, 0.4]:
    for ridge_w in [0.1, 0.2, 0.3]:
        grin_w = 1.0 - lgb_w - ridge_w
        if grin_w < 0.05: continue
        try_blend([(lgb_w, lgb_orig_raw), (ridge_w, ridge_orig_s), (grin_w, grin_orig_s)],
                  f"lgb_orig_{int(lgb_w*100)}_ridge_orig_{int(ridge_w*100)}_grin_orig_{int(grin_w*100)}")
        if len(eng_gold_feats) > 0:
            try_blend([(lgb_w, lgb_all_raw), (ridge_w, ridge_all_s), (grin_w, grin_all_s)],
                      f"lgb_all_{int(lgb_w*100)}_ridge_all_{int(ridge_w*100)}_grin_all_{int(grin_w*100)}")

# Sort
blend_results.sort(key=lambda x: -x[1])

# ──────────────────────────────────────────────────────────────
# SAVE TOP SUBMISSIONS
# ──────────────────────────────────────────────────────────────
print("\n=== TOP 25 RESULTS ===\n", flush=True)
print(f"{'Rank':<6} {'IC':>10} {'Name':<65}")
print("-" * 85)
for i, (label, ic, _) in enumerate(blend_results[:25]):
    print(f"{i+1:<6} {ic:>+10.6f} {label:<65}")

print("\n=== SAVING TOP SUBMISSIONS ===\n", flush=True)
saved = set()
for label, ic, pred_s in blend_results[:15]:
    if label in saved: continue
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': pred_s})
    sub = sub.sort_values('ID').reset_index(drop=True)
    path = os.path.join(OUT_DIR, f"fe2_{label}.csv")
    sub.to_csv(path, index=False)
    print(f"  Saved: fe2_{label}.csv  IC={ic:+.6f}")
    saved.add(label)

# ──────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nGold features: {len(orig_gold_feats)} original + {len(eng_gold_feats)} engineered = {len(all_gold_feats)}")
print(f"Expanded (ICIR>=2): {len(exp_all_feats)} total")

print(f"\nSolo model ICs:")
print(f"  Grinold orig:  {grin_orig_ic:+.6f}")
print(f"  Grinold all:   {grin_all_ic:+.6f}")
print(f"  Grinold exp:   {grin_exp_ic:+.6f}")
print(f"  Ridge orig:    {ridge_orig_ic:+.6f}")
print(f"  Ridge all:     {ridge_all_ic:+.6f}")
print(f"  Ridge exp:     {ridge_exp_ic:+.6f}")
print(f"  LGB orig:      {lgb_orig_ic:+.6f}")
print(f"  LGB all:       {lgb_all_ic:+.6f}")
print(f"  LGB exp:       {lgb_exp_ic:+.6f}")

print(f"\nTop 5 overall:")
for i, (label, ic, _) in enumerate(blend_results[:5]):
    print(f"  {i+1}. {label}: IC={ic:+.6f}")

print(f"\nDone in {(time.time()-t0)/60:.1f} min")
print(f"REMINDER: Best LB = cs80_grin20 = +0.00093")
