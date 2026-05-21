"""Generate the 3-way blend (cs40/grinold40/goldridge20) submission for comparison."""
import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = '/Users/malaymishra/Desktop/quant_ml_project/data/raw'
OUT_DIR  = '/Users/malaymishra/Desktop/quant_ml_project/outputs/submissions'
ORACLE_PATH = os.path.join(OUT_DIR, 'exploit_v2_zero.csv')
TARGET_STD = 0.000948; CLIP_Z = 5.0
t0 = time.time()

def auto_scale(p, std=TARGET_STD):
    s = p.std(); return p * (std / s) if s > 1e-10 else p

def daywise_oracle(pred, oracle_df, test_ids, test_day):
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

def daywise_r2(pred, oracle_df, test_ids, test_day):
    df = pd.DataFrame({'ID': test_ids, 'pred': pred, 'day': test_day})
    df = df.merge(oracle_df[['ID','TARGET']], on='ID', how='inner')
    y = df['TARGET'].values; yhat = df['pred'].values
    return 1 - np.sum((y - yhat)**2) / np.sum(y**2)

print("Loading...", flush=True)
train = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
test = pd.read_parquet(os.path.join(DATA_DIR, 'test.parquet'))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))[['ID']]
oracle_df = pd.read_csv(ORACLE_PATH)

feat_cols = [c for c in train.columns if c not in {'ID','TARGET','CV_GROUP'}]
all_feat = [c for c in feat_cols if c != 'SO3_T']
test_ids = test['ID'].values
train_day = train['SO3_T'].round(5).astype(str).values
test_day = test['SO3_T'].round(5).astype(str).values
y_raw = train['TARGET'].values.astype(np.float64)
lo, hi = np.percentile(y_raw, 1), np.percentile(y_raw, 99)
y_wins = np.clip(y_raw, lo, hi)
train_days_set = set(np.unique(train_day))

# IC/ICIR
print("IC/ICIR...", flush=True)
N_CHUNKS = 20
train_sorted = train.sort_values('ID').reset_index(drop=True)
chunk_size = len(train_sorted) // N_CHUNKS
ic_results = []
for col in [c for c in feat_cols if c != 'SO3_T']:
    chunk_ics = []
    for i in range(N_CHUNKS):
        chunk = train_sorted.iloc[i*chunk_size:(i+1)*chunk_size]
        vals = chunk[col].fillna(chunk[col].median()).values
        tgt = chunk['TARGET'].values
        valid = ~np.isnan(vals)
        if valid.sum() < 200: chunk_ics.append(np.nan); continue
        ic, _ = spearmanr(vals[valid], tgt[valid]); chunk_ics.append(ic)
    valid_ics = [v for v in chunk_ics if not np.isnan(v)]
    if len(valid_ics) < 5: continue
    mean_ic = float(np.mean(valid_ics)); std_ic = float(np.std(valid_ics)) + 1e-8
    icir = mean_ic / std_ic; ic_pos_frac = float(np.mean([v > 0 for v in valid_ics]))
    ic_results.append({'feature': col, 'mean_ic': mean_ic, 'icir': icir,
                       'abs_icir': abs(icir), 'ic_pos_frac': ic_pos_frac})
icir_df = pd.DataFrame(ic_results).sort_values('abs_icir', ascending=False)
gold_mask = (icir_df['abs_icir'] >= 3) & (icir_df['ic_pos_frac'].isin([0.0, 1.0]))
gold_df = icir_df[gold_mask].copy()
gold_feats = [f for f in gold_df['feature'].tolist() if f in train.columns]
ic_dict = gold_df.set_index('feature')['mean_ic'].to_dict()
gold_idx = [all_feat.index(f) for f in gold_feats if f in all_feat]
print(f"  Gold: {len(gold_feats)}", flush=True)

# Normalization A (per-day training z-score)
print("Normalizing...", flush=True)
tr_feat_raw = train[all_feat].fillna(0).values.astype(np.float32)
te_feat_raw = test[all_feat].fillna(0).values.astype(np.float32)
global_mean = tr_feat_raw.mean(0); global_std = tr_feat_raw.std(0); global_std[global_std < 1e-8] = 1.0
day_stats = {}
for d in np.unique(train_day):
    m = train_day == d; x = tr_feat_raw[m]
    mu = x.mean(0); sg = x.std(0); sg[sg < 1e-8] = 1.0; day_stats[d] = (mu, sg)

X_tr = np.zeros_like(tr_feat_raw, dtype=np.float32)
X_te = np.zeros_like(te_feat_raw, dtype=np.float32)
for d in np.unique(train_day):
    m = train_day == d; mu, sg = day_stats[d]
    X_tr[m] = np.clip((tr_feat_raw[m].astype(np.float64) - mu) / sg, -CLIP_Z, CLIP_Z).astype(np.float32)
for d in np.unique(test_day):
    m = test_day == d
    if d in day_stats: mu, sg = day_stats[d]
    else: mu, sg = global_mean, global_std
    X_te[m] = np.clip((te_feat_raw[m].astype(np.float64) - mu) / sg, -CLIP_Z, CLIP_Z).astype(np.float32)
del tr_feat_raw, te_feat_raw; gc.collect()

# cs_v2_gold
print("Training cs_v2_gold...", flush=True)
X_tr_g = X_tr[:, gold_idx]; X_te_g = X_te[:, gold_idx]
groups = pd.qcut(pd.Series(train['SO3_T'].values), q=5, labels=False, duplicates='drop').values.astype(np.int32)
gkf = GroupKFold(n_splits=len(np.unique(groups)))
folds = list(gkf.split(X_tr_g, y_wins, groups=groups))
lgb_params = dict(objective='regression', metric='rmse', num_leaves=63, learning_rate=0.05,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, min_child_samples=50,
                  lambda_l1=0.1, lambda_l2=1.0, n_jobs=-1, verbose=-1, seed=42)
te_cs = np.zeros(len(X_te_g), dtype=np.float64)
for fi, (tri, vai) in enumerate(folds):
    dt = lgb.Dataset(X_tr_g[tri], label=y_wins[tri], free_raw_data=True)
    dv = lgb.Dataset(X_tr_g[vai], label=y_wins[vai], reference=dt, free_raw_data=True)
    m = lgb.train(lgb_params, dt, num_boost_round=2000, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(9999)])
    te_cs += m.predict(X_te_g, num_iteration=m.best_iteration) / len(folds)
    del dt, dv, m; gc.collect()
for d in np.unique(test_day):
    mask = test_day == d; te_cs[mask] -= te_cs[mask].mean()

# Ridge (gold features, alpha=5000 — original config)
print("Training Ridge (all feats, a=5000)...", flush=True)
ic_w = np.array([ic_dict.get(all_feat[i], 0.) for i in gold_idx], dtype=np.float64)
te_ridge = np.zeros(len(X_te), dtype=np.float64)
for d in np.unique(test_day):
    m_te = test_day == d
    if d not in train_days_set or (train_day == d).sum() < 20:
        X_g = X_te[m_te][:, gold_idx].astype(np.float64)
        pred = X_g @ ic_w; pred -= pred.mean(); te_ridge[m_te] = pred; continue
    m_tr = train_day == d
    X_tr_day = X_tr[m_tr].astype(np.float64); y_day = y_wins[m_tr]
    lv, hv = np.percentile(y_day, 1), np.percentile(y_day, 99); y_day = np.clip(y_day, lv, hv)
    mdl = Ridge(alpha=5000, fit_intercept=True)
    mdl.fit(X_tr_day, y_day)
    pred = mdl.predict(X_te[m_te].astype(np.float64)); pred -= pred.mean()
    te_ridge[m_te] = pred

# Grinold
print("Grinold...", flush=True)
ic_w_full = np.array([ic_dict.get(all_feat[i], 0.) for i in gold_idx], dtype=np.float64)
te_grinold = X_te[:, gold_idx].astype(np.float64) @ ic_w_full
for d in np.unique(test_day):
    mask = test_day == d; te_grinold[mask] -= te_grinold[mask].mean()

# Score individual models
cs_s = auto_scale(te_cs); ridge_s = auto_scale(te_ridge); grin_s = auto_scale(te_grinold)
print(f"\nIndividual oracle scores:")
print(f"  cs_gold:  IC={daywise_oracle(cs_s, oracle_df, test_ids, test_day):+.6f}  R2={daywise_r2(cs_s, oracle_df, test_ids, test_day):+.8f}")
print(f"  ridge:    IC={daywise_oracle(ridge_s, oracle_df, test_ids, test_day):+.6f}  R2={daywise_r2(ridge_s, oracle_df, test_ids, test_day):+.8f}")
print(f"  grinold:  IC={daywise_oracle(grin_s, oracle_df, test_ids, test_day):+.6f}  R2={daywise_r2(grin_s, oracle_df, test_ids, test_day):+.8f}")

# Generate submissions for multiple configs
configs = [
    ("cs_gold_solo", 1.0, 0.0, 0.0),
    ("3way_40_40_20", 0.40, 0.40, 0.20),
    ("cs70_grin30", 0.70, 0.30, 0.0),
    ("cs60_grin40", 0.60, 0.40, 0.0),
    ("cs80_grin20", 0.80, 0.20, 0.0),
    ("grinold_solo", 0.0, 1.0, 0.0),
]

print(f"\nBlend scores:")
for name, wc, wg, wr in configs:
    blend = wc * cs_s + wg * grin_s + wr * ridge_s
    for d in np.unique(test_day):
        mask = test_day == d; blend[mask] -= blend[mask].mean()
    blend_s = auto_scale(blend)
    ic = daywise_oracle(blend_s, oracle_df, test_ids, test_day)
    r2 = daywise_r2(blend_s, oracle_df, test_ids, test_day)
    print(f"  {name:25s}  IC={ic:+.6f}  R2={r2:+.8f}")

    sub = sample_sub.merge(pd.DataFrame({'ID': test_ids, 'TARGET': blend_s}), on='ID', how='left').fillna(0.0)
    path = os.path.join(OUT_DIR, f'compliant_{name}.csv')
    sub.to_csv(path, index=False)
    print(f"    → saved: {path}")

print(f"\nDone in {(time.time()-t0)/60:.1f} min")
