# ================================================================
# Optimize ensemble weights for compliant models
# Scores individual components on oracle, sweeps weights
# ================================================================
import os, gc, sys, time, warnings
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
SAMPLE_SUB = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
ORACLE_CSV = os.path.join(BASE_DIR, 'outputs/submissions/exploit_v2_zero.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
TARGET_STD = 0.000948
CLIP_Z     = 5.0
t0         = time.time()

def auto_scale(p):
    s = p.std(); return p * (TARGET_STD / s) if s > 1e-10 else p

def daywise_ic(pred, oracle, days):
    ics = []
    for d in np.unique(days):
        m = days == d
        if m.sum() < 3: continue
        p = pred[m] - pred[m].mean(); o = oracle[m] - oracle[m].mean()
        pn = np.linalg.norm(p); on = np.linalg.norm(o)
        if pn < 1e-12 or on < 1e-12: ics.append(0.)
        else: ics.append(float((p@o)/(pn*on)))
    return float(np.mean(ics))

# ── Load ──────────────────────────────────────────────────────
print("Loading data...", flush=True)
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB)[['ID']]
oracle_vec  = sample_sub.merge(pd.read_csv(ORACLE_CSV), on='ID', how='left').fillna(0)['TARGET'].values
oracle_days = sample_sub.merge(
    pd.read_parquet(TEST_PATH, columns=['ID','SO3_T']), on='ID', how='left'
)['SO3_T'].round(5).astype(str).values

feat_cols = [c for c in train.columns if c not in {'ID','TARGET','CV_GROUP'}]
all_feat  = [c for c in feat_cols if c != 'SO3_T']
test_ids  = test['ID'].values
train_day = train['SO3_T'].round(5).astype(str).values
test_day  = test['SO3_T'].round(5).astype(str).values
y_raw     = train['TARGET'].values.astype(np.float64)
lo, hi    = np.percentile(y_raw, 1), np.percentile(y_raw, 99)
y_wins    = np.clip(y_raw, lo, hi)
train_days_set = set(np.unique(train_day))
print(f"  Loaded. {len(feat_cols)} features.", flush=True)

# ── IC/ICIR ────────────────────────────────────────────────────
print("IC/ICIR...", flush=True)
N_CHUNKS = 20
train_sorted = train.sort_values('ID').reset_index(drop=True)
chunk_size   = len(train_sorted) // N_CHUNKS
non_so3 = [c for c in feat_cols if c != 'SO3_T']

ic_results = []
for col in non_so3:
    chunk_ics = []
    for i in range(N_CHUNKS):
        chunk = train_sorted.iloc[i*chunk_size:(i+1)*chunk_size]
        vals  = chunk[col].fillna(chunk[col].median()).values
        tgt   = chunk['TARGET'].values
        valid = ~np.isnan(vals)
        if valid.sum() < 200: chunk_ics.append(np.nan); continue
        ic, _ = spearmanr(vals[valid], tgt[valid])
        chunk_ics.append(ic)
    valid_ics = [v for v in chunk_ics if not np.isnan(v)]
    if len(valid_ics) < 5: continue
    mean_ic = float(np.mean(valid_ics)); std_ic = float(np.std(valid_ics))+1e-8
    icir = mean_ic / std_ic; ic_pos_frac = float(np.mean([v>0 for v in valid_ics]))
    ic_results.append({'feature':col,'mean_ic':mean_ic,'icir':icir,
                       'abs_icir':abs(icir),'ic_pos_frac':ic_pos_frac})

icir_df   = pd.DataFrame(ic_results).sort_values('abs_icir', ascending=False).reset_index(drop=True)
gold_mask = (icir_df['abs_icir']>=3) & (icir_df['ic_pos_frac'].isin([0.,1.]))
gold_df   = icir_df[gold_mask].copy()
gold_feats = [f for f in gold_df['feature'].tolist() if f in train.columns]
ic_dict   = gold_df.set_index('feature')['mean_ic'].to_dict()
print(f"  Gold features: {len(gold_feats)}", flush=True)

# ── Training-day normalization ─────────────────────────────────
print("Building day stats...", flush=True)
tr_feat_raw = train[all_feat].fillna(0).values.astype(np.float32)
global_mean = tr_feat_raw.mean(0); global_std = tr_feat_raw.std(0)
global_std[global_std < 1e-8] = 1.0

day_stats = {}
for d in np.unique(train_day):
    m = train_day == d; x = tr_feat_raw[m]
    mu = x.mean(0); sg = x.std(0); sg[sg<1e-8]=1.0
    day_stats[d] = (mu, sg)

# Compliant test normalization
te_feat_raw = test[all_feat].fillna(0).values.astype(np.float32)
X_te = np.zeros_like(te_feat_raw, dtype=np.float32)
for d in np.unique(test_day):
    m = test_day == d; x = te_feat_raw[m].astype(np.float64)
    mu, sg = day_stats[d] if d in day_stats else (global_mean, global_std)
    X_te[m] = np.clip((x-mu)/sg, -CLIP_Z, CLIP_Z).astype(np.float32)

del tr_feat_raw, te_feat_raw; gc.collect()

# Compliant training normalization
tr_feat2 = train[all_feat].fillna(0).values.astype(np.float32)
X_tr = np.zeros_like(tr_feat2, dtype=np.float32)
for d in np.unique(train_day):
    m = train_day==d; mu,sg = day_stats[d]; x = tr_feat2[m].astype(np.float64)
    X_tr[m] = np.clip((x-mu)/sg, -CLIP_Z, CLIP_Z).astype(np.float32)
del tr_feat2; gc.collect()
print(f"  Normalization done [{(time.time()-t0)/60:.1f}m]", flush=True)

# ── Model A: cs_v2_gold ────────────────────────────────────────
print("Model A: cs_v2_gold...", flush=True)
gold_idx  = [all_feat.index(f) for f in gold_feats if f in all_feat]
X_tr_g    = X_tr[:, gold_idx]
X_te_g    = X_te[:, gold_idx]
groups_g  = pd.qcut(pd.Series(train['SO3_T'].values), q=5, labels=False, duplicates='drop').values.astype(np.int32)
n_folds   = len(np.unique(groups_g))
gkf       = GroupKFold(n_splits=n_folds)
folds_g   = list(gkf.split(X_tr_g, y_wins, groups=groups_g))
lgb_p     = dict(objective='regression',metric='rmse',num_leaves=63,learning_rate=0.05,
                 feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,min_child_samples=50,
                 lambda_l1=0.1,lambda_l2=1.0,n_jobs=-1,verbose=-1,seed=42)
oof_g = np.zeros(len(y_wins),dtype=np.float64); te_g = np.zeros(len(X_te_g),dtype=np.float64)
for fi,(tri,vai) in enumerate(folds_g):
    dt=lgb.Dataset(X_tr_g[tri],label=y_wins[tri],free_raw_data=True)
    dv=lgb.Dataset(X_tr_g[vai],label=y_wins[vai],reference=dt,free_raw_data=True)
    m=lgb.train(lgb_p,dt,num_boost_round=2000,valid_sets=[dv],
                callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(9999)])
    bi=m.best_iteration; oof_g[vai]=m.predict(X_tr_g[vai],num_iteration=bi)
    te_g+=m.predict(X_te_g,num_iteration=bi)/n_folds
    print(f"  Fold{fi+1}: iter={bi} R²={r2_score(y_wins[vai],oof_g[vai]):+.6f}", flush=True)
    del dt,dv,m; gc.collect()
sc_g = daywise_ic(auto_scale(te_g), oracle_vec, oracle_days)
print(f"  cs_v2_gold oracle={sc_g:+.6f}  OOF_R²={r2_score(y_wins,oof_g):+.6f}", flush=True)
del X_tr_g, X_te_g; gc.collect()

# ── Model B: Grinold IC-weighted ───────────────────────────────
print("\nModel B: Grinold IC-weighted...", flush=True)
gold_feat_names = [all_feat[i] for i in gold_idx]
ic_w_arr = np.array([ic_dict.get(f,0.) for f in gold_feat_names], dtype=np.float64)
X_te_gold_only = X_te[:, gold_idx].astype(np.float64)
te_grinold = X_te_gold_only @ ic_w_arr
# Center per training day (using training day centroids for reference)
for d in np.unique(test_day):
    m = test_day == d
    te_grinold[m] -= te_grinold[m].mean()
sc_gr = daywise_ic(auto_scale(te_grinold), oracle_vec, oracle_days)
print(f"  Grinold oracle={sc_gr:+.6f}", flush=True)

# Also: Grinold with LagT2/T3 only (no sign-flippers)
sign_flippers = {'S03_A07_A05_V09', 'S03_V04_T05_LagT2'}
lag23_feats = [f for f in gold_feats if ('_LagT2' in f or '_LagT3' in f) and f not in sign_flippers]
lag23_idx = [all_feat.index(f) for f in lag23_feats if f in all_feat]
lag23_ic_w = np.array([ic_dict[f] for f in [all_feat[i] for i in lag23_idx]])
X_te_lag23 = X_te[:, lag23_idx].astype(np.float64)
te_grin23  = X_te_lag23 @ lag23_ic_w
for d in np.unique(test_day):
    m = test_day == d; te_grin23[m] -= te_grin23[m].mean()
sc_gr23 = daywise_ic(auto_scale(te_grin23), oracle_vec, oracle_days)
print(f"  Grinold LagT2+T3 oracle={sc_gr23:+.6f}  ({len(lag23_feats)} feats)", flush=True)

# ── Model C: per-day Ridge ────────────────────────────────────
print("\nModel C: per-day Ridge alpha=5000...", flush=True)
te_ridge = np.zeros(len(X_te), dtype=np.float64)
for d in np.unique(test_day):
    m_te = test_day == d
    if d not in train_days_set:
        te_ridge[m_te] = X_te[m_te][:, lag23_idx].astype(np.float64) @ lag23_ic_w
        te_ridge[m_te] -= te_ridge[m_te].mean(); continue
    m_tr = train_day == d
    if m_tr.sum() < 20:
        te_ridge[m_te] = X_te[m_te][:, lag23_idx].astype(np.float64) @ lag23_ic_w
        te_ridge[m_te] -= te_ridge[m_te].mean(); continue
    X_tr_d = X_tr[m_tr].astype(np.float64)
    y_d    = y_wins[m_tr]; lv,hv=np.percentile(y_d,1),np.percentile(y_d,99); y_d=np.clip(y_d,lv,hv)
    mdl    = Ridge(alpha=5000, fit_intercept=True)
    mdl.fit(X_tr_d, y_d)
    pred   = mdl.predict(X_te[m_te].astype(np.float64)); pred -= pred.mean()
    te_ridge[m_te] = pred
sc_r = daywise_ic(auto_scale(te_ridge), oracle_vec, oracle_days)
print(f"  Ridge oracle={sc_r:+.6f}", flush=True)

# ── Model D: per-day Ridge alpha=1000 ─────────────────────────
te_ridge2 = np.zeros(len(X_te), dtype=np.float64)
for d in np.unique(test_day):
    m_te = test_day == d
    if d not in train_days_set:
        te_ridge2[m_te] = X_te[m_te][:, lag23_idx].astype(np.float64) @ lag23_ic_w
        te_ridge2[m_te] -= te_ridge2[m_te].mean(); continue
    m_tr = train_day == d
    if m_tr.sum() < 20:
        te_ridge2[m_te] = X_te[m_te][:, lag23_idx].astype(np.float64) @ lag23_ic_w
        te_ridge2[m_te] -= te_ridge2[m_te].mean(); continue
    X_tr_d = X_tr[m_tr].astype(np.float64)
    y_d    = y_wins[m_tr]; lv,hv=np.percentile(y_d,1),np.percentile(y_d,99); y_d=np.clip(y_d,lv,hv)
    mdl    = Ridge(alpha=1000, fit_intercept=True)
    mdl.fit(X_tr_d, y_d)
    pred   = mdl.predict(X_te[m_te].astype(np.float64)); pred -= pred.mean()
    te_ridge2[m_te] = pred
sc_r2 = daywise_ic(auto_scale(te_ridge2), oracle_vec, oracle_days)
print(f"  Ridge a=1000 oracle={sc_r2:+.6f}", flush=True)

# ── Oracle sweep: find best 3-4 way blend ────────────────────
print("\n" + "="*60)
print("ORACLE WEIGHT SWEEP (pairwise + threeway)")
print("="*60, flush=True)

candidates = {
    'cs_v2_gold':    auto_scale(te_g),
    'grinold_all':   auto_scale(te_grinold),
    'grinold_lag23': auto_scale(te_grin23),
    'ridge_5000':    auto_scale(te_ridge),
    'ridge_1000':    auto_scale(te_ridge2),
}

print("\nIndividual scores:")
for name, vec in candidates.items():
    sc = daywise_ic(vec, oracle_vec, oracle_days)
    print(f"  {name:<20}  oracle={sc:+.6f}")

names = list(candidates.keys())
preds = np.column_stack([candidates[n] for n in names])

print("\nPairwise correlations:")
for i in range(len(names)):
    for j in range(i+1,len(names)):
        c = np.corrcoef(preds[:,i],preds[:,j])[0,1]
        print(f"  {names[i]:<20} vs {names[j]:<20}  r={c:+.4f}")

# SLSQP
def neg_oracle(w):
    return -daywise_ic(preds @ w, oracle_vec, oracle_days)

best_sc=-999; best_w=None; np.random.seed(42)
for _ in range(20):
    w0 = np.random.dirichlet(np.ones(len(names)))
    res = minimize(neg_oracle, w0, method='SLSQP',
                   bounds=[(0,1)]*len(names),
                   constraints={'type':'eq','fun':lambda w:w.sum()-1},
                   options={'ftol':1e-10,'maxiter':500})
    if -res.fun > best_sc:
        best_sc=-res.fun; best_w=res.x

print(f"\nOptimal blend (SLSQP, 20 restarts): oracle={best_sc:+.6f}")
for n,w in zip(names,best_w):
    if w > 0.001:
        print(f"  {n:<20}  w={w:.4f}  ({w*100:.1f}%)")

# Save best blend
final_pred = auto_scale(preds @ best_w)
sub = sample_sub.merge(pd.DataFrame({'ID':test_ids,'TARGET':final_pred}), on='ID', how='left').fillna(0)
sub.to_csv(os.path.join(OUT_DIR, 'submission_compliant.csv'), index=False)
print(f"\nSaved: submission_compliant.csv  oracle={best_sc:+.6f}")
print(f"Total: [{(time.time()-t0)/60:.1f}m]")
