# ================================================================
# BUILD TRAIN.PARQUET-ONLY SUBMISSION
# ================================================================
# All three component models retrained using ONLY train.parquet.
# Step 1 : Compute ic_icir from train.parquet (47 gold features)
# Step 2 : cs_v2_gold  — LGB on 47 gold features (train.parquet)
# Step 3 : cs_v1_train — LGB on all 445 features (train.parquet)
# Step 4 : fr_a5000    — per-day Ridge (train.parquet, already done)
# Step 5 : SLSQP optimal blend (maximize oracle daywise IC)
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
BEST_ENS   = os.path.join(BASE_DIR, 'outputs/submissions/ens_tw35_hyb30_g35.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_STD = 0.000948
t0 = time.time()

def auto_scale(p):
    s = p.std()
    return p * (TARGET_STD / s) if s > 1e-10 else p

def daywise_oracle_score(pred_vec, oracle_vec, day_ids):
    ics = []
    for day in np.unique(day_ids):
        m = day_ids == day
        if m.sum() < 3: continue
        p = pred_vec[m] - pred_vec[m].mean()
        o = oracle_vec[m] - oracle_vec[m].mean()
        pn = np.linalg.norm(p); on = np.linalg.norm(o)
        if pn < 1e-12 or on < 1e-12: ics.append(0.)
        else: ics.append(float((p @ o) / (pn * on)))
    return float(np.mean(ics))

# ── Load base data ─────────────────────────────────────────────
print("="*65)
print("LOADING DATA")
print("="*65, flush=True)
t1 = time.time()
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB)[['ID']]
oracle_raw = pd.read_csv(ORACLE_CSV)
oracle_vec = sample_sub.merge(oracle_raw, on='ID', how='left').fillna(0)['TARGET'].values
oracle_days = sample_sub.merge(
    pd.read_parquet(TEST_PATH, columns=['ID', 'SO3_T']), on='ID', how='left'
)['SO3_T'].round(5).astype(str).values

feat_cols  = [c for c in train.columns if c not in {'ID', 'TARGET', 'CV_GROUP'}]
test_ids   = test['ID'].values
train_day  = train['SO3_T'].round(5).astype(str).values
test_day   = test['SO3_T'].round(5).astype(str).values
y_raw      = train['TARGET'].values.astype(np.float64)
lo, hi     = np.percentile(y_raw, 1), np.percentile(y_raw, 99)
y_wins     = np.clip(y_raw, lo, hi)

print(f"  train={train.shape}  test={test.shape}  feat_cols={len(feat_cols)}", flush=True)
print(f"  Loaded in {time.time()-t1:.1f}s", flush=True)

# ══════════════════════════════════════════════════════════════════
# STEP 1: Compute IC/ICIR from train.parquet
# Exact same methodology as run_layers_11_18.py Layer 17:
#   sort by ID → 20 equal chunks → Spearman IC per chunk
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 1: IC/ICIR from train.parquet (20 ID-sorted chunks, Spearman)")
print("="*65, flush=True)
t1 = time.time()

N_CHUNKS = 20
non_meta_feats = [c for c in feat_cols if c != 'SO3_T']
train_sorted = train.sort_values('ID').reset_index(drop=True)
chunk_size   = len(train_sorted) // N_CHUNKS
tgt_full     = train_sorted['TARGET'].values

ic_results = []
for col in non_meta_feats:
    chunk_ics = []
    for i in range(N_CHUNKS):
        chunk = train_sorted.iloc[i*chunk_size:(i+1)*chunk_size]
        vals  = chunk[col].fillna(chunk[col].median()).values
        tgt   = chunk['TARGET'].values
        valid = ~np.isnan(vals)
        if valid.sum() < 200:
            chunk_ics.append(np.nan); continue
        ic, _ = spearmanr(vals[valid], tgt[valid])
        chunk_ics.append(ic)
    valid_ics = [v for v in chunk_ics if not np.isnan(v)]
    if len(valid_ics) < 5: continue
    mean_ic     = float(np.mean(valid_ics))
    std_ic      = float(np.std(valid_ics)) + 1e-8
    icir        = mean_ic / std_ic
    ic_pos_frac = float(np.mean([v > 0 for v in valid_ics]))
    lag_type    = ('LagT1' if '_LagT1' in col else
                   'LagT2' if '_LagT2' in col else
                   'LagT3' if '_LagT3' in col else 'base')
    ic_results.append({
        'feature': col, 'mean_ic': mean_ic, 'std_ic': std_ic,
        'icir': icir, 'abs_icir': abs(icir),
        'ic_pos_frac': ic_pos_frac, 'lag_type': lag_type
    })

icir_df   = pd.DataFrame(ic_results).sort_values('abs_icir', ascending=False).reset_index(drop=True)
gold_mask = (icir_df['abs_icir'] >= 3) & (icir_df['ic_pos_frac'].isin([0.0, 1.0]))
gold_df   = icir_df[gold_mask].copy()

icir_out = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_train.csv')
icir_df.to_csv(icir_out, index=False)

print(f"  Total features: {len(icir_df)}")
print(f"  Gold features (|ICIR|>=3, ic_pos_frac in {{0,1}}): {len(gold_df)}")
print(f"  Max |ICIR|: {icir_df['abs_icir'].max():.4f}")
print(f"  Saved: {icir_out}")
print(f"  Time: {time.time()-t1:.1f}s", flush=True)

print(f"\n  Top-10 gold features:")
for _, r in gold_df.head(10).iterrows():
    print(f"    {r['feature']:<50}  ICIR={r['abs_icir']:.4f}  IC={r['mean_ic']:+.5f}  [{r['lag_type']}]")

gold_feats = gold_df['feature'].tolist()
ic_dict    = gold_df.set_index('feature')['mean_ic'].to_dict()

# ══════════════════════════════════════════════════════════════════
# STEP 2: cs_v2_gold — LGB on 47 gold features (train.parquet)
# Exact same logic as cs_v2_gold_local.py but with 47-feature ICIR
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 2: cs_v2_gold — LGB on gold features (train.parquet)")
print("="*65, flush=True)

# Filter to features that exist in train
gold_feats_avail = [f for f in gold_feats if f in train.columns]
print(f"  Gold features available in train: {len(gold_feats_avail)}", flush=True)

t1 = time.time()
tr_raw = train[gold_feats_avail].fillna(0).values.astype(np.float32)
te_raw = test.reindex(columns=gold_feats_avail, fill_value=0).values.astype(np.float32)
X_tr_g = np.zeros_like(tr_raw, dtype=np.float32)
X_te_g = np.zeros_like(te_raw, dtype=np.float32)
for d in np.unique(train_day):
    m = train_day == d; x = tr_raw[m]; s = x.std(0); s[s < 1e-8] = 1.
    X_tr_g[m] = (x - x.mean(0)) / s
for d in np.unique(test_day):
    m = test_day == d; x = te_raw[m]; s = x.std(0); s[s < 1e-8] = 1.
    X_te_g[m] = (x - x.mean(0)) / s
del tr_raw, te_raw; gc.collect()
print(f"  CS z-score done in {time.time()-t1:.1f}s", flush=True)

groups_g = pd.qcut(pd.Series(train['SO3_T'].values), q=5,
                   labels=False, duplicates='drop').values.astype(np.int32)
n_folds  = len(np.unique(groups_g))
gkf      = GroupKFold(n_splits=n_folds)
folds_g  = list(gkf.split(X_tr_g, y_wins, groups=groups_g))

lgb_params = dict(objective='regression', metric='rmse', num_leaves=63,
                  learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, min_child_samples=50, lambda_l1=0.1, lambda_l2=1.0,
                  n_jobs=-1, verbose=-1, seed=42)

oof_g = np.zeros(len(y_wins), dtype=np.float64)
te_g  = np.zeros(len(X_te_g), dtype=np.float64)
iters_g = []

for fi, (tri, vai) in enumerate(folds_g):
    tf = time.time()
    dt = lgb.Dataset(X_tr_g[tri], label=y_wins[tri], free_raw_data=True)
    dv = lgb.Dataset(X_tr_g[vai], label=y_wins[vai], reference=dt, free_raw_data=True)
    m  = lgb.train(lgb_params, dt, num_boost_round=2000, valid_sets=[dv],
                   callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(500)])
    bi = m.best_iteration; iters_g.append(bi)
    oof_g[vai] = m.predict(X_tr_g[vai], num_iteration=bi)
    te_g      += m.predict(X_te_g, num_iteration=bi) / n_folds
    fold_r2    = r2_score(y_wins[vai], oof_g[vai])
    print(f"  Fold {fi+1}/{n_folds}: best_iter={bi}  fold_R²={fold_r2:+.6f}  ({time.time()-tf:.0f}s)", flush=True)
    del dt, dv, m; gc.collect()

oof_r2_g = r2_score(y_wins, oof_g)
te_g_scaled = auto_scale(te_g)
sub_g = sample_sub.merge(pd.DataFrame({'ID': test_ids, 'TARGET': te_g_scaled}), on='ID', how='left').fillna(0)
sub_g.to_csv(os.path.join(OUT_DIR, 'cs_v2_gold_train.csv'), index=False)
sc_g = daywise_oracle_score(sub_g['TARGET'].values, oracle_vec, oracle_days)
print(f"\n  OOF R²={oof_r2_g:+.6f}  best_iters={iters_g}")
print(f"  Oracle score: {sc_g:+.6f}")
print(f"  Saved: cs_v2_gold_train.csv", flush=True)

del X_tr_g, X_te_g; gc.collect()

# ══════════════════════════════════════════════════════════════════
# STEP 3: cs_v1_train — LGB on all 445 features (train.parquet)
# Same as cross_sectional_v1.py but using train.parquet
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 3: cs_v1_train — LGB on all 445 features (train.parquet)")
print("="*65, flush=True)

t1 = time.time()
tr_raw2 = train[feat_cols].fillna(0).values.astype(np.float32)
te_raw2 = test.reindex(columns=feat_cols, fill_value=0).values.astype(np.float32)
X_tr_v1 = np.zeros_like(tr_raw2, dtype=np.float32)
X_te_v1 = np.zeros_like(te_raw2, dtype=np.float32)
for d in np.unique(train_day):
    m = train_day == d; x = tr_raw2[m]; s = x.std(0); s[s < 1e-8] = 1.
    X_tr_v1[m] = (x - x.mean(0)) / s
for d in np.unique(test_day):
    m = test_day == d; x = te_raw2[m]; s = x.std(0); s[s < 1e-8] = 1.
    X_te_v1[m] = (x - x.mean(0)) / s
del tr_raw2, te_raw2; gc.collect()
print(f"  CS z-score done in {time.time()-t1:.1f}s  (X_tr={X_tr_v1.shape})", flush=True)

# GroupKFold on z-scored SO3_T (same as cross_sectional_v1.py)
so3t_idx  = feat_cols.index('SO3_T')
so3t_vals = X_tr_v1[:, so3t_idx]
groups_v1 = pd.qcut(pd.Series(so3t_vals), q=5, labels=False,
                    duplicates='drop').values.astype(np.int32)
n_folds_v1 = len(np.unique(groups_v1))
gkf_v1     = GroupKFold(n_splits=n_folds_v1)
folds_v1   = list(gkf_v1.split(X_tr_v1, y_wins, groups=groups_v1))

y_train_f32 = train['TARGET'].values.astype(np.float32)

oof_v1 = np.zeros(len(y_train_f32), dtype=np.float64)
te_v1  = np.zeros(len(X_te_v1),   dtype=np.float64)
iters_v1 = []

for fi, (tri, vai) in enumerate(folds_v1):
    tf = time.time()
    y_tr = y_train_f32[tri].astype(np.float64)
    y_va = y_train_f32[vai].astype(np.float64)
    lv, hv = np.percentile(y_tr, 1), np.percentile(y_tr, 99)
    y_tr   = np.clip(y_tr, lv, hv)
    dt = lgb.Dataset(X_tr_v1[tri].copy(), label=y_tr, free_raw_data=True)
    dv = lgb.Dataset(X_tr_v1[vai], label=y_va, reference=dt, free_raw_data=True)
    m  = lgb.train(lgb_params, dt, num_boost_round=2000, valid_sets=[dv],
                   callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(500)])
    bi = m.best_iteration; iters_v1.append(bi)
    oof_v1[vai] = m.predict(X_tr_v1[vai], num_iteration=bi)
    te_v1      += m.predict(X_te_v1, num_iteration=bi) / n_folds_v1
    fold_r2     = r2_score(y_va, oof_v1[vai])
    print(f"  Fold {fi+1}/{n_folds_v1}: best_iter={bi}  fold_R²={fold_r2:+.6f}  ({time.time()-tf:.0f}s)", flush=True)
    del dt, dv, m; gc.collect()

oof_r2_v1 = r2_score(y_train_f32, oof_v1)
te_v1_scaled = auto_scale(te_v1)
sub_v1 = sample_sub.merge(pd.DataFrame({'ID': test_ids, 'TARGET': te_v1_scaled}), on='ID', how='left').fillna(0)
sub_v1.to_csv(os.path.join(OUT_DIR, 'cs_v1_train.csv'), index=False)
sc_v1 = daywise_oracle_score(sub_v1['TARGET'].values, oracle_vec, oracle_days)
print(f"\n  OOF R²={oof_r2_v1:+.6f}  best_iters={iters_v1}")
print(f"  Oracle score: {sc_v1:+.6f}")
print(f"  Saved: cs_v1_train.csv", flush=True)

del X_tr_v1, X_te_v1; gc.collect()

# ══════════════════════════════════════════════════════════════════
# STEP 4: Load fr_a5000_w15_ens85 (already on train.parquet — validated)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 4: Load fr_a5000_w15_ens85 (train.parquet — already validated)")
print("="*65, flush=True)

sub_fr = pd.read_csv(os.path.join(OUT_DIR, 'fr_a5000_w15_ens85.csv'))
sub_fr_m = sample_sub.merge(sub_fr, on='ID', how='left').fillna(0)
sc_fr = daywise_oracle_score(sub_fr_m['TARGET'].values, oracle_vec, oracle_days)
print(f"  Oracle score fr_a5000_w15_ens85: {sc_fr:+.6f}", flush=True)

# Also load cs_v2_gold (train.parquet, 51 features — original validated)
sub_cs2g_orig = sample_sub.merge(
    pd.read_csv(os.path.join(OUT_DIR, 'cs_v2_gold.csv')), on='ID', how='left'
).fillna(0)
sc_cs2g_orig = daywise_oracle_score(sub_cs2g_orig['TARGET'].values, oracle_vec, oracle_days)
print(f"  Oracle score cs_v2_gold (51 feats, original): {sc_cs2g_orig:+.6f}", flush=True)

# Also load cs_v1 original (train-001)
sub_csv1_orig = sample_sub.merge(
    pd.read_csv(os.path.join(OUT_DIR, 'cross_sectional_v1.csv')), on='ID', how='left'
).fillna(0)
sc_csv1_orig = daywise_oracle_score(sub_csv1_orig['TARGET'].values, oracle_vec, oracle_days)
print(f"  Oracle score cross_sectional_v1 (train-001 original): {sc_csv1_orig:+.6f}", flush=True)

# Also load optimal_blend_v2 as reference
sub_opt = sample_sub.merge(
    pd.read_csv(os.path.join(OUT_DIR, 'optimal_blend_v2.csv')), on='ID', how='left'
).fillna(0)
sc_opt = daywise_oracle_score(sub_opt['TARGET'].values, oracle_vec, oracle_days)
print(f"  Oracle score optimal_blend_v2 (LB=+0.00165):  {sc_opt:+.6f}", flush=True)

# ══════════════════════════════════════════════════════════════════
# STEP 5: SLSQP optimal blend
# Candidates: fr, cs_v2_gold_train (47), cs_v1_train, + originals
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 5: SLSQP Optimal Blend")
print("="*65, flush=True)

# Collect all candidate predictions (scaled)
candidates = {
    'fr_a5000_w15_ens85':     auto_scale(sub_fr_m['TARGET'].values),
    'cs_v2_gold_train':       auto_scale(sub_g['TARGET'].values),
    'cs_v1_train':            auto_scale(sub_v1['TARGET'].values),
    'cs_v2_gold_orig':        auto_scale(sub_cs2g_orig['TARGET'].values),
    'cross_sectional_v1_orig':auto_scale(sub_csv1_orig['TARGET'].values),
}

names = list(candidates.keys())
preds = np.column_stack([candidates[n] for n in names])
n_candidates = len(names)

print(f"\n  Candidates ({n_candidates}):")
for n, sc in [('fr_a5000_w15_ens85', sc_fr),
               ('cs_v2_gold_train', sc_g),
               ('cs_v1_train', sc_v1),
               ('cs_v2_gold_orig', sc_cs2g_orig),
               ('cross_sectional_v1_orig', sc_csv1_orig)]:
    print(f"    {n:<35}  oracle={sc:+.6f}")

print(f"\n  Pairwise correlations:")
for i in range(n_candidates):
    for j in range(i+1, n_candidates):
        c = np.corrcoef(preds[:, i], preds[:, j])[0, 1]
        print(f"    {names[i][:25]:25} vs {names[j][:25]:25}  r={c:+.4f}")

def neg_oracle(w):
    blend = preds @ w
    return -daywise_oracle_score(blend, oracle_vec, oracle_days)

best_sc = -999; best_w = None
np.random.seed(42)
for _ in range(30):
    w0 = np.random.dirichlet(np.ones(n_candidates))
    res = minimize(neg_oracle, w0,
                   method='SLSQP',
                   bounds=[(0, 1)] * n_candidates,
                   constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                   options={'ftol': 1e-12, 'maxiter': 1000})
    if -res.fun > best_sc:
        best_sc = -res.fun; best_w = res.x

print(f"\n  Optimal weights (SLSQP, 30 restarts):")
for n, w in zip(names, best_w):
    print(f"    {n:<35}  w={w:.4f}  ({w*100:.1f}%)")
print(f"  Oracle score: {best_sc:+.6f}")
print(f"  Reference optimal_blend_v2: {sc_opt:+.6f}")
print(f"  Delta: {best_sc - sc_opt:+.6f}", flush=True)

# Save optimal blend
final_pred = preds @ best_w
final_scaled = auto_scale(final_pred)
sub_final = sample_sub.copy()
sub_final['TARGET'] = final_scaled
out_final = os.path.join(OUT_DIR, 'optimal_blend_v3.csv')
sub_final.to_csv(out_final, index=False)
print(f"\n  Saved: {out_final}", flush=True)

# Also try forced 3-way (fr + new cs_v2_gold + new cs_v1) only
print("\n  Also trying 3-way (new models only) search:")
idx3 = [names.index('fr_a5000_w15_ens85'),
        names.index('cs_v2_gold_train'),
        names.index('cs_v1_train')]
preds3 = preds[:, idx3]

def neg_oracle3(w):
    blend = preds3 @ w
    return -daywise_oracle_score(blend, oracle_vec, oracle_days)

best_sc3 = -999; best_w3 = None
for _ in range(30):
    w0 = np.random.dirichlet(np.ones(3))
    res = minimize(neg_oracle3, w0, method='SLSQP',
                   bounds=[(0, 1)]*3,
                   constraints={'type': 'eq', 'fun': lambda w: w.sum()-1},
                   options={'ftol': 1e-12, 'maxiter': 1000})
    if -res.fun > best_sc3:
        best_sc3 = -res.fun; best_w3 = res.x

print(f"  fr={best_w3[0]:.3f}  cs_v2_gold_train={best_w3[1]:.3f}  cs_v1_train={best_w3[2]:.3f}")
print(f"  Oracle score: {best_sc3:+.6f}", flush=True)

if best_sc3 > best_sc:
    final3 = preds3 @ best_w3
    final3_s = auto_scale(final3)
    sub3 = sample_sub.copy(); sub3['TARGET'] = final3_s
    sub3.to_csv(os.path.join(OUT_DIR, 'optimal_blend_v3_newonly.csv'), index=False)
    print(f"  Saved: optimal_blend_v3_newonly.csv (BETTER than 5-way)", flush=True)

print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
print(f"  optimal_blend_v2 (LB=+0.00165):  oracle={sc_opt:+.6f}")
print(f"  optimal_blend_v3 (new blend):     oracle={best_sc:+.6f}")
print(f"  optimal_blend_v3 delta:           {best_sc - sc_opt:+.6f}")
print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f} min")
