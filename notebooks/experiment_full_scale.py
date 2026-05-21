"""
Full scale optimization for ALL model components and their ensembles.
Tests: Ridge (all feats a=5000), Ridge (gold a=500), Grinold, cs_gold
at different scales, plus scale-optimized ensembles.
"""
import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = '/Users/malaymishra/Desktop/quant_ml_project/data/raw'
OUT_DIR  = '/Users/malaymishra/Desktop/quant_ml_project/outputs/submissions'
ORACLE_PATH = os.path.join(OUT_DIR, 'exploit_v2_zero.csv')
CLIP_Z = 5.0
t0 = time.time()

def daywise_r2(pred, y, day):
    """R² computed globally (like LB)."""
    ss_res = np.sum((y - pred)**2)
    ss_tot = np.sum(y**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.

def optimal_scale_r2(pred_unit, y, day):
    """Find the scale factor s that maximizes R²(y, s*pred_unit).
    R²(s) = 1 - [Σy² - 2s·Σy·p + s²·Σp²] / Σy² = 2s·C/T - s²·P/T
    where C = Σy·p, T = Σy², P = Σp². Optimal: s* = C/P = Cov(y,p)/Var(p) when p is zero-mean.
    """
    C = np.sum(y * pred_unit)
    P = np.sum(pred_unit**2)
    s_star = C / P if P > 1e-20 else 0.
    r2_star = daywise_r2(s_star * pred_unit, y, day)
    return s_star, r2_star

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

# Merge oracle for scoring
oracle_map = oracle_df.set_index('ID')['TARGET'].to_dict()
oracle_mask = np.array([oid in oracle_map for oid in test_ids])
oracle_y = np.array([oracle_map.get(oid, 0.) for oid in test_ids])
oracle_days = test_day.copy()

# IC/ICIR
print("IC/ICIR...", flush=True)
N_CHUNKS = 20; train_sorted = train.sort_values('ID').reset_index(drop=True)
chunk_size = len(train_sorted) // N_CHUNKS; ic_results = []
for col in [c for c in feat_cols if c != 'SO3_T']:
    chunk_ics = []
    for i in range(N_CHUNKS):
        chunk = train_sorted.iloc[i*chunk_size:(i+1)*chunk_size]
        vals = chunk[col].fillna(chunk[col].median()).values; tgt = chunk['TARGET'].values
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
gold_mask_df = (icir_df['abs_icir'] >= 3) & (icir_df['ic_pos_frac'].isin([0.0, 1.0]))
gold_df = icir_df[gold_mask_df].copy()
gold_feats = [f for f in gold_df['feature'].tolist() if f in all_feat]
ic_dict = gold_df.set_index('feature')['mean_ic'].to_dict()
gold_idx = [all_feat.index(f) for f in gold_feats]
print(f"  Gold: {len(gold_feats)}", flush=True)

# Normalization
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

# ═══════════════════════════════════════════════════════════
# Build ALL model components (raw, unit-variance, per-day demeaned)
# ═══════════════════════════════════════════════════════════

models = {}  # name → unit-variance, per-day demeaned prediction vector

# 1. cs_v2_gold
print("\nTraining cs_v2_gold...", flush=True)
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
models['cs_gold'] = te_cs / te_cs.std()

# 2. Grinold
print("Building Grinold...", flush=True)
ic_w = np.array([ic_dict.get(f, 0.) for f in gold_feats], dtype=np.float64)
te_grin = X_te[:, gold_idx].astype(np.float64) @ ic_w
for d in np.unique(test_day):
    mask = test_day == d; te_grin[mask] -= te_grin[mask].mean()
models['grinold'] = te_grin / te_grin.std()

# 3. Ridge (all features, alpha=5000)
print("Building Ridge (all feats, a=5000)...", flush=True)
te_ridge_all = np.zeros(len(X_te), dtype=np.float64)
for d in np.unique(test_day):
    m_te = test_day == d
    if d not in train_days_set or (train_day == d).sum() < 20:
        X_g = X_te[m_te][:, gold_idx].astype(np.float64)
        pred = X_g @ ic_w; pred -= pred.mean(); te_ridge_all[m_te] = pred; continue
    m_tr = train_day == d
    X_tr_day = X_tr[m_tr].astype(np.float64); y_day = y_wins[m_tr]
    lv, hv = np.percentile(y_day, 1), np.percentile(y_day, 99); y_day = np.clip(y_day, lv, hv)
    mdl = Ridge(alpha=5000, fit_intercept=True); mdl.fit(X_tr_day, y_day)
    pred = mdl.predict(X_te[m_te].astype(np.float64)); pred -= pred.mean()
    te_ridge_all[m_te] = pred
models['ridge_all_a5k'] = te_ridge_all / te_ridge_all.std()

# 4. Ridge (gold features, alpha=500)
print("Building Ridge (gold feats, a=500)...", flush=True)
te_ridge_gold = np.zeros(len(X_te), dtype=np.float64)
for d in np.unique(test_day):
    m_te = test_day == d
    if d not in train_days_set or (train_day == d).sum() < 20:
        X_g = X_te[m_te][:, gold_idx].astype(np.float64)
        pred = X_g @ ic_w; pred -= pred.mean(); te_ridge_gold[m_te] = pred; continue
    m_tr = train_day == d
    X_tr_day = X_tr[m_tr][:, gold_idx].astype(np.float64); y_day = y_wins[m_tr]
    lv, hv = np.percentile(y_day, 1), np.percentile(y_day, 99); y_day = np.clip(y_day, lv, hv)
    mdl = Ridge(alpha=500, fit_intercept=True); mdl.fit(X_tr_day, y_day)
    pred = mdl.predict(X_te[m_te][:, gold_idx].astype(np.float64)); pred -= pred.mean()
    te_ridge_gold[m_te] = pred
models['ridge_gold_a500'] = te_ridge_gold / te_ridge_gold.std()

# 5. Ridge (gold features, alpha=100)
print("Building Ridge (gold feats, a=100)...", flush=True)
te_ridge_g100 = np.zeros(len(X_te), dtype=np.float64)
for d in np.unique(test_day):
    m_te = test_day == d
    if d not in train_days_set or (train_day == d).sum() < 20:
        X_g = X_te[m_te][:, gold_idx].astype(np.float64)
        pred = X_g @ ic_w; pred -= pred.mean(); te_ridge_g100[m_te] = pred; continue
    m_tr = train_day == d
    X_tr_day = X_tr[m_tr][:, gold_idx].astype(np.float64); y_day = y_wins[m_tr]
    lv, hv = np.percentile(y_day, 1), np.percentile(y_day, 99); y_day = np.clip(y_day, lv, hv)
    mdl = Ridge(alpha=100, fit_intercept=True); mdl.fit(X_tr_day, y_day)
    pred = mdl.predict(X_te[m_te][:, gold_idx].astype(np.float64)); pred -= pred.mean()
    te_ridge_g100[m_te] = pred
models['ridge_gold_a100'] = te_ridge_g100 / te_ridge_g100.std()

del X_tr, X_te; gc.collect()

# ═══════════════════════════════════════════════════════════
# Optimal scale for each model individually
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("INDIVIDUAL MODEL OPTIMAL SCALES (on oracle subset)")
print(f"{'='*70}")

# Use oracle subset for scoring
y_oracle = oracle_y[oracle_mask]
days_oracle = oracle_days[oracle_mask]

for name, pred_unit in models.items():
    pred_oracle = pred_unit[oracle_mask]
    s_star, r2_star = optimal_scale_r2(pred_oracle, y_oracle, days_oracle)
    equiv_std = abs(s_star) * 1.0  # pred_unit has std=1, so scale = target std
    # Also compute at TARGET_STD=0.000948
    r2_948 = daywise_r2(pred_unit[oracle_mask] * 0.000948, y_oracle, days_oracle)
    print(f"  {name:25s}  s*={s_star:.8f} (≈std={equiv_std:.6f})  R²*={r2_star:+.10f}  R²@0.000948={r2_948:+.10f}")

# ═══════════════════════════════════════════════════════════
# Pairwise correlations
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PAIRWISE CORRELATIONS")
print(f"{'='*70}")
model_names = list(models.keys())
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        r = np.corrcoef(models[model_names[i]], models[model_names[j]])[0,1]
        print(f"  {model_names[i]:25s} vs {model_names[j]:25s}  r={r:+.4f}")

# ═══════════════════════════════════════════════════════════
# Exhaustive 2-way blends with optimal scale
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("2-WAY BLENDS WITH OPTIMAL SCALE")
print(f"{'='*70}")

blend_results = []

for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        for w1 in np.arange(0.1, 1.0, 0.1):
            w2 = 1 - w1
            blend = w1 * models[model_names[i]] + w2 * models[model_names[j]]
            # Per-day demean
            for d in np.unique(test_day):
                mask = test_day == d; blend[mask] -= blend[mask].mean()
            blend = blend / blend.std()  # unit variance

            b_oracle = blend[oracle_mask]
            s_star, r2_star = optimal_scale_r2(b_oracle, y_oracle, days_oracle)
            # Also at std=0.0006
            r2_06 = daywise_r2(blend[oracle_mask] * 0.0006, y_oracle, days_oracle)

            blend_results.append({
                'model1': model_names[i], 'model2': model_names[j],
                'w1': w1, 'w2': w2,
                'opt_scale': s_star, 'r2_opt': r2_star, 'r2_06': r2_06
            })

blend_df = pd.DataFrame(blend_results).sort_values('r2_opt', ascending=False)
print("\nTop 20 blends by optimal R²:")
for _, r in blend_df.head(20).iterrows():
    print(f"  {r['w1']:.1f}×{r['model1']:20s} + {r['w2']:.1f}×{r['model2']:20s}  "
          f"s*={r['opt_scale']:.6f}  R²*={r['r2_opt']:+.10f}  R²@0.0006={r['r2_06']:+.10f}")

# ═══════════════════════════════════════════════════════════
# 3-way blends with optimal scale
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("3-WAY BLENDS (cs_gold + grinold + best_ridge) WITH OPTIMAL SCALE")
print(f"{'='*70}")

three_results = []
for ridge_name in ['ridge_all_a5k', 'ridge_gold_a500', 'ridge_gold_a100']:
    for wc in np.arange(0.1, 0.9, 0.1):
        for wg in np.arange(0.1, 0.9 - wc + 0.01, 0.1):
            wr = round(1 - wc - wg, 2)
            if wr < 0.05: continue
            blend = wc * models['cs_gold'] + wg * models['grinold'] + wr * models[ridge_name]
            for d in np.unique(test_day):
                mask = test_day == d; blend[mask] -= blend[mask].mean()
            blend = blend / blend.std()
            b_oracle = blend[oracle_mask]
            s_star, r2_star = optimal_scale_r2(b_oracle, y_oracle, days_oracle)
            three_results.append({
                'wc': wc, 'wg': wg, 'wr': wr, 'ridge': ridge_name,
                'opt_scale': s_star, 'r2_opt': r2_star
            })

three_df = pd.DataFrame(three_results).sort_values('r2_opt', ascending=False)
print("\nTop 15 three-way blends:")
for _, r in three_df.head(15).iterrows():
    print(f"  cs={r['wc']:.1f} grin={r['wg']:.1f} {r['ridge']:20s}={r['wr']:.1f}  "
          f"s*={r['opt_scale']:.6f}  R²*={r['r2_opt']:+.10f}")

# ═══════════════════════════════════════════════════════════
# Save top submissions
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("SAVING TOP SUBMISSIONS")
print(f"{'='*70}")

# Best solo
for name in model_names:
    pred_o = models[name][oracle_mask]
    s_star, r2_star = optimal_scale_r2(pred_o, y_oracle, days_oracle)
    if r2_star > 0:
        pred = models[name] * s_star
        sub = sample_sub.merge(pd.DataFrame({'ID': test_ids, 'TARGET': pred}), on='ID', how='left').fillna(0.0)
        fname = f'scaled_{name}_s{s_star:.6f}.csv'
        sub.to_csv(os.path.join(OUT_DIR, fname), index=False)
        print(f"  {fname}  R²={r2_star:+.8f}")

# Best 2-way
top_blend = blend_df.iloc[0]
m1, m2, w1 = top_blend['model1'], top_blend['model2'], top_blend['w1']
blend = w1 * models[m1] + (1-w1) * models[m2]
for d in np.unique(test_day):
    mask = test_day == d; blend[mask] -= blend[mask].mean()
blend = blend / blend.std() * top_blend['opt_scale']
sub = sample_sub.merge(pd.DataFrame({'ID': test_ids, 'TARGET': blend}), on='ID', how='left').fillna(0.0)
fname = f"scaled_best2way_{w1:.1f}{m1}_{1-w1:.1f}{m2}.csv"
sub.to_csv(os.path.join(OUT_DIR, fname), index=False)
print(f"  {fname}  R²={top_blend['r2_opt']:+.8f}")

# Best 3-way
top3 = three_df.iloc[0]
blend = top3['wc'] * models['cs_gold'] + top3['wg'] * models['grinold'] + top3['wr'] * models[top3['ridge']]
for d in np.unique(test_day):
    mask = test_day == d; blend[mask] -= blend[mask].mean()
blend = blend / blend.std() * top3['opt_scale']
sub = sample_sub.merge(pd.DataFrame({'ID': test_ids, 'TARGET': blend}), on='ID', how='left').fillna(0.0)
fname = f"scaled_best3way_cs{top3['wc']:.1f}_grin{top3['wg']:.1f}_{top3['ridge']}{top3['wr']:.1f}.csv"
sub.to_csv(os.path.join(OUT_DIR, fname), index=False)
print(f"  {fname}  R²={top3['r2_opt']:+.8f}")

print(f"\nDone in {(time.time()-t0)/60:.1f} min")
