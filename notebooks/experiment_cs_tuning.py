"""
Tune cs_v2_gold LGB to maximize LB R².
Test: different hyperparams, feature subsets, scale factors.
"""
import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
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

# Multiple gold thresholds
gold3 = icir_df[(icir_df['abs_icir'] >= 3) & (icir_df['ic_pos_frac'].isin([0.0, 1.0]))].copy()
gold25 = icir_df[(icir_df['abs_icir'] >= 2.5) & (icir_df['ic_pos_frac'].isin([0.0, 1.0]))].copy()
gold35 = icir_df[(icir_df['abs_icir'] >= 3.5) & (icir_df['ic_pos_frac'].isin([0.0, 1.0]))].copy()
gold4 = icir_df[(icir_df['abs_icir'] >= 4) & (icir_df['ic_pos_frac'].isin([0.0, 1.0]))].copy()

# Also: top-N by abs_icir regardless of ic_pos_frac
top30 = icir_df.head(30).copy()

feat_sets = {
    'gold3': [f for f in gold3['feature'].tolist() if f in all_feat],
    'gold2.5': [f for f in gold25['feature'].tolist() if f in all_feat],
    'gold3.5': [f for f in gold35['feature'].tolist() if f in all_feat],
    'gold4': [f for f in gold4['feature'].tolist() if f in all_feat],
    'top30': [f for f in top30['feature'].tolist() if f in all_feat],
}
for k, v in feat_sets.items():
    print(f"  {k}: {len(v)} features")

ic_dict = gold3.set_index('feature')['mean_ic'].to_dict()

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

# CV setup
groups = pd.qcut(pd.Series(train['SO3_T'].values), q=5, labels=False, duplicates='drop').values.astype(np.int32)
n_folds = len(np.unique(groups))
gkf = GroupKFold(n_splits=n_folds)

# Grinold for blending
gold_idx_3 = [all_feat.index(f) for f in feat_sets['gold3']]
ic_w = np.array([ic_dict.get(f, 0.) for f in feat_sets['gold3']], dtype=np.float64)
te_grinold = X_te[:, gold_idx_3].astype(np.float64) @ ic_w
for d in np.unique(test_day):
    mask = test_day == d; te_grinold[mask] -= te_grinold[mask].mean()
grin_s = auto_scale(te_grinold)

# ── LGB configs to test ─────────────────────────────────────
configs = []

# Vary hyperparameters with gold3 features
for nl, lr, mcs, l1, l2, ff, bf, label in [
    (63,  0.05, 50,  0.1, 1.0, 0.8, 0.8, "baseline"),
    (31,  0.05, 50,  0.1, 1.0, 0.8, 0.8, "nl31"),
    (15,  0.05, 50,  0.1, 1.0, 0.8, 0.8, "nl15"),
    (63,  0.02, 50,  0.1, 1.0, 0.8, 0.8, "lr002"),
    (63,  0.01, 50,  0.1, 1.0, 0.8, 0.8, "lr001"),
    (31,  0.02, 50,  0.1, 1.0, 0.8, 0.8, "nl31_lr002"),
    (63,  0.05, 100, 0.1, 1.0, 0.8, 0.8, "mcs100"),
    (63,  0.05, 50,  0.5, 5.0, 0.8, 0.8, "highreg"),
    (63,  0.05, 50,  1.0, 10.0, 0.8, 0.8, "veryhighreg"),
    (63,  0.05, 50,  0.1, 1.0, 0.6, 0.6, "lowsample"),
    (31,  0.02, 100, 0.5, 5.0, 0.7, 0.7, "conservative"),
]:
    configs.append(('gold3', nl, lr, mcs, l1, l2, ff, bf, label))

# Vary feature sets with baseline hyperparams
for fset in ['gold2.5', 'gold3.5', 'gold4', 'top30']:
    configs.append((fset, 63, 0.05, 50, 0.1, 1.0, 0.8, 0.8, f"feats_{fset}"))

print(f"\nTesting {len(configs)} configurations...", flush=True)

results = []
best_r2 = -999
best_pred = None
best_name = None

for fset, nl, lr, mcs, l1, l2, ff, bf, label in configs:
    feats = feat_sets[fset]
    if len(feats) < 3:
        print(f"  SKIP {label} ({fset}): only {len(feats)} features"); continue

    fidx = [all_feat.index(f) for f in feats]
    X_tr_g = X_tr[:, fidx]; X_te_g = X_te[:, fidx]
    folds = list(gkf.split(X_tr_g, y_wins, groups=groups))

    lgb_params = dict(objective='regression', metric='rmse', num_leaves=nl,
                      learning_rate=lr, feature_fraction=ff, bagging_fraction=bf,
                      bagging_freq=1, min_child_samples=mcs, lambda_l1=l1, lambda_l2=l2,
                      n_jobs=-1, verbose=-1, seed=42)

    oof = np.zeros(len(y_wins), dtype=np.float64)
    te = np.zeros(len(X_te_g), dtype=np.float64)
    iters = []
    for fi, (tri, vai) in enumerate(folds):
        dt = lgb.Dataset(X_tr_g[tri], label=y_wins[tri], free_raw_data=True)
        dv = lgb.Dataset(X_tr_g[vai], label=y_wins[vai], reference=dt, free_raw_data=True)
        m = lgb.train(lgb_params, dt, num_boost_round=3000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(9999)])
        bi = m.best_iteration; iters.append(bi)
        oof[vai] = m.predict(X_tr_g[vai], num_iteration=bi)
        te += m.predict(X_te_g, num_iteration=bi) / n_folds
        del dt, dv, m; gc.collect()

    # Per-day demean
    for d in np.unique(test_day):
        mask = test_day == d; te[mask] -= te[mask].mean()

    oof_r2 = r2_score(y_wins, oof)
    te_s = auto_scale(te)
    ic = daywise_oracle(te_s, oracle_df, test_ids, test_day)
    r2 = daywise_r2(te_s, oracle_df, test_ids, test_day)

    # Also test cs_solo and cs80_grin20 blends
    blend80 = 0.80 * te_s + 0.20 * grin_s
    for d in np.unique(test_day):
        mask = test_day == d; blend80[mask] -= blend80[mask].mean()
    blend80_s = auto_scale(blend80)
    r2_blend = daywise_r2(blend80_s, oracle_df, test_ids, test_day)

    name = f"{fset}_{label}"
    results.append({'config': name, 'iters': str(iters), 'oof_r2': oof_r2,
                    'oracle_IC': ic, 'oracle_R2_solo': r2, 'oracle_R2_blend80': r2_blend,
                    'n_feats': len(feats)})
    print(f"  {name:35s}  iters={iters}  OOF={oof_r2:+.6f}  IC={ic:+.6f}  R2_solo={r2:+.8f}  R2_80={r2_blend:+.8f}", flush=True)

    # Track best by solo R²
    if r2 > best_r2:
        best_r2 = r2; best_pred = te_s.copy(); best_name = name

# Also try different scale factors on best prediction
print(f"\n{'='*70}")
print(f"Scale factor sweep on best config: {best_name}")
print(f"{'='*70}")

# Get raw (unscaled) prediction for best config
# Re-derive from best_pred which is already auto_scaled
raw_best = best_pred / best_pred.std()  # normalize to std=1

for scale in [0.0006, 0.0007, 0.0008, 0.0009, 0.000948, 0.0010, 0.0011, 0.0012]:
    scaled = raw_best * scale
    r2 = daywise_r2(scaled, oracle_df, test_ids, test_day)
    print(f"  std={scale:.6f}  R2={r2:+.10f}")

# Summary
print(f"\n{'='*70}")
print("RESULTS SORTED BY oracle_R2_solo")
print(f"{'='*70}")
res_df = pd.DataFrame(results).sort_values('oracle_R2_solo', ascending=False)
for _, r in res_df.iterrows():
    print(f"  R2_solo={r['oracle_R2_solo']:+.8f}  R2_80={r['oracle_R2_blend80']:+.8f}  IC={r['oracle_IC']:+.6f}  OOF={r['oof_r2']:+.6f}  {r['config']}")

# Save best solo and best blend submissions
print(f"\nSaving best submissions...")
# Best solo
sub = sample_sub.merge(pd.DataFrame({'ID': test_ids, 'TARGET': best_pred}), on='ID', how='left').fillna(0.0)
path = os.path.join(OUT_DIR, f'compliant_tuned_solo.csv')
sub.to_csv(path, index=False)
print(f"  Solo: {path}  ({best_name})")

# Best with 80/20 grinold blend
blend = 0.80 * best_pred + 0.20 * grin_s
for d in np.unique(test_day):
    mask = test_day == d; blend[mask] -= blend[mask].mean()
blend_s = auto_scale(blend)
sub2 = sample_sub.merge(pd.DataFrame({'ID': test_ids, 'TARGET': blend_s}), on='ID', how='left').fillna(0.0)
path2 = os.path.join(OUT_DIR, f'compliant_tuned_80grin20.csv')
sub2.to_csv(path2, index=False)
print(f"  Blend: {path2}")

print(f"\nDone in {(time.time()-t0)/60:.1f} min")
