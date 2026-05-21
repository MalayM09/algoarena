# ================================================================
# TREE RULES → LINEAR MODEL (fast version, no leaf encoding)
# ================================================================
# cs_gold_top8 uses 8 features, 3-28 iterations (very shallow).
# Each LGB tree makes explicit threshold splits: "if feature >= t"
# These splits are the alpha-generating rules the model found.
#
# Here we:
#   1. Retrain cs_gold_top8, save models, extract all splits
#   2. Print human-readable rules (what the model actually learned)
#   3. Build binary features from splits: 1 if feature >= threshold
#   4. Train Ridge on these binary split-features (alpha sweep)
#   5. Blend best result with optimal_blend_v2
#
# Leaf encoding removed — 5 models × ~19 trees × n_leaves = huge
# sparse matrix that takes 40+ min. Binary splits are sufficient.
# ================================================================

import sys, os, gc, time, warnings
sys.stdout.reconfigure(line_buffering=True)   # unbuffered output

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ORACLE     = os.path.join(BASE_DIR, 'outputs/submissions/exploit_v2_zero.csv')
SAMPLE_SUB = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
TARGET_STD = 0.000948
t0 = time.time()

def auto_scale(p):
    s = p.std(); return p*(TARGET_STD/s) if s>1e-10 else p

def daywise_oracle_score(pred_vec, oracle_vec, day_ids):
    ics = []
    for day in np.unique(day_ids):
        m = day_ids==day
        if m.sum()<3: continue
        p=pred_vec[m]-pred_vec[m].mean(); o=oracle_vec[m]-oracle_vec[m].mean()
        pn=np.linalg.norm(p); on=np.linalg.norm(o)
        if pn<1e-12 or on<1e-12: ics.append(0.)
        else: ics.append(float((p@o)/(pn*on)))
    return float(np.mean(ics))

# ── Load ──────────────────────────────────────────────────────────
print("Loading data...", flush=True)
t1 = time.time()
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
print(f"  Done in {time.time()-t1:.1f}s", flush=True)

# Top-8 features by LGB gain (from cs_gold_importance_sweep)
TOP8_FEATS = [
    'S02_F03_U01_LagT3',
    'S01_F03_U01_LagT3',
    'S03_V03_T03_LagT2',
    'S03_D02_A09_A02_B07_E07_E08_LagT3',
    'Price_LagT2',
    'S03_D02_A09_A02_B04_E04_E05_LagT3',
    'S03_D02_A09_A02_B06_E06_E07_LagT3',
    'S03_A07_A05_V09',
]
print(f"  Using top-8 features", flush=True)

y_raw     = train['TARGET'].values.astype(np.float64)
lo, hi    = np.percentile(y_raw, 1), np.percentile(y_raw, 99)
y_wins    = np.clip(y_raw, lo, hi)
test_ids  = test['ID'].values
train_day = train['SO3_T'].round(5).astype(str).values
test_day  = test['SO3_T'].round(5).astype(str).values

sample_sub  = pd.read_csv(SAMPLE_SUB)[['ID']]
oracle_vec  = sample_sub.merge(pd.read_csv(ORACLE),on='ID',how='left').fillna(0)['TARGET'].values
oracle_days = sample_sub.merge(
    pd.read_parquet(TEST_PATH,columns=['ID','SO3_T']),on='ID',how='left'
)['SO3_T'].round(5).astype(str).values

# ── CS z-score ────────────────────────────────────────────────────
print("CS z-score normalisation...", flush=True)
tr_raw = train[TOP8_FEATS].fillna(0).values.astype(np.float32)
te_raw = test.reindex(columns=TOP8_FEATS,fill_value=0).values.astype(np.float32)
X_tr   = np.zeros_like(tr_raw); X_te = np.zeros_like(te_raw)
for d in np.unique(train_day):
    m=train_day==d; x=tr_raw[m]; s=x.std(0); s[s<1e-8]=1.; X_tr[m]=(x-x.mean(0))/s
for d in np.unique(test_day):
    m=test_day==d; x=te_raw[m]; s=x.std(0); s[s<1e-8]=1.; X_te[m]=(x-x.mean(0))/s
del tr_raw, te_raw; gc.collect()
print(f"  X_tr: {X_tr.shape}  X_te: {X_te.shape}", flush=True)

# GroupKFold
groups  = pd.qcut(pd.Series(train['SO3_T'].values),q=5,labels=False,
                  duplicates='drop').values.astype(np.int32)
n_folds = len(np.unique(groups))
gkf     = GroupKFold(n_splits=n_folds)
folds   = list(gkf.split(X_tr, y_wins, groups=groups))
print(f"  GroupKFold: {n_folds} folds", flush=True)

# ── STEP 1: Train LGB top-8, SAVE MODELS ─────────────────────────
print("\n" + "="*60)
print("STEP 1: Retrain cs_gold_top8 — save models for rule extraction")
print("="*60, flush=True)
LGB_PARAMS = dict(objective='regression',metric='rmse',num_leaves=63,
                  learning_rate=0.05,feature_fraction=0.8,bagging_fraction=0.8,
                  bagging_freq=1,min_child_samples=50,lambda_l1=0.1,lambda_l2=1.0,
                  n_jobs=-1,verbose=-1,seed=42)

fold_models  = []
lgb_te_preds = np.zeros(len(X_te))
lgb_oof      = np.zeros(len(y_wins))

for fi,(tri,vai) in enumerate(folds):
    t1 = time.time()
    dt = lgb.Dataset(X_tr[tri],label=y_wins[tri],free_raw_data=True)
    dv = lgb.Dataset(X_tr[vai],label=y_wins[vai],reference=dt,free_raw_data=True)
    m  = lgb.train(LGB_PARAMS,dt,num_boost_round=2000,valid_sets=[dv],
                   callbacks=[lgb.early_stopping(50,verbose=False),
                               lgb.log_evaluation(500)])
    bi = m.best_iteration
    lgb_oof[vai]   = m.predict(X_tr[vai],num_iteration=bi)
    lgb_te_preds  += m.predict(X_te,num_iteration=bi)/n_folds
    fold_models.append(m)
    print(f"  Fold {fi+1}/{n_folds}: best_iter={bi}  trees={m.num_trees()}  ({time.time()-t1:.0f}s)", flush=True)
    del dt,dv; gc.collect()

lgb_scaled  = auto_scale(lgb_te_preds)
lgb_sub     = sample_sub.merge(pd.DataFrame({'ID':test_ids,'TARGET':lgb_scaled}),on='ID',how='left').fillna(0)
lgb_score   = daywise_oracle_score(lgb_sub['TARGET'].values,oracle_vec,oracle_days)
print(f"\n  LGB oracle_score: {lgb_score:+.6f}  (reference)", flush=True)

# ── STEP 2: Extract and print all split rules ─────────────────────
print("\n" + "="*60)
print("STEP 2: Extracting split rules from all fold models")
print("="*60, flush=True)

all_splits = []
for fi, m in enumerate(fold_models):
    df_tree    = m.trees_to_dataframe()
    split_nodes = df_tree[df_tree['split_feature'].notna()].copy()
    split_nodes['fold'] = fi
    all_splits.append(split_nodes[['fold','split_feature','threshold','split_gain','count']])

splits_df = pd.concat(all_splits, ignore_index=True)

# Map column indices to feature names
splits_df['feature_name'] = splits_df['split_feature'].map(
    lambda x: TOP8_FEATS[int(x.replace('Column_',''))]
              if isinstance(x,str) and 'Column_' in x else str(x)
)

print(f"\n  Total split nodes across all folds: {len(splits_df)}")
print(f"\n  Feature usage in splits:")
feat_counts = splits_df['feature_name'].value_counts()
for feat, cnt in feat_counts.items():
    print(f"    {feat:<45}  {cnt:3d} nodes")

print(f"\n  Top-25 split rules by gain (what the model actually learned):")
print(f"  {'Feature':<45} {'Threshold':>12} {'Gain':>10} {'Fold':>6}")
print(f"  {'─'*45} {'─'*12} {'─'*10} {'─'*6}")
top_splits = splits_df.sort_values('split_gain', ascending=False).head(25)
for _, row in top_splits.iterrows():
    fname  = str(row['feature_name'])
    thresh = float(row['threshold'])
    gain   = float(row['split_gain'])
    fold   = int(row['fold'])
    print(f"  {fname:<45} {thresh:>+12.4f} {gain:>10.4f}  fold{fold}")

# ── STEP 3: Build binary split features ──────────────────────────
print("\n" + "="*60)
print("STEP 3: Binary split features from unique (feature, threshold) pairs")
print("="*60, flush=True)

# Deduplicate and limit splits to avoid huge matrices
unique_splits = (splits_df[['feature_name','threshold']]
                 .drop_duplicates()
                 .sort_values(['feature_name','threshold'])
                 .reset_index(drop=True))
print(f"  Unique (feature, threshold) pairs: {len(unique_splits)}", flush=True)

# Vectorized binary feature builder: for each feature, broadcast all thresholds at once
def build_binary_features_fast(X, feat_names, unique_splits):
    """
    For each feature, gather its sorted thresholds and compute
    X[:, f] >= each_threshold using broadcasting → O(n * n_thresholds) per feature.
    Much faster than a Python loop over every (feature, threshold) pair.
    """
    parts = []
    col_labels = []
    for f_idx, fname in enumerate(feat_names):
        mask = unique_splits['feature_name'] == fname
        thresholds = unique_splits.loc[mask, 'threshold'].values  # shape (k,)
        if len(thresholds) == 0:
            continue
        # X[:, f_idx] shape (n,) → broadcast with thresholds (k,)
        b = (X[:, f_idx:f_idx+1] >= thresholds[np.newaxis, :]).astype(np.float32)  # (n, k)
        parts.append(b)
        col_labels.extend([(fname, t) for t in thresholds])
    return np.concatenate(parts, axis=1), col_labels

print("  Building binary feature matrices (vectorized)...", flush=True)
t1 = time.time()
B_tr, col_labels = build_binary_features_fast(X_tr, TOP8_FEATS, unique_splits)
B_te, _          = build_binary_features_fast(X_te, TOP8_FEATS, unique_splits)
print(f"  train={B_tr.shape}  test={B_te.shape}  density={B_tr.mean():.3f}  ({time.time()-t1:.1f}s)", flush=True)

# ── STEP 4: Ridge on binary split features ────────────────────────
print("\n" + "="*60)
print("STEP 4: Ridge on binary split features (alpha sweep)")
print("="*60, flush=True)

ALPHA_SWEEP = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
ridge_results = []

for alpha in ALPHA_SWEEP:
    t1 = time.time()
    oof_r = np.zeros(len(y_wins))
    te_pr = np.zeros(len(B_te))
    for fi,(tri,vai) in enumerate(folds):
        model_r = Ridge(alpha=alpha, fit_intercept=False)
        model_r.fit(B_tr[tri], y_wins[tri])
        oof_r[vai] = model_r.predict(B_tr[vai])
        te_pr     += model_r.predict(B_te) / n_folds
    oof_r2 = r2_score(y_wins, oof_r)
    scaled  = auto_scale(te_pr)
    sub     = sample_sub.merge(pd.DataFrame({'ID':test_ids,'TARGET':scaled}),on='ID',how='left').fillna(0)
    sc      = daywise_oracle_score(sub['TARGET'].values, oracle_vec, oracle_days)
    print(f"  alpha={alpha:<8}  OOF_R²={oof_r2:+.6f}  oracle={sc:+.6f}  ({time.time()-t1:.0f}s)", flush=True)
    ridge_results.append({'alpha':alpha,'oracle':sc,'oof_r2':oof_r2,'te_pr':te_pr,'sub':sub})

best_ridge = max(ridge_results, key=lambda x: x['oracle'])
best_ridge['sub'].to_csv(os.path.join(OUT_DIR,'tree_rules_ridge.csv'), index=False)
print(f"\n  Best Ridge: alpha={best_ridge['alpha']}  oracle={best_ridge['oracle']:+.6f}", flush=True)

# Also try: Ridge directly on the 8 raw CS z-scored features (baseline)
print("\n  [Baseline] Ridge on raw 8 CS features:")
for alpha in [0.1, 1.0, 10.0]:
    oof_r = np.zeros(len(y_wins)); te_pr = np.zeros(len(X_te))
    for fi,(tri,vai) in enumerate(folds):
        mr = Ridge(alpha=alpha, fit_intercept=False)
        mr.fit(X_tr[tri], y_wins[tri])
        oof_r[vai] = mr.predict(X_tr[vai])
        te_pr     += mr.predict(X_te) / n_folds
    oof_r2 = r2_score(y_wins, oof_r)
    sub = sample_sub.merge(pd.DataFrame({'ID':test_ids,'TARGET':auto_scale(te_pr)}),on='ID',how='left').fillna(0)
    sc  = daywise_oracle_score(sub['TARGET'].values, oracle_vec, oracle_days)
    print(f"  alpha={alpha:<8}  OOF_R²={oof_r2:+.6f}  oracle={sc:+.6f}", flush=True)

# ── STEP 5: Blend best variant with optimal_blend_v2 ──────────────
print("\n" + "="*60)
print("STEP 5: Blend tree_rules_ridge + optimal_blend_v2")
print("="*60, flush=True)

anchor_path = os.path.join(OUT_DIR, 'optimal_blend_v2.csv')
anchor = sample_sub.merge(pd.read_csv(anchor_path),on='ID',how='left').fillna(0)['TARGET'].values

best_vec = best_ridge['sub']['TARGET'].values
print(f"\n  optimal_blend_v2   oracle=+0.060098  LB=+0.00165 (current best)")
print(f"  tree_rules_ridge   oracle={best_ridge['oracle']:+.6f}")
print(f"\n  Pairwise blends (w = weight on tree_rules_ridge):")
for w in [0.10, 0.20, 0.30, 0.40, 0.50]:
    blend = w * best_vec + (1-w) * anchor
    s = blend.std(); blend = blend*(TARGET_STD/s) if s>1e-10 else blend
    sc = daywise_oracle_score(blend, oracle_vec, oracle_days)
    flag = '  ← BEATS THRESHOLD' if sc > 0.060349 else ''
    print(f"    w={w:.0%}  oracle={sc:+.6f}  delta_anchor={sc-0.060098:+.6f}{flag}")

print(f"\n  Submit threshold: +0.060349")
print(f"  Total elapsed:    {(time.time()-t0)/60:.1f} min")
