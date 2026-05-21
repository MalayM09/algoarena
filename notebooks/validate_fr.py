# ================================================================
# VALIDATE fullfeature_ridge on train.parquet
# Runs exact same logic as fullfeature_ridge.py
# Compares new fr_a5000_w15_ens85_validate.csv with saved original
# ================================================================

import os, time, warnings, sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR    = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH  = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH   = os.path.join(BASE_DIR, 'data/raw/test.parquet')
SAMPLE_PATH = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
BEST_ENS    = os.path.join(BASE_DIR, 'outputs/submissions/ens_tw35_hyb30_g35.csv')
ICIR_PATH   = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'outputs/submissions')
SAVED_FR    = os.path.join(OUT_DIR, 'fr_a5000_w15_ens85.csv')

TARGET_STD = 0.000948
CLIP_Z     = 5.0
ALPHA      = 5000
W_FR       = 0.15

print("Loading data...", flush=True)
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days = set(train['day_id'].unique())
y_train    = train['TARGET'].values.astype(np.float64)
n_test     = len(test)
test_ids   = test['ID'].values
sample_sub = pd.read_csv(SAMPLE_PATH)[['ID']]
print(f"  train={len(train):,}  test={n_test:,}", flush=True)

all_feat = [c for c in train.columns
            if c not in ['ID', 'TARGET', 'CV_GROUP', 'SO3_T', 'day_id']]

icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = ((icir_df['abs_icir'] >= 3) &
             ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0)))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
top10     = [f for f in gold_df['feature'].tolist()[:10] if f in train.columns]
ic_arr    = np.array([gold_df.set_index('feature')['mean_ic'].to_dict()[f] for f in top10])
print(f"  All features: {len(all_feat)}  Gold top-10: {len(top10)}", flush=True)

def zscore_fit(X, clip=CLIP_Z):
    m = X.mean(0); s = np.where(X.std(0) < 1e-8, 1.0, X.std(0))
    return np.clip((X - m) / s, -clip, clip), m, s

def zscore_apply(X, m, s, clip=CLIP_Z):
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip)

def winsorise(y, lo=1, hi=99):
    return np.clip(y, np.percentile(y, lo), np.percentile(y, hi))

def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d > 1e-12 else 0.0

# Grinold baseline (top-10)
print("\nGrinold baseline...", flush=True)
grinold_preds = np.zeros(n_test)
for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index.values
    X_te   = grp_te[top10].fillna(0).values.astype(np.float64)
    if day in train_days:
        grp_tr  = train[train['day_id'] == day]
        X_tr, m, s = zscore_fit(grp_tr[top10].fillna(0).values.astype(np.float64))
        X_te_z     = zscore_apply(X_te, m, s)
    else:
        X_te_z, _, _ = zscore_fit(X_te)
    pred = X_te_z @ ic_arr; pred -= pred.mean()
    grinold_preds[te_idx] = pred
print(f"  Done [{(time.time()-t0)/60:.1f}m]", flush=True)

# Full-feature Ridge alpha=5000
print(f"\nFull-feature Ridge alpha={ALPHA}...", flush=True)
ridge_preds = np.zeros(n_test)
days_with_train = 0
days_no_train   = 0

for day, grp_te in test.groupby('day_id'):
    te_idx   = grp_te.index.values
    X_te_raw = grp_te[all_feat].fillna(0).values.astype(np.float64)
    if day in train_days:
        grp_tr = train[train['day_id'] == day]
        if len(grp_tr) < 20:
            ridge_preds[te_idx] = grinold_preds[te_idx]; continue
        X_tr_raw = grp_tr[all_feat].fillna(0).values.astype(np.float64)
        y_tr     = winsorise(y_train[grp_tr.index])
        X_tr_z, m, s = zscore_fit(X_tr_raw)
        X_te_z        = zscore_apply(X_te_raw, m, s)
        mdl = Ridge(alpha=ALPHA, fit_intercept=True)
        mdl.fit(X_tr_z, y_tr)
        pred = mdl.predict(X_te_z); pred -= pred.mean()
        ridge_preds[te_idx] = pred
        days_with_train += 1
    else:
        ridge_preds[te_idx] = grinold_preds[te_idx]
        days_no_train += 1

print(f"  Overlap days: {days_with_train}  Future-only days: {days_no_train}", flush=True)
print(f"  [{(time.time()-t0)/60:.1f}m]", flush=True)

# Build blend: fr_a5000_w15_ens85
best_ens = (sample_sub.merge(pd.read_csv(BEST_ENS)[['ID','TARGET']]
                             .rename(columns={'TARGET':'b'}), on='ID', how='left')
            .fillna(0.0)['b'].values)
best_s    = auto_scale(best_ens)
ridge_s   = auto_scale(ridge_preds)

blend     = (1 - W_FR) * best_s + W_FR * ridge_s
blend_s   = auto_scale(blend)

sub_new = sample_sub.copy()
sub_new['TARGET'] = blend_s
out_path = os.path.join(OUT_DIR, 'fr_a5000_w15_ens85_validate.csv')
sub_new.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}", flush=True)

# ── COMPARE with saved CSV ──────────────────────────────────────
print(f"\n{'='*55}")
print("COMPARISON vs saved fr_a5000_w15_ens85.csv")
print(f"{'='*55}")
saved = pd.read_csv(SAVED_FR)
merged = sample_sub.merge(saved.rename(columns={'TARGET':'TARGET_saved'}), on='ID', how='left').fillna(0)
merged = merged.merge(sub_new.rename(columns={'TARGET':'TARGET_new'}), on='ID', how='left').fillna(0)

diff = (merged['TARGET_new'] - merged['TARGET_saved']).abs()
corr = merged['TARGET_new'].corr(merged['TARGET_saved'])
print(f"  Max abs diff  : {diff.max():.10f}")
print(f"  Mean abs diff : {diff.mean():.10f}")
print(f"  Correlation   : {corr:.8f}")
print(f"  Saved std     : {merged['TARGET_saved'].std():.6f}")
print(f"  New std       : {merged['TARGET_new'].std():.6f}")

if diff.max() < 1e-6:
    print(f"\n  PASS — predictions match within 1e-6")
else:
    print(f"\n  MISMATCH — max diff = {diff.max():.8f}")
    worst = merged.loc[diff.idxmax()]
    print(f"  Worst ID: {worst['ID']}  saved={worst['TARGET_saved']:.8f}  new={worst['TARGET_new']:.8f}")

print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f} min")
