# ================================================================
# FULL-FEATURE PER-DAY RIDGE — All 444 features, alpha sweep
# ================================================================
# Core hypothesis:
#   Our current per-day Ridge uses only top-10 gold features.
#   On each overlap day we observe ~1,900 liquid asset returns —
#   enough to estimate a richer factor structure from all 444 features.
#
#   With 444 features and 1,900 samples (ratio 4.3:1), we need
#   heavier regularization than alpha=10. We sweep alpha and blend
#   with our current best ensemble.
#
# Comparison baseline:
#   per_day_ridge_top10_alpha10  → contributes to threeway (+0.00124 LB)
#   per_day_ridge_all444_alpha?  → this script
#
# Generates:
#   fullridge_a100.csv   — all features, alpha=100
#   fullridge_a500.csv   — all features, alpha=500
#   fullridge_a1000.csv  — all features, alpha=1000
#   fullridge_a5000.csv  — all features, alpha=5000
#   fullridge_a500_g35_ens.csv  — best fullridge blended into best ensemble
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR    = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH  = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH   = os.path.join(BASE_DIR, 'data/raw/test.parquet')
SAMPLE_PATH = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
BEST_ENS    = os.path.join(BASE_DIR, 'outputs/submissions/ens_tw35_hyb30_g35.csv')
ICIR_PATH   = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'outputs/submissions')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_STD  = 0.000948
CLIP_Z      = 5.0
ALPHAS      = [100, 500, 1000, 5000]

print("=" * 65)
print("FULL-FEATURE PER-DAY RIDGE — Alpha Sweep")
print("=" * 65)

# ── Load data ──────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days      = set(train['day_id'].unique())
y_train         = train['TARGET'].values.astype(np.float64)
n_test          = len(test)
test_ids        = test['ID'].values
sample_sub      = pd.read_csv(SAMPLE_PATH)[['ID']]
print(f"  Train: {len(train):,}  Test: {n_test:,}")

# ── Feature sets ───────────────────────────────────────────────────
# All 444 features (everything except metadata columns)
all_feat = [c for c in train.columns
            if c not in ['ID', 'TARGET', 'CV_GROUP', 'SO3_T', 'day_id']]

# Gold top-10 for Grinold baseline (unchanged)
icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = ((icir_df['abs_icir'] >= 3) &
             ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0)))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
top10     = [f for f in gold_df['feature'].tolist()[:10] if f in train.columns]
ic_arr    = np.array([gold_df.set_index('feature')['mean_ic'].to_dict()[f] for f in top10])
print(f"  All features: {len(all_feat)}  Gold top-10: {len(top10)}")

# ── Helpers ────────────────────────────────────────────────────────
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

# ── Grinold baseline (top-10) ──────────────────────────────────────
print("\nComputing Grinold baseline (top-10 features)...")
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
    pred = X_te_z @ ic_arr
    pred -= pred.mean()
    grinold_preds[te_idx] = pred

print(f"  Grinold done  [{(time.time()-t0)/60:.1f}m]")

# ── Full-feature per-day Ridge — alpha sweep ───────────────────────
print(f"\nFull-feature Ridge sweep over alphas: {ALPHAS}...")
print(f"  Feature count: {len(all_feat)}  (vs 10 in current Ridge)")

ridge_preds = {a: np.zeros(n_test) for a in ALPHAS}
days_with_train = 0
days_no_train   = 0

for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index.values
    X_te_raw = grp_te[all_feat].fillna(0).values.astype(np.float64)

    if day in train_days:
        grp_tr  = train[train['day_id'] == day]
        if len(grp_tr) < 20:
            # Too few training samples — fall back to Grinold
            for a in ALPHAS:
                ridge_preds[a][te_idx] = grinold_preds[te_idx]
            continue

        X_tr_raw = grp_tr[all_feat].fillna(0).values.astype(np.float64)
        y_tr     = winsorise(y_train[grp_tr.index])

        # Z-score using TRAINING day stats (liquid population)
        X_tr_z, m, s = zscore_fit(X_tr_raw)
        X_te_z        = zscore_apply(X_te_raw, m, s)

        for a in ALPHAS:
            mdl = Ridge(alpha=a, fit_intercept=True)
            mdl.fit(X_tr_z, y_tr)
            pred = mdl.predict(X_te_z)
            pred -= pred.mean()
            ridge_preds[a][te_idx] = pred

        days_with_train += 1
    else:
        # No training data for this day — use Grinold fallback
        for a in ALPHAS:
            ridge_preds[a][te_idx] = grinold_preds[te_idx]
        days_no_train += 1

print(f"  Overlap days: {days_with_train}  Future-only days: {days_no_train}")
print(f"  [{(time.time()-t0)/60:.1f}m]")

# ── Load existing best ensemble for comparison ─────────────────────
best_ens = (sample_sub
            .merge(pd.read_csv(BEST_ENS)[['ID','TARGET']]
                   .rename(columns={'TARGET':'b'}), on='ID', how='left')
            .fillna(0.0)['b'].values)
best_s   = auto_scale(best_ens)

# ── Save and report ────────────────────────────────────────────────
print(f"\n{'Alpha':<8}  {'corr_vs_ens':>12}  {'corr_vs_grinold':>16}  File")
print("-" * 70)

saved = {}
for a in ALPHAS:
    ps = auto_scale(ridge_preds[a])
    gs = auto_scale(grinold_preds)
    corr_ens  = pearson_r(ps, best_s)
    corr_grin = pearson_r(ps, gs)
    name = f'fullridge_a{a}'
    sub  = pd.DataFrame({'ID': test_ids, 'TARGET': ps})
    sub  = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    saved[a] = ps
    print(f"  a={a:<6}  {corr_ens:+.4f}        {corr_grin:+.4f}           {name}.csv")

# ── Build blended submissions with best alpha ──────────────────────
# We don't know which alpha is best yet — build blends for all
print(f"\nBuilding ensemble blends (fullridge + best_ens)...")
g35_s = auto_scale(grinold_preds)

for a in ALPHAS:
    r_s = saved[a]
    # Replace Ridge component in our best ensemble:
    # ens_tw35_hyb30_g35 = 0.35*threeway + 0.30*hybrid + 0.35*grinold_probe
    # threeway = 0.30*ridge + 0.40*knn + 0.30*grinold (implicit)
    # Simple blend: replace the ridge inside threeway with fullridge
    # Approximate: blend 20% fullridge into best ensemble
    for w in [0.15, 0.20, 0.25]:
        blend = (1 - w) * best_s + w * r_s
        blend_s = auto_scale(blend)
        sub = pd.DataFrame({'ID': test_ids, 'TARGET': blend_s})
        sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
        name = f'fr_a{a}_w{int(w*100)}_ens{int((1-w)*100)}'
        sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
        c = pearson_r(blend_s, best_s)
        print(f"  {name:<40}  corr_vs_best={c:+.4f}")

print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
print("""
── INTERPRETATION ─────────────────────────────────────────────
corr_vs_ens:
  < 0.90: very different predictions — high risk/reward
  0.90-0.95: meaningfully different — worth submitting
  > 0.98: too similar — skip

SUBMISSION ORDER:
  1. fullridge_a500   (pure full-feature Ridge, standalone test)
  2. fullridge_a1000  (more regularized)
  3. fr_a500_w20_ens80 (20% fullridge + 80% current best — safe blend)
  4. fr_a1000_w20_ens80

KEY QUESTION the LB answers:
  Does full-feature Ridge score HIGHER than current Ridge (0.00086)?
  If yes by 2x+: the 50x gap to leader is partially explained by feature count.
  If no or marginal: the gap is likely from asset identification methods.
""")
