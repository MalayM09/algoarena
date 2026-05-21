# ================================================================
# THREEWAY G15 — CORRECT REBUILD
# ================================================================
# Replicates perday_ridge.py methodology exactly, with Grinold
# top-10 → top-15 upgrade.
#
# Original threeway_r30_k40_g29 was built in perday_ridge.py as:
#   auto_scale(0.30*auto_scale(ridge) + 0.40*hybrid_kernel_raw + 0.29*auto_scale(grinold10))
# where hybrid_kernel_raw = hybrid_grinold_kernel.csv = 0.70*NW + 0.30*Grinold10
#
# This script:
#   1. Recomputes Ridge (same as perday_ridge.py: fit_intercept=True, alpha=10, top10)
#   2. Recomputes Grinold top10 (for extracting pure NW from hybrid)
#   3. Computes Grinold top15
#   4. Extracts pure NW: pure_nw = (hybrid_raw - 0.30*auto_scale(g10)) / 0.70
#   5. Builds hybrid_g15 = 0.70*pure_nw + 0.30*auto_scale(g15)
#   6. Builds threeway_g15 = auto_scale(0.30*ridge_scaled + 0.40*hybrid_g15 + 0.29*g15_scaled)
#
# Also saves a SANITY reconstruction of original threeway to confirm methodology.
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR    = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH  = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH   = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH   = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
SAMPLE_PATH = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'outputs/submissions')
HYBRID_PATH = os.path.join(OUT_DIR, 'hybrid_grinold_kernel.csv')

RIDGE_ALPHA = 10.0
TARGET_STD  = 0.000948

print("=" * 65)
print("THREEWAY G15 V2 — Correct Rebuild (perday_ridge methodology)")
print("=" * 65)

# ── Load data ──────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days = set(train['day_id'].unique())
new_days   = set(test['day_id'].unique()) - train_days
y_train    = train['TARGET'].values.astype(np.float64)
test_ids   = test['ID'].values
print(f"  Train: {len(train):,}  Test: {len(test):,}")
print(f"  Overlap: {len(train_days & set(test['day_id'].unique()))}  New: {len(new_days)}")

# ── Gold features ──────────────────────────────────────────────────
icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
all_gold  = [f for f in gold_df['feature'].tolist() if f in train.columns]
ic_map    = gold_df.set_index('feature')['mean_ic'].to_dict()

top10 = all_gold[:10]
top15 = all_gold[:15]
ic_arr10 = np.array([ic_map[f] for f in top10])
ic_arr15 = np.array([ic_map[f] for f in top15])
print(f"  Gold features — top10: {len(top10)}  top15: {len(top15)}")

# ── Helpers ────────────────────────────────────────────────────────
def zscore_fit(X, clip=5.0):
    m = X.mean(0); s = X.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip), m, s

def zscore_apply(X, m, s, clip=5.0):
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
    return (a @ b) / d if d > 1e-10 else 0.0

# ── Load hybrid_grinold_kernel ─────────────────────────────────────
sample_sub = pd.read_csv(SAMPLE_PATH)[['ID']]
hybrid_df  = pd.read_csv(HYBRID_PATH)
hybrid_df  = sample_sub.merge(
    hybrid_df[['ID','TARGET']].rename(columns={'TARGET': 'k_pred'}),
    on='ID', how='left'
).fillna(0.0)
hybrid_raw = hybrid_df['k_pred'].values
print(f"\n  Loaded hybrid_grinold_kernel.csv  std={hybrid_raw.std():.7f}")

# ── Generate test predictions ──────────────────────────────────────
print("\nGenerating test predictions (Ridge + Grinold10 + Grinold15)...")
n_test = len(test)
te_ridge  = np.zeros(n_test)
te_g10    = np.zeros(n_test)
te_g15    = np.zeros(n_test)

for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index
    X_te10 = grp_te[top10].fillna(0).values.astype(np.float64)
    X_te15 = grp_te[top15].fillna(0).values.astype(np.float64)

    # Grinold: z-score on test assets (same as perday_ridge.py and grinold_engine.py)
    Xz10, _, _  = zscore_fit(X_te10)
    pred_g10     = Xz10 @ ic_arr10; pred_g10 -= pred_g10.mean()
    te_g10[te_idx] = pred_g10

    Xz15, _, _  = zscore_fit(X_te15)
    pred_g15     = Xz15 @ ic_arr15; pred_g15 -= pred_g15.mean()
    te_g15[te_idx] = pred_g15

    if day in train_days:
        grp_tr  = train[train['day_id'] == day]
        y_tr    = y_train[grp_tr.index]
        y_tr_w  = winsorise(y_tr)
        X_tr10  = grp_tr[top10].fillna(0).values.astype(np.float64)

        # Ridge: z-score using train stats, fit_intercept=True (perday_ridge.py)
        X_tr_z, m_tr, s_tr = zscore_fit(X_tr10)
        X_te_z  = zscore_apply(X_te10, m_tr, s_tr)
        ridge   = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        ridge.fit(X_tr_z, y_tr_w)
        pred_r  = ridge.predict(X_te_z); pred_r -= pred_r.mean()
        te_ridge[te_idx] = pred_r
    else:
        # Future day: Grinold fallback
        te_ridge[te_idx] = pred_g10

elapsed = (time.time() - t0) / 60
print(f"  Done.  [{elapsed:.1f}m elapsed]")

# ── Component correlations ─────────────────────────────────────────
print(f"\n  Component correlations:")
print(f"  Ridge    vs G10:   {pearson_r(te_ridge, te_g10):+.4f}")
print(f"  Ridge    vs G15:   {pearson_r(te_ridge, te_g15):+.4f}")
print(f"  G10      vs G15:   {pearson_r(te_g10,   te_g15):+.4f}")
print(f"  hybrid   vs G10:   {pearson_r(hybrid_raw, te_g10):+.4f}")

# ── Reconstruct sanity check ───────────────────────────────────────
# Replicate original: auto_scale(0.30*ridge_s + 0.40*hybrid_raw + 0.29*g10_s)
ridge_s = auto_scale(te_ridge)
g10_s   = auto_scale(te_g10)
g15_s   = auto_scale(te_g15)

sanity = auto_scale(0.30 * ridge_s + 0.40 * hybrid_raw + 0.29 * g10_s)

# Check correlation with original
orig_path = os.path.join(OUT_DIR, 'threeway_r30_k40_g29.csv')
if os.path.exists(orig_path):
    orig_df = pd.read_csv(orig_path)
    orig_df = sample_sub.merge(
        orig_df[['ID','TARGET']].rename(columns={'TARGET':'op'}),
        on='ID', how='left'
    ).fillna(0.0)
    orig_preds = orig_df['op'].values
    print(f"\n  Sanity vs original corr: {pearson_r(sanity, orig_preds):+.4f}")
    print(f"  (Target: ~0.99+  |  Earlier cosine-KNN was ~0.848)")

# ── Extract pure NW from hybrid ────────────────────────────────────
# hybrid_raw = 0.70 * auto_scale(nw) + 0.30 * auto_scale(g10)
# → pure_nw ≈ (hybrid_raw - 0.30 * g10_s) / 0.70
pure_nw = (hybrid_raw - 0.30 * g10_s) / 0.70
print(f"\n  Pure NW extracted:  std={pure_nw.std():.7f}")
print(f"  NW vs Ridge corr:   {pearson_r(pure_nw, ridge_s):+.4f}")
print(f"  NW vs G10 corr:     {pearson_r(pure_nw, g10_s):+.4f}")

# ── Build upgraded hybrid with G15 ────────────────────────────────
hybrid_g15 = 0.70 * pure_nw + 0.30 * g15_s
print(f"\n  hybrid_g15 std:             {hybrid_g15.std():.7f}")
print(f"  hybrid_g15 vs hybrid_g10:   {pearson_r(hybrid_g15, hybrid_raw):+.4f}")

# ── Build all submission variants ─────────────────────────────────
print("\nBuilding submissions...")

def save_sub(preds, name):
    ps = auto_scale(preds)
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': ps})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    t = sub['TARGET']
    print(f"  {name:<55}  std={t.std():.6f}  mean={t.mean():+.7f}")

# 1. SANITY: reconstruction of original (should correlate ~1.0 with threeway_r30_k40_g29)
save_sub(0.30 * ridge_s + 0.40 * hybrid_raw + 0.29 * g10_s,
         'threeway_g10_v2_sanity')

# 2. PRIMARY: full g15 upgrade
#    Replace hybrid's G10 with G15 AND upgrade the direct G component
save_sub(0.30 * ridge_s + 0.40 * hybrid_g15 + 0.29 * g15_s,
         'threeway_g15_v2_full')

# 3. PARTIAL: only upgrade the direct G component (keep hybrid as-is)
#    Simpler, changes only 29% of predictions
save_sub(0.30 * ridge_s + 0.40 * hybrid_raw + 0.29 * g15_s,
         'threeway_g15_v2_direct_only')

# 4. Sanity with g30 weight (check if g29 vs g30 matters)
save_sub(0.30 * ridge_s + 0.40 * hybrid_raw + 0.30 * g10_s,
         'threeway_g10_v2_g30_sanity')

# 5. Full g15 with g30 weight
save_sub(0.30 * ridge_s + 0.40 * hybrid_g15 + 0.30 * g15_s,
         'threeway_g15_v2_full_g30')

print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f} min")

# ── Cross-correlation with original ───────────────────────────────
if os.path.exists(orig_path):
    variants = {
        'g10_v2_sanity':      auto_scale(0.30 * ridge_s + 0.40 * hybrid_raw + 0.29 * g10_s),
        'g15_v2_full':        auto_scale(0.30 * ridge_s + 0.40 * hybrid_g15 + 0.29 * g15_s),
        'g15_v2_direct_only': auto_scale(0.30 * ridge_s + 0.40 * hybrid_raw + 0.29 * g15_s),
        'g15_v2_full_g30':    auto_scale(0.30 * ridge_s + 0.40 * hybrid_g15 + 0.30 * g15_s),
    }
    print(f"\n  Correlation with original threeway_r30_k40_g29:")
    for name, preds in variants.items():
        corr = pearson_r(preds, orig_preds)
        print(f"    {name:<30}  {corr:+.4f}")

print(f"""
  ── SUBMISSION GUIDE ─────────────────────────────────────────────
  PRIMARY:   threeway_g15_v2_full.csv
             Full g15 upgrade: hybrid rebuilt with g15 + direct g15.
             Expected marginally above +0.00124.

  SANITY:    threeway_g10_v2_sanity.csv
             Should correlate ~0.99+ with original threeway_r30_k40_g29.
             Confirms the rebuild is correct. Score should be near +0.00124.

  PARTIAL:   threeway_g15_v2_direct_only.csv
             Only the direct 29% Grinold upgraded; hybrid stays as-is.
             Smaller change, safer bet, still an improvement.
""")
