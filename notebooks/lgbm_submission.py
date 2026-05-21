# ================================================================
# LGBM SUBMISSION — Pooled cross-sectional LightGBM
# ================================================================
# Best CV config (from cv_lgbm.py):
#   objective=regression_l1 (MAE), max_depth=3, n_estimators=100
#   top-20 gold features, z-scored per day
#   CV R²=0.00543 vs Grinold baseline 0.00229 (+137%)
#
# Strategy:
#   1. Fit one LightGBM on ALL 660k training rows
#      (features z-scored within each training day)
#   2. Predict ALL test rows
#      (features z-scored within each test day, test assets only)
#   3. Mean-center predictions per day (cross-sectional signal)
#   4. Build ensemble variants with existing Grinold signal
#
# Note on z-scoring:
#   Training: z-score per day using ~1500 liquid assets
#   Test: z-score per day using ~800 illiquid test assets
#   This is the same distribution shift as the competition itself.
#   The cross-sectional z-score handles most of the non-stationarity.
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR    = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH  = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH   = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH   = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
SAMPLE_PATH = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'outputs/submissions')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_STD = 0.000948
CLIP_Z     = 5.0

# Best CV config
N_FEATS    = 20    # top-20 gold (CV R²=0.00543)
N_EST      = 100   # n_estimators=100 (peak n from CV)
MAX_DEPTH  = 3
NUM_LEAVES = 7     # 2^(depth-1) - 1 = 7 for depth=3

print("=" * 65)
print("LGBM SUBMISSION — Cross-sectional MAE LightGBM")
print("=" * 65)

# ── Load data ──────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
y_train    = train['TARGET'].values.astype(np.float64)
test_ids   = test['ID'].values
train_days = set(train['day_id'].unique())
new_days   = set(test['day_id'].unique()) - train_days
print(f"  Train: {len(train):,}  Test: {len(test):,}")
print(f"  Train days: {len(train_days)}  Overlap: {len(train_days & set(test['day_id']))}  New: {len(new_days)}")

# ── Gold features ──────────────────────────────────────────────────
icir_df  = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df  = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
all_gold = [f for f in gold_df['feature'].tolist() if f in train.columns]
feats    = all_gold[:N_FEATS]
print(f"  Gold features total: {len(all_gold)}  Using top-{N_FEATS}: {len(feats)}")

# ── Helper: z-score per day ────────────────────────────────────────
def zscore_day(X, clip=CLIP_Z):
    m = X.mean(0, keepdims=True)
    s = X.std(0, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip).astype(np.float32)

def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return (a @ b) / d if d > 1e-10 else 0.0

# ── Build training matrix (z-scored per day, pooled) ──────────────
print("\nBuilding training matrix (z-scored per day)...")
train_X_rows = []
train_y_rows = []
for day, grp in train.groupby('day_id'):
    if len(grp) < 5: continue
    X_raw = grp[feats].fillna(0).values.astype(np.float32)
    X_z   = zscore_day(X_raw)
    train_X_rows.append(X_z)
    train_y_rows.append(y_train[grp.index].astype(np.float32))

X_train = np.vstack(train_X_rows).astype(np.float32)
y_tr    = np.concatenate(train_y_rows).astype(np.float32)
print(f"  X_train: {X_train.shape}  y_train: {y_tr.shape}")
print(f"  [{(time.time()-t0)/60:.1f}m]")

# ── Fit LGBM on ALL training data ─────────────────────────────────
print(f"\nFitting LightGBM (MAE, depth={MAX_DEPTH}, n={N_EST})...")
model = lgb.LGBMRegressor(
    objective    = 'regression_l1',
    max_depth    = MAX_DEPTH,
    num_leaves   = NUM_LEAVES,
    n_estimators = N_EST,
    learning_rate= 0.05,
    subsample    = 0.8,
    colsample_bytree = 0.8,
    min_child_samples = 20,
    reg_alpha    = 0.1,
    reg_lambda   = 1.0,
    verbose      = -1,
    n_jobs       = 4
)
model.fit(X_train, y_tr)
print(f"  Fit complete.  [{(time.time()-t0)/60:.1f}m]")

# Feature importance
fi = pd.DataFrame({'feature': feats, 'importance': model.feature_importances_})
fi = fi.sort_values('importance', ascending=False)
print(f"\n  Feature importances (top-10):")
for _, row in fi.head(10).iterrows():
    print(f"    {row['feature']:<55}  {int(row['importance'])}")

# ── Also fit Grinold (top-10, Pearson IC) for ensemble ────────────
print("\nComputing Grinold IC (top-10, for ensemble)...")
g_feats = all_gold[:10]
ic_map  = gold_df.set_index('feature')['mean_ic'].to_dict()
ic_arr  = np.array([ic_map[f] for f in g_feats])
# Note: this uses full-dataset ICs (ic_icir_full.csv) — valid for
# final submission since we're fitting on ALL training data
# (same data used to compute ICs). For CV we computed in-fold.

# ── Generate test predictions ──────────────────────────────────────
print("\nGenerating test predictions (per day)...")
n_test     = len(test)
te_lgbm    = np.zeros(n_test, dtype=np.float32)
te_grinold = np.zeros(n_test, dtype=np.float32)

for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index.values

    # LGBM: z-score test assets within this day
    X_te = grp_te[feats].fillna(0).values.astype(np.float32)
    X_te_z = zscore_day(X_te)
    pred_l = model.predict(X_te_z).astype(np.float32)
    pred_l -= pred_l.mean()
    te_lgbm[te_idx] = pred_l

    # Grinold: z-score test assets + dot with IC
    X_g = grp_te[g_feats].fillna(0).values.astype(np.float32)
    X_g_z = zscore_day(X_g)
    pred_g = (X_g_z @ ic_arr).astype(np.float32)
    pred_g -= pred_g.mean()
    te_grinold[te_idx] = pred_g

elapsed = (time.time() - t0) / 60
print(f"  Done.  [{elapsed:.1f}m]")

# ── Component correlations ─────────────────────────────────────────
print(f"\n  LGBM vs Grinold corr:  {pearson_r(te_lgbm, te_grinold):+.4f}")
print(f"  LGBM std:    {te_lgbm.std():.7f}")
print(f"  Grinold std: {te_grinold.std():.7f}")

# ── Build submissions ──────────────────────────────────────────────
print("\nBuilding submissions...")
sample_sub = pd.read_csv(SAMPLE_PATH)[['ID']]

def save_sub(preds, name):
    ps = auto_scale(preds)
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': ps})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    t = sub['TARGET']
    print(f"  {name:<55}  std={t.std():.6f}  mean={t.mean():+.7f}")

l_s = auto_scale(te_lgbm)
g_s = auto_scale(te_grinold)

# 1. Pure LGBM (primary)
save_sub(te_lgbm,
         'lgbm_top20_l1_d3_n100')

# 2. LGBM + Grinold ensemble (50/50)
save_sub(0.70 * l_s + 0.30 * g_s,
         'lgbm70_grinold30')

# 3. LGBM + Grinold (80/20) — heavier LGBM weight
save_sub(0.80 * l_s + 0.20 * g_s,
         'lgbm80_grinold20')

# 4. Load existing threeway and blend with LGBM
threeway_path = os.path.join(OUT_DIR, 'threeway_g15_v2_full.csv')
if os.path.exists(threeway_path):
    tw_df = pd.read_csv(threeway_path)
    tw_df = sample_sub.merge(
        tw_df[['ID','TARGET']].rename(columns={'TARGET': 'tw'}),
        on='ID', how='left'
    ).fillna(0.0)
    tw_preds = tw_df['tw'].values.astype(np.float32)
    tw_s = auto_scale(tw_preds)

    print(f"\n  Loaded threeway_g15_v2_full  std={tw_preds.std():.7f}")
    print(f"  LGBM vs Threeway corr:  {pearson_r(l_s, tw_s):+.4f}")

    save_sub(0.50 * l_s + 0.50 * tw_s,
             'lgbm50_threeway50')
    save_sub(0.60 * l_s + 0.40 * tw_s,
             'lgbm60_threeway40')
    save_sub(0.70 * l_s + 0.30 * tw_s,
             'lgbm70_threeway30')

print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f} min")
print(f"""
  ── SUBMISSION GUIDE ─────────────────────────────────────────
  PRIMARY:   lgbm_top20_l1_d3_n100.csv
             Pure LGBM (MAE, depth=3, top-20 gold features)
             CV R²=0.00543  (+137% vs Grinold baseline 0.00229)

  ENSEMBLE:  lgbm70_threeway30.csv
             Blends LGBM with current best threeway (+0.00122 LB)
             Safer bet — keeps some threeway signal.

  AGGRESSIVE: lgbm80_grinold20.csv
             Mostly LGBM with small Grinold stabilizer.

  RECOMMENDED ORDER:
    1. lgbm50_threeway50  — safe blend, should beat threeway
    2. lgbm_top20_l1_d3_n100 — pure LGBM, highest CV score
    3. lgbm70_threeway30  — intermediate blend
""")
