# ================================================================
# CV LGBM — Leak-proof cross-sectional LightGBM
# ================================================================
# Design:
#   Features are z-scored WITHIN each day (using all day's assets
#   for normalization stats — same as production). Then pooled
#   across days. LightGBM learns one non-linear mapping across
#   all days. CV uses CV_GROUP to hold out ~20% of assets per fold.
#
# Motivation:
#   Grinold is a linear model. LightGBM can capture:
#     - Non-linear feature effects (e.g. saturation at extremes)
#     - Feature interactions (e.g. feature_A matters only if
#       feature_B is also high)
#     - Robustness to TARGET kurtosis=48 (via Huber/L1 objectives)
#
# CV structure:
#   70 CV_GROUPs → 5 superfolds
#   For each fold: train on ~80% of assets across all days,
#   validate on ~20% of assets across all days.
#   Per-day R² computed only on held-out assets.
#
# Experiments:
#   1. Feature set: top-5, top-10, top-20, top-51 gold
#   2. Objective: regression (MSE), huber (α=0.9), regression_l1 (MAE)
#   3. Tree depth: max_depth=3, 5, 7
#   4. Feature selection: unified top-K by in-fold IC (from prior run)
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/cv_results')
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS = 5
CLIP_Z  = 5.0

print("=" * 70)
print("CV LGBM — Leak-proof cross-sectional LightGBM")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
n_days = train['day_id'].nunique()
print(f"  Rows: {len(train):,} | Days: {n_days} | CV groups: {train['CV_GROUP'].nunique()}")

# ── Gold features ──────────────────────────────────────────────────────
icir_df  = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df  = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
all_gold = [f for f in gold_df['feature'].tolist() if f in train.columns]
print(f"  Gold features (ICIR≥3, never-flip): {len(all_gold)}")

# ── Assign superfolds ──────────────────────────────────────────────────
groups_sorted = sorted(train['CV_GROUP'].unique())
fold_map = {g: i % N_FOLDS for i, g in enumerate(groups_sorted)}
train['_fold'] = train['CV_GROUP'].map(fold_map)
fold_sizes = train['_fold'].value_counts().sort_index()
print(f"\n  {N_FOLDS}-fold CV:")
for f, n in fold_sizes.items():
    print(f"    fold {f}: {n:,} rows")

# ── Z-score per day (using ALL assets in day for stats) ───────────────
# This matches production where all assets (liquid+illiquid) are present.
# Compute once for all experiments.
print("\nZ-scoring features per day (all gold features, float32)...")
day_keys  = []
day_X_all = []   # shape: (n_assets, n_gold), z-scored, float32
day_y     = []
day_folds = []

def zscore_day(X, clip=CLIP_Z):
    m = X.mean(0, keepdims=True)
    s = X.std(0, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip).astype(np.float32)

for day, grp in train.groupby('day_id'):
    if len(grp) < 5: continue
    X_raw = grp[all_gold].fillna(0).values.astype(np.float32)
    day_keys.append(day)
    day_X_all.append(zscore_day(X_raw))
    day_y.append(grp['TARGET'].values.astype(np.float32))
    day_folds.append(grp['_fold'].values.astype(np.int8))

print(f"  Grouped {len(day_keys)} days  [{(time.time()-t0)/60:.1f}m]")

# ── Helper: build pooled train/val arrays for a fold ──────────────────
def build_fold_arrays(feat_indices, fold):
    """Returns (X_train, y_train, X_val, y_val, val_days_flat)."""
    tr_X, tr_y = [], []
    va_X, va_y, va_days = [], [], []
    for day_key, X_day, y_day, folds_day in zip(day_keys, day_X_all, day_y, day_folds):
        Xf = X_day[:, feat_indices]
        tr = folds_day != fold
        va = folds_day == fold
        if tr.sum() > 0:
            tr_X.append(Xf[tr]); tr_y.append(y_day[tr])
        if va.sum() > 0:
            va_X.append(Xf[va]); va_y.append(y_day[va])
            va_days.extend([day_key] * int(va.sum()))
    return (np.vstack(tr_X), np.concatenate(tr_y),
            np.vstack(va_X), np.concatenate(va_y), va_days)

# ── Per-day R² ────────────────────────────────────────────────────────
def compute_perday_r2(val_preds, val_targets, val_days_list):
    r2s = []
    df = pd.DataFrame({'p': val_preds, 't': val_targets, 'd': val_days_list})
    for _, grp in df.groupby('d'):
        p = grp['p'].values; t = grp['t'].values
        if len(p) < 3: continue
        p_dm = p - p.mean(); t_dm = t - t.mean()
        ss_tot = (t_dm ** 2).sum()
        if ss_tot < 1e-12: continue
        r2s.append(1.0 - ((p_dm - t_dm) ** 2).sum() / ss_tot)
    return (np.mean(r2s), np.std(r2s), len(r2s)) if r2s else (np.nan, np.nan, 0)

# ── Core CV function ───────────────────────────────────────────────────
def cv_lgbm(feat_indices, params, n_estimators=200, winsor_pct=1):
    """
    Leak-proof 5-fold CV for a LightGBM config.

    Args:
        feat_indices  : column indices into all_gold
        params        : LightGBM param dict (objective, depth, etc.)
        n_estimators  : number of trees (fixed; no early stopping needed)
        winsor_pct    : winsorize targets at (pct, 100-pct) before fitting

    Returns (mean_r2, std_r2, n_days)
    """
    all_val_preds   = []
    all_val_targets = []
    all_val_days    = []

    for fold in range(N_FOLDS):
        X_tr, y_tr, X_va, y_va, va_days = build_fold_arrays(feat_indices, fold)

        # Winsorize targets (handles kurtosis=48)
        lo = np.percentile(y_tr, winsor_pct)
        hi = np.percentile(y_tr, 100 - winsor_pct)
        y_tr_w = np.clip(y_tr, lo, hi).astype(np.float32)

        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            verbose=-1,
            n_jobs=4,
            **params
        )
        model.fit(X_tr, y_tr_w)

        pred = model.predict(X_va).astype(np.float32)
        # Mean-center per day (same as Grinold — we want cross-sectional signal)
        # Build day offsets for mean-centering
        day_counts = {}
        for d in va_days:
            day_counts[d] = day_counts.get(d, 0) + 1
        offset = 0
        for d in sorted(day_counts):
            n = day_counts[d]
            pred[offset:offset+n] -= pred[offset:offset+n].mean()
            offset += n

        all_val_preds.extend(pred.tolist())
        all_val_targets.extend(y_va.tolist())
        all_val_days.extend(va_days)

        del X_tr, y_tr, X_va, y_va

    return compute_perday_r2(
        np.array(all_val_preds, dtype=np.float32),
        np.array(all_val_targets, dtype=np.float32),
        all_val_days
    )

# ── Run experiments ────────────────────────────────────────────────────
results = []

def run(label, feat_indices, params, n_estimators=200, winsor_pct=1, extra=None):
    r2, std, nd = cv_lgbm(feat_indices, params, n_estimators=n_estimators,
                           winsor_pct=winsor_pct)
    row = {'config': label, 'r2': r2, 'std': std, 'n_days': nd,
           'n_feats': len(feat_indices), 'n_est': n_estimators,
           **params}
    if extra: row.update(extra)
    results.append(row)
    print(f"  {label:<52}  R²={r2:.5f}  ±{std:.5f}  days={nd}")
    return r2

# ── Baseline: reproduce Grinold R²=0.00229 for reference ──────────────
# (Grinold from prior run: R²=0.00229)
print("\n" + "=" * 70)
print("EXP 1 — Objective sweep  (top-10 gold, depth=5, n=200)")
print("=" * 70)
idx10 = list(range(10))

base_params = dict(objective='regression', max_depth=5, num_leaves=31,
                   learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                   min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0)
run("lgbm_top10_mse_d5",     idx10, base_params)

huber_params = {**base_params, 'objective': 'huber', 'alpha': 0.9}
run("lgbm_top10_huber09_d5", idx10, huber_params)

l1_params = {**base_params, 'objective': 'regression_l1'}
run("lgbm_top10_l1_d5",      idx10, l1_params)

huber95_params = {**base_params, 'objective': 'huber', 'alpha': 0.95}
run("lgbm_top10_huber95_d5", idx10, huber95_params)

print(f"  [{(time.time()-t0)/60:.1f}m elapsed]")

print("\n" + "=" * 70)
print("EXP 2 — Depth sweep  (top-10, best objective from EXP1)")
print("=" * 70)
# Pick the best objective from EXP1 (will use MSE as fallback)
best_obj_row = max(results, key=lambda r: r['r2'])
best_obj_params = {k: v for k, v in best_obj_row.items()
                   if k in ('objective', 'alpha', 'max_depth', 'num_leaves',
                             'learning_rate', 'subsample', 'colsample_bytree',
                             'min_child_samples', 'reg_alpha', 'reg_lambda')}

for depth, leaves in [(3, 7), (4, 15), (6, 63), (7, 127)]:
    p = {**best_obj_params, 'max_depth': depth, 'num_leaves': leaves}
    obj_short = p.get('objective', 'mse').replace('regression', 'mse').replace('_l1', 'l1')
    run(f"lgbm_top10_{obj_short}_d{depth}", idx10, p)

print(f"  [{(time.time()-t0)/60:.1f}m elapsed]")

print("\n" + "=" * 70)
print("EXP 3 — Feature count  (best depth + objective)")
print("=" * 70)
best_so_far = max(results, key=lambda r: r['r2'])
best_depth = best_so_far.get('max_depth', 5)
best_leaves = best_so_far.get('num_leaves', 31)
best_obj = best_so_far.get('objective', 'regression')
best_alpha = best_so_far.get('alpha', None)
best_params = {**base_params, 'objective': best_obj,
               'max_depth': best_depth, 'num_leaves': best_leaves}
if best_alpha:
    best_params['alpha'] = best_alpha

for k in [5, 15, 20, 30, 51]:
    idx = list(range(min(k, len(all_gold))))
    run(f"lgbm_top{k:02d}_best", idx, best_params)

print(f"  [{(time.time()-t0)/60:.1f}m elapsed]")

print("\n" + "=" * 70)
print("EXP 4 — n_estimators sweep  (top-10, best config)")
print("=" * 70)
for n_est in [50, 100, 300, 500]:
    run(f"lgbm_top10_best_n{n_est}", idx10, best_params, n_estimators=n_est)

print(f"  [{(time.time()-t0)/60:.1f}m elapsed]")

print("\n" + "=" * 70)
print("EXP 5 — Winsorization level  (top-10, best config, n=200)")
print("=" * 70)
for wp in [0, 2, 5]:
    run(f"lgbm_top10_best_winsor{wp}", idx10, best_params, winsor_pct=wp)

print(f"  [{(time.time()-t0)/60:.1f}m elapsed]")

# ── Save + summary ─────────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
out_path   = os.path.join(OUT_DIR, 'cv_lgbm.csv')
results_df.to_csv(out_path, index=False)

print("\n" + "=" * 70)
print("SUMMARY — All configs ranked by CV R²")
print("=" * 70)
print(results_df[['config', 'n_feats', 'objective', 'max_depth',
                   'n_est', 'r2', 'std']].to_string(index=False))

best  = results_df.iloc[0]
grinold_baseline = 0.00229
print(f"\n  Grinold baseline (top-10, Pearson, z-score) CV R²: {grinold_baseline:.5f}")
print(f"  Best LGBM config: {best['config']}")
print(f"  Best CV R²:       {best['r2']:.5f}  ({best['r2']-grinold_baseline:+.5f} vs Grinold)")
print(f"\n  Saved: {out_path}")
print(f"  Total: {(time.time()-t0)/60:.1f} min")
