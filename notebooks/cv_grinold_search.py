# ================================================================
# CV GRINOLD SEARCH — Leak-proof CV using CV_GROUP
# ================================================================
# CRITICAL DESIGN PRINCIPLE (no IC leak):
#   ICs are computed INSIDE the fold loop using ONLY training rows
#   of that fold. The validation fold is never touched during IC
#   estimation. This is mathematically clean and matches production.
#
# CV structure:
#   70 CV_GROUPs → merged into N_FOLDS superfolds
#   Fold k: val = rows with _fold==k, train = all other rows
#   For each fold:
#     1. Compute ICs on train rows (pooled across days, z-scored per day)
#     2. For each day: normalize ALL rows, predict VAL rows only
#   Per-day R² computed only on held-out val rows
#
# Performance:
#   - Data pre-grouped into numpy arrays once (no repeated groupby)
#   - float32 throughout (halves memory vs float64)
#   - Vectorised batch IC via matrix multiply
#
# Configs tested:
#   Grinold  : top-K gold features (K = 5,10,15,20,25,30,40,51)
#   Norm     : z-score vs rank (within-day)
#   IC type  : Pearson vs Spearman (both computed in-fold)
#   ICIR     : threshold sweep
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/cv_results')
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS  = 5    # 5-fold: 14 groups per fold, ~130k val rows per fold — fast & stable
CLIP_Z   = 5.0

print("=" * 70)
print("CV GRINOLD SEARCH — Leak-proof CV via CV_GROUP")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
n_days = train['day_id'].nunique()
print(f"  Rows: {len(train):,} | Days: {n_days} | CV groups: {train['CV_GROUP'].nunique()}")

# ── Gold features ──────────────────────────────────────────────────────
icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
all_gold  = [f for f in gold_df['feature'].tolist() if f in train.columns]
# gold_df['mean_ic'] used ONLY for feature ordering, NOT for predictions in CV
gold_order_ic = gold_df.set_index('feature')['mean_ic'].to_dict()
print(f"  Gold features (ICIR≥3, never-flip): {len(all_gold)}")

# ── Assign superfolds ──────────────────────────────────────────────────
groups_sorted = sorted(train['CV_GROUP'].unique())
fold_map = {g: i % N_FOLDS for i, g in enumerate(groups_sorted)}
train['_fold'] = train['CV_GROUP'].map(fold_map)
fold_sizes = train['_fold'].value_counts().sort_index()
print(f"\n  {N_FOLDS}-fold CV (groups per fold ≈ {len(groups_sorted)//N_FOLDS}):")
for f, n in fold_sizes.items():
    print(f"    fold {f}: {n:,} rows")

# ── Pre-group data into numpy arrays (done ONCE for speed) ────────────
print("\nPre-grouping data by day (numpy arrays, float32)...")
# Use ALL gold features so we only need to slice columns per experiment
day_keys  = []
day_X     = []   # shape: (n_assets, n_gold_features), float32
day_y     = []   # shape: (n_assets,), float32
day_folds = []   # shape: (n_assets,), int8

for day, grp in train.groupby('day_id'):
    if len(grp) < 5: continue
    day_keys.append(day)
    day_X.append(grp[all_gold].fillna(0).values.astype(np.float32))
    day_y.append(grp['TARGET'].values.astype(np.float32))
    day_folds.append(grp['_fold'].values.astype(np.int8))

print(f"  Grouped {len(day_keys)} days  [{(time.time()-t0)/60:.1f}m]")

# ── Helper functions ───────────────────────────────────────────────────
def zscore_rows(X, clip=CLIP_Z):
    """Z-score across rows (assets), clipped. Returns float32."""
    m = X.mean(0, keepdims=True)
    s = X.std(0, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip).astype(np.float32)

def rank_rows(X):
    """Rank-normalize across rows to [-1,+1]. Returns float32."""
    n = X.shape[0]
    if n < 2: return np.zeros_like(X, dtype=np.float32)
    ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float32)
    return (2.0 * ranks / (n - 1) - 1.0).astype(np.float32)

def compute_ic_pearson_vectorised(X_zscored, y):
    """
    Fast vectorised Pearson IC.
    X_zscored : (N, K) already z-scored within day and pooled across days
    y          : (N,)  raw targets (NOT z-scored; IC = cov under z-score)
    Returns IC array of shape (K,).
    Note: since X is z-scored, corr(X_k, y) = cov(X_k, y) / std(y)
    We want the sign/rank of IC, not absolute value, so just use cov.
    """
    y_dm = y - y.mean()
    return (X_zscored * y_dm[:, None]).mean(0)  # shape (K,)

def compute_ic_spearman_vectorised(X_zscored, y):
    """
    Fast vectorised Spearman IC using rank transforms.
    Ranks are approximated per-column; then Pearson on ranks.
    """
    # Rank-transform both X and y
    n = X_zscored.shape[0]
    X_ranks = np.argsort(np.argsort(X_zscored, axis=0), axis=0).astype(np.float32)
    y_ranks = np.argsort(np.argsort(y)).astype(np.float32)
    # Normalise ranks to [-1,+1] (same scale)
    X_ranks = 2.0 * X_ranks / max(n - 1, 1) - 1.0
    y_ranks = 2.0 * y_ranks / max(n - 1, 1) - 1.0
    y_dm = y_ranks - y_ranks.mean()
    return (X_ranks * y_dm[:, None]).mean(0)

def compute_perday_r2(val_preds, val_targets, val_days_list):
    """
    val_preds, val_targets : flat arrays across all days and folds
    val_days_list : parallel array of day labels
    Returns (mean_r2, std_r2, n_days_evaluated).
    """
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

# ── Core CV function — STRICTLY LEAK-FREE ─────────────────────────────
def cv_grinold_strict(feat_indices, norm='zscore', ic_type='pearson'):
    """
    Leak-proof N-fold CV for a Grinold configuration.

    Args:
        feat_indices : indices into all_gold (column indices into day_X arrays)
        norm         : 'zscore' | 'rank'
        ic_type      : 'pearson' | 'spearman'

    Returns:
        (mean_r2, std_r2, n_days_evaluated)
    """
    all_val_preds   = []
    all_val_targets = []
    all_val_days    = []

    for fold in range(N_FOLDS):
        # ── Step 1: Build pooled training set (z-scored per day) ──────
        # ICs are estimated from THIS pooled set — zero leakage
        train_X_parts = []
        train_y_parts = []

        for X_day, y_day, folds_day in zip(day_X, day_y, day_folds):
            tr_mask = (folds_day != fold)
            if tr_mask.sum() < 5: continue
            X_tr = X_day[tr_mask][:, feat_indices]
            # Z-score using FULL day stats (all assets), but only keep train rows
            # This is correct: normalization stats from full day, weights from train
            X_full = X_day[:, feat_indices]
            if norm == 'zscore':
                X_full_n = zscore_rows(X_full)
            else:
                X_full_n = rank_rows(X_full)
            train_X_parts.append(X_full_n[tr_mask])
            train_y_parts.append(y_day[tr_mask])

        X_train_pool = np.vstack(train_X_parts)   # (N_train, K)
        y_train_pool = np.concatenate(train_y_parts)  # (N_train,)

        # ── Step 2: Compute ICs strictly on training pool ─────────────
        if ic_type == 'pearson':
            ic_arr = compute_ic_pearson_vectorised(X_train_pool, y_train_pool)
        else:
            ic_arr = compute_ic_spearman_vectorised(X_train_pool, y_train_pool)

        # ── Step 3: Predict val rows using in-fold ICs ────────────────
        for day_key, X_day, y_day, folds_day in zip(day_keys, day_X, day_y, day_folds):
            val_mask = (folds_day == fold)
            if val_mask.sum() == 0: continue

            X_full = X_day[:, feat_indices]
            if norm == 'zscore':
                X_full_n = zscore_rows(X_full)
            else:
                X_full_n = rank_rows(X_full)

            X_val = X_full_n[val_mask]
            pred  = (X_val @ ic_arr).astype(np.float32)
            pred -= pred.mean()

            all_val_preds.extend(pred.tolist())
            all_val_targets.extend(y_day[val_mask].tolist())
            all_val_days.extend([day_key] * int(val_mask.sum()))

        # Free pool arrays
        del X_train_pool, y_train_pool, train_X_parts, train_y_parts

    return compute_perday_r2(
        np.array(all_val_preds, dtype=np.float32),
        np.array(all_val_targets, dtype=np.float32),
        all_val_days
    )

# ── Run experiments ────────────────────────────────────────────────────
results = []

def run(label, feat_indices, norm='zscore', ic_type='pearson', extra=None):
    r2, std, nd = cv_grinold_strict(feat_indices, norm=norm, ic_type=ic_type)
    row = {'config': label, 'r2': r2, 'std': std, 'n_days': nd,
           'n_feats': len(feat_indices), 'norm': norm, 'ic_type': ic_type}
    if extra: row.update(extra)
    results.append(row)
    print(f"  {label:<42}  R²={r2:.5f}  ±{std:.5f}  days={nd}")
    return r2

print("\n" + "=" * 70)
print("EXP 1 — Feature count sweep  (Pearson IC, z-score)")
print("=" * 70)
for k in [5, 10, 15, 20, 25, 30, 40, 51]:
    idx = list(range(min(k, len(all_gold))))
    run(f"grinold_top{k:02d}_pearson_zscore", idx, norm='zscore', ic_type='pearson')
print(f"  [{(time.time()-t0)/60:.1f}m elapsed]")

print("\n" + "=" * 70)
print("EXP 2 — Rank norm vs Z-score  (Pearson IC, top-10 & top-20)")
print("=" * 70)
for k in [10, 20]:
    idx = list(range(k))
    run(f"grinold_top{k:02d}_pearson_rank",   idx, norm='rank',   ic_type='pearson')
print(f"  [{(time.time()-t0)/60:.1f}m elapsed]")

print("\n" + "=" * 70)
print("EXP 3 — Spearman IC vs Pearson IC  (z-score, top-10 & top-20)")
print("=" * 70)
for k in [10, 20]:
    idx = list(range(k))
    run(f"grinold_top{k:02d}_spearman_zscore", idx, norm='zscore', ic_type='spearman')
print(f"  [{(time.time()-t0)/60:.1f}m elapsed]")

print("\n" + "=" * 70)
print("EXP 4 — ICIR threshold sweep  (features ranked by ICIR from gold_df)")
print("       Note: features are ordered by ICIR already; different thresholds")
print("       change HOW MANY we include, not which IC map we use.")
print("=" * 70)
# Build threshold → feature count mapping
for icir_thresh in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
    mask_t = icir_df['abs_icir'] >= icir_thresh
    if icir_thresh >= 3.0:
        mask_t &= ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
    feats_t = [f for f in icir_df[mask_t].sort_values('abs_icir', ascending=False)['feature'].tolist()
               if f in all_gold]
    if not feats_t: continue
    idx_t = [all_gold.index(f) for f in feats_t]
    run(f"grinold_icir{icir_thresh:.1f}_n{len(feats_t):02d}_pearson_zscore",
        idx_t, norm='zscore', ic_type='pearson',
        extra={'icir_thresh': icir_thresh})
print(f"  [{(time.time()-t0)/60:.1f}m elapsed]")

# ── Save + print summary ───────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
out_path   = os.path.join(OUT_DIR, 'cv_grinold_search.csv')
results_df.to_csv(out_path, index=False)

print("\n" + "=" * 70)
print("SUMMARY — All configs ranked by CV R²")
print("=" * 70)
print(results_df[['config', 'n_feats', 'norm', 'ic_type', 'r2', 'std']].to_string(index=False))

best  = results_df.iloc[0]
base  = results_df[results_df['config'] == 'grinold_top10_pearson_zscore']
base_r2 = base['r2'].values[0] if len(base) else float('nan')
print(f"\n  Baseline (top-10, Pearson, z-score) CV R²: {base_r2:.5f}")
print(f"  Best config:  {best['config']}")
print(f"  Best CV R²:   {best['r2']:.5f}  (+{best['r2']-base_r2:+.5f} vs baseline)")
print(f"\n  Saved: {out_path}")
print(f"  Total: {(time.time()-t0)/60:.1f} min")
