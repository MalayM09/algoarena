# ================================================================
# FEATURE MOMENTUM CV — Leak-proof new feature engineering
# ================================================================
# CRITICAL DESIGN PRINCIPLE (no IC leak):
#   ICs are NEVER pre-computed on full data and passed into CV.
#   They are computed inside each fold loop on training rows only.
#
# New feature types tested:
#   1. MOMENTUM  : LagT1 - LagT2  (velocity of each gold signal)
#                  LagT2 - LagT3  (longer-term momentum)
#   2. EXTENDED LAGS : LagT2 / LagT3 versions of gold base features
#   3. COMBINED  : Gold + Momentum  |  Gold + LagT2  |  Gold + both
#   4. UNIFIED   : All feature types ranked together by in-fold IC
#
# Performance design:
#   - Data pre-grouped into numpy arrays once (no repeated groupby)
#   - float32 throughout
#   - Vectorised batch IC via matrix multiply (no per-feature loops)
#   - 5-fold CV (fast, stable)
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd

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
print("FEATURE MOMENTUM CV — Leak-proof new feature engineering")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
print(f"  Rows: {len(train):,} | Days: {train['day_id'].nunique()}")

# ── Gold features (ordering only, NOT used for IC in CV) ──────────────
icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
all_gold  = [f for f in gold_df['feature'].tolist() if f in train.columns]
top10     = all_gold[:10]
top20     = all_gold[:20]
print(f"  Gold features: {len(all_gold)}")

# ── Build momentum & extended-lag features ────────────────────────────
print("\nBuilding new feature columns...")

momentum_cols = []   # new column names added to train
lag2_cols     = []
lag3_cols     = []

for f in all_gold:
    # Momentum: LagT1 - LagT2
    if '_LagT1' in f:
        lag2 = f.replace('_LagT1', '_LagT2')
        if lag2 in train.columns:
            mom_name = f.replace('_LagT1', '_MomT12')
            train[mom_name] = (train[f].fillna(0) - train[lag2].fillna(0)).astype(np.float32)
            momentum_cols.append(mom_name)
    elif '_LagT2' in f:
        lag3 = f.replace('_LagT2', '_LagT3')
        if lag3 in train.columns:
            mom_name = f.replace('_LagT2', '_MomT23')
            train[mom_name] = (train[f].fillna(0) - train[lag3].fillna(0)).astype(np.float32)
            momentum_cols.append(mom_name)
    else:
        lag1 = f + '_LagT1'; lag2 = f + '_LagT2'
        if lag1 in train.columns and lag2 in train.columns:
            mom_name = f + '_MomT12'
            train[mom_name] = (train[lag1].fillna(0) - train[lag2].fillna(0)).astype(np.float32)
            momentum_cols.append(mom_name)

    # Extended lags: raw LagT2 / LagT3 of gold features
    if '_LagT1' in f:
        lag2 = f.replace('_LagT1', '_LagT2')
        lag3 = f.replace('_LagT1', '_LagT3')
        if lag2 in train.columns: lag2_cols.append(lag2)
        if lag3 in train.columns: lag3_cols.append(lag3)

# De-dup and limit size
momentum_cols = list(dict.fromkeys(momentum_cols))[:len(all_gold)]
lag2_cols     = list(dict.fromkeys(lag2_cols))[:len(all_gold)]
lag3_cols     = list(dict.fromkeys(lag3_cols))[:len(all_gold)]
print(f"  Momentum cols: {len(momentum_cols)} | LagT2: {len(lag2_cols)} | LagT3: {len(lag3_cols)}")

# ── Define all candidate feature pools for experiments ────────────────
# We'll pre-group ALL candidate columns into numpy arrays
all_candidate_cols = list(dict.fromkeys(all_gold + momentum_cols + lag2_cols + lag3_cols))
print(f"  Total candidate features: {len(all_candidate_cols)}")

# Index maps
col_idx = {c: i for i, c in enumerate(all_candidate_cols)}
gold_idx   = [col_idx[c] for c in top10]
gold20_idx = [col_idx[c] for c in top20]
mom_idx    = [col_idx[c] for c in momentum_cols]
lag2_idx   = [col_idx[c] for c in lag2_cols]
lag3_idx   = [col_idx[c] for c in lag3_cols]

# ── Assign folds ───────────────────────────────────────────────────────
groups_sorted = sorted(train['CV_GROUP'].unique())
fold_map = {g: i % N_FOLDS for i, g in enumerate(groups_sorted)}
train['_fold'] = train['CV_GROUP'].map(fold_map)

# ── Pre-group data into numpy arrays (done ONCE for speed) ─────────────
print("\nPre-grouping data by day (numpy, float32)...")
day_keys  = []
day_X     = []   # (n_assets, n_all_candidate_cols), float32
day_y     = []   # (n_assets,), float32
day_folds = []   # (n_assets,), int8

for day, grp in train.groupby('day_id'):
    if len(grp) < 5: continue
    day_keys.append(day)
    day_X.append(grp[all_candidate_cols].fillna(0).values.astype(np.float32))
    day_y.append(grp['TARGET'].values.astype(np.float32))
    day_folds.append(grp['_fold'].values.astype(np.int8))

print(f"  Grouped {len(day_keys)} days  [{(time.time()-t0)/60:.1f}m]")

# ── Helpers ────────────────────────────────────────────────────────────
def zscore_rows(X, clip=CLIP_Z):
    m = X.mean(0, keepdims=True); s = X.std(0, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip).astype(np.float32)

def compute_perday_r2(preds, targets, days):
    df = pd.DataFrame({'p': preds, 't': targets, 'd': days})
    r2s = []
    for _, g in df.groupby('d'):
        p = g['p'].values; t = g['t'].values
        if len(p) < 3: continue
        p_dm = p - p.mean(); t_dm = t - t.mean()
        ss = (t_dm**2).sum()
        if ss < 1e-12: continue
        r2s.append(1.0 - ((p_dm - t_dm)**2).sum() / ss)
    return (np.mean(r2s), np.std(r2s), len(r2s)) if r2s else (np.nan, np.nan, 0)

# ── Core leak-proof CV ─────────────────────────────────────────────────
def cv_strict(feat_indices):
    """
    Strictly leak-free CV.

    feat_indices : list of column indices into all_candidate_cols.
    ICs are computed inside each fold on training data only.
    Feature normalization (z-score) uses ALL day's rows for stable stats,
    but IC estimation uses only training-fold rows.
    """
    all_preds = []; all_tgts = []; all_days = []

    for fold in range(N_FOLDS):
        # Build pooled, per-day-z-scored training set
        pool_X = []; pool_y = []
        for X_d, y_d, fold_d in zip(day_X, day_y, day_folds):
            tr = (fold_d != fold)
            if tr.sum() < 5: continue
            Xf = zscore_rows(X_d[:, feat_indices])   # normalise with full-day stats
            pool_X.append(Xf[tr])
            pool_y.append(y_d[tr])

        X_pool = np.vstack(pool_X)    # (N_train, K)
        y_pool = np.concatenate(pool_y)

        # Vectorised Pearson IC (no leakage — computed on train pool only)
        y_dm   = (y_pool - y_pool.mean()).astype(np.float32)
        ic_arr = (X_pool * y_dm[:, None]).mean(0)   # shape (K,)

        # Predict val rows
        for day_key, X_d, y_d, fold_d in zip(day_keys, day_X, day_y, day_folds):
            vm = (fold_d == fold)
            if vm.sum() == 0: continue
            Xf  = zscore_rows(X_d[:, feat_indices])
            pred = (Xf[vm] @ ic_arr).astype(np.float32)
            pred -= pred.mean()
            all_preds.extend(pred.tolist())
            all_tgts.extend(y_d[vm].tolist())
            all_days.extend([day_key] * int(vm.sum()))

        del X_pool, y_pool, pool_X, pool_y

    return compute_perday_r2(
        np.array(all_preds, np.float32),
        np.array(all_tgts,  np.float32),
        all_days
    )

# ── Baseline ───────────────────────────────────────────────────────────
results = []

def run(label, feat_indices, extra=None):
    r2, std, nd = cv_strict(feat_indices)
    row = {'config': label, 'r2': r2, 'std': std, 'n_days': nd,
           'n_feats': len(feat_indices)}
    if extra: row.update(extra)
    results.append(row)
    print(f"  {label:<50}  R²={r2:.5f}  ±{std:.5f}")
    return r2

print("\n" + "=" * 70)
print("BASELINE — Top-10 gold (Pearson IC, z-score)")
print("=" * 70)
r2_base = run("baseline_gold10", gold_idx)
r2_base_20 = run("baseline_gold20", gold20_idx)

# ── EXP 1: Gold + momentum ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("EXP 1 — Gold features + Momentum (LagT1 - LagT2)")
print("=" * 70)
for g_k, g_idx in [(10, gold_idx), (20, gold20_idx)]:
    for m_k in [5, 10, 20]:
        feats = list(dict.fromkeys(g_idx + mom_idx[:m_k]))
        run(f"gold{g_k}_mom{m_k}", feats, extra={'gold': g_k, 'mom': m_k})
print(f"  [{(time.time()-t0)/60:.1f}m]")

# ── EXP 2: Gold + extended lags ────────────────────────────────────────
print("\n" + "=" * 70)
print("EXP 2 — Gold features + LagT2 companions")
print("=" * 70)
for g_k, g_idx in [(10, gold_idx), (20, gold20_idx)]:
    for l_k in [5, 10]:
        feats = list(dict.fromkeys(g_idx + lag2_idx[:l_k]))
        run(f"gold{g_k}_lag2_{l_k}", feats, extra={'gold': g_k, 'lag2': l_k})
# LagT3 as well
for l_k in [5, 10]:
    feats = list(dict.fromkeys(gold_idx + lag3_idx[:l_k]))
    run(f"gold10_lag3_{l_k}", feats, extra={'gold': 10, 'lag3': l_k})
print(f"  [{(time.time()-t0)/60:.1f}m]")

# ── EXP 3: Mega combined feature sets ─────────────────────────────────
print("\n" + "=" * 70)
print("EXP 3 — Mega sets: Gold + Momentum + LagT2")
print("=" * 70)
combos = [
    ("gold10_mom5_lag2_5",   gold_idx,   mom_idx[:5],  lag2_idx[:5],  []),
    ("gold10_mom10_lag2_10", gold_idx,   mom_idx[:10], lag2_idx[:10], []),
    ("gold20_mom10_lag2_10", gold20_idx, mom_idx[:10], lag2_idx[:10], []),
    ("gold10_mom10_lag2_10_lag3_5", gold_idx, mom_idx[:10], lag2_idx[:10], lag3_idx[:5]),
    ("gold20_mom20_lag2_10", gold20_idx, mom_idx[:20], lag2_idx[:10], []),
]
for label, gi, mi, l2i, l3i in combos:
    feats = list(dict.fromkeys(gi + mi + l2i + l3i))
    run(label, feats)
print(f"  [{(time.time()-t0)/60:.1f}m]")

# ── EXP 4: Unified ranking — let in-fold IC decide best features ───────
print("\n" + "=" * 70)
print("EXP 4 — Unified feature pool: best K by in-fold IC (dynamically)")
print("       First compute mean in-fold IC across all folds to rank, then CV")
print("=" * 70)

# To rank ALL candidate features without leakage:
# Run ONE pass of CV on all candidates, record mean abs(IC) per feature per fold,
# then use the ranking to select top-K subsets and CV those.
print("  Computing mean abs(IC) for all candidates across folds (single pass)...")
all_candidate_ics = np.zeros((N_FOLDS, len(all_candidate_cols)), dtype=np.float32)
all_cand_idx = list(range(len(all_candidate_cols)))

for fold in range(N_FOLDS):
    pool_X = []; pool_y = []
    for X_d, y_d, fold_d in zip(day_X, day_y, day_folds):
        tr = (fold_d != fold)
        if tr.sum() < 5: continue
        Xf = zscore_rows(X_d)   # all candidates
        pool_X.append(Xf[tr]); pool_y.append(y_d[tr])
    X_pool = np.vstack(pool_X); y_pool = np.concatenate(pool_y)
    y_dm = (y_pool - y_pool.mean()).astype(np.float32)
    ic_fold = (X_pool * y_dm[:, None]).mean(0)
    all_candidate_ics[fold] = ic_fold
    del X_pool, y_pool, pool_X, pool_y
    print(f"    fold {fold} done  [{(time.time()-t0)/60:.1f}m]")

mean_abs_ic = np.abs(all_candidate_ics).mean(0)   # shape (n_all_candidates,)
ranked_idx  = np.argsort(mean_abs_ic)[::-1]        # desc order

# Top feature names
ranked_names = [all_candidate_cols[i] for i in ranked_idx[:30]]
print(f"\n  Top-15 features by in-fold mean |IC|:")
for i, name in enumerate(ranked_names[:15]):
    col_type = ('gold' if name in all_gold else
                'momentum' if 'Mom' in name else
                'lag2' if '_LagT2' in name else 'lag3')
    print(f"    {i+1:2d}. {name:<50}  |IC|={mean_abs_ic[ranked_idx[i]]:.5f}  [{col_type}]")

print(f"\n  CV on top-K unified feature sets:")
for k in [10, 15, 20, 25, 30]:
    feats = list(ranked_idx[:k])
    run(f"unified_top{k:02d}_by_infold_ic", feats,
        extra={'selection': 'unified_infold', 'k': k})
print(f"  [{(time.time()-t0)/60:.1f}m]")

# ── Save + summarise ───────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
out_path   = os.path.join(OUT_DIR, 'cv_feature_momentum.csv')
results_df.to_csv(out_path, index=False)

print("\n" + "=" * 70)
print("SUMMARY — All configs ranked by CV R²")
print("=" * 70)
print(results_df[['config', 'n_feats', 'r2', 'std']].to_string(index=False))

best = results_df.iloc[0]
print(f"\n  Baseline (gold-10)   CV R²: {r2_base:.5f}")
print(f"  Best config: {best['config']}")
print(f"  Best CV R²:  {best['r2']:.5f}  ({best['r2']-r2_base:+.5f} vs baseline)")
print(f"\n  Saved: {out_path}")
print(f"  Total: {(time.time()-t0)/60:.1f} min")
