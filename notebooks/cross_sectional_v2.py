# ================================================================
# CROSS-SECTIONAL NORMALIZED MODEL — V2 (Four Variants)
# ================================================================
# Generates 4 submission CSVs, each exploring a different axis
# relative to the v1 baseline (oracle_score ≈ 0.0517):
#
#   A  cs_v2_huber   — Huber loss (robust to kurtosis=48 fat tails)
#   B  cs_v2_rank    — Rank-normalize TARGET before training
#   C  cs_v2_gold    — Gold features only (abs_icir >= 3, ic_pos_frac=0|1)
#   D  cs_v2_shallow — Shallower trees (num_leaves=31)
#
# All variants share:
#   - CS z-score all features per day
#   - GroupKFold(5) on SO3_T quintiles
#   - auto_scale predictions to TARGET_STD=0.000948
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import norm as sp_norm

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths — auto-detect Kaggle vs local ───────────────────────────
import os as _os
if _os.path.exists('/kaggle/input'):
    # Kaggle environment
    # Competition data — update COMP_SLUG to match your competition input name
    COMP_SLUG  = 'irage-short-horizon'          # ← adjust if needed
    COMP_DIR   = f'/kaggle/input/{COMP_SLUG}'
    if not _os.path.exists(COMP_DIR):
        # fallback: first subfolder under /kaggle/input
        COMP_DIR = sorted(_os.listdir('/kaggle/input'))[0]
        COMP_DIR = f'/kaggle/input/{COMP_DIR}'
    TRAIN_PATH = os.path.join(COMP_DIR, 'train.parquet')
    TEST_PATH  = os.path.join(COMP_DIR, 'test.parquet')
    SAMPLE_SUB = os.path.join(COMP_DIR, 'sample_submission.csv')
    OUT_DIR    = '/kaggle/working'
    # ic_icir_full.csv must be uploaded as a Kaggle dataset
    # Upload ic_icir_full.csv as a dataset and set its slug below
    ICIR_SLUG  = 'quant-ml-summaries'           # ← adjust to your dataset slug
    ICIR_PATH  = f'/kaggle/input/{ICIR_SLUG}/ic_icir_full.csv'
    # Oracle (exploit_v2_zero.csv) — upload as dataset if available
    ORACLE     = f'/kaggle/input/{ICIR_SLUG}/exploit_v2_zero.csv'
else:
    # Local environment
    BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
    TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train.parquet')
    TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
    OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
    ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
    ORACLE     = os.path.join(BASE_DIR, 'outputs/submissions/exploit_v2_zero.csv')
    SAMPLE_SUB = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')

os.makedirs(OUT_DIR, exist_ok=True)
print(f"TRAIN_PATH : {TRAIN_PATH}")
print(f"OUT_DIR    : {OUT_DIR}")

TARGET_STD = 0.000948
N_FOLDS    = 5
N_EST      = 2000
LR         = 0.05
ES_ROUNDS  = 50

t0 = time.time()

# ── Helpers ───────────────────────────────────────────────────────
def auto_scale(p, target_std=TARGET_STD):
    """Scale predictions so std == TARGET_STD."""
    s = p.std()
    if s > 1e-10:
        return p * (target_std / s)
    return p


def daywise_oracle_score(pred_vec, oracle_vec, day_ids):
    """Day-wise cross-sectional Pearson IC averaged over days."""
    day_corrs = []
    for day in np.unique(day_ids):
        mask = day_ids == day
        if mask.sum() < 3:
            continue
        p = pred_vec[mask];  o = oracle_vec[mask]
        p = p - p.mean();    o = o - o.mean()
        np_norm = np.linalg.norm(p)
        no_norm = np.linalg.norm(o)
        if np_norm < 1e-12 or no_norm < 1e-12:
            day_corrs.append(0.0)
        else:
            day_corrs.append(float((p @ o) / (np_norm * no_norm)))
    return float(np.mean(day_corrs))


# ── Load data ─────────────────────────────────────────────────────
print("=" * 60)
print("Loading data...")
print("=" * 60)
t1 = time.time()
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET'}]
print(f"  Train: {len(train):,}  Test: {len(test):,}  Features: {len(feat_cols)}")
print(f"  Load time: {time.time()-t1:.1f}s")

# ── Trading day IDs ────────────────────────────────────────────────
train_time_ids = train['SO3_T'].round(5).astype(str).values
test_time_ids  = test['SO3_T'].round(5).astype(str).values
print(f"  Unique trading days — train: {len(np.unique(train_time_ids))}  "
      f"test: {len(np.unique(test_time_ids))}")

y_train  = train['TARGET'].values.astype(np.float64)
test_ids = test['ID'].values

# ── Cross-sectional z-score normalization ────────────────────────
print("\nApplying cross-sectional z-score normalization...")
t1 = time.time()

train_feat = train[feat_cols].fillna(0).values.astype(np.float32)
test_feat  = test.reindex(columns=feat_cols, fill_value=0).values.astype(np.float32)

train_norm = np.zeros_like(train_feat)
for tid in np.unique(train_time_ids):
    mask = train_time_ids == tid
    x    = train_feat[mask]
    m    = x.mean(axis=0)
    s    = x.std(axis=0)
    s    = np.where(s < 1e-8, 1.0, s)
    train_norm[mask] = (x - m) / s

test_norm = np.zeros_like(test_feat)
for tid in np.unique(test_time_ids):
    mask = test_time_ids == tid
    x    = test_feat[mask]
    m    = x.mean(axis=0)
    s    = x.std(axis=0)
    s    = np.where(s < 1e-8, 1.0, s)
    test_norm[mask] = (x - m) / s

print(f"  Done in {time.time()-t1:.1f}s")
print(f"  Sample stats after norm — mean: {train_norm.mean():.4f}  std: {train_norm.std():.4f}")

del train_feat, test_feat
gc.collect()

X_train_full = train_norm.astype(np.float32)
X_test_full  = test_norm.astype(np.float32)
del train_norm, test_norm
gc.collect()

# ── GroupKFold on SO3_T quintiles ─────────────────────────────────
so3t_idx  = feat_cols.index('SO3_T')
so3t_vals = X_train_full[:, so3t_idx]
groups    = pd.qcut(pd.Series(so3t_vals), q=N_FOLDS,
                    labels=False, duplicates='drop').values.astype(np.int32)
n_folds   = len(np.unique(groups))
gkf       = GroupKFold(n_splits=n_folds)
folds     = list(gkf.split(X_train_full, y_train, groups=groups))

print(f"\nGroupKFold: {n_folds} folds on SO3_T quintiles")
for i, (tr, va) in enumerate(folds):
    print(f"  Fold {i+1}: train={len(tr):,}  val={len(va):,}")

# ── Load oracle ───────────────────────────────────────────────────
oracle_available = os.path.exists(ORACLE)
if oracle_available:
    oracle_df   = pd.read_csv(ORACLE)
    sample_sub  = pd.read_csv(SAMPLE_SUB)[['ID']]
    # Align oracle to test ordering via sample_submission
    oracle_df   = sample_sub.merge(oracle_df, on='ID', how='left').fillna(0.0)
    oracle_vec  = oracle_df['TARGET'].values
    # test day ids aligned to sample_submission order
    test_df_tmp = pd.read_parquet(TEST_PATH, columns=['ID', 'SO3_T'])
    test_df_tmp = sample_sub.merge(test_df_tmp, on='ID', how='left')
    oracle_day_ids = test_df_tmp['SO3_T'].round(5).astype(str).values
    del test_df_tmp
    print(f"\nOracle loaded: {ORACLE}")
else:
    print(f"\nOracle NOT found at {ORACLE} — skipping oracle scoring")

# ── Gold features (Variant C) ─────────────────────────────────────
print(f"\nLoading gold features from {ICIR_PATH}...")
icir_df = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & (icir_df['ic_pos_frac'].isin([0.0, 1.0]))
gold_feats = icir_df.loc[gold_mask, 'feature'].tolist()
gold_feats = [f for f in gold_feats if f in feat_cols]
gold_idx   = [feat_cols.index(f) for f in gold_feats]
print(f"  Gold features (abs_icir>=3, ic_pos_frac=0|1): {len(gold_feats)}")

# ── Base LGB params ───────────────────────────────────────────────
BASE_LGB_PARAMS = dict(
    metric            = 'rmse',
    num_leaves        = 63,
    learning_rate     = LR,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 1,
    min_child_samples = 50,
    lambda_l1         = 0.1,
    lambda_l2         = 1.0,
    n_jobs            = -1,
    verbose           = -1,
    seed              = 42,
)

# ── Generic training function ─────────────────────────────────────
def train_variant(X_tr_full, X_te_full, y_raw, folds_list, n_f,
                  lgb_params, use_rank_target=False, winsorize=True,
                  feat_subset_idx=None, variant_name='variant'):
    """
    Train a LightGBM model across folds and return (test_preds, oof_preds).

    Parameters
    ----------
    feat_subset_idx : list of int or None
        If provided, only these column indices of X_tr_full are used.
    use_rank_target : bool
        If True, rank-normalize TARGET per day within training rows.
    winsorize : bool
        If True, winsorize TARGET at fold-level p1/p99.
    """
    if feat_subset_idx is not None:
        X_tr = X_tr_full[:, feat_subset_idx].astype(np.float32)
        X_te = X_te_full[:, feat_subset_idx].astype(np.float32)
    else:
        X_tr = X_tr_full
        X_te = X_te_full

    y_base = y_raw.copy()

    if use_rank_target:
        # Rank-normalize TARGET per day across ALL training rows
        # (applied once globally; fold winsorize is disabled for rank variant)
        y_rank = np.zeros_like(y_base, dtype=np.float64)
        for tid in np.unique(train_time_ids):
            mask = train_time_ids == tid
            y_d  = y_base[mask]
            ranks = pd.Series(y_d).rank(method='average') / (len(y_d) + 1)
            y_rank[mask] = sp_norm.ppf(ranks.values)
        y_base = y_rank

    oof_preds  = np.zeros(len(y_base), dtype=np.float64)
    test_preds = np.zeros(len(X_te),   dtype=np.float64)
    fold_r2s   = []
    best_iters = []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds_list):
        tf = time.time()
        print(f"\n  {'─'*50}")
        print(f"  Fold {fold_idx+1}/{n_f}  [{variant_name}]")

        X_fold_tr = X_tr[tr_idx].copy()
        y_fold_tr = y_base[tr_idx].copy()
        y_fold_va = y_base[va_idx].copy()

        if winsorize and not use_rank_target:
            lo, hi   = np.percentile(y_fold_tr, 1), np.percentile(y_fold_tr, 99)
            y_fold_tr = np.clip(y_fold_tr, lo, hi)

        dtrain = lgb.Dataset(X_fold_tr, label=y_fold_tr, free_raw_data=True)
        dval   = lgb.Dataset(X_tr[va_idx], label=y_fold_va,
                             reference=dtrain, free_raw_data=True)

        del X_fold_tr
        gc.collect()

        model = lgb.train(
            lgb_params, dtrain,
            num_boost_round = N_EST,
            valid_sets      = [dval],
            callbacks       = [lgb.early_stopping(ES_ROUNDS, verbose=False),
                               lgb.log_evaluation(200)],
        )

        best_iter = model.best_iteration
        best_iters.append(best_iter)

        oof_preds[va_idx] = model.predict(X_tr[va_idx], num_iteration=best_iter)
        test_preds       += model.predict(X_te, num_iteration=best_iter) / n_f

        fold_r2 = r2_score(y_fold_va, oof_preds[va_idx])
        fold_r2s.append(fold_r2)
        print(f"  best_iter={best_iter}  fold_R²={fold_r2:+.6f}  ({time.time()-tf:.0f}s)")

        del dtrain, dval, model
        gc.collect()

    oof_r2 = r2_score(y_base, oof_preds)
    print(f"\n  [{variant_name}] OOF R²={oof_r2:+.6f}  best_iters={best_iters}")
    return test_preds, oof_preds, oof_r2


# ── Submission helper ─────────────────────────────────────────────
sample_sub = pd.read_csv(SAMPLE_SUB)[['ID']]

def save_submission(test_preds, name):
    """auto_scale, align to sample_submission, save CSV."""
    scaled = auto_scale(test_preds)
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': scaled})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    out_path = os.path.join(OUT_DIR, f'{name}.csv')
    sub.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return scaled, sub


def eval_oracle(scaled_preds, name):
    if not oracle_available:
        return None
    # Align scaled_preds (indexed by test_ids order) to oracle order
    pred_df  = pd.DataFrame({'ID': test_ids, 'TARGET': scaled_preds})
    aligned  = sample_sub.merge(pred_df, on='ID', how='left').fillna(0.0)
    score    = daywise_oracle_score(aligned['TARGET'].values, oracle_vec, oracle_day_ids)
    print(f"  oracle_score [{name}]: {score:+.6f}")
    return score


# ================================================================
# VARIANT A — cs_v2_huber
# Huber loss (alpha=0.9) is robust to kurtosis=48 fat tails.
# All 444 features, num_leaves=63, winsorize target.
# ================================================================
print("\n" + "=" * 60)
print("VARIANT A: cs_v2_huber  (objective=huber, alpha=0.9)")
print("=" * 60)
t_a = time.time()

params_a = {**BASE_LGB_PARAMS,
            'objective': 'huber',
            'alpha':     0.9}

test_preds_a, oof_preds_a, oof_r2_a = train_variant(
    X_train_full, X_test_full, y_train, folds, n_folds,
    lgb_params=params_a, winsorize=True, variant_name='cs_v2_huber'
)
scaled_a, _ = save_submission(test_preds_a, 'cs_v2_huber')
score_a      = eval_oracle(test_preds_a, 'cs_v2_huber')
print(f"  Variant A elapsed: {(time.time()-t_a)/60:.1f} min")

# ================================================================
# VARIANT B — cs_v2_rank
# Rank-normalize TARGET per day via normal quantile transform.
# Predictions are in rank-space; auto_scale before saving.
# No winsorize (rank transform already handles outliers).
# ================================================================
print("\n" + "=" * 60)
print("VARIANT B: cs_v2_rank  (rank-normalize TARGET per day)")
print("=" * 60)
t_b = time.time()

params_b = {**BASE_LGB_PARAMS, 'objective': 'regression'}

test_preds_b, oof_preds_b, oof_r2_b = train_variant(
    X_train_full, X_test_full, y_train, folds, n_folds,
    lgb_params=params_b, use_rank_target=True, winsorize=False,
    variant_name='cs_v2_rank'
)
scaled_b, _ = save_submission(test_preds_b, 'cs_v2_rank')
score_b      = eval_oracle(test_preds_b, 'cs_v2_rank')
print(f"  Variant B elapsed: {(time.time()-t_b)/60:.1f} min")

# ================================================================
# VARIANT C — cs_v2_gold
# Gold features only: abs_icir >= 3 AND ic_pos_frac in {0, 1}.
# All other settings identical to v1 (num_leaves=63, winsorize).
# ================================================================
print("\n" + "=" * 60)
print(f"VARIANT C: cs_v2_gold  ({len(gold_feats)} gold features)")
print("=" * 60)
t_c = time.time()

params_c = {**BASE_LGB_PARAMS, 'objective': 'regression'}

test_preds_c, oof_preds_c, oof_r2_c = train_variant(
    X_train_full, X_test_full, y_train, folds, n_folds,
    lgb_params=params_c, winsorize=True, feat_subset_idx=gold_idx,
    variant_name='cs_v2_gold'
)
scaled_c, _ = save_submission(test_preds_c, 'cs_v2_gold')
score_c      = eval_oracle(test_preds_c, 'cs_v2_gold')
print(f"  Variant C elapsed: {(time.time()-t_c)/60:.1f} min")

# ================================================================
# VARIANT D — cs_v2_shallow
# Shallower trees (num_leaves=31) to reduce overfit to liquid noise.
# All 444 features, winsorize target.
# ================================================================
print("\n" + "=" * 60)
print("VARIANT D: cs_v2_shallow  (num_leaves=31)")
print("=" * 60)
t_d = time.time()

params_d = {**BASE_LGB_PARAMS,
            'objective':  'regression',
            'num_leaves': 31}

test_preds_d, oof_preds_d, oof_r2_d = train_variant(
    X_train_full, X_test_full, y_train, folds, n_folds,
    lgb_params=params_d, winsorize=True, variant_name='cs_v2_shallow'
)
scaled_d, _ = save_submission(test_preds_d, 'cs_v2_shallow')
score_d      = eval_oracle(test_preds_d, 'cs_v2_shallow')
print(f"  Variant D elapsed: {(time.time()-t_d)/60:.1f} min")

# ================================================================
# SUMMARY TABLE
# ================================================================
print("\n" + "=" * 60)
print("CROSS-SECTIONAL V2 — COMPARISON TABLE")
print("=" * 60)
print(f"{'Variant':<20} {'OOF R²':>12} {'oracle_score':>14}")
print(f"{'─'*20} {'─'*12} {'─'*14}")
print(f"{'cs_v1 (baseline)':<20} {'N/A':>12} {'+0.051700':>14}  (reference)")
print(f"{'cs_v2_huber':<20} {oof_r2_a:>+12.6f} {(str(f'{score_a:+.6f}') if score_a is not None else 'N/A'):>14}")
print(f"{'cs_v2_rank':<20} {oof_r2_b:>+12.6f} {(str(f'{score_b:+.6f}') if score_b is not None else 'N/A'):>14}")
print(f"{'cs_v2_gold':<20} {oof_r2_c:>+12.6f} {(str(f'{score_c:+.6f}') if score_c is not None else 'N/A'):>14}")
print(f"{'cs_v2_shallow':<20} {oof_r2_d:>+12.6f} {(str(f'{score_d:+.6f}') if score_d is not None else 'N/A'):>14}")
print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
print("\nNOTE: OOF R² is inversely correlated with LB. Use oracle_score for selection.")
