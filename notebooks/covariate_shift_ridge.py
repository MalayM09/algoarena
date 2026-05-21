# ================================================================
# COVARIATE SHIFT REWEIGHTING — Fix Liquid→Illiquid Calibration
# ================================================================
# Problem confirmed from pseudo_illiquid_oof.py:
#   - Standard OOF is 10-130x inflated (liquid validates on liquid)
#   - Pseudo-illiquid R² is negative (predictions too large for
#     illiquid assets) despite positive IC (+0.022 to +0.056)
#   - Calibration failure, not signal failure
#
# Fix: reweight training samples so model focuses on observations
# that "look like" the test (illiquid) population.
#
# Method: Kullback-Leibler Importance Estimation Procedure (KLIEP)
# via logistic density ratio — simpler: fit logistic classifier
# (train=0, test=1) on gold features. Sample weight = P(test|x) /
# P(train|x) = density ratio.
#
# Key difference from previous adv_weighted_ridge.py:
#   - Use only gold features (51) not all 445 → gentler weights
#   - Use LogisticRegression not LightGBM → calibrated probabilities
#   - Evaluate with pseudo-illiquid OOF (not standard KFold)
#   - Three weight variants: logistic, bookshape-inverse, combined
#
# Experiments:
#   baseline     : gold_z Ridge, no reweighting (reference)
#   cov_logistic : density ratio weights from LogisticRegression
#   cov_bookshape: within-day BookShape-inverse weights (direct proxy)
#   cov_combined : product of both weights
#   huber_cov    : Huber loss + cov_logistic (robust to kurtosis=48)
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression, HuberRegressor
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
t0 = time.time()

print("=" * 65)
print("COVARIATE SHIFT REWEIGHTING — Liquid→Illiquid Calibration Fix")
print("=" * 65)

# ── Feature selection ─────────────────────────────────────────────
icir_df    = pd.read_csv(ICIR_PATH)
gold_mask  = (icir_df['abs_icir'] >= 3) & \
             ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_feats = icir_df[gold_mask].sort_values('abs_icir', ascending=False)['feature'].tolist()
gold_ic_map = dict(zip(icir_df['feature'], icir_df['mean_ic']))
print(f"Gold features (ICIR>=3, never flip): {len(gold_feats)}")

# ── Load data ─────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)

all_cols   = set(train.columns) - {'ID', 'TARGET'}
gold_feats = [f for f in gold_feats if f in all_cols]
print(f"Gold features in dataset: {len(gold_feats)}")

train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days = set(train['day_id'].unique())
overlap    = train_days & set(test['day_id'].unique())
new_days   = set(test['day_id'].unique()) - train_days

y_train  = train['TARGET'].values.astype(np.float64)
test_ids = test['ID'].values

print(f"Train days: {len(train_days)}  Overlap: {len(overlap)}  New: {len(new_days)}")

# ── BookShape features ────────────────────────────────────────────
b_near = [c for c in all_cols if 'Lag' not in c and
          any(f'_B0{i}' in c for i in range(5))]
b_far  = [c for c in all_cols if 'Lag' not in c and
          any(f'_B{i}' in c for i in ['06','07','08','09','10'])]

bookshape_train = (train[b_near].fillna(0).sum(axis=1).values -
                   train[b_far].fillna(0).sum(axis=1).values).astype(np.float64)
bookshape_test  = (test[b_near].fillna(0).sum(axis=1).values -
                   test[b_far].fillna(0).sum(axis=1).values).astype(np.float64)

print(f"\nBookShape: train mean={bookshape_train.mean():.0f}  "
      f"test mean={bookshape_test.mean():.0f}")


# ================================================================
# STEP 1: FIT DENSITY RATIO CLASSIFIER
# ================================================================
# Logistic regression on gold features (train=0, test=1).
# CRITICAL FIX: use per-day cross-sectional z-scores as input,
# NOT global StandardScaler. Reason: liquid/illiquid distinction
# is within-day (same day, different BookShape). Global scaling
# mixes different days' magnitudes and teaches the classifier the
# wrong question. Per-day z-scores teach it: "within a day, does
# this asset's relative position look like test (illiquid)?"
# ================================================================
print("\n" + "="*65)
print("STEP 1: DENSITY RATIO (Logistic on per-day z-scored gold features)")
print("="*65)

print("  Computing per-day cross-sectional z-scores for all rows...")
# Pre-compute per-day z-scored gold features for both train and test
def compute_daily_zscores(df, feat_cols, clip=5.0):
    """Z-score each feature within each trading day."""
    out = np.zeros((len(df), len(feat_cols)), dtype=np.float32)
    for day, grp in df.groupby('day_id'):
        X = grp[feat_cols].fillna(0).values.astype(np.float64)
        m = X.mean(0);  s = X.std(0)
        s = np.where(s < 1e-8, 1.0, s)
        out[grp.index] = np.clip((X - m) / s, -clip, clip).astype(np.float32)
    return out

# Reset index so grp.index works correctly
train = train.reset_index(drop=True)
test  = test.reset_index(drop=True)

X_tr_z = compute_daily_zscores(train, gold_feats)   # (n_train, 51)
X_te_z = compute_daily_zscores(test,  gold_feats)   # (n_test,  51)
print(f"  Done. Train z-matrix: {X_tr_z.shape}  Test z-matrix: {X_te_z.shape}")

# Subsample for speed — 100k each class max, use z-scored features
rng = np.random.default_rng(42)
n_tr_sub = min(len(train), 100_000)
n_te_sub = min(len(test),  100_000)
tr_sub_idx = rng.choice(len(train), n_tr_sub, replace=False)
te_sub_idx = rng.choice(len(test),  n_te_sub, replace=False)

X_adv = np.vstack([X_tr_z[tr_sub_idx], X_te_z[te_sub_idx]]).astype(np.float64)
y_adv = np.concatenate([np.zeros(n_tr_sub), np.ones(n_te_sub)])

# Features are already z-scored per day — no further scaling needed
# (Logistic regression convergence is fine on [-5, 5] bounded inputs)
clf = LogisticRegression(C=0.1, max_iter=500, solver='saga', n_jobs=-1, random_state=42)
clf.fit(X_adv, y_adv)

# Evaluation via 3-fold CV on the subsampled adversarial set
from sklearn.model_selection import cross_val_score
cv_auc = cross_val_score(clf, X_adv, y_adv, cv=3, scoring='roc_auc', n_jobs=-1).mean()
print(f"\n  Logistic CV AUC (per-day z-scored): {cv_auc:.4f}")
if   cv_auc > 0.90: print("  → STRONG shift (weights will be meaningful)")
elif cv_auc > 0.75: print("  → MODERATE shift")
elif cv_auc > 0.60: print("  → MILD shift")
else:               print("  → WEAK shift (covariate reweighting may not help much)")

# Score ALL training rows using per-day z-scored features
p_test_given_x = clf.predict_proba(X_tr_z.astype(np.float64))[:, 1]  # P(test | x)

# Density ratio = P(test|x) / P(train|x)
eps               = 1e-6
density_ratio_raw = p_test_given_x / (1 - p_test_given_x + eps)

# Clip at 99th percentile (prevent extreme upweighting of a few rows)
clip_val      = np.percentile(density_ratio_raw, 99)
density_ratio = np.clip(density_ratio_raw, 0, clip_val)
density_ratio = density_ratio / density_ratio.mean()   # normalise to mean=1

print(f"\n  Density ratio stats (after clip at p99={clip_val:.4f}):")
print(f"    mean={density_ratio.mean():.3f}  "
      f"std={density_ratio.std():.3f}  "
      f"p5={np.percentile(density_ratio,5):.4f}  "
      f"p25={np.percentile(density_ratio,25):.4f}  "
      f"p50={np.percentile(density_ratio,50):.4f}  "
      f"p75={np.percentile(density_ratio,75):.4f}  "
      f"p99={np.percentile(density_ratio,99):.4f}  "
      f"max={density_ratio.max():.4f}")
print(f"    % rows with weight > 2x: {(density_ratio > 2).mean()*100:.1f}%")
print(f"    % rows with weight > 5x: {(density_ratio > 5).mean()*100:.1f}%")

# Sanity check on test rows — most should have high P(test|x)
p_test_test = clf.predict_proba(X_te_z.astype(np.float64))[:, 1]
print(f"\n  P(test|x) for TEST rows (sanity — should be high):")
print(f"    mean={p_test_test.mean():.3f}  median={np.median(p_test_test):.3f}  "
      f"% > 0.5: {(p_test_test > 0.5).mean()*100:.1f}%")

# Store on train df
train['density_ratio'] = density_ratio


# ================================================================
# STEP 2: BOOKSHAPE-INVERSE WEIGHT (within-day proxy)
# ================================================================
# Within each day, invert the BookShape percentile rank so
# illiquid-looking assets (low BookShape) get higher weight.
# This is a purely structural reweighting — no classifier needed.
# ================================================================
print("\n" + "="*65)
print("STEP 2: BOOKSHAPE-INVERSE WEIGHT (per-day)")
print("="*65)

train['bookshape'] = bookshape_train
bookshape_weight   = np.ones(len(train))

for day, grp in train.groupby('day_id'):
    n     = len(grp)
    bs    = grp['bookshape'].values
    # rank from 0 (most illiquid) to 1 (most liquid)
    bs_rank = rankdata(bs, method='average') / n
    # invert: illiquid (low rank) gets high weight
    inv_rank  = 1.0 - bs_rank + 1.0/n        # range [1/n, 1]
    # normalise to mean=1 within day
    inv_rank  = inv_rank / inv_rank.mean()
    bookshape_weight[grp.index] = inv_rank

bsw_clip = np.percentile(bookshape_weight, 99)
bookshape_weight = np.clip(bookshape_weight, 0, bsw_clip)
bookshape_weight = bookshape_weight / bookshape_weight.mean()

train['bs_weight']  = bookshape_weight

print(f"  BookShape-inverse weight stats:")
print(f"    mean={bookshape_weight.mean():.3f}  "
      f"std={bookshape_weight.std():.3f}  "
      f"p5={np.percentile(bookshape_weight,5):.4f}  "
      f"p50={np.percentile(bookshape_weight,50):.4f}  "
      f"p95={np.percentile(bookshape_weight,95):.4f}  "
      f"max={bookshape_weight.max():.4f}")

# Combined weight
combined_weight = density_ratio * bookshape_weight
combined_weight = combined_weight / combined_weight.mean()
combined_clip   = np.percentile(combined_weight, 99)
combined_weight = np.clip(combined_weight, 0, combined_clip)
combined_weight = combined_weight / combined_weight.mean()
train['combined_weight'] = combined_weight


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def zscore_day(X_tr, X_te=None, clip=5.0):
    m = X_tr.mean(0);  s = X_tr.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    Xtz = np.clip((X_tr - m) / s, -clip, clip)
    Xez = np.clip((X_te - m) / s, -clip, clip) if X_te is not None else None
    return Xtz, Xez

def rank_day(X_tr, X_te=None):
    if X_te is None:
        n = X_tr.shape[0]
        return np.apply_along_axis(
            lambda c: rankdata(c, 'average') / n, 0, X_tr), None
    n_tr  = X_tr.shape[0]
    X_all = np.vstack([X_tr, X_te])
    n_all = X_all.shape[0]
    Xr    = np.apply_along_axis(
        lambda c: rankdata(c, 'average') / n_all, 0, X_all)
    return Xr[:n_tr], Xr[n_tr:]

def winsorise(y, pct=5):
    lo, hi = np.percentile(y, pct), np.percentile(y, 100-pct)
    return np.clip(y, lo, hi)

def fit_ridge_weighted(X_tr, y_tr, X_te, alpha=100, weights=None):
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_tr, winsorise(y_tr), sample_weight=weights)
    return model.predict(X_te)

def fit_huber_weighted(X_tr, y_tr, X_te, epsilon=1.5, alpha=0.01, weights=None):
    """Huber regression — robust to fat tails (kurtosis=48)."""
    model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=300)
    # HuberRegressor doesn't support sample_weight directly in fit,
    # but we simulate it by duplicating high-weight samples
    if weights is not None:
        # integer approximation: round weights, clip to reasonable max
        w_int = np.clip(np.round(weights * 3).astype(int), 1, 20)
        idx   = np.repeat(np.arange(len(X_tr)), w_int)
        model.fit(X_tr[idx], winsorise(y_tr)[idx])
    else:
        model.fit(X_tr, winsorise(y_tr))
    return model.predict(X_te)

def per_day_ic(y_true, y_pred):
    if len(y_true) < 5: return np.nan
    corr, _ = spearmanr(y_true, y_pred)
    return corr


# ================================================================
# STEP 3: GLOBAL FALLBACK (for 84 new days)
# ================================================================
print("\n" + "="*65)
print("STEP 3: GLOBAL FALLBACKS (for 84 new test days)")
print("="*65)

def fit_global_fallback(weights=None, tag='unweighted'):
    X_g = train[gold_feats].fillna(0).values.astype(np.float64)
    m, s = X_g.mean(0), X_g.std(0);  s = np.where(s < 1e-8, 1.0, s)
    X_z  = np.clip((X_g - m) / s, -5, 5)
    model = Ridge(alpha=1000, fit_intercept=True)
    model.fit(X_z, y_train, sample_weight=weights)
    print(f"  Fallback [{tag}] fitted.")
    return model, m, s

fb_base,    m_b, s_b = fit_global_fallback(weights=None,             tag='no_weight')
fb_logistic, m_l, s_l = fit_global_fallback(weights=density_ratio,  tag='density_ratio')
fb_combined, m_c, s_c = fit_global_fallback(weights=combined_weight, tag='combined')


# ================================================================
# STEP 4: MAIN EXPERIMENT LOOP
# ================================================================
# For each experiment:
#   (a) Per-day transductive Ridge with sample weights
#   (b) Evaluate via PSEUDO-ILLIQUID OOF (BookShape split)
#   (c) Collect test predictions
# ================================================================

experiments = {
    'baseline'     : (None,             'zscore', fb_base,     m_b, s_b),
    'cov_logistic' : (density_ratio,    'zscore', fb_logistic, m_l, s_l),
    'cov_bookshape': (bookshape_weight, 'zscore', fb_combined, m_c, s_c),
    'cov_combined' : (combined_weight,  'zscore', fb_combined, m_c, s_c),
    'cov_rank'     : (density_ratio,    'rank',   fb_logistic, m_l, s_l),
}

print("\n" + "="*65)
print("STEP 4: EXPERIMENT LOOP — with Pseudo-Illiquid OOF eval")
print("="*65)

pi_results   = {}   # pseudo-illiquid OOF R²
pi_ic_dist   = {}   # per-day IC on pseudo-illiquid
te_preds_all = {}   # test predictions

rng2 = np.random.default_rng(42)

for exp_name, (weights, norm_type, fb_model, fb_mean, fb_std) in experiments.items():
    t_exp = time.time()
    print(f"\n  [{exp_name}]  norm={norm_type}  "
          f"weights={'none' if weights is None else 'yes'}")

    pi_oof    = np.full(len(train), np.nan)
    te_preds  = np.zeros(len(test))
    day_ics   = {}
    day_r2s   = {}

    for day, grp in train.groupby('day_id'):
        n = len(grp)
        if n < 20:
            continue

        y_day  = y_train[grp.index]
        bs_day = grp['bookshape'].values

        # Pseudo-illiquid split (BookShape median)
        bs_med     = np.median(bs_day)
        liq_mask   = bs_day >= bs_med
        illiq_mask = ~liq_mask

        if liq_mask.sum() < 10 or illiq_mask.sum() < 5:
            continue

        liq_idx   = grp.index[liq_mask]
        illiq_idx = grp.index[illiq_mask]

        X_liq_raw   = grp.loc[liq_idx,   gold_feats].fillna(0).values.astype(np.float64)
        X_illiq_raw = grp.loc[illiq_idx, gold_feats].fillna(0).values.astype(np.float64)
        y_liq       = y_train[liq_idx]
        y_illiq     = y_train[illiq_idx]

        # Sample weights for the liquid training half
        w_liq = None
        if weights is not None:
            w_liq = weights[liq_idx]
            if w_liq.sum() < 1e-8:
                w_liq = None

        if norm_type == 'zscore':
            X_liq_n, X_illiq_n = zscore_day(X_liq_raw, X_illiq_raw)
        else:
            X_liq_n, X_illiq_n = rank_day(X_liq_raw, X_illiq_raw)

        pi_pred = fit_ridge_weighted(X_liq_n, y_liq, X_illiq_n, alpha=100, weights=w_liq)
        pi_oof[illiq_idx] = pi_pred

        if illiq_mask.sum() >= 5:
            day_r2s[day] = r2_score(y_illiq, pi_pred)
            day_ics[day] = per_day_ic(y_illiq, pi_pred)

    # Build test predictions (full day train set)
    for day, grp_te in test.groupby('day_id'):
        te_idx   = grp_te.index
        X_te_raw = grp_te[gold_feats].fillna(0).values.astype(np.float64)

        if day in train_days:
            grp_tr = train[train['day_id'] == day]
            if len(grp_tr) < 15:
                X_z = np.clip((X_te_raw - fb_mean) / fb_std, -5, 5)
                te_preds[te_idx] = fb_model.predict(X_z)
                continue

            X_tr_raw = grp_tr[gold_feats].fillna(0).values.astype(np.float64)
            y_tr_day = y_train[grp_tr.index]

            w_day = None
            if weights is not None:
                w_day = weights[grp_tr.index]
                if w_day.sum() < 1e-8:
                    w_day = None

            if norm_type == 'zscore':
                X_tr_n, X_te_n = zscore_day(X_tr_raw, X_te_raw)
            else:
                X_tr_n, X_te_n = rank_day(X_tr_raw, X_te_raw)

            te_preds[te_idx] = fit_ridge_weighted(X_tr_n, y_tr_day, X_te_n,
                                                   alpha=100, weights=w_day)
        else:
            X_z = np.clip((X_te_raw - fb_mean) / fb_std, -5, 5)
            te_preds[te_idx] = fb_model.predict(X_z)

    # Clip test predictions
    clip_b     = 3.0 * y_train.std()
    te_preds   = np.clip(te_preds, -clip_b, clip_b)

    # Aggregate metrics
    pi_mask    = ~np.isnan(pi_oof)
    pi_r2      = r2_score(y_train[pi_mask], pi_oof[pi_mask]) if pi_mask.sum() > 0 else np.nan
    pi_med_r2  = np.nanmedian(list(day_r2s.values()))
    pi_med_ic  = np.nanmedian(list(day_ics.values()))
    ic_arr     = np.array(list(day_ics.values()))
    pct_pos_ic = (ic_arr > 0).mean() * 100
    pct_neg_ic = (ic_arr < 0).mean() * 100

    pi_results[exp_name] = {
        'pi_r2': pi_r2, 'pi_med_r2': pi_med_r2,
        'pi_med_ic': pi_med_ic,
        'pct_pos_ic': pct_pos_ic, 'pct_neg_ic': pct_neg_ic,
    }
    te_preds_all[exp_name] = te_preds.copy()

    print(f"    Pseudo-illiquid R²  : {pi_r2:+.6f}")
    print(f"    Median day R²       : {pi_med_r2:+.6f}")
    print(f"    Median day IC       : {pi_med_ic:+.6f}")
    print(f"    IC dist: {pct_pos_ic:.1f}% pos  {pct_neg_ic:.1f}% neg")
    print(f"    Test pred std       : {te_preds.std():.7f}")
    print(f"    Time                : {time.time()-t_exp:.0f}s")


# ================================================================
# STEP 5: ALSO TEST HUBER LOSS + COV_LOGISTIC
# ================================================================
print("\n" + "="*65)
print("STEP 5: HUBER REGRESSION + COV_LOGISTIC (robust to kurtosis=48)")
print("="*65)

t_h = time.time()
pi_oof_h  = np.full(len(train), np.nan)
te_preds_h = np.zeros(len(test))
day_ics_h  = {}
day_r2s_h  = {}

for day, grp in train.groupby('day_id'):
    n = len(grp)
    if n < 20:
        continue

    y_day  = y_train[grp.index]
    bs_day = grp['bookshape'].values
    bs_med = np.median(bs_day)

    liq_mask   = bs_day >= bs_med
    illiq_mask = ~liq_mask
    if liq_mask.sum() < 10 or illiq_mask.sum() < 5:
        continue

    liq_idx   = grp.index[liq_mask]
    illiq_idx = grp.index[illiq_mask]

    X_liq_raw   = grp.loc[liq_idx,   gold_feats].fillna(0).values.astype(np.float64)
    X_illiq_raw = grp.loc[illiq_idx, gold_feats].fillna(0).values.astype(np.float64)
    y_liq  = y_train[liq_idx]
    y_illiq = y_train[illiq_idx]
    w_liq  = density_ratio[liq_idx]

    X_liq_n, X_illiq_n = zscore_day(X_liq_raw, X_illiq_raw)

    # Huber: simulate sample weights by integer replication
    w_int = np.clip(np.round(w_liq * 3).astype(int), 1, 15)
    idx_r = np.repeat(np.arange(len(X_liq_n)), w_int)
    y_liq_w = winsorise(y_liq)

    try:
        model = HuberRegressor(epsilon=1.5, alpha=0.01, max_iter=300)
        model.fit(X_liq_n[idx_r], y_liq_w[idx_r])
        pi_pred = model.predict(X_illiq_n)
    except Exception:
        # fallback to Ridge on failure
        pi_pred = fit_ridge_weighted(X_liq_n, y_liq, X_illiq_n, alpha=100, weights=w_liq)

    pi_oof_h[illiq_idx] = pi_pred

    if illiq_mask.sum() >= 5:
        day_r2s_h[day] = r2_score(y_illiq, pi_pred)
        day_ics_h[day]  = per_day_ic(y_illiq, pi_pred)

# Build test predictions for Huber
for day, grp_te in test.groupby('day_id'):
    te_idx   = grp_te.index
    X_te_raw = grp_te[gold_feats].fillna(0).values.astype(np.float64)

    if day in train_days:
        grp_tr = train[train['day_id'] == day]
        if len(grp_tr) < 15:
            X_z = np.clip((X_te_raw - m_l) / s_l, -5, 5)
            te_preds_h[te_idx] = fb_logistic.predict(X_z)
            continue

        X_tr_raw = grp_tr[gold_feats].fillna(0).values.astype(np.float64)
        y_tr_day = y_train[grp_tr.index]
        w_day    = density_ratio[grp_tr.index]
        w_int    = np.clip(np.round(w_day * 3).astype(int), 1, 15)
        idx_r    = np.repeat(np.arange(len(X_tr_raw)), w_int)

        X_tr_n, X_te_n = zscore_day(X_tr_raw, X_te_raw)

        try:
            model = HuberRegressor(epsilon=1.5, alpha=0.01, max_iter=300)
            model.fit(X_tr_n[idx_r], winsorise(y_tr_day)[idx_r])
            te_preds_h[te_idx] = model.predict(X_te_n)
        except Exception:
            te_preds_h[te_idx] = fit_ridge_weighted(X_tr_n, y_tr_day, X_te_n,
                                                     alpha=100, weights=w_day)
    else:
        X_z = np.clip((X_te_raw - m_l) / s_l, -5, 5)
        te_preds_h[te_idx] = fb_logistic.predict(X_z)

te_preds_h = np.clip(te_preds_h, -3.0*y_train.std(), 3.0*y_train.std())

pi_mask_h   = ~np.isnan(pi_oof_h)
pi_r2_h     = r2_score(y_train[pi_mask_h], pi_oof_h[pi_mask_h])
pi_med_r2_h = np.nanmedian(list(day_r2s_h.values()))
pi_med_ic_h = np.nanmedian(list(day_ics_h.values()))
ic_arr_h    = np.array(list(day_ics_h.values()))
pi_results['huber_cov'] = {
    'pi_r2': pi_r2_h, 'pi_med_r2': pi_med_r2_h,
    'pi_med_ic': pi_med_ic_h,
    'pct_pos_ic': (ic_arr_h > 0).mean()*100,
    'pct_neg_ic': (ic_arr_h < 0).mean()*100,
}
te_preds_all['huber_cov'] = te_preds_h.copy()

print(f"  Pseudo-illiquid R²  : {pi_r2_h:+.6f}")
print(f"  Median day R²       : {pi_med_r2_h:+.6f}")
print(f"  Median day IC       : {pi_med_ic_h:+.6f}")
print(f"  IC: {(ic_arr_h>0).mean()*100:.1f}% pos  {(ic_arr_h<0).mean()*100:.1f}% neg")
print(f"  Test pred std       : {te_preds_h.std():.7f}")
print(f"  Time                : {time.time()-t_h:.0f}s")


# ================================================================
# STEP 6: FULL COMPARISON TABLE
# ================================================================
print("\n" + "="*65)
print("FULL RESULTS: Covariate Shift Experiments")
print("="*65)
print(f"""
Known LB scores:
  fold_safe_v1        : LB = +0.00005  (best)
  transductive_v4_005 : LB = +0.00003
  knn_K3_3pct         : LB = -0.00042  (worst)
  lgbm_baseline_v1    : LB = -0.00002

Pseudo-illiquid R² from previous run (no weights):
  gold_z  : PI R² = -0.100   Med IC = +0.022   62.4% pos IC
  gold_r  : PI R² = -0.029   Med IC = +0.029   64.0% pos IC
  grinold : PI R² = -411     Med IC = +0.056   76.5% pos IC  ← cleanest signal
""")

print(f"  {'Experiment':<16}  {'PI R²':>10}  {'Med Day R²':>12}  {'Med IC':>10}  {'%pos IC':>8}  {'%neg IC':>8}  {'Te std':>10}")
print(f"  {'-'*82}")

for name, res in pi_results.items():
    te = te_preds_all[name]
    print(f"  {name:<16}  {res['pi_r2']:+10.6f}  {res['pi_med_r2']:+12.6f}  "
          f"{res['pi_med_ic']:+10.6f}  {res['pct_pos_ic']:7.1f}%  "
          f"{res['pct_neg_ic']:7.1f}%  {te.std():10.7f}")


# ================================================================
# STEP 7: SAVE SUBMISSIONS
# ================================================================
print("\n" + "="*65)
print("STEP 7: SAVING SUBMISSIONS (5% scale)")
print("="*65)

sample_sub = pd.read_csv(os.path.join(BASE_DIR, 'data/raw/sample_submission.csv'))[['ID']]

def save_sub(preds, name, scale=0.05):
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': preds * scale})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    path = os.path.join(OUT_DIR, f'{name}.csv')
    sub.to_csv(path, index=False)
    t = sub['TARGET']
    print(f"  {name}: std={t.std():.7f}  mean={t.mean():+.8f}  "
          f"pct_pos={(t>0).mean()*100:.1f}%  skew={t.skew():+.3f}")
    return t.std()

print("\n  Saving at 5% scale for all variants:")
stds = {}
for name in pi_results:
    stds[name] = save_sub(te_preds_all[name], f'covshift_{name}_5pct')

# Best variant by PI R²
best = max(pi_results, key=lambda k: pi_results[k]['pi_r2'])
print(f"\n  Best by pseudo-illiquid R²: {best} (PI R² = {pi_results[best]['pi_r2']:+.6f})")
print(f"  Best by pseudo-illiquid IC: "
      f"{max(pi_results, key=lambda k: pi_results[k]['pi_med_ic'])}")
print(f"  Best by % positive IC days: "
      f"{max(pi_results, key=lambda k: pi_results[k]['pct_pos_ic'])}")

print(f"""
INTERPRETATION:
───────────────
Improvement in PI R² vs baseline confirms covariate reweighting works.
Improvement in % positive IC days is equally important.

What to watch:
  • If cov_logistic PI R² > baseline PI R²  → density ratio helps
  • If cov_bookshape PI R² > cov_logistic   → within-day proxy is better
  • If huber_cov PI R² > cov_logistic       → Huber loss adds value
  • If any PI R² > 0                         → calibration is now correct
  • If all PI R² < 0 but IC improved         → still need scale shrinkage

Reference: gold_r baseline had PI R²=-0.029, IC=+0.029, 64% positive days.
""")

print(f"Total elapsed: {(time.time()-t0)/60:.1f} min")
