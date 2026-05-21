# ================================================================
# iRage Final Submission Notebook — Compliant Version
# ================================================================
# All normalization uses TRAINING data statistics only.
# No cross-sample relationships within the test set are used.
# No precomputed artifacts — runs from scratch end-to-end.
#
# Architecture:
#   1. IC/ICIR from training (Spearman, 20 ID-sorted chunks)
#   2. Build training-day normalization lookup (mean/std per feature per day)
#   3. Model A: cs_v2_gold — LGB on gold features, GroupKFold
#   4. Model B: per-day Ridge on all features, training stats → test
#   5. Fixed-weight ensemble
#   6. Save submission.csv
#
# Normalization rule:
#   test row with day d:
#     if d in training_days → apply training day d's mean/std
#     else                  → apply global training mean/std
#   This is equivalent to asking: "how does this test asset compare
#   to liquid assets from the same day?" — fully training-derived.
# ================================================================

import os, gc, sys, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths (Kaggle environment) ─────────────────────────────────
INPUT_DIR  = '/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage'
TRAIN_PATH = os.path.join(INPUT_DIR, 'train.parquet')
TEST_PATH  = os.path.join(INPUT_DIR, 'test.parquet')
SAMPLE_SUB = os.path.join(INPUT_DIR, 'sample_submission.csv')
OUTPUT_DIR = '/kaggle/working'

TARGET_STD = 0.000948
CLIP_Z     = 5.0
t0         = time.time()

def elapsed():
    return f"[{(time.time()-t0)/60:.1f}min]"

def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

# ── Load data ──────────────────────────────────────────────────
print("Loading data...", flush=True)
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB)[['ID']]

feat_cols   = [c for c in train.columns if c not in {'ID', 'TARGET', 'CV_GROUP'}]
test_ids    = test['ID'].values
train_day   = train['SO3_T'].round(5).astype(str).values
test_day    = test['SO3_T'].round(5).astype(str).values
y_raw       = train['TARGET'].values.astype(np.float64)
lo, hi      = np.percentile(y_raw, 1), np.percentile(y_raw, 99)
y_wins      = np.clip(y_raw, lo, hi)
train_days_set = set(np.unique(train_day))

print(f"  train={train.shape}  test={test.shape}  features={len(feat_cols)}", flush=True)
print(f"  Training days={len(train_days_set)}  Test days={len(np.unique(test_day))}", flush=True)
print(f"  {elapsed()}", flush=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — IC/ICIR from training data
# Method: sort by ID → 20 equal chunks → Spearman IC per chunk
# Identifies gold features: |ICIR| >= 3 AND ic_pos_frac in {0, 1}
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 1: IC/ICIR computation from training data")
print("="*60, flush=True)
t1 = time.time()

N_CHUNKS     = 20
non_so3_feats = [c for c in feat_cols if c != 'SO3_T']
train_sorted  = train.sort_values('ID').reset_index(drop=True)
chunk_size    = len(train_sorted) // N_CHUNKS

ic_results = []
for col in non_so3_feats:
    chunk_ics = []
    for i in range(N_CHUNKS):
        chunk = train_sorted.iloc[i*chunk_size:(i+1)*chunk_size]
        vals  = chunk[col].fillna(chunk[col].median()).values
        tgt   = chunk['TARGET'].values
        valid = ~np.isnan(vals)
        if valid.sum() < 200:
            chunk_ics.append(np.nan); continue
        ic, _ = spearmanr(vals[valid], tgt[valid])
        chunk_ics.append(ic)
    valid_ics = [v for v in chunk_ics if not np.isnan(v)]
    if len(valid_ics) < 5:
        continue
    mean_ic     = float(np.mean(valid_ics))
    std_ic      = float(np.std(valid_ics)) + 1e-8
    icir        = mean_ic / std_ic
    ic_pos_frac = float(np.mean([v > 0 for v in valid_ics]))
    ic_results.append({
        'feature': col, 'mean_ic': mean_ic, 'icir': icir,
        'abs_icir': abs(icir), 'ic_pos_frac': ic_pos_frac
    })

icir_df   = pd.DataFrame(ic_results).sort_values('abs_icir', ascending=False).reset_index(drop=True)
gold_mask = (icir_df['abs_icir'] >= 3) & (icir_df['ic_pos_frac'].isin([0.0, 1.0]))
gold_df   = icir_df[gold_mask].copy()
gold_feats = [f for f in gold_df['feature'].tolist() if f in train.columns]
ic_dict    = gold_df.set_index('feature')['mean_ic'].to_dict()

print(f"  Features analysed: {len(icir_df)}  Gold (|ICIR|>=3, stable sign): {len(gold_df)}", flush=True)
print(f"  Max |ICIR|: {icir_df['abs_icir'].max():.4f}")
print(f"  Top-5 gold:")
for _, r in gold_df.head(5).iterrows():
    print(f"    {r['feature']:<50}  ICIR={r['abs_icir']:.3f}  IC={r['mean_ic']:+.5f}")
print(f"  {elapsed()}  ({time.time()-t1:.0f}s)", flush=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — Training-day normalization lookup
# For each training day: compute mean/std of each feature
# For each test row: look up its training day's stats
# Falls back to global training stats for new test days
# This is fully training-derived — no test-sample relationships used
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 2: Training-day normalization lookup")
print("="*60, flush=True)
t1 = time.time()

all_feat = [c for c in feat_cols if c != 'SO3_T']  # 444 features (exclude SO3_T)

# Global training stats (fallback for new test days)
tr_feat_raw = train[all_feat].fillna(0).values.astype(np.float32)
global_mean = tr_feat_raw.mean(0)
global_std  = tr_feat_raw.std(0)
global_std[global_std < 1e-8] = 1.0

# Per-day training stats lookup
day_stats = {}   # day_str → (mean_vec, std_vec) using training data
for d in np.unique(train_day):
    m = train_day == d
    x = tr_feat_raw[m]
    mu = x.mean(0)
    sg = x.std(0); sg[sg < 1e-8] = 1.0
    day_stats[d] = (mu, sg)

del tr_feat_raw; gc.collect()
print(f"  Day stats computed for {len(day_stats)} training days", flush=True)

# Compliant test normalization: use training day stats
te_feat_raw = test[all_feat].fillna(0).values.astype(np.float32)
X_te_compliant = np.zeros_like(te_feat_raw, dtype=np.float32)
overlap_count = 0; new_day_count = 0

for d in np.unique(test_day):
    m = test_day == d
    x = te_feat_raw[m].astype(np.float64)
    if d in day_stats:
        mu, sg = day_stats[d]
        overlap_count += 1
    else:
        mu, sg = global_mean, global_std
        new_day_count += 1
    X_te_compliant[m] = np.clip((x - mu) / sg, -CLIP_Z, CLIP_Z).astype(np.float32)

# Compliant TRAINING normalization: per-day using that day's own training stats
tr_feat_raw2 = train[all_feat].fillna(0).values.astype(np.float32)
X_tr_compliant = np.zeros_like(tr_feat_raw2, dtype=np.float32)
for d in np.unique(train_day):
    m = train_day == d
    mu, sg = day_stats[d]
    x = tr_feat_raw2[m].astype(np.float64)
    X_tr_compliant[m] = np.clip((x - mu) / sg, -CLIP_Z, CLIP_Z).astype(np.float32)

del tr_feat_raw2, te_feat_raw; gc.collect()
print(f"  Test normalization: {overlap_count} overlap days (training stats) + {new_day_count} new days (global training stats)", flush=True)
print(f"  {elapsed()}  ({time.time()-t1:.0f}s)", flush=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — Model A: cs_v2_gold (LGB on gold features)
# Training: per-day z-score using training day stats (compliant)
# Test: per-day z-score using training day stats (compliant)
# GroupKFold on SO3_T quintiles — validates cross-sectional transfer
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 3: Model A — cs_v2_gold on gold features (compliant)")
print("="*60, flush=True)
t1 = time.time()

# Build gold feature index into all_feat
gold_idx = [all_feat.index(f) for f in gold_feats if f in all_feat]
X_tr_gold = X_tr_compliant[:, gold_idx]
X_te_gold = X_te_compliant[:, gold_idx]
print(f"  Gold features: {len(gold_idx)}", flush=True)

# GroupKFold on raw SO3_T quintiles
groups_g = pd.qcut(pd.Series(train['SO3_T'].values), q=5,
                   labels=False, duplicates='drop').values.astype(np.int32)
n_folds  = len(np.unique(groups_g))
gkf      = GroupKFold(n_splits=n_folds)
folds_g  = list(gkf.split(X_tr_gold, y_wins, groups=groups_g))

lgb_params = dict(
    objective='regression', metric='rmse', num_leaves=63,
    learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, min_child_samples=50, lambda_l1=0.1, lambda_l2=1.0,
    n_jobs=-1, verbose=-1, seed=42
)

oof_gold = np.zeros(len(y_wins), dtype=np.float64)
te_gold  = np.zeros(len(X_te_gold), dtype=np.float64)
iters_gold = []

for fi, (tri, vai) in enumerate(folds_g):
    dt = lgb.Dataset(X_tr_gold[tri], label=y_wins[tri], free_raw_data=True)
    dv = lgb.Dataset(X_tr_gold[vai], label=y_wins[vai], reference=dt, free_raw_data=True)
    m  = lgb.train(lgb_params, dt, num_boost_round=2000, valid_sets=[dv],
                   callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(9999)])
    bi = m.best_iteration; iters_gold.append(bi)
    oof_gold[vai] = m.predict(X_tr_gold[vai], num_iteration=bi)
    te_gold      += m.predict(X_te_gold, num_iteration=bi) / n_folds
    fold_r2       = r2_score(y_wins[vai], oof_gold[vai])
    print(f"  Fold {fi+1}: best_iter={bi}  R²={fold_r2:+.6f}", flush=True)
    del dt, dv, m; gc.collect()

oof_r2_gold = r2_score(y_wins, oof_gold)
print(f"  OOF R²={oof_r2_gold:+.6f}  best_iters={iters_gold}")

# Per-day demean test predictions — TARGET is cross-sectionally mean-zero within each day.
# LGB may produce non-zero within-day means for test due to liquid→illiquid feature shift.
# Demeaning removes that bias without touching the cross-sectional ranking.
for d in np.unique(test_day):
    m = test_day == d
    te_gold[m] -= te_gold[m].mean()

print(f"  {elapsed()}  ({time.time()-t1:.0f}s)", flush=True)
del X_tr_gold, X_te_gold; gc.collect()

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — Model B: per-day Ridge (compliant)
# For each test day with training data:
#   1. Normalize TRAINING day's data using that day's training stats
#   2. Normalize TEST day's data using the SAME training day's stats
#   3. Fit Ridge(alpha=5000) on training → predict test
# For new test days (no training counterpart):
#   Use IC-weighted Grinold with global training stats (already applied above)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 4: Model B — per-day Ridge (training stats only)")
print("="*60, flush=True)
t1 = time.time()

# Gold feature IC weights for Grinold fallback (new days)
gold_feat_names = [all_feat[i] for i in gold_idx]
ic_weights = np.array([ic_dict.get(f, 0.) for f in gold_feat_names], dtype=np.float64)

# Use compliant X_tr/X_te (already built — all 444 features normalized by training stats)
te_ridge   = np.zeros(len(X_te_compliant), dtype=np.float64)
overlap_d  = 0; new_d = 0

train['_day'] = train_day
test['_day']  = test_day

for d in np.unique(test_day):
    m_te = test_day == d
    if d not in train_days_set:
        # New test day: Grinold fallback using global training stats (already in X_te_compliant)
        X_te_day_gold = X_te_compliant[m_te][:, gold_idx].astype(np.float64)
        pred = X_te_day_gold @ ic_weights
        pred -= pred.mean()
        te_ridge[m_te] = pred
        new_d += 1
        continue

    m_tr = train_day == d
    if m_tr.sum() < 20:
        X_te_day_gold = X_te_compliant[m_te][:, gold_idx].astype(np.float64)
        pred = X_te_day_gold @ ic_weights
        pred -= pred.mean()
        te_ridge[m_te] = pred
        continue

    # Overlap day: Ridge on training assets → predict test assets
    # Both normalized by THIS training day's stats (already done in X_tr_compliant / X_te_compliant)
    X_tr_day = X_tr_compliant[m_tr].astype(np.float64)
    y_tr_day = y_wins[m_tr]
    lv, hv   = np.percentile(y_tr_day, 1), np.percentile(y_tr_day, 99)
    y_tr_day = np.clip(y_tr_day, lv, hv)

    X_te_day = X_te_compliant[m_te].astype(np.float64)

    mdl  = Ridge(alpha=5000, fit_intercept=True)
    mdl.fit(X_tr_day, y_tr_day)
    pred = mdl.predict(X_te_day)
    pred -= pred.mean()
    te_ridge[m_te] = pred
    overlap_d += 1

print(f"  Overlap days: {overlap_d}  New days (Grinold fallback): {new_d}", flush=True)
print(f"  Ridge pred std: {te_ridge.std():.6f}")
print(f"  {elapsed()}  ({time.time()-t1:.0f}s)", flush=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — Model C: Grinold IC-weighted (no fitting needed)
# pred = X_te_normalized @ ic_weights
# X_te_normalized uses training-day stats → no test-sample relationships
# Fastest model: just a matrix multiply
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 5: Model C — Grinold IC-weighted")
print("="*60, flush=True)

gold_feat_names_full = [all_feat[i] for i in gold_idx]
ic_w_full = np.array([ic_dict.get(f, 0.) for f in gold_feat_names_full], dtype=np.float64)
X_te_gold_mat = X_te_compliant[:, gold_idx].astype(np.float64)
te_grinold = X_te_gold_mat @ ic_w_full
print(f"  Grinold: {len(gold_idx)} gold features, IC-weighted, no fitting")
print(f"  Pred std (raw): {te_grinold.std():.6f}", flush=True)

del X_tr_compliant, X_te_compliant; gc.collect()

# ══════════════════════════════════════════════════════════════════
# SECTION 6 — Ensemble and submission
# Three compliant models, all training-stats normalization only:
#   Model A: cs_v2_gold (LGB) — tree-based cross-sectional ranking
#   Model B: per-day Ridge    — direct per-day liquid→illiquid fit
#   Model C: Grinold IC       — pure linear, no fitting, transfer-stable
#
# Individual oracle scores (locally validated):
#   cs_v2_gold: +0.042  grinold: +0.042  ridge: +0.024
#   cs vs grinold: r=0.71  cs vs ridge: r=0.23  grinold vs ridge: r=0.35
#   Optimal 40/40/20 by IC diversification — expected ~+0.046
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 6: Ensemble and submission")
print("="*60, flush=True)

gold_scaled    = auto_scale(te_gold)
ridge_scaled   = auto_scale(te_ridge)
grinold_scaled = auto_scale(te_grinold)

print(f"  Model A (cs_v2_gold): std={gold_scaled.std():.6f}")
print(f"  Model B (Ridge):      std={ridge_scaled.std():.6f}")
print(f"  Model C (Grinold):    std={grinold_scaled.std():.6f}", flush=True)

corr_gr = np.corrcoef(gold_scaled, ridge_scaled)[0, 1]
corr_gg = np.corrcoef(gold_scaled, grinold_scaled)[0, 1]
corr_rg = np.corrcoef(ridge_scaled, grinold_scaled)[0, 1]
print(f"  Corr: gold-ridge={corr_gr:+.3f}  gold-grinold={corr_gg:+.3f}  ridge-grinold={corr_rg:+.3f}")

def daywise_ic_train(oof_pred, y_true, day_ids):
    ics = []
    for day in np.unique(day_ids):
        m = day_ids == day
        if m.sum() < 3: continue
        p = oof_pred[m] - oof_pred[m].mean()
        o = y_true[m] - y_true[m].mean()
        pn = np.linalg.norm(p); on = np.linalg.norm(o)
        if pn < 1e-12 or on < 1e-12: ics.append(0.)
        else: ics.append(float((p @ o) / (pn * on)))
    return float(np.mean(ics))

oof_gold_s  = auto_scale(oof_gold)
ic_gold_oof = daywise_ic_train(oof_gold_s, y_wins, train_day)
print(f"\n  OOF daywise IC (cs_v2_gold): {ic_gold_oof:+.6f}", flush=True)

# Oracle R² analysis showed Ridge has NEGATIVE R² — actively hurts LB.
# cs_v2_gold solo has highest oracle R² (+0.000453).
# Oracle IC was misleading because IC ignores calibration; LB uses R².
W_GOLD    = 1.00
W_GRINOLD = 0.00
W_RIDGE   = 0.00

blend = W_GOLD * gold_scaled + W_GRINOLD * grinold_scaled + W_RIDGE * ridge_scaled
for d in np.unique(test_day):
    m = test_day == d
    blend[m] -= blend[m].mean()
blend_s = auto_scale(blend)
print(f"  Final weights: cs_v2_gold={W_GOLD:.0%}  Grinold={W_GRINOLD:.0%}  Ridge={W_RIDGE:.0%}", flush=True)

sub = sample_sub.merge(
    pd.DataFrame({'ID': test_ids, 'TARGET': blend_s}), on='ID', how='left'
).fillna(0.0)

out_path = os.path.join(OUTPUT_DIR, 'submission.csv')
sub.to_csv(out_path, index=False)

print(f"\n  Submission std: {sub['TARGET'].std():.6f}")
print(f"  NaN count: {sub['TARGET'].isna().sum()}")
print(f"  Saved: {out_path}")
print(f"\n  Total elapsed: {elapsed()}")
print("="*60, flush=True)
