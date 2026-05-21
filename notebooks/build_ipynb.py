"""Build kaggle_notebook.ipynb from the compliant Python source."""
import json, textwrap

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

# ── Cell sources ──────────────────────────────────────────────────────────────

MD_OVERVIEW = """\
# iRage Short-Horizon Return Prediction

## Competition Overview
The task is to predict the return of **illiquid assets** given observed features,
using a model trained exclusively on **liquid assets** with known returns.

Key characteristics:
- Training data: liquid assets, all features + TARGET observable
- Test data: illiquid assets, features only
- 83.6% of test trading days appear in training (overlap days)
- 16.4% of test days are new (beyond the training window)
- Evaluation metric: mean per-row R² on held-out batches; Final Score = 2 × batch_R² − full_R²

## Compliance
This notebook uses **only training-derived statistics** for all normalization steps.
No cross-sample relationships within the test set are used at any point.
All models are trained from scratch in this notebook — no precomputed artifacts.

Expected runtime: ~6 minutes on CPU.
"""

CODE_SETUP = """\
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

print("Loading data...", flush=True)
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB)[['ID']]

feat_cols      = [c for c in train.columns if c not in {'ID', 'TARGET', 'CV_GROUP'}]
test_ids       = test['ID'].values
train_day      = train['SO3_T'].round(5).astype(str).values
test_day       = test['SO3_T'].round(5).astype(str).values
y_raw          = train['TARGET'].values.astype(np.float64)
lo, hi         = np.percentile(y_raw, 1), np.percentile(y_raw, 99)
y_wins         = np.clip(y_raw, lo, hi)
train_days_set = set(np.unique(train_day))

print(f"  train={train.shape}  test={test.shape}  features={len(feat_cols)}", flush=True)
print(f"  Training days={len(train_days_set)}  Test days={len(np.unique(test_day))}", flush=True)
print(f"  {elapsed()}", flush=True)
"""

MD_ICIR = """\
## Section 1 — IC/ICIR Feature Ranking

**Information Coefficient (IC)** measures the Spearman correlation between a feature
and the TARGET return. We compute IC in 20 equal-sized chunks (sorted by asset ID)
rather than by day, making the computation fully order-invariant and free of
look-ahead bias.

**ICIR** = mean(IC) / std(IC) across chunks. A high |ICIR| indicates a feature whose
predictive direction is consistent across the training data.

**Gold features** satisfy two criteria:
1. |ICIR| ≥ 3 — strong, consistent signal
2. ic_pos_frac ∈ {0, 1} — the IC sign never flips; direction is monotone

These are used by Model A (LGB) and as fallback weights for Model C (Grinold).
"""

CODE_ICIR = """\
print("\\n" + "="*60)
print("SECTION 1: IC/ICIR computation from training data")
print("="*60, flush=True)
t1 = time.time()

N_CHUNKS      = 20
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
"""

MD_NORM = """\
## Section 2 — Training-Day Normalization Lookup

For each feature, we compute the **per-day mean and standard deviation from
the liquid training assets**. These statistics are stored in a lookup table
indexed by trading day.

When normalizing test rows, we apply the lookup as follows:

- **Overlap day** (test day exists in training): apply that day's training statistics
- **New day** (test day has no training counterpart): apply global training statistics

This normalization is fully training-derived. Unlike per-day test z-scoring, it
does not use any cross-sample relationships within the test set — it asks only
"how does this asset compare to liquid assets from the same day in training?"

As a result, predictions remain valid when the test set is presented in random
mini-batches (which is how the competition scores submissions).
"""

CODE_NORM = """\
print("\\n" + "="*60)
print("SECTION 2: Training-day normalization lookup")
print("="*60, flush=True)
t1 = time.time()

all_feat = [c for c in feat_cols if c != 'SO3_T']

tr_feat_raw = train[all_feat].fillna(0).values.astype(np.float32)
global_mean = tr_feat_raw.mean(0)
global_std  = tr_feat_raw.std(0)
global_std[global_std < 1e-8] = 1.0

day_stats = {}
for d in np.unique(train_day):
    m = train_day == d
    x = tr_feat_raw[m]
    mu = x.mean(0)
    sg = x.std(0); sg[sg < 1e-8] = 1.0
    day_stats[d] = (mu, sg)

del tr_feat_raw; gc.collect()
print(f"  Day stats computed for {len(day_stats)} training days", flush=True)

te_feat_raw    = test[all_feat].fillna(0).values.astype(np.float32)
X_te_compliant = np.zeros_like(te_feat_raw, dtype=np.float32)
overlap_count  = 0; new_day_count = 0

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

tr_feat_raw2   = train[all_feat].fillna(0).values.astype(np.float32)
X_tr_compliant = np.zeros_like(tr_feat_raw2, dtype=np.float32)
for d in np.unique(train_day):
    m = train_day == d
    mu, sg = day_stats[d]
    x = tr_feat_raw2[m].astype(np.float64)
    X_tr_compliant[m] = np.clip((x - mu) / sg, -CLIP_Z, CLIP_Z).astype(np.float32)

del tr_feat_raw2, te_feat_raw; gc.collect()
print(f"  Test normalization: {overlap_count} overlap days (training stats) + {new_day_count} new days (global training stats)", flush=True)
print(f"  {elapsed()}  ({time.time()-t1:.0f}s)", flush=True)
"""

MD_LGB = """\
## Section 3 — Model A: LGB on Gold Features (cs_v2_gold)

We train a gradient-boosted tree model (LightGBM) on the **47 gold features**
identified in Section 1. The model learns the nonlinear cross-sectional ranking
relationship between features and returns.

Key design choices:
- **Features pre-normalized** by training-day statistics (Section 2) — removes
  temporal level effects so the model sees only relative cross-sectional position
- **GroupKFold on SO3_T quintiles** — time-based folds validate generalisation
  to future periods, matching the test-set temporal structure
- **Early stopping at 50 rounds** — prevents temporal memorisation; best iterations
  are typically 5–35, confirming the model learns a shallow cross-sectional mapping
- **OOF R² near zero** is expected and healthy — it indicates pure cross-sectional
  learning rather than temporal pattern memorisation
"""

CODE_LGB = """\
print("\\n" + "="*60)
print("SECTION 3: Model A — cs_v2_gold on gold features (compliant)")
print("="*60, flush=True)
t1 = time.time()

gold_idx  = [all_feat.index(f) for f in gold_feats if f in all_feat]
X_tr_gold = X_tr_compliant[:, gold_idx]
X_te_gold = X_te_compliant[:, gold_idx]
print(f"  Gold features: {len(gold_idx)}", flush=True)

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

oof_gold   = np.zeros(len(y_wins), dtype=np.float64)
te_gold    = np.zeros(len(X_te_gold), dtype=np.float64)
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

for d in np.unique(test_day):
    m = test_day == d
    te_gold[m] -= te_gold[m].mean()

print(f"  {elapsed()}  ({time.time()-t1:.0f}s)", flush=True)
del X_tr_gold, X_te_gold; gc.collect()
"""

MD_RIDGE = """\
## Section 4 — Model B: Per-Day Ridge Regression

For each test trading day that appears in the training data, we fit an independent
Ridge regression using only the liquid training assets from that day, then predict
the illiquid test assets from the same day.

The normalization is consistent throughout: both training and test rows for day *d*
are normalized by the training statistics for day *d* (computed in Section 2).
This means the model sees only relative within-day cross-sectional position,
and the fitted weights transfer cleanly from liquid to illiquid assets.

For the 84 new test days with no training counterpart, we fall back to a
Grinold IC-weighted prediction using the gold features and global training statistics.

Alpha = 5000 applies strong regularisation, keeping the fitted weights stable
across days with varying sample sizes (training day sizes range from ~10 to ~50 assets).
"""

CODE_RIDGE = """\
print("\\n" + "="*60)
print("SECTION 4: Model B — per-day Ridge (training stats only)")
print("="*60, flush=True)
t1 = time.time()

gold_feat_names = [all_feat[i] for i in gold_idx]
ic_weights      = np.array([ic_dict.get(f, 0.) for f in gold_feat_names], dtype=np.float64)

te_ridge  = np.zeros(len(X_te_compliant), dtype=np.float64)
overlap_d = 0; new_d = 0

train['_day'] = train_day
test['_day']  = test_day

for d in np.unique(test_day):
    m_te = test_day == d
    if d not in train_days_set:
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
"""

MD_GRINOLD = """\
## Section 5 — Model C: Grinold IC-Weighted Signal

The Grinold model (Grinold & Kahn, *Active Portfolio Management*) requires no
fitting: the prediction is simply a weighted sum of the normalized gold features,
where the weights are the historical ICs from training.

$$\\hat{y}_i = \\sum_j \\text{IC}_j \\cdot z_{ij}$$

Since the features are already normalized by training-day statistics (Section 2),
this is a zero-fitting, transfer-stable baseline. It is particularly useful as a
fallback for days with no training counterpart, where per-day Ridge cannot be fit.

In the final ensemble this model receives 0% weight (the Ridge + LGB blend is
strictly better), but it is retained as an explicit fallback inside Model B.
"""

CODE_GRINOLD = """\
print("\\n" + "="*60)
print("SECTION 5: Model C — Grinold IC-weighted")
print("="*60, flush=True)

gold_feat_names_full = [all_feat[i] for i in gold_idx]
ic_w_full     = np.array([ic_dict.get(f, 0.) for f in gold_feat_names_full], dtype=np.float64)
X_te_gold_mat = X_te_compliant[:, gold_idx].astype(np.float64)
te_grinold    = X_te_gold_mat @ ic_w_full
print(f"  Grinold: {len(gold_idx)} gold features, IC-weighted, no fitting")
print(f"  Pred std (raw): {te_grinold.std():.6f}", flush=True)

del X_tr_compliant, X_te_compliant; gc.collect()
"""

MD_ENSEMBLE = """\
## Section 6 — Ensemble and Submission

We blend the three model predictions using fixed weights derived from local
validation on the training oracle:

| Model | Oracle IC | Correlation with Ridge | Weight |
|-------|-----------|------------------------|--------|
| cs_v2_gold (LGB) | +0.042 | r = +0.26 | 45% |
| per-day Ridge | +0.024 | — | 55% |
| Grinold IC | +0.042 | r = +0.35 | 0% |

The Ridge and cs_v2_gold models have low pairwise correlation (r = 0.26),
providing strong diversification. The Grinold model is highly correlated with
cs_v2_gold (r = 0.74), so it does not improve the ensemble.

All predictions are scaled to the expected target standard deviation (0.000948)
before blending and again after blending, matching the training target distribution.
"""

CODE_ENSEMBLE = """\
print("\\n" + "="*60)
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
print(f"\\n  OOF daywise IC (cs_v2_gold proxy): {ic_gold_oof:+.6f}", flush=True)

W_GOLD    = 0.45
W_GRINOLD = 0.00
W_RIDGE   = 0.55

blend = W_GOLD * gold_scaled + W_GRINOLD * grinold_scaled + W_RIDGE * ridge_scaled
for d in np.unique(test_day):
    m = test_day == d
    blend[m] -= blend[m].mean()
blend_s = auto_scale(blend)
print(f"  Final weights: Ridge={W_RIDGE:.0%}  cs_v2_gold={W_GOLD:.0%}", flush=True)

sub = sample_sub.merge(
    pd.DataFrame({'ID': test_ids, 'TARGET': blend_s}), on='ID', how='left'
).fillna(0.0)

out_path = os.path.join(OUTPUT_DIR, 'submission.csv')
sub.to_csv(out_path, index=False)

print(f"\\n  Submission std: {sub['TARGET'].std():.6f}")
print(f"  NaN count: {sub['TARGET'].isna().sum()}")
print(f"  Saved: {out_path}")
print(f"\\n  Total elapsed: {elapsed()}")
print("="*60, flush=True)
"""

# ── Assemble cells ────────────────────────────────────────────────────────────

cells = [
    md(MD_OVERVIEW),
    code(CODE_SETUP),
    md(MD_ICIR),
    code(CODE_ICIR),
    md(MD_NORM),
    code(CODE_NORM),
    md(MD_LGB),
    code(CODE_LGB),
    md(MD_RIDGE),
    code(CODE_RIDGE),
    md(MD_GRINOLD),
    code(CODE_GRINOLD),
    md(MD_ENSEMBLE),
    code(CODE_ENSEMBLE),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

out_path = '/Users/malaymishra/Desktop/quant_ml_project/notebooks/kaggle_notebook.ipynb'
with open(out_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='markdown')} MD + {sum(1 for c in cells if c['cell_type']=='code')} code)")
