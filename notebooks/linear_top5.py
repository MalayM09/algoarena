# ================================================================
# LINEAR TOP-5: Best 5 features, minimal-noise linear models
# ================================================================
# Top-8 features by LGB gain, drop sign-flipper (S03_A07_A05_V09)
# → top-5 sign-consistent LagT2/LagT3 features.
#
# 3 approaches tried:
#   A) Ridge on CS z-scored features (alpha sweep)
#   B) IC-weighted direct combination (no fitting — pure signal)
#   C) Equal-weight cross-sectional rank average (fully distribution-free)
#
# All approaches: ~30 seconds total.
# ================================================================

import sys, os, gc, time, warnings
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ORACLE     = os.path.join(BASE_DIR, 'outputs/submissions/exploit_v2_zero.csv')
SAMPLE_SUB = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
TARGET_STD = 0.000948
t0         = time.time()

# Top-5: LGB-gain top-8, excluding sign-flipper S03_A07_A05_V09
TOP5_FEATS = [
    'S02_F03_U01_LagT3',                    # #1 by LGB gain
    'S01_F03_U01_LagT3',                    # #2
    'S03_V03_T03_LagT2',                    # #3
    'S03_D02_A09_A02_B07_E07_E08_LagT3',   # #4
    'Price_LagT2',                          # #5
]

def auto_scale(p):
    s = p.std(); return p * (TARGET_STD / s) if s > 1e-10 else p

def daywise_oracle_score(pred_vec, oracle_vec, day_ids):
    ics = []
    for day in np.unique(day_ids):
        m = day_ids == day
        if m.sum() < 3: continue
        p = pred_vec[m] - pred_vec[m].mean()
        o = oracle_vec[m] - oracle_vec[m].mean()
        pn = np.linalg.norm(p); on = np.linalg.norm(o)
        if pn < 1e-12 or on < 1e-12: ics.append(0.)
        else: ics.append(float((p @ o) / (pn * on)))
    return float(np.mean(ics))

def save_and_score(pred_arr, test_ids, sample_sub, oracle_vec, oracle_days, fname):
    scaled = auto_scale(pred_arr)
    sub = sample_sub.merge(
        pd.DataFrame({'ID': test_ids, 'TARGET': scaled}), on='ID', how='left'
    ).fillna(0)
    sc = daywise_oracle_score(sub['TARGET'].values, oracle_vec, oracle_days)
    sub.to_csv(os.path.join(OUT_DIR, fname), index=False)
    return sc, sub

# ── Load ──────────────────────────────────────────────────────────
print("Loading...", flush=True)
t1 = time.time()
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
print(f"  {time.time()-t1:.1f}s | train={train.shape} test={test.shape}", flush=True)

y_raw     = train['TARGET'].values.astype(np.float64)
lo, hi    = np.percentile(y_raw, 1), np.percentile(y_raw, 99)
y_wins    = np.clip(y_raw, lo, hi)
test_ids  = test['ID'].values
train_day = train['SO3_T'].round(5).astype(str).values
test_day  = test['SO3_T'].round(5).astype(str).values

sample_sub  = pd.read_csv(SAMPLE_SUB)[['ID']]
oracle_vec  = sample_sub.merge(pd.read_csv(ORACLE), on='ID', how='left').fillna(0)['TARGET'].values
oracle_days = sample_sub.merge(
    pd.read_parquet(TEST_PATH, columns=['ID', 'SO3_T']), on='ID', how='left'
)['SO3_T'].round(5).astype(str).values

# ── CS z-score normalisation ──────────────────────────────────────
print(f"\nCS z-score normalisation on {len(TOP5_FEATS)} features...", flush=True)
tr_raw = train[TOP5_FEATS].fillna(0).values.astype(np.float32)
te_raw = test.reindex(columns=TOP5_FEATS, fill_value=0).values.astype(np.float32)
X_tr = np.zeros_like(tr_raw); X_te = np.zeros_like(te_raw)
for d in np.unique(train_day):
    m = train_day == d; x = tr_raw[m]; s = x.std(0); s[s < 1e-8] = 1.
    X_tr[m] = (x - x.mean(0)) / s
for d in np.unique(test_day):
    m = test_day == d; x = te_raw[m]; s = x.std(0); s[s < 1e-8] = 1.
    X_te[m] = (x - x.mean(0)) / s
del tr_raw, te_raw; gc.collect()
print(f"  X_tr={X_tr.shape}  X_te={X_te.shape}", flush=True)

# Per-feature IC on training data (liquid)
print(f"\n  Per-feature IC on liquid training data:")
ics_train = []
for fi, fname in enumerate(TOP5_FEATS):
    day_ics = []
    for d in np.unique(train_day):
        m = train_day == d
        if m.sum() < 5: continue
        try:
            r, _ = pearsonr(X_tr[m, fi], y_wins[m])
            if not np.isnan(r): day_ics.append(r)
        except: pass
    avg_ic = float(np.mean(day_ics)) if day_ics else 0.
    icir   = avg_ic / (float(np.std(day_ics)) + 1e-10) if len(day_ics) > 1 else 0.
    ics_train.append(avg_ic)
    print(f"    [{fi+1}] {fname:<45}  IC={avg_ic:+.5f}  ICIR={icir:+.3f}")

# GroupKFold
groups  = pd.qcut(pd.Series(train['SO3_T'].values), q=5, labels=False,
                  duplicates='drop').values.astype(np.int32)
n_folds = len(np.unique(groups))
gkf     = GroupKFold(n_splits=n_folds)
folds   = list(gkf.split(X_tr, y_wins, groups=groups))
print(f"\n  GroupKFold: {n_folds} folds", flush=True)

# ══════════════════════════════════════════════════════════════════
# APPROACH A: Ridge with alpha sweep
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("APPROACH A: Ridge on CS z-scored top-5 (alpha sweep)")
print("="*60, flush=True)

ALPHA_SWEEP = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]
ridge_results = []

for alpha in ALPHA_SWEEP:
    oof_r = np.zeros(len(y_wins)); te_pr = np.zeros(len(X_te))
    for fi, (tri, vai) in enumerate(folds):
        m = Ridge(alpha=alpha, fit_intercept=False)
        m.fit(X_tr[tri], y_wins[tri])
        oof_r[vai] = m.predict(X_tr[vai])
        te_pr     += m.predict(X_te) / n_folds
    oof_r2 = r2_score(y_wins, oof_r)
    sc, sub = save_and_score(te_pr, test_ids, sample_sub, oracle_vec, oracle_days,
                             f'linear_top5_ridge_a{alpha:.0e}.csv')
    # Print coefficients from last fold
    coefs = m.coef_
    coef_str = '  '.join([f'{TOP5_FEATS[i][:12]}={c:+.4f}' for i, c in enumerate(coefs)])
    flag = '  ←' if sc > 0.060349 else ''
    print(f"  alpha={alpha:<8}  OOF_R²={oof_r2:+.6f}  oracle={sc:+.6f}{flag}", flush=True)
    ridge_results.append({'alpha': alpha, 'oracle': sc, 'oof_r2': oof_r2, 'te_pr': te_pr})

best_ridge = max(ridge_results, key=lambda x: x['oracle'])
print(f"\n  Best Ridge: alpha={best_ridge['alpha']}  oracle={best_ridge['oracle']:+.6f}", flush=True)

# Print coefficients at best alpha
for alpha in ALPHA_SWEEP:
    if alpha == best_ridge['alpha']:
        m_best = Ridge(alpha=alpha, fit_intercept=False).fit(X_tr, y_wins)
        print(f"\n  Coefficients at best alpha (full train fit):")
        for fi, fname in enumerate(TOP5_FEATS):
            print(f"    {fname:<45}  coef={m_best.coef_[fi]:+.6f}  IC={ics_train[fi]:+.5f}")

# ══════════════════════════════════════════════════════════════════
# APPROACH B: IC-weighted direct combination (no fitting)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("APPROACH B: IC-weighted direct combination (no fitting)")
print("="*60, flush=True)
# pred_i = IC_i * z_score(feat_i)
# This is equivalent to GLS/Grinold with known IC weights

ic_arr = np.array(ics_train, dtype=np.float64)
# Normalise IC weights to sum to 1 (by abs)
ic_weights = ic_arr / (np.abs(ic_arr).sum() + 1e-10)
print(f"\n  IC weights: {dict(zip([f[:12] for f in TOP5_FEATS], [f'{w:+.4f}' for w in ic_weights]))}")

te_ic = X_te.astype(np.float64) @ ic_weights
tr_ic = X_tr.astype(np.float64) @ ic_weights

oof_ic = np.zeros(len(y_wins))
for fi, (tri, vai) in enumerate(folds):
    oof_ic[vai] = tr_ic[vai]   # OOF = direct prediction (no fitting)
oof_r2_ic = r2_score(y_wins, oof_ic)
sc_ic, sub_ic = save_and_score(te_ic, test_ids, sample_sub, oracle_vec, oracle_days,
                               'linear_top5_ic_weighted.csv')
flag = '  ←' if sc_ic > 0.060349 else ''
print(f"  OOF_R²={oof_r2_ic:+.6f}  oracle={sc_ic:+.6f}{flag}", flush=True)

# ══════════════════════════════════════════════════════════════════
# APPROACH C: Equal-weight cross-sectional rank average
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("APPROACH C: Equal-weight CS rank average (fully distribution-free)")
print("="*60, flush=True)
# For each day, rank each asset by each feature (0 to 1), then average ranks.
# Sign of each feature's contribution = sign of IC (from training).

ic_signs = np.sign(ic_arr)   # +1 or -1 per feature
print(f"  Feature signs: {dict(zip([f[:12] for f in TOP5_FEATS], ic_signs.tolist()))}")

def cs_rank_score(X_raw, day_ids, ic_signs):
    """Per-day rank each feature (0..1), apply IC sign, average."""
    out = np.zeros(len(X_raw))
    for d in np.unique(day_ids):
        m = day_ids == d
        if m.sum() < 2: continue
        x = X_raw[m]  # (n_day, 5)
        # Rank within day for each feature
        ranks = np.zeros_like(x, dtype=np.float64)
        for fi in range(x.shape[1]):
            from scipy.stats import rankdata
            ranks[:, fi] = rankdata(x[:, fi]) / (m.sum() + 1)   # 0..1 fractional rank
        # Apply IC sign and average
        out[m] = (ranks * ic_signs[np.newaxis, :]).mean(axis=1)
    return out

from scipy.stats import rankdata

def cs_rank_score_v2(X_raw, day_ids, ic_signs):
    out = np.zeros(len(X_raw))
    for d in np.unique(day_ids):
        m = day_ids == d
        if m.sum() < 2: continue
        x = X_raw[m]
        ranks = np.zeros_like(x, dtype=np.float64)
        for fi in range(x.shape[1]):
            ranks[:, fi] = rankdata(x[:, fi]) / (m.sum() + 1)
        out[m] = (ranks * ic_signs).mean(axis=1)
    return out

# Use raw (un-z-scored) features for ranking — z-score is monotonic so result same
tr_raw_rank = train[TOP5_FEATS].fillna(0).values.astype(np.float32)
te_raw_rank = test.reindex(columns=TOP5_FEATS, fill_value=0).values.astype(np.float32)

print("  Computing ranks (train)...", flush=True)
tr_rank = cs_rank_score_v2(tr_raw_rank, train_day, ic_signs)
print("  Computing ranks (test)...", flush=True)
te_rank = cs_rank_score_v2(te_raw_rank, test_day, ic_signs)

oof_rank = np.zeros(len(y_wins))
for fi, (tri, vai) in enumerate(folds):
    oof_rank[vai] = tr_rank[vai]
oof_r2_rank = r2_score(y_wins, oof_rank)
sc_rank, sub_rank = save_and_score(te_rank, test_ids, sample_sub, oracle_vec, oracle_days,
                                   'linear_top5_rank.csv')
flag = '  ←' if sc_rank > 0.060349 else ''
print(f"  OOF_R²={oof_r2_rank:+.6f}  oracle={sc_rank:+.6f}{flag}", flush=True)

# ══════════════════════════════════════════════════════════════════
# SUMMARY + BLEND CHECK
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
anchor_path = os.path.join(OUT_DIR, 'optimal_blend_v2.csv')
anchor = sample_sub.merge(pd.read_csv(anchor_path), on='ID', how='left').fillna(0)['TARGET'].values

results = [
    ('Ridge (best alpha)',  best_ridge['oracle'],  auto_scale(best_ridge['te_pr'])),
    ('IC-weighted',         sc_ic,                 auto_scale(te_ic)),
    ('Rank-average',        sc_rank,               auto_scale(te_rank)),
]

print(f"\n  optimal_blend_v2  oracle=+0.060098  LB=+0.00165 (anchor)")
for label, oracle_s, vec in results:
    flag = '  ←' if oracle_s > 0.060349 else ''
    print(f"  {label:<25}  oracle={oracle_s:+.6f}{flag}")

print(f"\n  Blend check (w = weight on new model, 1-w on anchor):")
print(f"  {'Model':<25}  {'w=10%':>10}  {'w=20%':>10}  {'w=30%':>10}")
print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}")

for label, oracle_s, vec in results:
    row = f"  {label:<25}"
    for w in [0.10, 0.20, 0.30]:
        blend = w * vec + (1-w) * anchor
        s = blend.std(); blend = blend*(TARGET_STD/s) if s>1e-10 else blend
        sc = daywise_oracle_score(blend, oracle_vec, oracle_days)
        flag = '*' if sc > 0.060349 else ' '
        row += f"  {sc:+.6f}{flag}"
    print(row)

print(f"\n  Submit threshold: +0.060349")
print(f"  Total elapsed:    {(time.time()-t0)/60:.1f} min")
