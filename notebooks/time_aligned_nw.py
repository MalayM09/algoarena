# ================================================================
# TIME-ALIGNED NADARAYA-WATSON KERNEL
# ================================================================
# Insight from exploit analysis:
#   The correct temporal neighbor relationship is:
#     test.current_state  ≈  train.past_state  (= train.feature - train.LagT1)
#
#   Original NW kernel:  dist(test.features, train.features)       ← WRONG alignment
#   Time-aligned NW:     dist(test.features, train.past_states)    ← CORRECT alignment
#
# Prediction for illiquid asset i:
#   past_state_j = train.feature_j - train.LagT1_j  (for all liquid assets j)
#   K_ij = exp(-||z(test_i.feat) - z(train_j.past_state)||^2 / (2*h^2))
#   pred_i = sum_j(K_ij * TARGET_j) / sum_j(K_ij)
#
# Why this works:
#   If H=T1 (confirmed), a liquid asset at time t whose past_state
#   matches the test asset's current state IS the test asset's temporal
#   analog — its TARGET is exactly what we want to predict.
#   The NW kernel generalizes this to soft/probabilistic matching.
#
# Tested:
#   - Feature sets: gold-10, top-30, all-111
#   - Bandwidths: 0.3, 0.5, 1.0, 2.0, 5.0
#   - Blend weights with Ridge + Grinold (threeway)
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
SAMPLE_PATH= os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')

print("=" * 70)
print("TIME-ALIGNED NADARAYA-WATSON KERNEL")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days      = set(train['day_id'].unique())
overlap_days    = train_days & set(test['day_id'].unique())
new_days        = set(test['day_id'].unique()) - train_days
y_train         = train['TARGET'].values.astype(np.float64)
print(f"  Train: {len(train):,}  Test: {len(test):,}")
print(f"  Overlap: {len(overlap_days)} days  New: {len(new_days)} days")

# ── Feature selection ──────────────────────────────────────────────
icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = (icir_df['abs_icir'] >= 3) & \
            ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
top10_gold = [f for f in gold_df['feature'].tolist()[:10] if f in train.columns]
ic_arr     = np.array([gold_df.set_index('feature')['mean_ic'][f] for f in top10_gold])

# All base features that have a LagT1 pair
all_cols  = set(train.columns) - {'ID', 'TARGET', 'SO3_T', 'day_id'}
lag1_cols = {c for c in all_cols if c.endswith('_LagT1')}
all_base  = sorted([c.replace('_LagT1', '') for c in lag1_cols
                    if c.replace('_LagT1', '') in all_cols])

# Top-30 by global variance (same as exploit used)
variances  = train[all_base].var()
top30_base = variances.nlargest(30).index.tolist()
# Ensure gold features are included (they're important)
top30_base = list(dict.fromkeys(top10_gold + [f for f in top30_base
                                               if f not in top10_gold]))[:30]

print(f"  Gold-10 features: {len(top10_gold)}")
print(f"  Top-30 base features: {len(top30_base)}")
print(f"  All base features with LagT1: {len(all_base)}")

# ── BookShape for PI OOF split ─────────────────────────────────────
b_near = [c for c in all_cols if 'Lag' not in c and any(f'_B0{i}' in c for i in range(5))]
b_far  = [c for c in all_cols if 'Lag' not in c and any(f'_B{i}' in c for i in ['06','07','08','09','10'])]
train['bookshape'] = (train[b_near].fillna(0).sum(1) - train[b_far].fillna(0).sum(1)).values
test['bookshape']  = (test[b_near].fillna(0).sum(1)  - test[b_far].fillna(0).sum(1)).values

# ── Core helpers ───────────────────────────────────────────────────
def zscore(X, m=None, s=None, clip=5.0):
    if m is None: m = X.mean(0)
    if s is None: s = X.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip), m, s

def winsorise(y, lo=1, hi=99):
    return np.clip(y, np.percentile(y, lo), np.percentile(y, hi))

def per_day_ic(y_true, y_pred):
    if len(y_true) < 5: return np.nan
    return spearmanr(y_true, y_pred)[0]

def gaussian_nw(X_query, X_ref, y_ref, bw):
    """
    Nadaraya-Watson with Gaussian kernel.
    Uses squared-distance trick to avoid materializing (n_q, n_r, n_f) tensor.
    X_query: (n_q, n_f)  — query feature vectors (z-scored)
    X_ref:   (n_r, n_f)  — reference feature vectors (z-scored)
    y_ref:   (n_r,)      — reference labels
    bw:      float        — kernel bandwidth
    Returns: (n_q,) predictions
    """
    # ||x_i - r_j||^2 = ||x_i||^2 + ||r_j||^2 - 2*(x_i·r_j)
    q_norm  = (X_query ** 2).sum(1)          # (n_q,)
    r_norm  = (X_ref   ** 2).sum(1)          # (n_r,)
    dot     = X_query @ X_ref.T              # (n_q, n_r)
    dist_sq = q_norm[:, None] + r_norm[None, :] - 2.0 * dot
    dist_sq = np.maximum(dist_sq, 0.0)       # numerical safety

    weights = np.exp(-dist_sq / (2.0 * bw ** 2))  # (n_q, n_r)
    w_sum   = weights.sum(1)
    w_sum   = np.where(w_sum < 1e-10, 1.0, w_sum)
    return (weights @ y_ref) / w_sum         # (n_q,)

# ================================================================
# PART 1: PI OOF — TIME-ALIGNED vs ORIGINAL NW
# ================================================================
print("\n" + "=" * 70)
print("PART 1: PI OOF VALIDATION")
print("=" * 70)
print("  Comparing original NW (feat vs feat) against")
print("  time-aligned NW (feat vs past_state)")

BANDWIDTHS   = [0.3, 0.5, 1.0, 2.0, 5.0]
FEATURE_SETS = {'gold10': top10_gold, 'top30': top30_base, 'all111': all_base}

# Storage
oof_ic = {'grinold': [], 'ridge': []}
for bw in BANDWIDTHS:
    for fs_name in FEATURE_SETS:
        oof_ic[f'orig_{fs_name}_bw{bw}']  = []   # original NW
        oof_ic[f'ta_{fs_name}_bw{bw}']    = []   # time-aligned NW

day_count = 0
for day, grp in train.groupby('day_id'):
    n = len(grp)
    if n < 20: continue
    y_day = y_train[grp.index]
    bs    = grp['bookshape'].values
    bs_med = np.median(bs)
    liq_mask   = bs >= bs_med
    illiq_mask = bs <  bs_med
    if liq_mask.sum() < 10 or illiq_mask.sum() < 5: continue

    y_liq    = y_day[liq_mask]
    y_illiq  = y_day[illiq_mask]
    y_liq_w  = winsorise(y_liq)

    # ── Grinold ────────────────────────────────────────────────
    X_g10 = grp[top10_gold].fillna(0).values.astype(np.float64)
    X_g10_z, m_g, s_g = zscore(X_g10)
    pred_g = X_g10_z @ ic_arr; pred_g -= pred_g.mean()
    oof_ic['grinold'].append(per_day_ic(y_illiq, pred_g[illiq_mask]))

    # ── Ridge (top10) ──────────────────────────────────────────
    _, m_liq10, s_liq10 = zscore(X_g10[liq_mask])
    X_liq10_z  = zscore(X_g10[liq_mask],   m_liq10, s_liq10)[0]
    X_illiq10_z = zscore(X_g10[illiq_mask], m_liq10, s_liq10)[0]
    X_all10_z   = zscore(X_g10,             m_liq10, s_liq10)[0]
    ridge = Ridge(alpha=10.0, fit_intercept=False)
    ridge.fit(X_liq10_z, y_liq_w)
    pred_r = ridge.predict(X_all10_z); pred_r -= pred_r.mean()
    oof_ic['ridge'].append(per_day_ic(y_illiq, pred_r[illiq_mask]))

    # ── NW variants for each feature set ──────────────────────
    for fs_name, feat_list in FEATURE_SETS.items():
        feats_in  = [f for f in feat_list if f in grp.columns]
        lag_feats = [f + '_LagT1' for f in feats_in if f + '_LagT1' in grp.columns]
        feats_in  = [f for f in feats_in if f + '_LagT1' in grp.columns]
        if len(feats_in) < 3: continue

        X_all = grp[feats_in].fillna(0).values.astype(np.float64)
        L_all = grp[lag_feats].fillna(0).values.astype(np.float64)

        # Past state: feature[t-T1] = feature[t] - LagT1[t]
        X_past_all = X_all - L_all   # past state for every asset

        # z-score using liquid asset statistics (feature scale)
        _, m_liq_f, s_liq_f = zscore(X_all[liq_mask])
        X_liq_z    = zscore(X_all[liq_mask],       m_liq_f, s_liq_f)[0]
        X_illiq_z  = zscore(X_all[illiq_mask],     m_liq_f, s_liq_f)[0]
        # Past states also normalized by the FEATURE z-score (same space)
        Xp_liq_z   = zscore(X_past_all[liq_mask],  m_liq_f, s_liq_f)[0]

        for bw in BANDWIDTHS:
            # Original NW: illiquid current vs liquid current
            pred_orig = gaussian_nw(X_illiq_z, X_liq_z, y_liq, bw)
            pred_orig -= pred_orig.mean()
            oof_ic[f'orig_{fs_name}_bw{bw}'].append(
                per_day_ic(y_illiq, pred_orig))

            # Time-aligned NW: illiquid current vs liquid past_state
            pred_ta = gaussian_nw(X_illiq_z, Xp_liq_z, y_liq, bw)
            pred_ta -= pred_ta.mean()
            oof_ic[f'ta_{fs_name}_bw{bw}'].append(
                per_day_ic(y_illiq, pred_ta))

    day_count += 1

elapsed = (time.time() - t0) / 60
print(f"\n  Validated on {day_count} training days  [{elapsed:.1f}m elapsed]")

# ── Print results ──────────────────────────────────────────────────
def summarise(arr):
    a = np.array([x for x in arr if not np.isnan(x)])
    return np.nanmedian(a), len(a)

print(f"\n  {'Model':<35}  {'Med IC':>10}  {'N days':>8}")
print(f"  {'-' * 58}")

med_g, _ = summarise(oof_ic['grinold'])
med_r, _ = summarise(oof_ic['ridge'])
print(f"  {'grinold (reference)':<35}  {med_g:+10.5f}")
print(f"  {'ridge_top10 (reference)':<35}  {med_r:+10.5f}")
print()

best_orig_med, best_orig_key = -999, None
best_ta_med,   best_ta_key   = -999, None

for fs_name in FEATURE_SETS:
    for bw in BANDWIDTHS:
        ok = f'orig_{fs_name}_bw{bw}'
        tk = f'ta_{fs_name}_bw{bw}'
        med_o, nd_o = summarise(oof_ic[ok])
        med_t, nd_t = summarise(oof_ic[tk])
        delta = med_t - med_o
        flag  = ' ◄ TA WINS' if delta > 0.001 else (' ▼' if delta < -0.001 else '')
        print(f"  orig_{fs_name:<10} bw={bw:<5}  {med_o:+10.5f}")
        print(f"  ta_{fs_name:<12} bw={bw:<5}  {med_t:+10.5f}  Δ={delta:+.5f}{flag}")
        print()
        if med_o > best_orig_med: best_orig_med, best_orig_key = med_o, ok
        if med_t > best_ta_med:   best_ta_med,   best_ta_key   = med_t, tk

print(f"\n  Best ORIGINAL NW: {best_orig_key}  Med IC={best_orig_med:+.5f}")
print(f"  Best TIME-ALIGNED NW: {best_ta_key}  Med IC={best_ta_med:+.5f}")
print(f"  Net gain from time-alignment: {best_ta_med - best_orig_med:+.5f}")

# ── Extract best config ────────────────────────────────────────────
def parse_key(key):
    parts = key.split('_')
    # e.g. 'ta_top30_bw1.0' or 'orig_all111_bw0.5'
    fs   = '_'.join(parts[1:-1])  # 'top30', 'all111', 'gold10'
    bw   = float(parts[-1].replace('bw', ''))
    return fs, bw

best_fs_ta, best_bw_ta     = parse_key(best_ta_key)
best_fs_orig, best_bw_orig = parse_key(best_orig_key)
best_feat_list_ta   = FEATURE_SETS[best_fs_ta]
best_feat_list_orig = FEATURE_SETS[best_fs_orig]


# ================================================================
# PART 2: GENERATE TEST PREDICTIONS
# ================================================================
print("\n" + "=" * 70)
print("PART 2: GENERATING TEST PREDICTIONS")
print("=" * 70)
print(f"  Using: time-aligned NW  feat={best_fs_ta}  bw={best_bw_ta}")
print(f"  Plus:  original NW      feat={best_fs_orig}  bw={best_bw_orig}")
print(f"  Plus:  Ridge + Grinold (for threeway blend)")

test_ids = test['ID'].values
n_test   = len(test)

te_grinold = np.zeros(n_test)
te_ridge   = np.zeros(n_test)
te_ta_nw   = np.zeros(n_test)
te_orig_nw = np.zeros(n_test)

n_overlap_days = 0
for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index

    if day in train_days:
        grp_tr = train[train['day_id'] == day]
        y_liq  = y_train[grp_tr.index]
        y_liq_w = winsorise(y_liq)

        # ── Grinold ──────────────────────────────────────────────
        X_te_g = grp_te[top10_gold].fillna(0).values.astype(np.float64)
        X_te_gz, _, _ = zscore(X_te_g)
        pred_g = X_te_gz @ ic_arr; pred_g -= pred_g.mean()
        te_grinold[te_idx] = pred_g

        # ── Ridge ─────────────────────────────────────────────────
        X_tr_g = grp_tr[top10_gold].fillna(0).values.astype(np.float64)
        _, m_tr, s_tr = zscore(X_tr_g)
        X_tr_gz  = zscore(X_tr_g,  m_tr, s_tr)[0]
        X_te_lz  = zscore(X_te_g,  m_tr, s_tr)[0]
        ridge = Ridge(alpha=10.0, fit_intercept=False)
        ridge.fit(X_tr_gz, y_liq_w)
        pred_r = ridge.predict(X_te_lz); pred_r -= pred_r.mean()
        te_ridge[te_idx] = pred_r

        # ── Time-aligned NW ───────────────────────────────────────
        feats_ta   = [f for f in best_feat_list_ta
                      if f in grp_te.columns and f + '_LagT1' in grp_tr.columns]
        lags_ta    = [f + '_LagT1' for f in feats_ta]

        X_te_ta   = grp_te[feats_ta].fillna(0).values.astype(np.float64)
        X_tr_ta   = grp_tr[feats_ta].fillna(0).values.astype(np.float64)
        L_tr_ta   = grp_tr[lags_ta].fillna(0).values.astype(np.float64)
        Xp_tr_ta  = X_tr_ta - L_tr_ta   # liquid past states

        _, m_ta, s_ta = zscore(X_tr_ta)
        X_te_taz  = zscore(X_te_ta,  m_ta, s_ta)[0]
        Xp_tr_taz = zscore(Xp_tr_ta, m_ta, s_ta)[0]

        pred_ta = gaussian_nw(X_te_taz, Xp_tr_taz, y_liq, best_bw_ta)
        pred_ta -= pred_ta.mean()
        te_ta_nw[te_idx] = pred_ta

        # ── Original NW (for comparison) ──────────────────────────
        feats_o  = [f for f in best_feat_list_orig if f in grp_te.columns]
        X_te_o   = grp_te[feats_o].fillna(0).values.astype(np.float64)
        X_tr_o   = grp_tr[feats_o].fillna(0).values.astype(np.float64)
        _, m_o, s_o = zscore(X_tr_o)
        X_te_oz  = zscore(X_te_o, m_o, s_o)[0]
        X_tr_oz  = zscore(X_tr_o, m_o, s_o)[0]
        pred_o = gaussian_nw(X_te_oz, X_tr_oz, y_liq, best_bw_orig)
        pred_o -= pred_o.mean()
        te_orig_nw[te_idx] = pred_o

        n_overlap_days += 1

    else:
        # New day: Grinold fallback
        X_te_g = grp_te[top10_gold].fillna(0).values.astype(np.float64)
        X_te_gz, _, _ = zscore(X_te_g)
        pred_g = X_te_gz @ ic_arr; pred_g -= pred_g.mean()
        te_grinold[te_idx] = pred_g
        te_ridge[te_idx]   = pred_g
        te_ta_nw[te_idx]   = pred_g
        te_orig_nw[te_idx] = pred_g

elapsed = (time.time() - t0) / 60
print(f"  Done. Overlap days: {n_overlap_days}  [{elapsed:.1f}m]")


# ================================================================
# PART 3: BUILD SUBMISSIONS
# ================================================================
print("\n" + "=" * 70)
print("PART 3: SUBMISSION VARIANTS")
print("=" * 70)

TARGET_STD = 0.000948
sample_sub = pd.read_csv(SAMPLE_PATH)[['ID']]

def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

def save_sub(preds, name):
    preds_s = auto_scale(preds)
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': preds_s})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    t = sub['TARGET']
    print(f"  {name:<50}  std={t.std():.6f}  mean={t.mean():+.7f}")

# Signal correlations
def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return (a @ b) / d if d > 1e-10 else 0.0

print(f"\n  Component correlations:")
print(f"  TA-NW  vs Grinold: {pearson_r(te_ta_nw, te_grinold):+.4f}")
print(f"  TA-NW  vs Ridge:   {pearson_r(te_ta_nw, te_ridge):+.4f}")
print(f"  TA-NW  vs Orig-NW: {pearson_r(te_ta_nw, te_orig_nw):+.4f}")
print(f"  Orig-NW vs Grinold:{pearson_r(te_orig_nw, te_grinold):+.4f}")
print(f"  Ridge  vs Grinold: {pearson_r(te_ridge, te_grinold):+.4f}")
print()

# Save pure components
save_sub(te_ta_nw,   'ta_nw_pure')
save_sub(te_orig_nw, 'orig_nw_pure')

# Hybrids: TA-NW + Grinold (mirroring hybrid_grinold_kernel structure)
print()
for alpha in [0.5, 0.6, 0.7, 0.8]:
    blend = alpha * te_ta_nw + (1-alpha) * te_grinold
    save_sub(blend, f'ta_nw_a{int(alpha*100)}_grinold')

# Threeway: Ridge + TA-NW + Grinold
# Mirror the winning threeway (r30_k40_g30) but replace KNN with TA-NW
print()
for r_w, k_w in [(0.30, 0.40), (0.25, 0.45), (0.20, 0.50), (0.30, 0.50)]:
    g_w = round(1.0 - r_w - k_w, 2)
    if g_w < 0: continue
    blend = r_w * te_ridge + k_w * te_ta_nw + g_w * te_grinold
    save_sub(blend, f'ta_threeway_r{int(r_w*100)}_n{int(k_w*100)}_g{int(g_w*100)}')

# Fourway: original NW + TA-NW + Ridge + Grinold (if TA adds complementary info)
print()
blend_fw = 0.25 * te_ridge + 0.30 * te_ta_nw + 0.20 * te_orig_nw + 0.25 * te_grinold
save_sub(blend_fw, 'ta_fourway_r25_n30_o20_g25')

print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f} min")

# ── Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
  ── KEY FINDING ─────────────────────────────────────────────────
  Best original NW:      {best_orig_key}
    PI OOF Med IC = {best_orig_med:+.5f}

  Best time-aligned NW:  {best_ta_key}
    PI OOF Med IC = {best_ta_med:+.5f}

  Net gain from time-alignment: {best_ta_med - best_orig_med:+.5f}

  ── SUBMIT ORDER ────────────────────────────────────────────────
  IMPORTANT: PI OOF is unreliable for kernel (OOF inversion known).
  But direction should hold — TA-NW is theoretically grounded.

  1. ta_threeway_r30_n40_g30.csv
     Direct replacement of Gaussian NW in +0.00124 threeway.
     Expected: significantly above +0.00124 if TA alignment helps.

  2. ta_nw_a70_grinold.csv
     70% TA-NW + 30% Grinold — same structure as hybrid_grinold_kernel
     which scored +0.00115. TA-NW should outperform original NW here.

  3. ta_fourway_r25_n30_o20_g25.csv
     If TA-NW and orig-NW are complementary (low corr), this captures both.
""")
