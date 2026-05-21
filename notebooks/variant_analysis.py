# ================================================================
# VARIANT ANALYSIS: Three improvements + all combinations
# ================================================================
# Comparing against baseline threeway_r30_k40_g29 (+0.00124):
#
#   A. ICIR_G   — Grinold with ICIR weights (ICIR×sign(IC)) instead of mean IC
#   B. WIN_KNN  — KNN copies winsorized neighbor returns (y_liq_w, not raw y_liq)
#   C. GOLD_NW  — Gaussian NW kernel (RBF) restricted to top-10 gold features
#
# Blends tested (all as 30% Ridge + 40% Kernel + 30% Grinold):
#   0. Baseline  (KNN_raw  + IC_G)        ← current best +0.00124
#   A. ICIR only (KNN_raw  + ICIR_G)
#   B. WIN  only (KNN_win  + IC_G)
#   C. NW   only (GOLD_NW  + IC_G)
#   AB.          (KNN_win  + ICIR_G)
#   AC.          (GOLD_NW  + ICIR_G)
#   BC.          (GOLD_NW_win + IC_G)     NW with winsorized y_ref
#   ABC.         (GOLD_NW_win + ICIR_G)   all three
#
# PI OOF shown for reference; submit by LB since OOF is inverted
# for kernel methods.
# ================================================================
import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
t0 = time.time()

print("=" * 70)
print("VARIANT ANALYSIS — ICIR_G / WIN_KNN / GOLD_NW + all combinations")
print("=" * 70)

# ── Feature selection ──────────────────────────────────────────────────────
icir_df  = pd.read_csv(ICIR_PATH)
gold_mask = ((icir_df['abs_icir'] >= 3) &
             ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0)))
gold_df  = icir_df[gold_mask].sort_values('abs_icir', ascending=False).copy()

ic_w_map   = gold_df.set_index('feature')['mean_ic'].to_dict()
icir_w_map = gold_df.set_index('feature')['abs_icir'].to_dict()

# ── Load data ──────────────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)

all_cols = set(train.columns) - {'ID', 'TARGET'}
top10    = [f for f in gold_df['feature'].tolist()[:10] if f in all_cols]
print(f"  Top-10 gold features: {top10}")

ic_arr   = np.array([ic_w_map[f]                          for f in top10])
icir_arr = np.array([icir_w_map[f] * np.sign(ic_w_map[f]) for f in top10])

print(f"  IC   weights: {np.round(ic_arr, 4)}")
print(f"  ICIR weights: {np.round(icir_arr, 4)}")
print(f"  ICIR/IC ratio (mean): {(np.abs(icir_arr) / np.abs(ic_arr)).mean():.1f}x larger magnitude")

train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days = set(train['day_id'].unique())
overlap    = train_days & set(test['day_id'].unique())
new_days   = set(test['day_id'].unique()) - train_days

y_train  = train['TARGET'].values.astype(np.float64)
test_ids = test['ID'].values

# BookShape for PI OOF split
b_near = [c for c in all_cols if 'Lag' not in c and
          any(f'_B0{i}' in c for i in range(5))]
b_far  = [c for c in all_cols if 'Lag' not in c and
          any(f'_B{i}' in c for i in ['06','07','08','09','10'])]
train['bookshape'] = (train[b_near].fillna(0).sum(1) -
                      train[b_far].fillna(0).sum(1)).astype(np.float64)
test['bookshape']  = (test[b_near].fillna(0).sum(1) -
                      test[b_far].fillna(0).sum(1)).astype(np.float64)

print(f"  Train: {len(train):,}  Test: {len(test):,}")
print(f"  Overlap: {len(overlap)} days  Future: {len(new_days)} days")

# ── Helpers ────────────────────────────────────────────────────────────────
def zscore_fit(X, clip=5.0):
    m = X.mean(0); s = X.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip), m, s

def zscore_apply(X, m, s, clip=5.0):
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip)

def winsorise(y, lo=1, hi=99):
    return np.clip(y, np.percentile(y, lo), np.percentile(y, hi))

def per_day_ic(y_true, y_pred):
    if len(y_true) < 5: return np.nan
    return spearmanr(y_true, y_pred)[0]

KNN_K = 5           # confirmed best K from prior analysis
RIDGE_ALPHA = 10.0  # confirmed best from perday_ridge.py

def knn_predict(X_query, X_ref, y_ref, K=KNN_K):
    """Cosine-similarity weighted KNN — top-K neighbors."""
    sim   = cosine_similarity(X_query, X_ref)
    K_    = min(K, sim.shape[1])
    topk  = np.argpartition(sim, -K_, axis=1)[:, -K_:]
    rows  = np.arange(len(X_query))[:, None]
    w     = np.maximum(sim[rows, topk], 0)
    w_sum = w.sum(1, keepdims=True)
    w     = w / np.where(w_sum < 1e-10, 1.0, w_sum)
    return (w * y_ref[topk]).sum(1)

def gauss_nw_predict(X_query, X_ref, y_ref, gamma=None):
    """Gaussian (RBF) Nadaraya-Watson kernel.
    gamma = 1 / (2 * sigma^2). Default: median heuristic."""
    if gamma is None:
        # Median heuristic: sigma^2 = median(||x-y||^2) / log(n_ref)
        dists2 = np.sum((X_query[:5, None, :] - X_ref[None, :, :]) ** 2, axis=2)
        med = np.median(dists2)
        sigma2 = max(med / max(np.log(len(X_ref)), 1.0), 1e-3)
        gamma = 1.0 / (2.0 * sigma2)
    K_mat = rbf_kernel(X_query, X_ref, gamma=gamma)  # (n_q, n_r)
    w_sum = K_mat.sum(1, keepdims=True)
    w     = K_mat / np.where(w_sum < 1e-10, 1.0, w_sum)
    return w @ y_ref

# ================================================================
# PART 1: PI OOF COMPARISON
# ================================================================
print("\n" + "=" * 70)
print("PART 1: PI OOF — all 8 model variants")
print("=" * 70)
print("  (Remember: PI OOF is unreliable for kernel methods — OOF inversion)")
print()

variant_keys = ['baseline', 'A_icir', 'B_win', 'C_nw', 'AB', 'AC', 'BC', 'ABC']
ic_store = {k: [] for k in variant_keys}

day_count = 0
for day, grp in train.groupby('day_id'):
    n = len(grp)
    if n < 20: continue

    y_day  = y_train[grp.index]
    bs     = grp['bookshape'].values
    bs_med = np.median(bs)

    liq_mask   = bs >= bs_med
    illiq_mask = bs <  bs_med
    if liq_mask.sum() < 10 or illiq_mask.sum() < 5: continue

    y_liq   = y_day[liq_mask]
    y_illiq = y_day[illiq_mask]
    y_liq_w = winsorise(y_liq)           # winsorized for Win_KNN + Ridge

    X_all = grp[top10].fillna(0).values.astype(np.float64)
    _, m_liq, s_liq = zscore_fit(X_all[liq_mask])
    X_liq_z   = zscore_apply(X_all[liq_mask],   m_liq, s_liq)
    X_illiq_z = zscore_apply(X_all[illiq_mask], m_liq, s_liq)

    # ── Ridge (shared) ─────────────────────────────────────────────────────
    ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
    ridge.fit(X_liq_z, y_liq_w)
    pred_r = ridge.predict(X_illiq_z)

    # ── Grinold variants ────────────────────────────────────────────────────
    pred_g_ic   = X_illiq_z @ ic_arr
    pred_g_icir = X_illiq_z @ icir_arr

    # ── Kernel variants ─────────────────────────────────────────────────────
    pred_knn_raw = knn_predict(X_illiq_z, X_liq_z, y_liq)
    pred_knn_win = knn_predict(X_illiq_z, X_liq_z, y_liq_w)
    pred_nw_raw  = gauss_nw_predict(X_illiq_z, X_liq_z, y_liq)
    pred_nw_win  = gauss_nw_predict(X_illiq_z, X_liq_z, y_liq_w)

    # ── Blend weights ────────────────────────────────────────────────────────
    rw, kw, gw = 0.30, 0.40, 0.30
    def tw(k, g):
        b = rw*pred_r + kw*k + gw*g
        return b - b.mean()

    preds = {
        'baseline': tw(pred_knn_raw, pred_g_ic),
        'A_icir':   tw(pred_knn_raw, pred_g_icir),
        'B_win':    tw(pred_knn_win, pred_g_ic),
        'C_nw':     tw(pred_nw_raw,  pred_g_ic),
        'AB':       tw(pred_knn_win, pred_g_icir),
        'AC':       tw(pred_nw_raw,  pred_g_icir),
        'BC':       tw(pred_nw_win,  pred_g_ic),
        'ABC':      tw(pred_nw_win,  pred_g_icir),
    }

    for k, p in preds.items():
        ic_store[k].append(per_day_ic(y_illiq, p))

    day_count += 1

elapsed = (time.time() - t0) / 60
print(f"  Validated on {day_count} training days [{elapsed:.1f}m elapsed]\n")

results = {}
print(f"  {'Variant':<12}  {'Description':<32}  {'Med IC':>8}  {'%pos':>7}  {'vs base':>8}")
print(f"  {'-' * 76}")

descriptions = {
    'baseline': 'KNN_raw  + IC_G       (current)',
    'A_icir':   'KNN_raw  + ICIR_G     (A only)',
    'B_win':    'KNN_win  + IC_G       (B only)',
    'C_nw':     'GOLD_NW  + IC_G       (C only)',
    'AB':       'KNN_win  + ICIR_G     (A+B)',
    'AC':       'GOLD_NW  + ICIR_G     (A+C)',
    'BC':       'GOLD_NW_win + IC_G    (B+C)',
    'ABC':      'GOLD_NW_win + ICIR_G  (A+B+C)',
}

base_med = None
for k in variant_keys:
    arr = np.array([x for x in ic_store[k] if not np.isnan(x)])
    med = np.nanmedian(arr)
    ppos = (arr > 0).mean() * 100
    results[k] = {'med': med, 'ppos': ppos}
    if k == 'baseline': base_med = med
    delta = med - base_med if base_med is not None else 0
    marker = ' ◄ BEST' if med == max(results[v]['med'] for v in results) else ''
    print(f"  {k:<12}  {descriptions[k]:<32}  {med:+8.5f}  {ppos:6.1f}%  {delta:+8.5f}{marker}")

print(f"\n  NOTE: Higher PI OOF does NOT guarantee better LB for kernel variants.")
print(f"  Submit in priority order decided by the LB-evidence-based reasoning below.")


# ================================================================
# PART 2: TEST PREDICTIONS — all 8 variants
# ================================================================
print("\n" + "=" * 70)
print("PART 2: GENERATING TEST PREDICTIONS (all 8 variants)")
print("=" * 70)

# Storage: one array per variant
te_preds = {k: np.zeros(len(test)) for k in variant_keys}

n_overlap = 0
for day, grp_te in test.groupby('day_id'):
    te_idx   = grp_te.index
    X_te     = grp_te[top10].fillna(0).values.astype(np.float64)

    if day in train_days:
        grp_tr  = train[train['day_id'] == day]
        y_liq   = y_train[grp_tr.index]
        y_liq_w = winsorise(y_liq)
        X_tr    = grp_tr[top10].fillna(0).values.astype(np.float64)

        _, m_liq, s_liq = zscore_fit(X_tr)
        X_tr_z = zscore_apply(X_tr, m_liq, s_liq)
        X_te_z = zscore_apply(X_te, m_liq, s_liq)

        # Ridge
        ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        ridge.fit(X_tr_z, y_liq_w)
        pred_r = ridge.predict(X_te_z)

        # Grinold IC
        pred_g_ic   = X_te_z @ ic_arr
        # Grinold ICIR
        pred_g_icir = X_te_z @ icir_arr

        # KNN raw
        pred_knn_raw = knn_predict(X_te_z, X_tr_z, y_liq)
        # KNN win
        pred_knn_win = knn_predict(X_te_z, X_tr_z, y_liq_w)
        # Gauss NW raw
        pred_nw_raw  = gauss_nw_predict(X_te_z, X_tr_z, y_liq)
        # Gauss NW win
        pred_nw_win  = gauss_nw_predict(X_te_z, X_tr_z, y_liq_w)

        rw, kw, gw = 0.30, 0.40, 0.30
        def tw_te(k, g):
            b = rw*pred_r + kw*k + gw*g
            return b - b.mean()

        blends = {
            'baseline': tw_te(pred_knn_raw, pred_g_ic),
            'A_icir':   tw_te(pred_knn_raw, pred_g_icir),
            'B_win':    tw_te(pred_knn_win, pred_g_ic),
            'C_nw':     tw_te(pred_nw_raw,  pred_g_ic),
            'AB':       tw_te(pred_knn_win, pred_g_icir),
            'AC':       tw_te(pred_nw_raw,  pred_g_icir),
            'BC':       tw_te(pred_nw_win,  pred_g_ic),
            'ABC':      tw_te(pred_nw_win,  pred_g_icir),
        }
        n_overlap += 1

    else:
        # Future day: Grinold IC fallback for all
        X_te_z, _, _ = zscore_fit(X_te)
        pred_g_ic   = X_te_z @ ic_arr
        pred_g_icir = X_te_z @ icir_arr
        blends = {k: pred_g_ic - pred_g_ic.mean() for k in variant_keys}
        blends['A_icir'] = pred_g_icir - pred_g_icir.mean()
        blends['AB']     = pred_g_icir - pred_g_icir.mean()
        blends['AC']     = pred_g_icir - pred_g_icir.mean()
        blends['ABC']    = pred_g_icir - pred_g_icir.mean()

    for k, b in blends.items():
        te_preds[k][te_idx] = b

elapsed = (time.time() - t0) / 60
print(f"  Overlap: {n_overlap}  Future: {len(new_days)}  [{elapsed:.1f}m]")


# ================================================================
# PART 3: SAVE SUBMISSIONS + STATISTICS
# ================================================================
print("\n" + "=" * 70)
print("PART 3: SUBMISSION STATISTICS")
print("=" * 70)

TARGET_STD = 0.000948
sample_sub = pd.read_csv(
    os.path.join(BASE_DIR, 'data/raw/sample_submission.csv'))[['ID']]

def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

def save_sub(preds, name):
    ps = auto_scale(preds)
    sub = pd.DataFrame({'ID': test_ids, 'TARGET': ps})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    return sub['TARGET'].std(), (sub['TARGET'] > 0).mean() * 100

sub_names = {
    'baseline': 'var_baseline',
    'A_icir':   'var_A_icir',
    'B_win':    'var_B_win',
    'C_nw':     'var_C_nw',
    'AB':       'var_AB_win_icir',
    'AC':       'var_AC_nw_icir',
    'BC':       'var_BC_nw_win',
    'ABC':      'var_ABC_all',
}

print(f"\n  {'Variant':<8}  {'Filename':<28}  {'std':>10}  {'%pos':>8}  {'PI_med':>10}")
print(f"  {'-' * 70}")
for k in variant_keys:
    std, ppos = save_sub(te_preds[k], sub_names[k])
    med_ic    = results[k]['med']
    print(f"  {k:<8}  {sub_names[k]:<28}  {std:.6f}  {ppos:7.1f}%  {med_ic:+10.5f}")


# ================================================================
# PART 4: CORRELATION ANALYSIS
# ================================================================
print("\n" + "=" * 70)
print("PART 4: CORRELATION WITH BASELINE (var_baseline)")
print("=" * 70)
print("  High correlation → similar predictions → marginal LB effect")
print("  Low correlation  → different predictions → potentially complementary")
print()

base_p = te_preds['baseline']
base_p_norm = base_p - base_p.mean()
print(f"  {'Variant':<8}  {'Corr vs baseline':>18}  {'Sign agree':>12}  {'Description'}")
print(f"  {'-' * 72}")
for k in variant_keys:
    p = te_preds[k]
    p_norm = p - p.mean()
    n1 = np.linalg.norm(base_p_norm)
    n2 = np.linalg.norm(p_norm)
    corr = (base_p_norm @ p_norm) / (n1*n2) if n1>1e-10 and n2>1e-10 else 0.0
    sa   = (np.sign(base_p) == np.sign(p)).mean() * 100
    print(f"  {k:<8}  {corr:+18.4f}  {sa:11.1f}%  {descriptions[k]}")

# Pairwise correlations between the 3 main new signals vs baseline
print(f"\n  Key component-level correlations:")
knn_raw_te  = np.zeros(len(test))
knn_win_te  = np.zeros(len(test))
nw_raw_te   = np.zeros(len(test))
g_ic_te     = np.zeros(len(test))
g_icir_te   = np.zeros(len(test))

# Recompute pure components for correlation analysis
for day, grp_te in test.groupby('day_id'):
    te_idx = grp_te.index
    X_te   = grp_te[top10].fillna(0).values.astype(np.float64)
    if day in train_days:
        grp_tr = train[train['day_id'] == day]
        y_liq  = y_train[grp_tr.index]
        y_liq_w = winsorise(y_liq)
        X_tr   = grp_tr[top10].fillna(0).values.astype(np.float64)
        _, m, s = zscore_fit(X_tr)
        X_tr_z = zscore_apply(X_tr, m, s)
        X_te_z = zscore_apply(X_te, m, s)
        knn_raw_te[te_idx] = knn_predict(X_te_z, X_tr_z, y_liq)
        knn_win_te[te_idx] = knn_predict(X_te_z, X_tr_z, y_liq_w)
        nw_raw_te[te_idx]  = gauss_nw_predict(X_te_z, X_tr_z, y_liq)
        g_ic_te[te_idx]    = X_te_z @ ic_arr
        g_icir_te[te_idx]  = X_te_z @ icir_arr
    else:
        X_te_z, _, _ = zscore_fit(X_te)
        g_ic_te[te_idx]   = X_te_z @ ic_arr
        g_icir_te[te_idx] = X_te_z @ icir_arr

def corr(a, b):
    a = a-a.mean(); b = b-b.mean()
    n1=np.linalg.norm(a); n2=np.linalg.norm(b)
    return (a@b)/(n1*n2) if n1>1e-10 and n2>1e-10 else 0.0

print(f"  KNN_raw vs GOLD_NW:   {corr(knn_raw_te, nw_raw_te):+.4f}")
print(f"  KNN_raw vs KNN_win:   {corr(knn_raw_te, knn_win_te):+.4f}")
print(f"  IC_G    vs ICIR_G:    {corr(g_ic_te, g_icir_te):+.4f}")
print(f"  KNN_raw vs IC_G:      {corr(knn_raw_te, g_ic_te):+.4f}")
print(f"  GOLD_NW vs IC_G:      {corr(nw_raw_te,  g_ic_te):+.4f}")


# ================================================================
# PART 5: SUBMISSION PRIORITY
# ================================================================
print("\n" + "=" * 70)
print("PART 5: SUBMISSION PRIORITY GUIDE")
print("=" * 70)

# Rank variants by PI OOF
pi_ranking = sorted(variant_keys, key=lambda k: results[k]['med'], reverse=True)
best_pi    = pi_ranking[0]

# Rank by distance from baseline (corr-based — lower corr = more different)
base_p_norm = base_p - base_p.mean()
base_n = np.linalg.norm(base_p_norm)
diffs = {}
for k in variant_keys:
    if k == 'baseline': continue
    p = te_preds[k]
    p_n = p - p.mean()
    n2 = np.linalg.norm(p_n)
    c = (base_p_norm @ p_n) / (base_n*n2) if base_n>1e-10 and n2>1e-10 else 1.0
    diffs[k] = 1 - c  # directional difference from baseline

most_different = sorted(diffs, key=lambda k: diffs[k], reverse=True)

print(f"""
  ── CONFIRMED LB SCORES (reference) ────────────────────────────────
  threeway_r30_k40_g29   +0.00124  ← CURRENT BEST
  hybrid_grinold_kernel  +0.00115  (70% NW + 30% Grinold)
  grinold_allday_p005    +0.00096  (pure Grinold)

  ── PI OOF ORDER (unreliable for kernel — treat as weak signal) ──────""")
for i, k in enumerate(pi_ranking[:4]):
    print(f"  {i+1}. {k:<8}  {descriptions[k]:<36}  PI={results[k]['med']:+.5f}")

print(f"""
  ── MOST DIFFERENT FROM CURRENT BEST ────────────────────────────────""")
for k in most_different[:4]:
    print(f"  {k:<8}  {descriptions[k]:<36}  diff={diffs[k]:.4f}  "
          f"PI={results[k]['med']:+.5f}")

print(f"""
  ── RECOMMENDED SUBMIT ORDER ────────────────────────────────────────
  Reasoning: OOF inversion means PI OOF is unreliable for kernel.
  Prioritise variants that are (a) different from baseline and
  (b) have theoretical backing.

  1. var_C_nw.csv    — GOLD_NW only
     Gaussian kernel targets true similarity vs cosine KNN hard cutoff.
     Different enough to test independently.

  2. var_ABC_all.csv — all three combined
     If any single improvement helps, the combined should help more.
     Best case upside; downside bounded by component quality.

  3. var_A_icir.csv  — ICIR_G only
     Pure theory: ICIR penalises inconsistent features. No kernel risk.
     Lowest variance bet — if it helps, the improvement is clean.

  Total elapsed: {(time.time()-t0)/60:.1f} min
""")
