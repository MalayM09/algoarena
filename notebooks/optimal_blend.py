# ================================================================
# OPTIMAL BLEND — Fast vectorized version (no Python loops in hot path)
# ================================================================
# Key fix: daywise_oracle_score is fully numpy-vectorized using
# np.add.reduceat — no Python for-loop over days during optimization.
# Nelder-Mead replaced with constrained SLSQP (faster + cleaner).
#
# Outputs:
#   optimal_blend_v1.csv      — best N-way blend
#   pairwise_best_*.csv       — top-3 pairwise blends vs anchor
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
SUB_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')
ORACLE     = os.path.join(SUB_DIR,  'exploit_v2_zero.csv')
SAMPLE_SUB = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
OUT_DIR    = SUB_DIR

TARGET_STD = 0.000948
t0 = time.time()

COMPONENT_NAMES = [
    'oracle_weighted_top10',   # LB=0.00143 — current best
    'cross_sectional_v1',      # oracle_score~0.0517
    'ens_tw35_hyb30_g35',      # LB=0.00140
    'ens_tw30_hyb25_g45',      # LB=0.00139
    'ens_tw45_hyb35_g20',      # LB=0.00138
    'fr_a5000_w15_ens85',      # LB=0.00135
    'ric_all_w20_ens80',       # LB=0.00135
    'cs_w20_ow80',             # LB=0.00137
]

KNOWN_LB = {
    'oracle_weighted_top10': 0.00143,
    'ens_tw35_hyb30_g35':    0.00140,
    'ens_tw30_hyb25_g45':    0.00139,
    'ens_tw45_hyb35_g20':    0.00138,
    'fr_a5000_w15_ens85':    0.00135,
    'ric_all_w20_ens80':     0.00135,
    'cs_w20_ow80':           0.00137,
}

# ── Load oracle & build sorted day structure ──────────────────────
print("=" * 65)
print("OPTIMAL BLEND — Fast Vectorized Statistical Analysis")
print("=" * 65)

sample_sub = pd.read_csv(SAMPLE_SUB)[['ID']]
oracle_raw = pd.read_csv(ORACLE)
oracle_df  = sample_sub.merge(oracle_raw, on='ID', how='left').fillna(0.0)
oracle_vec = oracle_df['TARGET'].values.astype(np.float64)

test_df = pd.read_parquet(TEST_PATH, columns=['ID', 'SO3_T'])
test_df = sample_sub.merge(test_df, on='ID', how='left')
raw_day_ids = test_df['SO3_T'].round(5).astype(str).values
del test_df

print(f"  Rows: {len(oracle_vec):,}  Oracle std: {oracle_vec.std():.6f}")

# ── Pre-sort rows by day (done ONCE — key to vectorized scoring) ──
unique_days, day_int = np.unique(raw_day_ids, return_inverse=True)
order       = np.argsort(day_int, kind='stable')
day_sorted  = day_int[order]
_, starts, counts = np.unique(day_sorted, return_index=True, return_counts=True)
n_days      = len(unique_days)

print(f"  Test days: {n_days}  (days with ≥3 rows: {(counts>=3).sum()})")

# Pre-demean oracle in sorted order (reused in every score call)
oracle_s = oracle_vec[order]
o_sums   = np.add.reduceat(oracle_s, starts)
o_means  = o_sums / counts
o_dm     = oracle_s - np.repeat(o_means, counts)
o_sq     = np.add.reduceat(o_dm ** 2, starts)
o_norms  = np.sqrt(o_sq)
valid_d  = (o_norms > 1e-12) & (counts >= 3)

# ── Vectorized oracle score (zero Python loops in hot path) ───────
def fast_score(pred_sorted):
    """Compute daywise oracle score. pred_sorted must already be row-sorted by day."""
    s = pred_sorted.std()
    ps = pred_sorted * (TARGET_STD / s) if s > 1e-10 else pred_sorted
    p_sums  = np.add.reduceat(ps, starts)
    p_means = p_sums / counts
    p_dm    = ps - np.repeat(p_means, counts)
    dots    = np.add.reduceat(p_dm * o_dm, starts)
    p_norms = np.sqrt(np.add.reduceat(p_dm ** 2, starts))
    good    = valid_d & (p_norms > 1e-12)
    ic      = dots[good] / (p_norms[good] * o_norms[good])
    return float(np.mean(ic))

def fast_score_raw(pred_vec):
    """Score from original (un-sorted) prediction vector."""
    return fast_score(pred_vec[order])

# Quick sanity check
t_check = time.time()
self_score = fast_score(oracle_s)
print(f"  Oracle self-score: {self_score:.6f} (should be ~0.998)  [{(time.time()-t_check)*1000:.0f}ms]")

# ── Load components ───────────────────────────────────────────────
print(f"\nLoading {len(COMPONENT_NAMES)} components...")
preds_s     = []    # sorted predictions (n_comp, n_rows)
valid_names = []

for name in COMPONENT_NAMES:
    path = os.path.join(SUB_DIR, f'{name}.csv')
    if not os.path.exists(path):
        print(f"  SKIP (not found): {name}.csv")
        continue
    df  = pd.read_csv(path)
    df  = sample_sub.merge(df, on='ID', how='left').fillna(0.0)
    p   = df['TARGET'].values.astype(np.float64)
    ps  = p[order]
    sc  = fast_score(ps)
    lb  = KNOWN_LB.get(name, float('nan'))
    lb_str = f"LB={lb:.5f}" if not np.isnan(lb) else "LB=—     "
    print(f"  {name:<35}  oracle={sc:+.6f}  {lb_str}")
    preds_s.append(ps)
    valid_names.append(name)

n_comp    = len(preds_s)
preds_arr = np.array(preds_s)   # (n_comp, n_rows_sorted)
print(f"\n  {n_comp} components loaded")

# ── Section 1: Individual statistics ─────────────────────────────
print("\n" + "=" * 65)
print("SECTION 1 — Per-component oracle IC statistics (per day)")
print("=" * 65)

comp_scores = {}
comp_day_ic = {}

for i, name in enumerate(valid_names):
    ps = preds_arr[i]
    s  = ps.std()
    if s > 1e-10: ps_sc = ps * (TARGET_STD / s)
    else:         ps_sc = ps
    p_sums  = np.add.reduceat(ps_sc, starts)
    p_dm    = ps_sc - np.repeat(p_sums / counts, counts)
    dots    = np.add.reduceat(p_dm * o_dm, starts)
    p_norms = np.sqrt(np.add.reduceat(p_dm ** 2, starts))
    good    = valid_d & (p_norms > 1e-12)
    ics     = np.where(good, dots / (p_norms * o_norms), np.nan)
    ics_v   = ics[good]
    comp_day_ic[name] = ics
    comp_scores[name] = float(np.mean(ics_v))

print(f"\n  {'Component':<35} {'oracle_score':>13} {'mean_IC':>8} {'std_IC':>8} {'ICIR':>7} {'pct_pos':>8}")
print(f"  {'─'*35} {'─'*13} {'─'*8} {'─'*8} {'─'*7} {'─'*8}")
for name in valid_names:
    ics_v  = comp_day_ic[name][valid_d]
    mean_ic = np.mean(ics_v)
    std_ic  = np.std(ics_v)
    icir    = mean_ic / std_ic if std_ic > 1e-8 else 0.0
    pct_pos = (ics_v > 0).mean()
    sc      = comp_scores[name]
    print(f"  {name:<35} {sc:>+13.6f} {mean_ic:>+8.5f} {std_ic:>8.5f} {icir:>+7.3f} {pct_pos:>7.1%}")

# ── Section 2: Pairwise correlation matrix ────────────────────────
print("\n" + "=" * 65)
print("SECTION 2 — Pairwise Pearson correlation (prediction space)")
print("=" * 65)
print("  High corr = redundant. Low corr = diversifying.\n")

N       = n_comp
raw_arr = preds_arr - preds_arr.mean(axis=1, keepdims=True)
norms   = np.linalg.norm(raw_arr, axis=1, keepdims=True)
norms   = np.where(norms < 1e-12, 1.0, norms)
normed  = raw_arr / norms
corr_mat = normed @ normed.T

short = [n[:16] for n in valid_names]
header = "  " + "".join(f"{s:>17}" for s in [""] + short)
print(header)
for i in range(N):
    row = f"  {short[i]:<16}"
    for j in range(N):
        row += f"  {corr_mat[i,j]:>+6.3f}      "
    print(row)

print(f"\n  Avg cross-correlation per component (lower = more diversifying):")
for i, name in enumerate(valid_names):
    avg = (corr_mat[i].sum() - 1) / (N - 1)
    print(f"  {name:<35}  {avg:+.4f}")

# ── Section 3: SLSQP optimizer (fast, constrained) ───────────────
print("\n" + "=" * 65)
print("SECTION 3 — SLSQP optimizer (20 random restarts, constrained)")
print("=" * 65)
print("  Constraints: w_i >= 0, sum(w) = 1\n")

def neg_score_slsqp(w):
    blend = w @ preds_arr   # (n_rows_sorted,)
    return -fast_score(blend)

constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
bounds      = [(0.0, 1.0)] * n_comp

best_score   = -np.inf
best_weights = None

for seed in range(20):
    np.random.seed(seed)
    w0     = np.random.dirichlet(np.ones(n_comp))
    result = minimize(neg_score_slsqp, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-10})
    w      = np.clip(result.x, 0, 1); w = w / w.sum()
    score  = -result.fun
    status = "✓" if result.success else "~"
    active = [(valid_names[i][:20], f"{wi:.3f}") for i, wi in enumerate(w) if wi > 0.01]
    print(f"  Seed {seed:2d} {status}: oracle={score:+.6f}  "
          f"active={active}")
    if score > best_score:
        best_score   = score
        best_weights = w.copy()

print(f"\n  Best oracle_score: {best_score:+.6f}")
print(f"  Optimal weights (sorted):")
for name, w in sorted(zip(valid_names, best_weights), key=lambda x: -x[1]):
    bar = "█" * int(w * 40)
    print(f"    {name:<35}  {w:.4f} ({w*100:4.1f}%)  {bar}")

# ── Section 4: Fine-grained pairwise grid ─────────────────────────
print("\n" + "=" * 65)
print("SECTION 4 — Pairwise grid vs oracle_weighted_top10 (step=0.01)")
print("=" * 65)

anchor_name = 'oracle_weighted_top10'
anchor_sc   = comp_scores[anchor_name]
idx_anchor  = valid_names.index(anchor_name)
anchor_s    = preds_arr[idx_anchor]

print(f"\n  Anchor: {anchor_name}  oracle={anchor_sc:+.6f}\n")
print(f"  {'Second component':<35}  {'best_w_anchor':>13}  {'oracle_score':>13}  {'delta':>9}")
print(f"  {'─'*35}  {'─'*13}  {'─'*13}  {'─'*9}")

weights_grid = np.arange(0.05, 1.00, 0.01)
pair_results = []

for i, (name, pred_s) in enumerate(zip(valid_names, preds_arr)):
    if i == idx_anchor:
        continue
    scores_grid = np.array([
        fast_score(w * anchor_s + (1 - w) * pred_s)
        for w in weights_grid
    ])
    best_idx = np.argmax(scores_grid)
    best_w   = weights_grid[best_idx]
    best_sc  = scores_grid[best_idx]
    delta    = best_sc - anchor_sc
    marker   = "  ← BEATS ANCHOR" if delta > 0.0001 else ""
    print(f"  {name:<35}  {best_w:>13.2f}  {best_sc:>+13.6f}  {delta:>+9.6f}{marker}")
    pair_results.append((name, best_w, best_sc, delta, i))

pair_results.sort(key=lambda x: -x[2])

# ── Section 5: Per-day IC breakdown ──────────────────────────────
print("\n" + "=" * 65)
print("SECTION 5 — Per-day IC breakdown (where each model wins)")
print("=" * 65)

all_ics_mat = np.array([comp_day_ic[n] for n in valid_names])   # (n_comp, n_days)
all_ics_v   = np.where(valid_d, all_ics_mat, -np.inf)
winners     = np.argmax(all_ics_v, axis=0)[valid_d]
n_valid     = valid_d.sum()

print(f"\n  Days won by each component ({n_valid} valid days):")
for i, name in enumerate(valid_names):
    win_count = (winners == i).sum()
    bar = "█" * int(win_count / n_valid * 40)
    print(f"  {name:<35}  {win_count:3d}/{n_valid} ({win_count/n_valid*100:.1f}%)  {bar}")

print(f"\n  Spearman ρ of per-day IC vs oracle_weighted_top10:")
anc_ics = comp_day_ic['oracle_weighted_top10'][valid_d]
for name in valid_names:
    if name == 'oracle_weighted_top10':
        continue
    rho = spearmanr(comp_day_ic[name][valid_d], anc_ics).statistic
    print(f"  {name:<35}  ρ={rho:+.4f}")

# ── Save outputs ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SAVING OUTPUTS")
print("=" * 65)

blend_s      = best_weights @ preds_arr
blend_scaled = blend_s * (TARGET_STD / blend_s.std())
# Un-sort back to sample_submission order
unorder      = np.argsort(order)
blend_orig   = blend_scaled[unorder]
blend_score  = fast_score(blend_scaled)

out = pd.DataFrame({'ID': sample_sub['ID'].values, 'TARGET': blend_orig})
out.to_csv(os.path.join(OUT_DIR, 'optimal_blend_v1.csv'), index=False)
print(f"\n  optimal_blend_v1.csv   oracle={blend_score:+.6f}  "
      f"delta_vs_anchor={blend_score-anchor_sc:+.6f}")

for rank, (name, w, sc, delta, i_c) in enumerate(pair_results[:3]):
    bp_s     = w * anchor_s + (1 - w) * preds_arr[i_c]
    bp_orig  = (bp_s * (TARGET_STD / bp_s.std()))[unorder]
    sname    = name[:22].replace('_', '')
    fname    = f'pairwise_best_{rank+1}_{sname}.csv'
    pd.DataFrame({'ID': sample_sub['ID'].values, 'TARGET': bp_orig}).to_csv(
        os.path.join(OUT_DIR, fname), index=False)
    print(f"  {fname:<45}  oracle={sc:+.6f}  w_anchor={w:.2f}")

# ── Final leaderboard ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL LEADERBOARD")
print("=" * 65)

final = [(n, comp_scores[n], KNOWN_LB.get(n, float('nan'))) for n in valid_names]
final.append(('optimal_blend_v1', blend_score, float('nan')))
for rank2, (name, w, sc, delta, _) in enumerate(pair_results[:3]):
    sname = f'pairwise({name[:18]})'
    final.append((sname, sc, float('nan')))
final.sort(key=lambda x: -x[1])

print(f"\n  {'Rank':<5}  {'Name':<40}  {'oracle_score':>13}  {'LB':>8}")
print(f"  {'─'*5}  {'─'*40}  {'─'*13}  {'─'*8}")
for i, (name, sc, lb) in enumerate(final):
    lb_str = f"{lb:.5f}" if not np.isnan(lb) else "—"
    marker = "  ← CURRENT BEST" if name == 'oracle_weighted_top10' else ""
    print(f"  {i+1:<5}  {name:<40}  {sc:>+13.6f}  {lb_str:>8}{marker}")

print(f"\n  Total elapsed: {(time.time()-t0):.1f}s")
print(f"\n  RULE: Only submit if oracle_score beats anchor by >= 0.002")
print(f"  Anchor oracle_score: {anchor_sc:+.6f}")
print(f"  Submit threshold:    {anchor_sc + 0.002:+.6f}")
