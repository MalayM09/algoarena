# ================================================================
# LB-SCORE ENSEMBLE — Combine best submissions by correlation analysis
# ================================================================
# Strategy:
#   1. Load all candidate CSVs
#   2. Compute pairwise Pearson correlations
#   3. Try: equal-weight, score-weighted, diversity-penalised blends
#   4. Save all candidates for LB testing
#
# Key insight: ensemble helps only when predictions are NOT collinear.
# If two submissions have corr > 0.98 they are nearly identical —
# averaging them gives almost no benefit. Use pairwise corr to
# identify the most diverse high-performers.
# ================================================================

import os
import numpy as np
import pandas as pd

BASE_DIR  = '/Users/malaymishra/Desktop/quant_ml_project'
SUB_DIR   = os.path.join(BASE_DIR, 'outputs/submissions')
SAMPLE    = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
TARGET_STD = 0.000948

# ── Known LB scores for all candidates ───────────────────────────
CANDIDATES = {
    'threeway_r30_k40_g29':      0.00124,
    'threeway_g15_v2_full':      0.00122,
    'fourway_r27_k36_g27_l10':   0.00119,
    'hybrid_grinold_kernel':     0.00115,
    'lgbm50_threeway50':         0.00098,
    'grinold_top10_probe_005':   0.00096,
    'grinold_top10_probe_006':   0.00094,
    'm1m2_upgraded_threeway':    0.00091,
    'ridge_hybrid_a070':         0.00086,
    'grinold_top10_probe_007':   0.00083,
    'grinold_top10_probe_003':   0.00077,
}

# ── Helpers ───────────────────────────────────────────────────────
def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d > 1e-12 else 0.0

def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

sample_sub = pd.read_csv(SAMPLE)[['ID']]

# ── Load all CSVs ─────────────────────────────────────────────────
print("Loading CSVs...")
data = {}
missing = []
for name in CANDIDATES:
    path = os.path.join(SUB_DIR, f'{name}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = sample_sub.merge(df[['ID','TARGET']], on='ID', how='left').fillna(0.0)
        data[name] = df['TARGET'].values.astype(np.float64)
        print(f"  ✓  {name:<45}  LB={CANDIDATES[name]:.5f}")
    else:
        missing.append(name)
        print(f"  ✗  {name}  (NOT FOUND)")

if missing:
    print(f"\n  Warning: {len(missing)} file(s) not found: {missing}")

names  = list(data.keys())
scores = np.array([CANDIDATES[n] for n in names])
preds  = np.vstack([data[n] for n in names])  # (n_subs, n_rows)
N      = len(names)

# ── Pairwise correlation matrix ───────────────────────────────────
print(f"\nPairwise Pearson correlations ({N} subs):")
corr_mat = np.ones((N, N))
for i in range(N):
    for j in range(i+1, N):
        r = pearson_r(preds[i], preds[j])
        corr_mat[i, j] = r
        corr_mat[j, i] = r

# Print matrix
header = ''.join([f'{n[:8]:>10}' for n in names])
print(f"{'':>40}" + header)
for i, ni in enumerate(names):
    row = ''.join([f'{corr_mat[i,j]:>10.4f}' for j in range(N)])
    print(f"  {ni[:38]:<40}" + row)

# Summary: average correlation of each submission with all others
avg_corr = (corr_mat.sum(1) - 1) / (N - 1)
print(f"\n  Average correlation with others:")
for i, n in enumerate(names):
    print(f"    {n:<45}  avg_corr={avg_corr[i]:.4f}  LB={CANDIDATES[n]:.5f}")

# ── Ensemble strategies ───────────────────────────────────────────
print(f"\n{'='*60}")
print("ENSEMBLE STRATEGIES")
print('='*60)

def save_ensemble(blend, name):
    bs = auto_scale(blend)
    sub = pd.DataFrame({'ID': sample_sub['ID'], 'TARGET': bs})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(SUB_DIR, f'ens_{name}.csv'), index=False)
    best = data['threeway_r30_k40_g29']
    print(f"  ens_{name:<45}  std={bs.std():.6f}  corr_vs_best={pearson_r(bs, best):+.4f}")

# Strategy 1: Top-3 equal weight (threeway, g15, fourway)
top3 = ['threeway_r30_k40_g29', 'threeway_g15_v2_full', 'fourway_r27_k36_g27_l10']
if all(n in data for n in top3):
    blend = sum(data[n] for n in top3) / 3
    save_ensemble(blend, 'top3_equal')

# Strategy 2: Top-5 equal weight (add hybrid + lgbm)
top5 = ['threeway_r30_k40_g29', 'threeway_g15_v2_full', 'fourway_r27_k36_g27_l10',
        'hybrid_grinold_kernel', 'lgbm50_threeway50']
if all(n in data for n in top5):
    blend = sum(data[n] for n in top5) / 5
    save_ensemble(blend, 'top5_equal')

# Strategy 3: LB-score weighted (all available)
w = scores / scores.sum()
blend = (preds * w[:, None]).sum(0)
save_ensemble(blend, 'lb_weighted_all')

# Strategy 4: LB-score weighted, top-4 only (≥0.00115)
top4_mask = scores >= 0.00115
top4_names = [n for n, m in zip(names, top4_mask) if m]
top4_scores = scores[top4_mask]
top4_preds  = preds[top4_mask]
if top4_preds.shape[0] > 1:
    w4 = top4_scores / top4_scores.sum()
    blend = (top4_preds * w4[:, None]).sum(0)
    save_ensemble(blend, 'lb_weighted_top4')
    print(f"    Top-4 members: {top4_names}")

# Strategy 5: Diversity-aware — pick subs with low mutual correlation
# Use threeway (best) as anchor, add the submission that adds most
# new information (lowest corr with current ensemble)
print(f"\n  Greedy diversity selection:")
anchor = ['threeway_r30_k40_g29']
pool   = [n for n in names if n != 'threeway_r30_k40_g29']
selected = list(anchor)

for step in range(4):  # add up to 4 more
    best_cand = None
    best_score = -999
    current_blend = sum(data[n] for n in selected) / len(selected)
    for cand in pool:
        if cand in selected: continue
        trial = (current_blend * len(selected) + data[cand]) / (len(selected) + 1)
        # Score = avg LB of selected - diversity penalty (correlation with current)
        div_penalty = pearson_r(data[cand], current_blend)
        lb_gain = CANDIDATES[cand]
        composite = lb_gain - 0.5 * div_penalty  # tune tradeoff here
        if composite > best_score:
            best_score = composite
            best_cand  = cand
    if best_cand:
        selected.append(best_cand)
        print(f"    Step {step+1}: add {best_cand:<45}  LB={CANDIDATES[best_cand]:.5f}  composite={best_score:.5f}")

greedy_blend = sum(data[n] for n in selected) / len(selected)
save_ensemble(greedy_blend, 'greedy_diversity')
print(f"    Final selection: {selected}")

# Strategy 6: Anchor-heavy — 60% best + 40% average of rest
anchor_w = 0.60
rest_names = [n for n in names if n != 'threeway_r30_k40_g29']
if rest_names and 'threeway_r30_k40_g29' in data:
    rest_blend = sum(data[n] for n in rest_names if n in data) / len(rest_names)
    blend = anchor_w * data['threeway_r30_k40_g29'] + (1-anchor_w) * rest_blend
    save_ensemble(blend, 'anchor60_rest40')

# Strategy 7: Best pair — find the 2-sub combination with highest
# theoretical gain (both high LB + low mutual corr)
print(f"\n  Best 2-submission pairs by LB×diversity:")
pair_results = []
for i in range(N):
    for j in range(i+1, N):
        avg_lb  = (scores[i] + scores[j]) / 2
        corr_ij = corr_mat[i, j]
        # Higher avg LB and lower corr = better pair
        pair_score = avg_lb * (1 - corr_ij)
        pair_results.append((pair_score, names[i], names[j], avg_lb, corr_ij))
pair_results.sort(reverse=True)
print(f"  {'Score':>8}  {'Sub A':<45}  {'Sub B':<45}  avg_LB   corr")
for ps, na, nb, avg_lb, corr_ij in pair_results[:5]:
    print(f"  {ps:8.6f}  {na:<45}  {nb:<45}  {avg_lb:.5f}  {corr_ij:.4f}")
# Save the best pair
_, best_a, best_b, _, _ = pair_results[0]
if best_a in data and best_b in data:
    blend = (data[best_a] + data[best_b]) / 2
    save_ensemble(blend, f'best_pair')
    print(f"\n    Saved best pair: {best_a} + {best_b}")

print(f"""
{'='*60}
SUMMARY — WHICH ENSEMBLES TO SUBMIT
{'='*60}

  Priority order:
  1. ens_greedy_diversity  — best theoretical combination
  2. ens_lb_weighted_top4  — score-weighted top performers
  3. ens_top3_equal        — safest (highest-scoring 3 subs)
  4. ens_best_pair         — most information-diverse pair

  corr_vs_best < 0.99 → meaningfully different, worth submitting
  corr_vs_best > 0.99 → nearly identical to current best, skip

  If ensemble LB > 0.00124: averaging is helping (diverse errors).
  If ensemble LB ≈ 0.00124: submissions are too correlated to gain.
  If ensemble LB < 0.00124: individual best is more stable.
""")
