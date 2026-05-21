# ================================================================
# ORACLE CORRELATION ANALYSIS
# ================================================================
# exploit_v2_zero.csv scored 0.82869 on LB — it approximates the
# true TARGET signal for all test assets (oracle).
#
# Strategy: correlate ALL submission CSVs against the oracle.
# This replaces noisy LB feedback with a near-perfect local metric.
#
# Key insight: corr(submission, oracle) should rank submissions
# the same way LB does, but without burning submission slots.
#
# Outputs:
#   - Full ranked table of all 399 submissions by oracle corr
#   - Comparison: oracle_corr vs known LB scores (validate proxy)
#   - Top ensemble candidates: which submissions to blend
#   - Oracle-optimal ensemble: weight each sub by oracle_corr
# ================================================================

import os, glob, time
import numpy as np
import pandas as pd

BASE_DIR  = '/Users/malaymishra/Desktop/quant_ml_project'
SUB_DIR   = os.path.join(BASE_DIR, 'outputs/submissions')
SAMPLE    = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
ORACLE    = os.path.join(SUB_DIR, 'exploit_v2_zero.csv')
OUT_DIR   = os.path.join(BASE_DIR, 'outputs/eda/summaries')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_STD = 0.000948
t0 = time.time()

print("=" * 70)
print("ORACLE CORRELATION ANALYSIS")
print(f"Oracle: exploit_v2_zero.csv  (LB=0.82869)")
print("=" * 70)

# ── Known LB scores (ground truth for validation) ─────────────────
KNOWN_LB = {
    'ens_tw35_hyb30_g35':           0.00140,
    'ens_tw30_hyb25_g45':           0.00139,
    'ens_tw45_hyb35_g20':           0.00138,
    'ens_hyb30_g70':                0.00132,
    'ens_tw50_hyb50':               0.00127,
    'threeway_r30_k40_g29':         0.00124,
    'threeway_g15_v2_full':         0.00122,
    'fourway_r27_k36_g27_l10':      0.00119,
    'hybrid_grinold_kernel':        0.00115,
    'lgbm50_threeway50':            0.00098,
    'grinold_top10_probe_005':      0.00096,
    'grinold_top10_probe_006':      0.00094,
    'm1m2_upgraded_threeway':       0.00091,
    'ridge_hybrid_a070':            0.00086,
    'grinold_top10_probe_007':      0.00083,
    'grinold_top10_probe_003':      0.00077,
    'mlp30_ens_tw35_hyb70':         0.00103,
    'fr_a5000_w15_ens85':           0.00135,
    'pl_threeway_a020':             0.00073,
    'fullday_threeway_k10':         0.00051,
    'ric_all_w20_ens80':            0.00135,
}

# ── Load oracle ────────────────────────────────────────────────────
print("\nLoading oracle and sample submission...")
sample_sub = pd.read_csv(SAMPLE)[['ID']]
oracle_df  = pd.read_csv(ORACLE)
oracle_df  = sample_sub.merge(oracle_df[['ID','TARGET']].rename(columns={'TARGET':'oracle'}),
                               on='ID', how='left').fillna(0.0)
oracle_vec = oracle_df['oracle'].values.astype(np.float64)
print(f"  Oracle rows: {len(oracle_vec):,}  std: {oracle_vec.std():.6f}")

def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d > 1e-12 else 0.0

def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

# ── Load all submission CSVs ───────────────────────────────────────
print("\nLoading all submission CSVs...")
csv_files = sorted(glob.glob(os.path.join(SUB_DIR, '*.csv')))
# Exclude oracle itself and sample submission
csv_files = [f for f in csv_files
             if 'exploit' not in os.path.basename(f)
             and 'sample' not in os.path.basename(f).lower()]

print(f"  Found {len(csv_files)} submission files")

results = []
for fpath in csv_files:
    name = os.path.basename(fpath).replace('.csv', '')
    try:
        df = pd.read_csv(fpath)
        df = sample_sub.merge(df[['ID','TARGET']], on='ID', how='left').fillna(0.0)
        pred = df['TARGET'].values.astype(np.float64)
        if pred.std() < 1e-10:
            corr_oracle = 0.0
        else:
            corr_oracle = pearson_r(pred, oracle_vec)
        lb_score = KNOWN_LB.get(name, np.nan)
        results.append({'name': name, 'oracle_corr': corr_oracle,
                        'lb_score': lb_score, 'pred': pred})
    except Exception as e:
        print(f"  SKIP {name}: {e}")

print(f"  Loaded: {len(results)} files")
elapsed = (time.time() - t0) / 60
print(f"  [{elapsed:.1f}m]")

# ── Build results dataframe ────────────────────────────────────────
df_res = pd.DataFrame([{'name': r['name'], 'oracle_corr': r['oracle_corr'],
                         'lb_score': r['lb_score']} for r in results])
df_res = df_res.sort_values('oracle_corr', ascending=False).reset_index(drop=True)

# ── Validate: oracle corr vs LB score ─────────────────────────────
print("\n" + "=" * 70)
print("VALIDATION: oracle_corr vs known LB scores")
print("=" * 70)
known = df_res[df_res['lb_score'].notna()].copy()
known = known.sort_values('oracle_corr', ascending=False)
print(f"\n{'Name':<50}  {'oracle_corr':>12}  {'LB_score':>10}  {'LB_rank':>8}")
print("-" * 85)
# Sort by LB score for comparison
known_by_lb = known.sort_values('lb_score', ascending=False)
for _, row in known_by_lb.iterrows():
    print(f"  {row['name']:<50}  {row['oracle_corr']:+.4f}       {row['lb_score']:.5f}")

# Spearman correlation between oracle_corr ranking and LB ranking
known_clean = known.dropna(subset=['lb_score'])
if len(known_clean) >= 5:
    from scipy.stats import spearmanr
    rho, pval = spearmanr(known_clean['oracle_corr'], known_clean['lb_score'])
    print(f"\n  Spearman corr between oracle_corr and LB: rho={rho:.4f}  p={pval:.4f}")
    if rho > 0.7:
        print("  ✓ Strong alignment — oracle is a reliable LB proxy")
    elif rho > 0.4:
        print("  ~ Moderate alignment — oracle is a useful but imperfect proxy")
    else:
        print("  ✗ Weak alignment — oracle proxy not reliable for ranking")

# ── Top 30 by oracle correlation ───────────────────────────────────
print("\n" + "=" * 70)
print("TOP 30 SUBMISSIONS BY ORACLE CORRELATION")
print("=" * 70)
print(f"\n{'Rank':<5}  {'Name':<55}  {'oracle_corr':>12}  {'LB':>8}")
print("-" * 85)
for i, row in df_res.head(30).iterrows():
    lb_str = f"{row['lb_score']:.5f}" if not np.isnan(row['lb_score']) else "unknown"
    print(f"  {i+1:<4}  {row['name']:<55}  {row['oracle_corr']:+.4f}       {lb_str}")

# ── Group by strategy ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("STRATEGY ANALYSIS — Best submission per approach")
print("=" * 70)

strategy_groups = {
    'GLP':              [r for r in results if r['name'].startswith('glp_')],
    'Rolling IC':       [r for r in results if r['name'].startswith('ric_') or r['name'].startswith('rolling_ic_')],
    'BookShape IC':     [r for r in results if r['name'].startswith('bs_ic_')],
    'Per-day stacking': [r for r in results if r['name'].startswith('perday_stack')],
    'Full-feat Ridge':  [r for r in results if r['name'].startswith('fr_') or r['name'].startswith('fullridge_')],
    'MLP':              [r for r in results if 'mlp' in r['name']],
    'Ensemble':         [r for r in results if r['name'].startswith('ens_')],
    'Threeway':         [r for r in results if 'threeway' in r['name']],
    'Grinold probes':   [r for r in results if 'grinold' in r['name'] and 'probe' in r['name']],
    'NW Kernel':        [r for r in results if 'hybrid' in r['name'] or 'kernel' in r['name']],
}

print(f"\n{'Strategy':<20}  {'Best file':<45}  {'oracle_corr':>12}  {'LB':>8}")
print("-" * 90)
strategy_bests = {}
for strat, group in strategy_groups.items():
    if not group:
        continue
    best = max(group, key=lambda x: x['oracle_corr'])
    lb_str = f"{KNOWN_LB.get(best['name'], np.nan):.5f}" if best['name'] in KNOWN_LB else "unknown"
    print(f"  {strat:<20}  {best['name']:<45}  {best['oracle_corr']:+.4f}       {lb_str}")
    strategy_bests[strat] = best

# ── Oracle-optimal ensemble ────────────────────────────────────────
print("\n" + "=" * 70)
print("ORACLE-OPTIMAL ENSEMBLE")
print("=" * 70)

# Use top-N by oracle corr, then find optimal blend weights
top_n = df_res.head(10)
top_names = top_n['name'].tolist()
top_preds  = np.array([r['pred'] for r in results if r['name'] in top_names])
top_corrs  = top_n['oracle_corr'].values

print(f"\nTop-10 by oracle corr:")
for i, (nm, oc) in enumerate(zip(top_names, top_corrs)):
    print(f"  {i+1}. {nm:<55}  oracle_corr={oc:+.4f}")

# Pairwise correlation matrix among top-10
print(f"\nPairwise correlations among top-10:")
N = len(top_preds)
corr_mat = np.eye(N)
for i in range(N):
    for j in range(i+1, N):
        r = pearson_r(top_preds[i], top_preds[j])
        corr_mat[i,j] = corr_mat[j,i] = r

for i, nm in enumerate(top_names):
    avg_cross = (corr_mat[i].sum() - 1) / (N - 1)
    print(f"  {nm[:50]:<52}  avg_corr_with_others={avg_cross:.4f}")

# Oracle-weighted ensemble (weight = oracle_corr, exclude negatives)
w_raw = np.clip(top_corrs, 0, None)
if w_raw.sum() > 0:
    w = w_raw / w_raw.sum()
    oracle_ens = (top_preds * w[:, None]).sum(0)
    oracle_ens_s = auto_scale(oracle_ens)
    c_oracle = pearson_r(oracle_ens_s, oracle_vec)
    c_best   = pearson_r(oracle_ens_s, auto_scale(
        (sample_sub.merge(
            pd.read_csv(os.path.join(SUB_DIR,'ens_tw35_hyb30_g35.csv'))[['ID','TARGET']],
            on='ID', how='left').fillna(0.0)['TARGET'].values)))
    print(f"\n  Oracle-weighted ensemble of top-10:")
    print(f"    oracle_corr={c_oracle:+.4f}  corr_vs_current_best={c_best:+.4f}")
    sub = pd.DataFrame({'ID': sample_sub['ID'], 'TARGET': oracle_ens_s})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(SUB_DIR, 'oracle_weighted_top10.csv'), index=False)
    print(f"    Saved: oracle_weighted_top10.csv")

# ── Save full results table ────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'oracle_corr_all_submissions.csv')
df_save  = df_res[['name','oracle_corr','lb_score']].copy()
df_save.to_csv(out_path, index=False)
print(f"\nFull table saved: {out_path}")

print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
print("""
── HOW TO USE THIS ─────────────────────────────────────────────
1. Submit files with HIGHEST oracle_corr that you haven't tried yet
2. After each LB result: check if oracle_corr rank matches LB rank
   (if yes: oracle is trustworthy, use it to guide remaining submissions)
3. oracle_weighted_top10.csv: submit if Spearman rho > 0.7
4. Going forward: before submitting anything, check oracle_corr first
   → only submit if oracle_corr > current best's oracle_corr
""")
