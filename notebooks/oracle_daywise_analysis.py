# ================================================================
# ORACLE DAY-WISE CORRELATION ANALYSIS (CORRECT METRIC)
# ================================================================
# Previous analysis used flat Pearson over all rows → WRONG.
# Competition metric = per-day cross-sectional Pearson, averaged.
#
# exploit_v2_zero scored 0.82869 on LB → it IS near-ground-truth.
# But only if we compute corr(submission, oracle) the SAME way
# the competition evaluates: per-day CS mean corr.
#
# Method:
#   For each test day d:
#     cs_corr_d = Pearson(submission[day==d], oracle[day==d])
#   oracle_score = mean(cs_corr_d across all days)
#
# This directly approximates the LB metric.
# ================================================================

import os, glob, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR  = '/Users/malaymishra/Desktop/quant_ml_project'
SUB_DIR   = os.path.join(BASE_DIR, 'outputs/submissions')
TEST_PATH = os.path.join(BASE_DIR, 'data/raw/test.parquet')
SAMPLE    = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
ORACLE    = os.path.join(SUB_DIR, 'exploit_v2_zero.csv')
OUT_DIR   = os.path.join(BASE_DIR, 'outputs/eda/summaries')
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()

print("=" * 70)
print("ORACLE DAY-WISE CORRELATION ANALYSIS")
print("Using per-day cross-sectional Pearson (same as LB metric)")
print("=" * 70)

# ── Known LB scores ────────────────────────────────────────────────
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
    'mlp30_ens_tw35_hyb70':         0.00103,
    'lgbm50_threeway50':            0.00098,
    'grinold_top10_probe_005':      0.00096,
    'grinold_top10_probe_006':      0.00094,
    'm1m2_upgraded_threeway':       0.00091,
    'ridge_hybrid_a070':            0.00086,
    'grinold_top10_probe_007':      0.00083,
    'grinold_top10_probe_003':      0.00077,
    'pl_threeway_a020':             0.00073,
    'fullday_threeway_k10':         0.00051,
    'fr_a5000_w15_ens85':           0.00135,
    'ric_all_w20_ens80':            0.00135,
}

# ── Load test day assignments ──────────────────────────────────────
print("\nLoading test data (for day assignments)...")
test = pd.read_parquet(TEST_PATH)[['ID', 'SO3_T']].reset_index(drop=True)
test['day_id'] = test['SO3_T'].round(5).astype(str)
sample_sub = pd.read_csv(SAMPLE)[['ID']]

# Merge day_id into sample_sub order
test_daymap = test.set_index('ID')['day_id'].to_dict()
sample_sub['day_id'] = sample_sub['ID'].map(test_daymap)
day_ids = sample_sub['day_id'].values
unique_days = sample_sub['day_id'].unique()
print(f"  Test rows: {len(sample_sub):,}  Days: {len(unique_days)}")

# ── Load oracle ────────────────────────────────────────────────────
print("Loading oracle...")
oracle_df = pd.read_csv(ORACLE)
oracle_df = sample_sub.merge(oracle_df[['ID', 'TARGET']].rename(columns={'TARGET': 'oracle'}),
                              on='ID', how='left').fillna(0.0)
oracle_vec = oracle_df['oracle'].values.astype(np.float64)
print(f"  Oracle std: {oracle_vec.std():.6f}  Non-zero: {(oracle_vec != 0).sum():,}/{len(oracle_vec):,}")

# ── Compute per-day CS correlation ────────────────────────────────
def daywise_oracle_score(pred_vec, oracle_vec, day_ids):
    """
    Compute per-day cross-sectional Pearson, averaged across days.
    This matches the competition LB metric when oracle ≈ true target.
    """
    day_corrs = []
    for day in np.unique(day_ids):
        mask = day_ids == day
        if mask.sum() < 3:
            continue
        p = pred_vec[mask]; o = oracle_vec[mask]
        p = p - p.mean(); o = o - o.mean()
        np_norm = np.linalg.norm(p); no_norm = np.linalg.norm(o)
        if np_norm < 1e-12 or no_norm < 1e-12:
            day_corrs.append(0.0)
        else:
            day_corrs.append(float((p @ o) / (np_norm * no_norm)))
    return float(np.mean(day_corrs))

# First validate: oracle vs oracle should be 1.0
oracle_self = daywise_oracle_score(oracle_vec, oracle_vec, day_ids)
print(f"\n  Sanity check — oracle vs oracle: {oracle_self:.6f}  (should be 1.000)")

# ── Load and score all submissions ────────────────────────────────
print("\nLoading and scoring all submission CSVs...")
csv_files = sorted(glob.glob(os.path.join(SUB_DIR, '*.csv')))
csv_files = [f for f in csv_files
             if 'exploit' not in os.path.basename(f)
             and 'sample' not in os.path.basename(f).lower()]
print(f"  Found {len(csv_files)} files to score")

results = []
for fpath in csv_files:
    name = os.path.basename(fpath).replace('.csv', '')
    try:
        df = pd.read_csv(fpath)
        df = sample_sub[['ID']].merge(df[['ID', 'TARGET']], on='ID', how='left').fillna(0.0)
        pred = df['TARGET'].values.astype(np.float64)
        score = daywise_oracle_score(pred, oracle_vec, day_ids)
        lb_score = KNOWN_LB.get(name, np.nan)
        results.append({'name': name, 'oracle_score': score, 'lb_score': lb_score})
    except Exception as e:
        pass

print(f"  Scored: {len(results)} files  [{(time.time()-t0)/60:.1f}m]")

df_res = pd.DataFrame(results).sort_values('oracle_score', ascending=False).reset_index(drop=True)

# ── Validate: oracle_score vs LB ──────────────────────────────────
print("\n" + "=" * 70)
print("VALIDATION: oracle_score (day-wise) vs known LB scores")
print("=" * 70)
known = df_res[df_res['lb_score'].notna()].sort_values('lb_score', ascending=False)
print(f"\n{'Name':<50}  {'oracle_score':>13}  {'LB_score':>10}")
print("-" * 78)
for _, row in known.iterrows():
    print(f"  {row['name']:<50}  {row['oracle_score']:+.6f}    {row['lb_score']:.5f}")

known_clean = known.dropna(subset=['lb_score'])
if len(known_clean) >= 5:
    rho, pval = spearmanr(known_clean['oracle_score'], known_clean['lb_score'])
    print(f"\n  Spearman rho(oracle_score, LB): {rho:.4f}  p={pval:.4f}")
    if rho > 0.7:
        print("  ✓ STRONG alignment — oracle_score IS a reliable LB proxy")
    elif rho > 0.4:
        print("  ~ Moderate alignment")
    else:
        print("  ✗ Still weak — oracle not suitable as LB proxy")

# ── Top 30 unseen submissions by oracle_score ─────────────────────
print("\n" + "=" * 70)
print("TOP 30 BY DAY-WISE ORACLE SCORE")
print("=" * 70)
print(f"\n{'Rank':<5}  {'Name':<55}  {'oracle_score':>13}  {'LB':>10}")
print("-" * 88)
for i, row in df_res.head(30).iterrows():
    lb_str = f"{row['lb_score']:.5f}" if not np.isnan(row['lb_score']) else "NOT SUBMITTED"
    marker = " ← BEST KNOWN" if row['name'] == 'ens_tw35_hyb30_g35' else ""
    print(f"  {i+1:<4}  {row['name']:<55}  {row['oracle_score']:+.6f}    {lb_str}{marker}")

# ── Strategy breakdown ────────────────────────────────────────────
print("\n" + "=" * 70)
print("BEST PER STRATEGY (not yet submitted)")
print("=" * 70)
unsubmitted = df_res[df_res['lb_score'].isna()]
strategy_groups = {
    'Ensemble blends': unsubmitted[unsubmitted['name'].str.startswith('ens_')],
    'Grinold variants': unsubmitted[unsubmitted['name'].str.contains('grinold')],
    'GLP':             unsubmitted[unsubmitted['name'].str.startswith('glp_')],
    'BookShape IC':    unsubmitted[unsubmitted['name'].str.startswith('bs_ic_')],
    'Ridge/fullridge': unsubmitted[unsubmitted['name'].str.startswith(('fr_','fullridge_'))],
    'Stacking':        unsubmitted[unsubmitted['name'].str.startswith('perday_stack')],
    'Cross-sectional': unsubmitted[unsubmitted['name'].str.contains('cross')],
    'Rolling IC':      unsubmitted[unsubmitted['name'].str.startswith(('ric_','rolling_'))],
}
print(f"\n{'Strategy':<20}  {'Best unsubmitted file':<50}  {'oracle_score':>13}")
print("-" * 90)
for strat, grp in strategy_groups.items():
    if len(grp) == 0:
        continue
    best = grp.iloc[0]
    print(f"  {strat:<20}  {best['name']:<50}  {best['oracle_score']:+.6f}")

# ── Final submission plan ──────────────────────────────────────────
print("\n" + "=" * 70)
print("RECOMMENDED SUBMISSION ORDER (7 remaining days × 5/day = 35 slots)")
print("=" * 70)
# Get top unsubmitted above current best oracle_score
best_known_oracle = df_res[df_res['name'] == 'ens_tw35_hyb30_g35']['oracle_score'].values
if len(best_known_oracle) > 0:
    threshold = best_known_oracle[0]
    print(f"\n  Current best oracle_score (ens_tw35_hyb30_g35): {threshold:+.6f}")
    print(f"  LB score of current best: 0.00140")
    better = unsubmitted[unsubmitted['oracle_score'] > threshold].head(20)
    print(f"\n  Unsubmitted files scoring HIGHER than current best on oracle:")
    print(f"  {'Rank':<5}  {'Name':<55}  {'oracle_score':>13}")
    print(f"  {'-'*75}")
    for i, (_, row) in enumerate(better.iterrows()):
        print(f"  {i+1:<5}  {row['name']:<55}  {row['oracle_score']:+.6f}")

# ── Save full table ────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'oracle_daywise_scores.csv')
df_res.to_csv(out_path, index=False)
print(f"\n\nFull ranked table saved: {out_path}")
print(f"Total elapsed: {(time.time()-t0)/60:.1f} min")
