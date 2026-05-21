# ================================================================
# TRAIN-TEST DISTRIBUTION SHIFT ANALYSIS
# ================================================================
# Answers 4 questions:
#
#   Q1. How different are train and test? (confirm the 99.8% shift)
#   Q2. Which features drift the most between train and test?
#   Q3. How many train rows genuinely look like the test set?
#   Q4. What should we do next given the answers?
#
# Outputs:
#   outputs/analysis/drift_report.csv      — per-feature drift stats
#   outputs/analysis/row_similarity.csv    — per-train-row similarity score
#   outputs/analysis/shift_summary.txt     — plain-English summary
# ================================================================

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import os, warnings
warnings.filterwarnings('ignore')

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
FMAP_PATH  = os.path.join(BASE_DIR, 'feature_map.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/analysis')
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 65)
print("TRAIN-TEST DISTRIBUTION SHIFT ANALYSIS")
print("=" * 65)

# ── Load ─────────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)

feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET'}]
n_feat    = len(feat_cols)
n_train   = len(train)
n_test    = len(test)

print(f"  Train : {n_train:,} rows  × {n_feat} features")
print(f"  Test  : {n_test:,}  rows  × {n_feat} features")
print(f"  Test  is {n_test/n_train*100:.1f}% the size of train")

y_train = train['TARGET'].values

# ── Load feature map (sign-stable features) ───────────────────────
stable_features = []
if os.path.exists(FMAP_PATH):
    fmap = pd.read_csv(FMAP_PATH)
    if 'sign_consistency' in fmap.columns:
        stable_features = fmap[fmap['sign_consistency'] == 1.0]['feature'].tolist() \
                          if 'feature' in fmap.columns else []
        # fallback column name
        if not stable_features and fmap.columns[0] != 'feature':
            col0 = fmap.columns[0]
            stable_features = fmap[fmap['sign_consistency'] == 1.0][col0].tolist()
    print(f"\n  Feature map loaded: {len(fmap)} features, "
          f"{len(stable_features)} sign-stable (sign_consistency=1.0)")
else:
    print("\n  feature_map.csv not found — will use all features")


# ================================================================
# Q1. HOW DIFFERENT ARE TRAIN AND TEST?
# ================================================================
# Method A: Train-vs-test separability using a single feature threshold
# Method B: Mean z-score across all features per row
# ================================================================
print("\n" + "=" * 65)
print("Q1. OVERALL TRAIN-TEST SEPARABILITY")
print("=" * 65)

# Compute per-feature test stats (mean, std)
test_means = test[feat_cols].mean()
test_stds  = test[feat_cols].std().replace(0, 1e-8)
train_means = train[feat_cols].mean()
train_stds  = train[feat_cols].std().replace(0, 1e-8)

# For each train row: average squared z-score under the TEST distribution
# Low value = train row looks like a typical test row
# High value = train row is far from test distribution
print("\nComputing per-row distance from test distribution...")
X_train = train[feat_cols].fillna(0).values
X_test  = test[feat_cols].fillna(0).values

t_mean_arr = test_means.values    # (n_feat,)
t_std_arr  = test_stds.values     # (n_feat,)

# Mean absolute z-score per row under test distribution
# z_ij = (X_train_ij - test_mean_j) / test_std_j
# row_dist_i = mean(|z_ij|)  across all features

# Process in chunks to avoid memory issues
chunk = 5000
row_dist_train = np.zeros(n_train, dtype=np.float32)

for start in range(0, n_train, chunk):
    end   = min(start + chunk, n_train)
    block = X_train[start:end]                            # (chunk, n_feat)
    z     = np.abs((block - t_mean_arr) / t_std_arr)     # (chunk, n_feat)
    row_dist_train[start:end] = z.mean(axis=1)

row_dist_test = np.zeros(n_test, dtype=np.float32)
for start in range(0, n_test, chunk):
    end   = min(start + chunk, n_test)
    block = X_test[start:end]
    z     = np.abs((block - t_mean_arr) / t_std_arr)
    row_dist_test[start:end] = z.mean(axis=1)

print(f"\n  Mean |z-score| distribution under TEST distribution:")
print(f"  {'':12}  {'mean':>8}  {'median':>8}  {'p25':>8}  {'p75':>8}")
print(f"  {'train rows':12}  {row_dist_train.mean():>8.3f}  "
      f"{np.median(row_dist_train):>8.3f}  "
      f"{np.percentile(row_dist_train,25):>8.3f}  "
      f"{np.percentile(row_dist_train,75):>8.3f}")
print(f"  {'test rows':12}  {row_dist_test.mean():>8.3f}  "
      f"{np.median(row_dist_test):>8.3f}  "
      f"{np.percentile(row_dist_test,25):>8.3f}  "
      f"{np.percentile(row_dist_test,75):>8.3f}")

# Separability: what fraction of train rows are MORE distant than median test row?
test_median_dist = np.median(row_dist_test)
pct_train_far    = (row_dist_train > test_median_dist).mean() * 100
pct_train_close  = (row_dist_train <= test_median_dist).mean() * 100

print(f"\n  Test median distance: {test_median_dist:.3f}")
print(f"  Train rows ABOVE test median distance (far from test): {pct_train_far:.1f}%")
print(f"  Train rows BELOW test median distance (close to test): {pct_train_close:.1f}%")
print(f"\n  → Separability interpretation:")
if pct_train_far > 90:
    print(f"    SEVERE shift: {pct_train_far:.1f}% of train rows are "
          f"far from test distribution")
    print(f"    This CONFIRMS the ~99.8% shift finding from EDA")
elif pct_train_far > 70:
    print(f"    MODERATE shift: {pct_train_far:.1f}% of train rows are far from test")
else:
    print(f"    MILD shift: distributions overlap reasonably well")


# ================================================================
# Q2. WHICH FEATURES DRIFT MOST?
# ================================================================
print("\n" + "=" * 65)
print("Q2. PER-FEATURE DRIFT ANALYSIS")
print("=" * 65)

print("\nRunning KS test for all features (may take ~1 min)...")
drift_records = []

for j, col in enumerate(feat_cols):
    tr = train[col].fillna(0).values
    te = test[col].fillna(0).values

    # KS statistic: 0=identical distribution, 1=completely different
    ks_stat, ks_pval = ks_2samp(tr, te)

    # Mean and std drift
    mean_drift = abs(tr.mean() - te.mean()) / (tr.std() + 1e-8)
    std_ratio  = te.std() / (tr.std() + 1e-8)  # 1.0 = same spread

    # Sign of mean: does the feature even point in the same direction?
    mean_sign_flip = int(np.sign(tr.mean()) != np.sign(te.mean()))

    # What fraction of test rows are within train's IQR?
    tr_q25, tr_q75 = np.percentile(tr, 25), np.percentile(tr, 75)
    pct_test_in_train_iqr = ((te >= tr_q25) & (te <= tr_q75)).mean()

    drift_records.append({
        'feature':              col,
        'train_mean':           tr.mean(),
        'test_mean':            te.mean(),
        'train_std':            tr.std(),
        'test_std':             te.std(),
        'mean_drift_z':         mean_drift,          # normalised mean shift
        'std_ratio':            std_ratio,            # test_std / train_std
        'ks_statistic':         ks_stat,              # 0=same, 1=completely different
        'ks_pvalue':            ks_pval,
        'mean_sign_flip':       mean_sign_flip,       # 1 = mean flipped sign
        'pct_test_in_train_iqr': pct_test_in_train_iqr,
        'is_stable_feature':    col in stable_features,
    })

    if (j + 1) % 100 == 0:
        print(f"  {j+1}/{n_feat} features done...")

drift_df = pd.DataFrame(drift_records)
drift_df['drift_score'] = (
    0.4 * drift_df['ks_statistic'] +
    0.4 * drift_df['mean_drift_z'].clip(0, 5) / 5 +
    0.2 * (1 - drift_df['pct_test_in_train_iqr'])
)
drift_df = drift_df.sort_values('drift_score', ascending=False).reset_index(drop=True)

# Summary
n_severe = (drift_df['ks_statistic'] > 0.5).sum()
n_moderate = ((drift_df['ks_statistic'] > 0.2) & (drift_df['ks_statistic'] <= 0.5)).sum()
n_mild = (drift_df['ks_statistic'] <= 0.2).sum()
n_sign_flip = drift_df['mean_sign_flip'].sum()

print(f"\n  KS statistic distribution across {n_feat} features:")
print(f"  SEVERE  drift (KS > 0.5) : {n_severe:>4} features  ({n_severe/n_feat*100:.1f}%)")
print(f"  MODERATE drift (KS 0.2-0.5): {n_moderate:>4} features  ({n_moderate/n_feat*100:.1f}%)")
print(f"  MILD    drift (KS < 0.2) : {n_mild:>4} features  ({n_mild/n_feat*100:.1f}%)")
print(f"  Mean sign flip in test   : {n_sign_flip:>4} features  ({n_sign_flip/n_feat*100:.1f}%)")

print(f"\n  Top 15 most-drifted features:")
cols_show = ['feature','train_mean','test_mean','ks_statistic',
             'mean_drift_z','pct_test_in_train_iqr','is_stable_feature']
print(drift_df[cols_show].head(15).to_string(index=False))

print(f"\n  15 least-drifted features (most stable train→test):")
print(drift_df[cols_show].tail(15).to_string(index=False))

# Focus on sign-stable features specifically
if stable_features:
    stable_drift = drift_df[drift_df['is_stable_feature']]
    print(f"\n  Drift stats for sign-stable features ({len(stable_drift)}):")
    print(f"  Mean KS statistic : {stable_drift['ks_statistic'].mean():.3f}")
    print(f"  Max  KS statistic : {stable_drift['ks_statistic'].max():.3f}")
    print(f"  Stable features with KS>0.3 (drifted despite sign-stability): "
          f"{(stable_drift['ks_statistic']>0.3).sum()}")


# ================================================================
# Q3. HOW MANY TRAIN ROWS LOOK LIKE THE TEST SET?
# ================================================================
print("\n" + "=" * 65)
print("Q3. TRAIN ROW TEST-SIMILARITY SCORING")
print("=" * 65)

# Use the LEAST-DRIFTED features for row-level similarity
# (High-drift features confuse the similarity score — a train row
#  can't ever be "test-like" on a feature that has completely shifted)
low_drift_features = drift_df[drift_df['ks_statistic'] <= 0.2]['feature'].tolist()
med_drift_features = drift_df[drift_df['ks_statistic'] <= 0.4]['feature'].tolist()

print(f"\n  Low-drift features (KS≤0.2): {len(low_drift_features)}")
print(f"  Med-drift features (KS≤0.4): {len(med_drift_features)}")

def compute_similarity(X_rows, feat_list, t_mean, t_std, threshold_z=2.0):
    """
    For each row, compute:
      coverage : fraction of features where |z| < threshold_z
                 (how many features are "in range" for test)
      mean_abs_z: average absolute z-score (lower = more test-like)
    """
    if not feat_list:
        return np.zeros(len(X_rows)), np.ones(len(X_rows)) * 99

    feat_idx = [feat_cols.index(f) for f in feat_list if f in feat_cols]
    X_sub    = X_rows[:, feat_idx]
    tm       = t_mean[feat_idx]
    ts       = t_std[feat_idx]

    z         = np.abs((X_sub - tm) / ts)           # (n_rows, n_feat)
    coverage  = (z < threshold_z).mean(axis=1)       # fraction in range
    mean_abs_z = z.mean(axis=1)
    return coverage, mean_abs_z

# Use low-drift features for scoring
print(f"\nScoring train rows using low-drift features (KS≤0.2)...")
cov_train_low, z_train_low = compute_similarity(
    X_train, low_drift_features, t_mean_arr, t_std_arr)
cov_test_low, z_test_low   = compute_similarity(
    X_test, low_drift_features, t_mean_arr, t_std_arr)

print(f"\n  Feature coverage (fraction of low-drift features within 2σ of test):")
print(f"  {'':12}  {'mean':>7}  {'p10':>7}  {'p25':>7}  {'p50':>7}  {'p75':>7}  {'p90':>7}")
print(f"  {'train rows':12}  {cov_train_low.mean():>7.3f}  "
      f"{np.percentile(cov_train_low,10):>7.3f}  "
      f"{np.percentile(cov_train_low,25):>7.3f}  "
      f"{np.percentile(cov_train_low,50):>7.3f}  "
      f"{np.percentile(cov_train_low,75):>7.3f}  "
      f"{np.percentile(cov_train_low,90):>7.3f}")
print(f"  {'test rows':12}  {cov_test_low.mean():>7.3f}  "
      f"{np.percentile(cov_test_low,10):>7.3f}  "
      f"{np.percentile(cov_test_low,25):>7.3f}  "
      f"{np.percentile(cov_test_low,50):>7.3f}  "
      f"{np.percentile(cov_test_low,75):>7.3f}  "
      f"{np.percentile(cov_test_low,90):>7.3f}")

# How many train rows pass various coverage thresholds?
print(f"\n  Train rows passing coverage threshold (on low-drift features):")
print(f"  {'Threshold':>12}  {'n_rows':>10}  {'%_of_train':>12}  "
      f"{'expected_pred_std':>18}")
for thr in [0.9, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]:
    mask  = cov_train_low >= thr
    n     = mask.sum()
    pct   = n / n_train * 100
    # If we trained only on these rows, how different would TARGET std be?
    if n > 0:
        y_sub_std = y_train[mask].std()
    else:
        y_sub_std = 0
    print(f"  coverage≥{thr:.2f}   :  {n:>10,}  {pct:>11.2f}%  {y_sub_std:>18.6f}")

# Also score using ALL features (total picture)
print(f"\n  Train rows passing coverage threshold (ALL {n_feat} features):")
cov_train_all, z_train_all = compute_similarity(
    X_train, feat_cols, t_mean_arr, t_std_arr)
for thr in [0.9, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]:
    mask = cov_train_all >= thr
    n    = mask.sum()
    pct  = n / n_train * 100
    print(f"  coverage≥{thr:.2f}   :  {n:>10,}  {pct:>11.2f}%")


# ================================================================
# SO3_T REGIME COMPARISON
# ================================================================
print("\n" + "=" * 65)
print("SO3_T REGIME DISTRIBUTION: TRAIN vs TEST")
print("=" * 65)

so3_train = train['SO3_T'].fillna(train['SO3_T'].median()).values
so3_test  = test['SO3_T'].fillna(test['SO3_T'].median()).values

# Use same quintile edges from train to classify both
edges = np.percentile(so3_train, [0, 20, 40, 60, 80, 100])
edges[0]  -= 1e-6
edges[-1] += 1e-6

regime_train = np.digitize(so3_train, edges) - 1
regime_test  = np.digitize(so3_test,  edges) - 1

print(f"\n  {'Regime':>8}  {'Train %':>9}  {'Test %':>9}  "
      f"{'Test/Train ratio':>18}  {'Interpretation':>20}")
for r in range(5):
    tr_pct  = (regime_train == r).mean() * 100
    te_pct  = (regime_test  == r).mean() * 100
    ratio   = te_pct / (tr_pct + 1e-6)
    if ratio > 2:
        interp = "TEST-HEAVY ← critical"
    elif ratio > 1.3:
        interp = "slightly test-heavy"
    elif ratio < 0.5:
        interp = "TRAIN-HEAVY → less relevant"
    elif ratio < 0.7:
        interp = "slightly train-heavy"
    else:
        interp = "balanced"
    print(f"  {r:>8}  {tr_pct:>9.1f}%  {te_pct:>9.1f}%  "
          f"{ratio:>18.2f}x  {interp:>20}")

print(f"\n  SO3_T range in train: [{so3_train.min():.4f}, {so3_train.max():.4f}]")
print(f"  SO3_T range in test : [{so3_test.min():.4f},  {so3_test.max():.4f}]")

so3_ks, so3_p = ks_2samp(so3_train, so3_test)
print(f"  SO3_T KS statistic  : {so3_ks:.4f}  (p={so3_p:.2e})")
if so3_ks > 0.3:
    print(f"  → STRONG regime shift: test is in different SO3_T conditions than train")
elif so3_ks > 0.1:
    print(f"  → MODERATE regime shift")
else:
    print(f"  → MILD regime shift: similar SO3_T distribution")


# ================================================================
# SAVE OUTPUTS
# ================================================================
print("\n" + "=" * 65)
print("SAVING OUTPUTS")
print("=" * 65)

# 1. Drift report
drift_df.to_csv(os.path.join(OUT_DIR, 'drift_report.csv'), index=False)
print(f"  drift_report.csv saved ({len(drift_df)} features)")

# 2. Row similarity scores
row_sim = pd.DataFrame({
    'row_idx':            np.arange(n_train),
    'ID':                 train['ID'].values,
    'TARGET':             y_train,
    'regime':             regime_train,
    'coverage_low_drift': cov_train_low,     # using KS≤0.2 features
    'coverage_all':       cov_train_all,      # using all features
    'mean_z_low_drift':   z_train_low,        # lower = more test-like
    'mean_z_all':         z_train_all,
    'row_dist_from_test': row_dist_train,     # overall distance from test
})

# Composite test-similarity score (higher = more like test)
row_sim['test_similarity'] = (
    0.5 * row_sim['coverage_low_drift'] +
    0.3 * row_sim['coverage_all'] +
    0.2 * (1 / (1 + row_sim['row_dist_from_test']))
)
row_sim['test_similarity_rank'] = row_sim['test_similarity'].rank(pct=True)

# Flag tiers
row_sim['tier'] = pd.cut(
    row_sim['test_similarity_rank'],
    bins=[0, 0.5, 0.75, 0.90, 1.001],
    labels=['bottom_50pct', 'mid_25pct', 'top_25pct', 'top_10pct']
)

row_sim.to_csv(os.path.join(OUT_DIR, 'row_similarity.csv'), index=False)
print(f"  row_similarity.csv saved ({len(row_sim):,} rows)")

# 3. Low-drift feature list
low_drift_df = drift_df[drift_df['ks_statistic'] <= 0.2][['feature','ks_statistic','drift_score']]
low_drift_df.to_csv(os.path.join(OUT_DIR, 'low_drift_features.csv'), index=False)
print(f"  low_drift_features.csv saved ({len(low_drift_df)} features)")


# ================================================================
# Q4. KEY NUMBERS AND WHAT TO DO NEXT
# ================================================================
print("\n" + "=" * 65)
print("Q4. KEY FINDINGS AND NEXT STEPS")
print("=" * 65)

top10_mask = row_sim['test_similarity_rank'] >= 0.90
top25_mask = row_sim['test_similarity_rank'] >= 0.75
top50_mask = row_sim['test_similarity_rank'] >= 0.50

print(f"\n  ── HOW MANY TRAIN ROWS ARE ACTUALLY USEFUL ──")
print(f"  Top 10% most test-like rows : {top10_mask.sum():>8,}  "
      f"({top10_mask.sum()/n_train*100:.2f}% of train)")
print(f"  Top 25% most test-like rows : {top25_mask.sum():>8,}  "
      f"({top25_mask.sum()/n_train*100:.2f}% of train)")
print(f"  Top 50% most test-like rows : {top50_mask.sum():>8,}  "
      f"({top50_mask.sum()/n_train*100:.2f}% of train)")

print(f"\n  ── TARGET STATS BY TIER ──")
print(f"  {'Tier':>20}  {'n_rows':>8}  {'TARGET_mean':>12}  "
      f"{'TARGET_std':>11}  {'TARGET_abs_mean':>15}")
for tier_name, mask in [('top_10pct (best)',  top10_mask),
                         ('top_25pct',         top25_mask),
                         ('top_50pct',         top50_mask),
                         ('bottom_50pct',      ~top50_mask),
                         ('ALL train',         np.ones(n_train, dtype=bool))]:
    y_sub = y_train[mask]
    if len(y_sub) == 0:
        continue
    print(f"  {tier_name:>20}  {mask.sum():>8,}  "
          f"{y_sub.mean():>+12.7f}  {y_sub.std():>11.6f}  "
          f"{np.abs(y_sub).mean():>15.7f}")

print(f"\n  ── DRIFT SUMMARY ──")
print(f"  Features with KS>0.5 (severe) : {n_severe} / {n_feat}"
      f"  ({n_severe/n_feat*100:.1f}%) ← drop from models")
print(f"  Features with KS≤0.2 (stable) : {n_mild} / {n_feat}"
      f"  ({n_mild/n_feat*100:.1f}%) ← use these in feature search")
print(f"  Sign-stable features that also have low drift:")
if stable_features:
    stable_low_drift = set(stable_features) & set(low_drift_features)
    print(f"    {len(stable_low_drift)} features are BOTH sign-stable AND low-drift")
    print(f"    These are your GOLD features — use them first for everything")
    gold = drift_df[drift_df['feature'].isin(stable_low_drift)][
        ['feature','ks_statistic','drift_score']
    ].sort_values('drift_score')
    print(gold.to_string(index=False))

# Save summary to text file
summary_lines = [
    "TRAIN-TEST SHIFT ANALYSIS SUMMARY",
    "=" * 50,
    f"Train rows: {n_train:,}",
    f"Test rows:  {n_test:,}",
    f"",
    f"SEPARABILITY",
    f"  {pct_train_far:.1f}% of train rows are FARTHER from test distribution",
    f"  than the median test row",
    f"",
    f"FEATURE DRIFT",
    f"  Severe (KS>0.5):   {n_severe} features ({n_severe/n_feat*100:.1f}%)",
    f"  Moderate (KS 0.2-0.5): {n_moderate} features ({n_moderate/n_feat*100:.1f}%)",
    f"  Mild (KS<=0.2):    {n_mild} features ({n_mild/n_feat*100:.1f}%)",
    f"",
    f"RELEVANT TRAIN ROWS",
    f"  Top 10% (most test-like): {top10_mask.sum():,} rows",
    f"  Top 25%:                  {top25_mask.sum():,} rows",
    f"  Top 50%:                  {top50_mask.sum():,} rows",
]
with open(os.path.join(OUT_DIR, 'shift_summary.txt'), 'w') as f:
    f.write('\n'.join(summary_lines))
print(f"\n  shift_summary.txt saved")

print("\n" + "=" * 65)
print("DONE")
print("=" * 65)
