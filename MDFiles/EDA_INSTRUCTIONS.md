# EDA Instructions — Short-Horizon Return Prediction
## For Claude Code Agent

---

## Project Context

You are performing exhaustive exploratory data analysis on a **de-identified multivariate tabular time-series dataset** for a Kaggle regression competition.

**Task:** Predict `TARGET` = `100 * (Price[t+H] - Price[t]) / Price[t]` — the H-window-forward percentage return of `Price`.

**Evaluation metric:** R² (coefficient of determination)

**Key structural facts you must keep in mind at all times:**
- Rows are **shuffled** — temporal order is not preserved in the released files
- IDs are integers but do NOT reliably encode time order after shuffling
- 445 predictive features = 112 base features + 111 LagT1 + 111 LagT2 + 111 LagT3
- `SO3_T` is a standalone covariate with **no lag versions**
- `Price` is both a feature AND defines the target
- Lag features: `feat_LagT1 = feat[t] - feat[t-T1]`, `T3 > T2 > T1`
- Train: 661,574 rows | Test: 410,139 rows

---

## Notebook Location

**All EDA must be performed in:** `notebooks/01_eda.ipynb`

Save all intermediate outputs (summary dataframes, arrays, plots) to `outputs/eda/` so downstream notebooks can load them without recomputing.

---

## Environment Setup Cell (Always First)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import (
    skew, kurtosis, shapiro, jarque_bera,
    ks_2samp, kruskal, levene, spearmanr, pearsonr
)
from scipy.spatial.distance import mahalanobis
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
import warnings
import os
import json
import pickle

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', '{:.6f}'.format)

# Output directory
os.makedirs('outputs/eda', exist_ok=True)
os.makedirs('outputs/eda/plots', exist_ok=True)
os.makedirs('outputs/eda/summaries', exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("Environment ready.")
```

---

## LAYER 1 — Data Audit & Sanity Checks

### 1.1 Load Data and Memory Audit

```python
# Load
train = pd.read_parquet('data/raw/train.parquet')
test  = pd.read_parquet('data/raw/test.parquet')

print(f"Train shape : {train.shape}")
print(f"Test  shape : {test.shape}")
print(f"\nTrain memory (float64): {train.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# Downcast to float32 for all feature columns (not ID, not TARGET)
feature_cols = [c for c in train.columns if c not in ['ID', 'TARGET']]
test_feature_cols = [c for c in test.columns if c != 'ID']

for col in feature_cols:
    train[col] = train[col].astype(np.float32)
for col in test_feature_cols:
    test[col] = test[col].astype(np.float32)

print(f"\nTrain memory (float32): {train.memory_usage(deep=True).sum() / 1e9:.2f} GB")
print(f"Test  memory (float32): {test.memory_usage(deep=True).sum() / 1e9:.2f} GB")
```

### 1.2 Column Taxonomy — Parse Feature Families

```python
# Separate feature groups
base_features   = [c for c in feature_cols if '_LagT' not in c]
lag_t1_features = [c for c in feature_cols if '_LagT1' in c]
lag_t2_features = [c for c in feature_cols if '_LagT2' in c]
lag_t3_features = [c for c in feature_cols if '_LagT3' in c]

print(f"Base features   : {len(base_features)}")
print(f"LagT1 features  : {len(lag_t1_features)}")
print(f"LagT2 features  : {len(lag_t2_features)}")
print(f"LagT3 features  : {len(lag_t3_features)}")
print(f"Total predictive: {len(feature_cols)}")

# Parse prefix families from column names
# Extract top-level prefix (everything before second underscore group)
def extract_family(col):
    col_clean = col.replace('_LagT1','').replace('_LagT2','').replace('_LagT3','')
    parts = col_clean.split('_')
    return parts[0] if len(parts) >= 1 else 'UNKNOWN'

family_map = {col: extract_family(col) for col in feature_cols}
families   = pd.Series(family_map).value_counts()
print(f"\nTop-level feature families:\n{families}")

# Save taxonomy
taxonomy = {
    'base_features'  : base_features,
    'lag_t1_features': lag_t1_features,
    'lag_t2_features': lag_t2_features,
    'lag_t3_features': lag_t3_features,
    'family_map'     : family_map
}
with open('outputs/eda/summaries/taxonomy.pkl', 'wb') as f:
    pickle.dump(taxonomy, f)
```

### 1.3 Zero-Variance & Constant Column Detection

```python
# Any column with std == 0 is useless
feature_stds = train[feature_cols].std()
zero_var_cols = feature_stds[feature_stds == 0].index.tolist()
near_zero_var_cols = feature_stds[feature_stds < 1e-6].index.tolist()

print(f"Zero-variance columns    : {len(zero_var_cols)}")
print(f"Near-zero variance (<1e-6): {len(near_zero_var_cols)}")
if zero_var_cols:
    print(f"  -> Will be dropped: {zero_var_cols[:10]}")

# Save drop list
drop_zero_var = zero_var_cols
```

### 1.4 Duplicate Row Detection

```python
# Exact duplicates on feature columns
n_exact_dupes = train[feature_cols].duplicated().sum()
print(f"Exact duplicate feature rows: {n_exact_dupes} ({n_exact_dupes/len(train)*100:.3f}%)")

# Near-duplicate detection: rows where >90% of features are identical
# Sample 10k rows for speed
sample_idx = np.random.choice(len(train), size=10000, replace=False)
sample = train[feature_cols].iloc[sample_idx]
dupe_counts = sample.duplicated(keep=False).sum()
print(f"Near-duplicate estimate (10k sample): {dupe_counts}")
```

### 1.5 ID Structure Analysis

```python
print(f"\nID range  : {train['ID'].min()} — {train['ID'].max()}")
print(f"ID unique : {train['ID'].nunique()} / {len(train)}")
print(f"ID gaps   : {train['ID'].max() - train['ID'].min() + 1 - train['ID'].nunique()} missing IDs in range")

# Check if IDs in test overlap with train
id_overlap = len(set(train['ID']) & set(test['ID']))
print(f"Train/Test ID overlap: {id_overlap} rows")

# Plot ID distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(train['ID'], bins=100, color='steelblue', alpha=0.7)
axes[0].set_title('Train ID Distribution')
axes[1].hist(test['ID'],  bins=100, color='coral',     alpha=0.7)
axes[1].set_title('Test ID Distribution')
plt.tight_layout()
plt.savefig('outputs/eda/plots/01_id_distribution.png', dpi=150)
plt.show()
```

---

## LAYER 2 — Target Variable Deep Analysis

### 2.1 TARGET Distribution — Full Statistical Profile

```python
target = train['TARGET'].values

# Core moments
t_mean     = np.mean(target)
t_std      = np.std(target)
t_skew     = stats.skew(target)
t_kurt     = stats.kurtosis(target)  # excess kurtosis (normal = 0)
t_median   = np.median(target)
t_mad      = np.mean(np.abs(target - t_median))  # median absolute deviation

print("=== TARGET Distribution Statistics ===")
print(f"  N              : {len(target):,}")
print(f"  Mean           : {t_mean:.6f}")
print(f"  Std Dev        : {t_std:.6f}")
print(f"  Median         : {t_median:.6f}")
print(f"  MAD            : {t_mad:.6f}")
print(f"  Skewness       : {t_skew:.4f}  (normal=0; >0 right tail)")
print(f"  Excess Kurtosis: {t_kurt:.4f}  (normal=0; >0 leptokurtic/fat tails)")
print(f"  Min            : {target.min():.4f}")
print(f"  Max            : {target.max():.4f}")
print(f"  1st pct        : {np.percentile(target, 1):.4f}")
print(f"  99th pct       : {np.percentile(target, 99):.4f}")

# Jarque-Bera normality test: H0 = normal
# JB = (n/6) * [S^2 + (K^2/4)]  where S=skewness, K=excess kurtosis
jb_stat, jb_p = jarque_bera(target)
print(f"\nJarque-Bera test: stat={jb_stat:.2f}, p={jb_p:.4e}")
print(f"  -> {'REJECT normality' if jb_p < 0.05 else 'Cannot reject normality'} at 5% level")
```

### 2.2 TARGET Tail Analysis & Outlier Thresholds

```python
# Fraction of observations in each sigma band
for sigma in [1, 2, 3, 4, 5]:
    frac_inside = np.mean(np.abs(target - t_mean) <= sigma * t_std)
    print(f"  |TARGET - mean| <= {sigma}σ : {frac_inside*100:.2f}%  "
          f"(normal would be {stats.norm.cdf(sigma)*2-1:.4f})")

# Exact zeros and sign distribution
pct_zero = np.mean(target == 0) * 100
pct_pos  = np.mean(target > 0)  * 100
pct_neg  = np.mean(target < 0)  * 100
print(f"\nTARGET = 0   : {pct_zero:.3f}%")
print(f"TARGET > 0   : {pct_pos:.2f}%")
print(f"TARGET < 0   : {pct_neg:.2f}%")

# Winsorization thresholds — determine best cap
for cap_sigma in [3, 4, 5, 10]:
    lo = t_mean - cap_sigma * t_std
    hi = t_mean + cap_sigma * t_std
    pct_winsorized = np.mean((target < lo) | (target > hi)) * 100
    print(f"  Winsorize at ±{cap_sigma}σ: {pct_winsorized:.3f}% of rows affected")
```

### 2.3 TARGET Visualization

```python
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig)

# 1. Histogram
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(target, bins=200, color='steelblue', alpha=0.7, density=True)
xr = np.linspace(target.min(), target.max(), 500)
ax1.plot(xr, stats.norm.pdf(xr, t_mean, t_std), 'r-', lw=2, label='Normal fit')
ax1.set_xlim([np.percentile(target,0.5), np.percentile(target,99.5)])
ax1.set_title('TARGET Distribution (clipped to 0.5–99.5 pct)')
ax1.legend()

# 2. Q-Q plot
ax2 = fig.add_subplot(gs[0, 1])
stats.probplot(target, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot vs Normal')

# 3. Log-scale histogram (shows tail behavior)
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(target, bins=500, color='coral', alpha=0.7)
ax3.set_yscale('log')
ax3.set_xlim([np.percentile(target,0.1), np.percentile(target,99.9)])
ax3.set_title('TARGET Histogram (log y-scale)')

# 4. Cumulative distribution
ax4 = fig.add_subplot(gs[1, 0])
sorted_t = np.sort(target)
cdf = np.arange(1, len(sorted_t)+1) / len(sorted_t)
ax4.plot(sorted_t, cdf, lw=1.5, color='steelblue')
ax4.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Median')
ax4.set_xlim([np.percentile(target,1), np.percentile(target,99)])
ax4.set_title('CDF of TARGET')
ax4.legend()

# 5. TARGET by sorted-ID chunks (temporal proxy)
ax5 = fig.add_subplot(gs[1, 1])
train_sorted = train.sort_values('ID').reset_index(drop=True)
chunk_size = len(train_sorted) // 20
chunk_means = [train_sorted['TARGET'].iloc[i*chunk_size:(i+1)*chunk_size].mean()
               for i in range(20)]
ax5.plot(chunk_means, marker='o', color='purple')
ax5.axhline(0, color='red', linestyle='--', alpha=0.5)
ax5.set_title('TARGET Mean per ID-sorted Chunk (regime proxy)')
ax5.set_xlabel('Chunk (0=lowest IDs, 19=highest IDs)')

# 6. Box plot
ax6 = fig.add_subplot(gs[1, 2])
ax6.boxplot(target, vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            flierprops=dict(marker='.', markersize=1, alpha=0.3))
ax6.set_title('TARGET Boxplot')
ax6.set_ylim([np.percentile(target,1), np.percentile(target,99)])

plt.suptitle('TARGET Variable — Complete Distribution Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda/plots/02_target_analysis.png', dpi=150)
plt.show()
```

---

## LAYER 3 — Missing Value Analysis

### 3.1 Per-Column Missingness

```python
# Missingness fractions for all columns
miss_train = train[feature_cols].isnull().mean().sort_values(ascending=False)
miss_test  = test[test_feature_cols].isnull().mean().sort_values(ascending=False)

print(f"Columns with ANY missing (train): {(miss_train > 0).sum()}")
print(f"Columns with >50% missing (train): {(miss_train > 0.5).sum()}")
print(f"Columns with >80% missing (train): {(miss_train > 0.8).sum()}")

# Missingness tiers
miss_tiers = {
    'zero_missing'   : miss_train[miss_train == 0].index.tolist(),
    'low_missing'    : miss_train[(miss_train > 0)    & (miss_train <= 0.2)].index.tolist(),
    'medium_missing' : miss_train[(miss_train > 0.2)  & (miss_train <= 0.5)].index.tolist(),
    'high_missing'   : miss_train[(miss_train > 0.5)  & (miss_train <= 0.8)].index.tolist(),
    'very_high_miss' : miss_train[miss_train > 0.8].index.tolist(),
}
for tier, cols in miss_tiers.items():
    print(f"  {tier:<20}: {len(cols)} features")

with open('outputs/eda/summaries/missingness_tiers.pkl', 'wb') as f:
    pickle.dump(miss_tiers, f)
```

### 3.2 Train vs Test Missingness Consistency

```python
# Compare missingness rates train vs test
miss_comparison = pd.DataFrame({
    'miss_train': miss_train,
    'miss_test' : miss_test.reindex(miss_train.index)
}).dropna()

miss_comparison['delta'] = (miss_comparison['miss_test'] - miss_comparison['miss_train']).abs()
large_drift_cols = miss_comparison[miss_comparison['delta'] > 0.1].sort_values('delta', ascending=False)

print(f"\nFeatures with >10% missingness drift (train vs test): {len(large_drift_cols)}")
print(large_drift_cols.head(20))

miss_comparison.to_csv('outputs/eda/summaries/missingness_comparison.csv')
```

### 3.3 Missingness as a Signal — Does NaN Predict TARGET?

```python
# For each feature, test whether being NaN correlates with TARGET magnitude or direction
# This tells us if we need "was_missing" indicator features

miss_signal_results = []
for col in feature_cols:
    miss_mask = train[col].isnull()
    n_miss = miss_mask.sum()
    if n_miss < 100 or n_miss > len(train) - 100:
        continue
    # Mean TARGET when missing vs not missing
    mean_target_missing    = train.loc[miss_mask,  'TARGET'].mean()
    mean_target_present    = train.loc[~miss_mask, 'TARGET'].mean()
    std_target_present     = train.loc[~miss_mask, 'TARGET'].std()
    # Standardized difference
    z_diff = (mean_target_missing - mean_target_present) / (std_target_present + 1e-8)
    # Mann-Whitney U test (non-parametric)
    u_stat, u_p = stats.mannwhitneyu(
        train.loc[miss_mask,  'TARGET'].values,
        train.loc[~miss_mask, 'TARGET'].values,
        alternative='two-sided'
    )
    miss_signal_results.append({
        'feature'             : col,
        'n_missing'           : n_miss,
        'miss_rate'           : n_miss / len(train),
        'mean_target_missing' : mean_target_missing,
        'mean_target_present' : mean_target_present,
        'z_diff'              : z_diff,
        'mwu_pvalue'          : u_p
    })

miss_signal_df = pd.DataFrame(miss_signal_results).sort_values('mwu_pvalue')
significant_miss_signals = miss_signal_df[miss_signal_df['mwu_pvalue'] < 0.01]
print(f"\nFeatures where NaN is a significant TARGET predictor (p<0.01): {len(significant_miss_signals)}")
print(significant_miss_signals.head(20))
miss_signal_df.to_csv('outputs/eda/summaries/missingness_signal.csv', index=False)
```

### 3.4 Missingness Correlation Heatmap (Family-level)

```python
# Binary missingness matrix
miss_binary = train[feature_cols].isnull().astype(int)
# Aggregate to family level for readability
family_miss = {}
for fam in families.index:
    fam_cols = [c for c in feature_cols if extract_family(c) == fam]
    family_miss[fam] = miss_binary[fam_cols].mean(axis=1)
family_miss_df = pd.DataFrame(family_miss)

family_miss_corr = family_miss_df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(family_miss_corr, cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            annot=False, linewidths=0.5)
plt.title('Family-Level Missingness Correlation\n(1 = always missing together)')
plt.tight_layout()
plt.savefig('outputs/eda/plots/03_missingness_family_corr.png', dpi=150)
plt.show()
```

---

## LAYER 4 — Univariate Distribution Analysis

### 4.1 Compute All Moments Efficiently

```python
# Vectorized moment computation across all features
print("Computing moments for all features (this may take 2-3 minutes)...")

moments_list = []
for col in feature_cols:
    vals = train[col].dropna().values
    if len(vals) < 100:
        continue
    p1, p25, p50, p75, p99 = np.percentile(vals, [1, 25, 50, 75, 99])
    col_skew = stats.skew(vals)
    col_kurt = stats.kurtosis(vals)   # excess kurtosis
    col_std  = np.std(vals)
    col_mean = np.mean(vals)

    # Tail asymmetry ratio: (p99 - p50) / (p50 - p01)
    tail_asym = (p99 - p50) / (p50 - p1 + 1e-8)
    # Zero fraction
    zero_frac = np.mean(vals == 0)
    # IQR
    iqr = p75 - p25
    # Coefficient of variation
    cv = col_std / (abs(col_mean) + 1e-8)

    moments_list.append({
        'feature'     : col,
        'n_valid'     : len(vals),
        'mean'        : col_mean,
        'std'         : col_std,
        'skewness'    : col_skew,
        'kurt_excess' : col_kurt,
        'p1'          : p1,
        'p25'         : p25,
        'p50'         : p50,
        'p75'         : p75,
        'p99'         : p99,
        'iqr'         : iqr,
        'tail_asym'   : tail_asym,
        'zero_frac'   : zero_frac,
        'cv'          : cv,
        'family'      : extract_family(col),
        'lag_type'    : ('LagT1' if '_LagT1' in col else
                         'LagT2' if '_LagT2' in col else
                         'LagT3' if '_LagT3' in col else 'base')
    })

moments_df = pd.DataFrame(moments_list)
moments_df.to_csv('outputs/eda/summaries/feature_moments.csv', index=False)
print(f"Moments computed for {len(moments_df)} features.")
print(moments_df[['skewness','kurt_excess','zero_frac','tail_asym']].describe())
```

### 4.2 Distribution Shape Summary

```python
# Classify features by distribution shape
high_skew_cols    = moments_df[moments_df['skewness'].abs() > 2]['feature'].tolist()
high_kurt_cols    = moments_df[moments_df['kurt_excess'] > 5]['feature'].tolist()
zero_inflated_cols = moments_df[moments_df['zero_frac'] > 0.3]['feature'].tolist()
heavy_right_tail  = moments_df[moments_df['tail_asym'] > 3]['feature'].tolist()

print(f"High skewness (|skew|>2)   : {len(high_skew_cols)} features")
print(f"Heavy kurtosis (excess>5)  : {len(high_kurt_cols)} features")
print(f"Zero-inflated (>30% zeros) : {len(zero_inflated_cols)} features")
print(f"Heavy right tail (asym>3)  : {len(heavy_right_tail)} features")

# Distribution of skewness and kurtosis across all features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].hist(moments_df['skewness'],    bins=80, color='steelblue', alpha=0.7)
axes[0,0].axvline(0, color='red', lw=2); axes[0,0].set_title('Distribution of Skewness Across Features')

axes[0,1].hist(moments_df['kurt_excess'], bins=80, color='coral',     alpha=0.7)
axes[0,1].axvline(0, color='red', lw=2); axes[0,1].set_title('Distribution of Excess Kurtosis Across Features')

axes[1,0].hist(moments_df['zero_frac'],   bins=80, color='green',     alpha=0.7)
axes[1,0].set_title('Distribution of Zero Fraction Across Features')

axes[1,1].scatter(moments_df['skewness'], moments_df['kurt_excess'],
                  alpha=0.3, s=10, color='purple')
axes[1,1].set_title('Skewness vs Excess Kurtosis (all features)')
axes[1,1].set_xlabel('Skewness'); axes[1,1].set_ylabel('Excess Kurtosis')

plt.tight_layout()
plt.savefig('outputs/eda/plots/04_distribution_shapes.png', dpi=150)
plt.show()

# Save shape classification
shape_classes = {
    'high_skew'       : high_skew_cols,
    'high_kurt'       : high_kurt_cols,
    'zero_inflated'   : zero_inflated_cols,
    'heavy_right_tail': heavy_right_tail,
}
with open('outputs/eda/summaries/shape_classes.pkl', 'wb') as f:
    pickle.dump(shape_classes, f)
```

### 4.3 Lag Variance Consistency Check

```python
# For each base feature that has all 3 lags:
# Variance should increase: Var(LagT1) < Var(LagT2) < Var(LagT3)
# Violation = possible data construction error

lag_consistency = []
for base_col in base_features:
    l1 = base_col + '_LagT1'
    l2 = base_col + '_LagT2'
    l3 = base_col + '_LagT3'
    if l1 not in feature_cols or l2 not in feature_cols or l3 not in feature_cols:
        continue
    var1 = train[l1].var()
    var2 = train[l2].var()
    var3 = train[l3].var()
    monotone_ok = (var1 <= var2 <= var3)
    lag_consistency.append({
        'base_feature' : base_col,
        'var_LagT1'    : var1,
        'var_LagT2'    : var2,
        'var_LagT3'    : var3,
        'monotone_ok'  : monotone_ok
    })

lag_cons_df = pd.DataFrame(lag_consistency)
violations  = lag_cons_df[~lag_cons_df['monotone_ok']]
print(f"\nLag variance monotonicity violations: {len(violations)} / {len(lag_cons_df)} base features")
print(violations.head(15))
lag_cons_df.to_csv('outputs/eda/summaries/lag_variance_consistency.csv', index=False)
```

---

## LAYER 5 — Target Correlation Analysis

### 5.1 Pearson and Spearman Correlations with TARGET

```python
# This is the core feature importance signal — compute for ALL features
print("Computing Pearson and Spearman correlations with TARGET...")
print("(Spearman is rank-based and robust to outliers and nonlinearity)")

corr_results = []
target_vals = train['TARGET'].values

for col in feature_cols:
    vals = train[col].values
    valid_mask = ~np.isnan(vals)
    if valid_mask.sum() < 1000:
        corr_results.append({'feature': col, 'pearson': np.nan,
                              'pearson_p': np.nan, 'spearman': np.nan,
                              'spearman_p': np.nan})
        continue
    v = vals[valid_mask]
    t = target_vals[valid_mask]

    # Pearson: measures LINEAR relationship
    # r = Σ[(xi - x̄)(yi - ȳ)] / [n * σx * σy]
    r_p, p_p = pearsonr(v, t)

    # Spearman: measures MONOTONIC relationship (rank-based)
    # rs = 1 - 6Σd²/[n(n²-1)]  where d = rank difference
    r_s, p_s = spearmanr(v, t)

    corr_results.append({
        'feature'   : col,
        'pearson'   : r_p,
        'pearson_p' : p_p,
        'spearman'  : r_s,
        'spearman_p': p_s,
        'family'    : extract_family(col),
        'lag_type'  : ('LagT1' if '_LagT1' in col else
                       'LagT2' if '_LagT2' in col else
                       'LagT3' if '_LagT3' in col else 'base')
    })

corr_df = pd.DataFrame(corr_results)
corr_df['pearson_abs']  = corr_df['pearson'].abs()
corr_df['spearman_abs'] = corr_df['spearman'].abs()
# Nonlinearity indicator: large gap between Spearman and Pearson means nonlinear signal
corr_df['nonlinearity_gap'] = (corr_df['spearman_abs'] - corr_df['pearson_abs'])
corr_df = corr_df.sort_values('spearman_abs', ascending=False)
corr_df.to_csv('outputs/eda/summaries/target_correlations.csv', index=False)

print(f"\nTop 20 features by |Spearman| with TARGET:")
print(corr_df[['feature','pearson','spearman','nonlinearity_gap']].head(20).to_string())
print(f"\nFeatures with |Pearson| > 0.05:  {(corr_df['pearson_abs'] > 0.05).sum()}")
print(f"Features with |Spearman| > 0.05: {(corr_df['spearman_abs'] > 0.05).sum()}")
```

### 5.2 Mutual Information with TARGET

```python
# MI captures ANY statistical dependence (linear + nonlinear + regime effects)
# More expensive to compute — use on subset or full depending on time
print("Computing Mutual Information (this takes ~10-20 minutes for all 445 features)...")
print("Using 100k row sample for speed...")

sample_mask = np.random.choice(len(train), size=100_000, replace=False)
X_sample = train[feature_cols].iloc[sample_mask]
y_sample = train['TARGET'].iloc[sample_mask].values

# Impute NaN with column median for MI computation
X_imp = X_sample.copy()
for col in feature_cols:
    med = X_imp[col].median()
    X_imp[col] = X_imp[col].fillna(med)

mi_scores = mutual_info_regression(
    X_imp.values, y_sample,
    discrete_features=False,
    random_state=RANDOM_SEED,
    n_neighbors=5
)
mi_df = pd.DataFrame({'feature': feature_cols, 'mutual_info': mi_scores})
mi_df = mi_df.sort_values('mutual_info', ascending=False)
mi_df.to_csv('outputs/eda/summaries/mutual_information.csv', index=False)

print(f"\nTop 20 features by Mutual Information:")
print(mi_df.head(20).to_string())
print(f"\nFeatures with MI > 0: {(mi_df['mutual_info'] > 0).sum()}")
```

### 5.3 Correlation Stability Across Temporal Chunks

```python
# Critical test: is the feature-TARGET correlation stable over time?
# Sort by ID as temporal proxy. Divide into N_CHUNKS equal windows.
# A feature whose correlation sign FLIPS is dangerous — it overfits to one regime.

N_CHUNKS = 10
train_sorted = train.sort_values('ID').reset_index(drop=True)
chunk_size   = len(train_sorted) // N_CHUNKS

# Focus on top 50 features by absolute Spearman
top50_cols = corr_df.dropna(subset=['spearman']).head(50)['feature'].tolist()

stability_results = {}
for col in top50_cols:
    chunk_corrs = []
    for i in range(N_CHUNKS):
        chunk = train_sorted.iloc[i*chunk_size : (i+1)*chunk_size]
        vals  = chunk[col].fillna(chunk[col].median()).values
        tgt   = chunk['TARGET'].values
        r, _  = pearsonr(vals, tgt)
        chunk_corrs.append(r)
    stability_results[col] = chunk_corrs

stability_df = pd.DataFrame(stability_results, index=[f'chunk_{i}' for i in range(N_CHUNKS)])
# Sign flip = correlation changes sign between chunks
sign_flips = (stability_df > 0).any() & (stability_df < 0).any()
unstable_features = sign_flips[sign_flips].index.tolist()
print(f"\nFeatures with correlation SIGN FLIP across chunks: {len(unstable_features)}")
print(f"  -> These are regime-dependent and risky: {unstable_features[:10]}")
stability_df.to_csv('outputs/eda/summaries/correlation_stability.csv')

# Visualize stability for top 20
plt.figure(figsize=(16, 8))
for col in top50_cols[:20]:
    plt.plot(range(N_CHUNKS), stability_results[col], alpha=0.6, marker='o', lw=1.5, label=col[:25])
plt.axhline(0, color='black', lw=1.5, linestyle='--')
plt.xlabel('Chunk (sorted by ID)')
plt.ylabel('Pearson r with TARGET')
plt.title('Correlation Stability — Top 20 Features\n(sign flip = regime-dependent/unstable feature)')
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig('outputs/eda/plots/05_correlation_stability.png', dpi=150)
plt.show()
```

### 5.4 Partial Correlation Controlling for Price

```python
# Many features may correlate with TARGET simply because they're correlated with Price
# We want the UNIQUE predictive information beyond Price

# Partial correlation formula:
# r(X,Y | Z) = [r(X,Y) - r(X,Z)*r(Y,Z)] / sqrt([1-r(X,Z)^2] * [1-r(Y,Z)^2])

price_vals = train['Price'].fillna(train['Price'].median()).values

partial_corr_results = []
for col in corr_df.head(80)['feature'].tolist():
    vals = train[col].fillna(train[col].median()).values
    r_xy, _ = pearsonr(vals, target_vals)           # feature vs TARGET
    r_xz, _ = pearsonr(vals, price_vals)             # feature vs Price
    r_yz, _ = pearsonr(target_vals, price_vals)      # TARGET vs Price
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2) + 1e-10)
    partial_r = (r_xy - r_xz * r_yz) / denom
    partial_corr_results.append({
        'feature'    : col,
        'pearson_r'  : r_xy,
        'partial_r'  : partial_r,
        'r_vs_price' : r_xz,
        'info_gain'  : abs(partial_r) - abs(r_xy)  # positive = MORE info after controlling for Price
    })

partial_corr_df = pd.DataFrame(partial_corr_results).sort_values('partial_r', key=abs, ascending=False)
partial_corr_df.to_csv('outputs/eda/summaries/partial_correlations.csv', index=False)
print("Top features by partial correlation (controlling for Price):")
print(partial_corr_df.head(20).to_string())
```

---

## LAYER 6 — Inter-Feature Correlation & Multicollinearity

### 6.1 Full Correlation Matrix & High-Correlation Pairs

```python
# Compute on a sample to manage memory (use all rows but float32)
print("Computing 445x445 correlation matrix...")
X_corr = train[feature_cols].copy()
# Fill NaN with column median before correlating
for col in feature_cols:
    X_corr[col] = X_corr[col].fillna(X_corr[col].median())

corr_matrix = X_corr.astype(np.float32).corr()
corr_matrix.to_parquet('outputs/eda/summaries/full_corr_matrix.parquet')
print("Correlation matrix saved.")

# Extract upper triangle pairs with |r| > threshold
threshold = 0.90
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = []
rows, cols_ = np.where(np.triu(corr_matrix.values, k=1) > threshold)
for r, c in zip(rows, cols_):
    high_corr_pairs.append({
        'feature_a': feature_cols[r],
        'feature_b': feature_cols[c],
        'correlation': corr_matrix.iloc[r, c]
    })
rows_neg, cols_neg = np.where(np.triu(corr_matrix.values, k=1) < -threshold)
for r, c in zip(rows_neg, cols_neg):
    high_corr_pairs.append({
        'feature_a': feature_cols[r],
        'feature_b': feature_cols[c],
        'correlation': corr_matrix.iloc[r, c]
    })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)
print(f"\nPairs with |r| > {threshold}: {len(high_corr_df)}")
print(high_corr_df.head(20).to_string())
high_corr_df.to_csv('outputs/eda/summaries/high_corr_pairs.csv', index=False)
```

### 6.2 Clustered Correlation Heatmap

```python
# Visualize block structure — groups of correlated features
# Use hierarchical clustering to reorder rows/columns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# Distance = 1 - |correlation|
dist_matrix = 1 - corr_matrix.abs().values
np.fill_diagonal(dist_matrix, 0)
dist_condensed = squareform(dist_matrix, checks=False)
linkage_matrix  = linkage(dist_condensed, method='ward')
cluster_order   = dendrogram(linkage_matrix, no_plot=True)['leaves']

# Reorder correlation matrix by cluster
corr_reordered = corr_matrix.iloc[cluster_order, cluster_order]

plt.figure(figsize=(18, 16))
sns.heatmap(corr_reordered, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            xticklabels=False, yticklabels=False, linewidths=0)
plt.title('Feature Correlation Matrix — Hierarchically Clustered\n(block structure = feature families)', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/eda/plots/06_clustered_corr_heatmap.png', dpi=150)
plt.show()
```

### 6.3 Within-Family vs Between-Family Correlation

```python
fam_list = [f for f in families.index if families[f] >= 3]
intra_corr = {}
for fam in fam_list:
    fam_cols = [c for c in feature_cols if extract_family(c) == fam]
    if len(fam_cols) < 2:
        continue
    sub_corr = corr_matrix.loc[fam_cols, fam_cols]
    upper = sub_corr.values[np.triu_indices_from(sub_corr.values, k=1)]
    intra_corr[fam] = {
        'mean_abs_corr' : np.mean(np.abs(upper)),
        'max_abs_corr'  : np.max(np.abs(upper)),
        'n_features'    : len(fam_cols)
    }

intra_df = pd.DataFrame(intra_corr).T.sort_values('mean_abs_corr', ascending=False)
print("Intra-family correlation (higher = more redundancy within family):")
print(intra_df.to_string())
intra_df.to_csv('outputs/eda/summaries/intra_family_correlation.csv')
```

### 6.4 VIF for Top Features (Linear Model Multicollinearity)

```python
# VIF_j = 1 / (1 - R²_j)
# where R²_j = R² from regressing feature j on all other features
# VIF > 10 → severe multicollinearity for that feature in a linear model

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Use top 50 features by spearman (after imputation)
top50 = corr_df.dropna(subset=['spearman']).head(50)['feature'].tolist()
X_vif = X_corr[top50].values

vif_data = []
for i, col in enumerate(top50):
    try:
        vif_val = variance_inflation_factor(X_vif, i)
    except Exception:
        vif_val = np.nan
    vif_data.append({'feature': col, 'VIF': vif_val})

vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
print("\nVIF for top 50 features (>10 = problematic for linear models):")
print(vif_df.head(20).to_string())
vif_df.to_csv('outputs/eda/summaries/vif_top50.csv', index=False)
high_vif = vif_df[vif_df['VIF'] > 10]['feature'].tolist()
print(f"\nFeatures with VIF > 10: {len(high_vif)}")
```

---

## LAYER 7 — Statistical Tests for Feature Validity

### 7.1 Kolmogorov-Smirnov Test — Train vs Test Distribution Shift

```python
# KS statistic D = max|F_train(x) - F_test(x)|
# Large D (small p) = distributions differ = covariate shift = leaderboard risk

print("Running KS tests for train vs test distribution shift...")
print("(Features with significant shift may underperform on private leaderboard)")

ks_results = []
for col in feature_cols:
    train_vals = train[col].dropna().values
    test_vals  = test[col].dropna().values
    if len(train_vals) < 100 or len(test_vals) < 100:
        continue
    ks_stat, ks_p = ks_2samp(train_vals, test_vals)
    ks_results.append({
        'feature' : col,
        'ks_stat' : ks_stat,
        'ks_p'    : ks_p,
        'shift'   : 'HIGH' if ks_p < 0.001 else 'MEDIUM' if ks_p < 0.01 else 'LOW'
    })

ks_df = pd.DataFrame(ks_results).sort_values('ks_stat', ascending=False)
ks_df.to_csv('outputs/eda/summaries/ks_train_test.csv', index=False)

print(f"\nFeatures with HIGH distribution shift (p<0.001): {(ks_df['ks_p'] < 0.001).sum()}")
print(f"Features with MEDIUM shift (p<0.01):            {(ks_df['ks_p'] < 0.01).sum()}")
print(f"Features with LOW/NO shift (p>=0.01):           {(ks_df['ks_p'] >= 0.01).sum()}")
print(f"\nTop 20 most-shifted features:")
print(ks_df[['feature','ks_stat','ks_p','shift']].head(20).to_string())
```

### 7.2 Augmented Dickey-Fuller Stationarity Test

```python
# ADF tests H0: unit root (non-stationary / random walk)
# Small p → reject H0 → feature IS stationary (good)
# Large p → fail to reject → feature has unit root (bad for shuffled dataset)

print("Running ADF stationarity tests on ID-sorted data...")
print("(Non-stationary features in a shuffled dataset are problematic)")

adf_results = []
train_sorted = train.sort_values('ID').reset_index(drop=True)
# Test only base features (lag features are differences, more likely stationary)
for col in base_features[:50]:  # first 50 base features for speed
    vals = train_sorted[col].dropna().values
    if len(vals) < 200:
        continue
    try:
        adf_stat, adf_p, _, _, _, _ = adfuller(vals[:10000], maxlag=5, autolag='AIC')
        adf_results.append({
            'feature'   : col,
            'adf_stat'  : adf_stat,
            'adf_p'     : adf_p,
            'stationary': adf_p < 0.05
        })
    except Exception as e:
        pass

adf_df = pd.DataFrame(adf_results)
print(f"\nStationary (p<0.05): {adf_df['stationary'].sum()} / {len(adf_df)} base features tested")
print(f"Non-stationary:      {(~adf_df['stationary']).sum()} features — may need differencing")
adf_df.sort_values('adf_p', ascending=False).to_csv('outputs/eda/summaries/adf_stationarity.csv', index=False)
```

### 7.3 Kruskal-Wallis Test — Feature Distribution Across TARGET Quintiles

```python
# KW test: is the distribution of a feature different across TARGET quintiles?
# H0: all quintile groups have the same distribution
# Significant result = feature carries information about TARGET quantile (even nonlinear)
# KW statistic H = (12/N(N+1)) * Σ(n_i * (R̄_i - R̄)²)

print("Running Kruskal-Wallis tests across TARGET quintiles...")
target_quintiles = pd.qcut(train['TARGET'], q=5, labels=False, duplicates='drop')

kw_results = []
for col in feature_cols:
    groups = []
    for q in range(5):
        vals = train.loc[target_quintiles == q, col].dropna().values
        if len(vals) > 50:
            groups.append(vals)
    if len(groups) < 3:
        continue
    try:
        kw_stat, kw_p = kruskal(*groups)
        kw_results.append({'feature': col, 'kw_stat': kw_stat, 'kw_p': kw_p})
    except Exception:
        pass

kw_df = pd.DataFrame(kw_results).sort_values('kw_stat', ascending=False)
kw_df.to_csv('outputs/eda/summaries/kruskal_wallis.csv', index=False)
print(f"\nFeatures significant in KW test (p<0.001): {(kw_df['kw_p'] < 0.001).sum()}")
print(f"Top 20 by KW statistic:")
print(kw_df.head(20).to_string())
```

### 7.4 Levene's Test — Variance Homogeneity Across TARGET Quintiles

```python
# Levene tests if variance of a feature is equal across TARGET quintiles
# Features that fail Levene's test are heteroskedastic —
# their signal strength varies across the TARGET range (regime-dependent)

print("Running Levene's test for variance homogeneity...")
levene_results = []
for col in feature_cols[:100]:  # first 100 for speed
    groups = []
    for q in range(5):
        vals = train.loc[target_quintiles == q, col].dropna().values
        if len(vals) > 50:
            groups.append(vals)
    if len(groups) < 3:
        continue
    try:
        lev_stat, lev_p = levene(*groups)
        levene_results.append({'feature': col, 'levene_stat': lev_stat, 'levene_p': lev_p})
    except Exception:
        pass

levene_df = pd.DataFrame(levene_results).sort_values('levene_stat', ascending=False)
levene_df.to_csv('outputs/eda/summaries/levene_test.csv', index=False)
print(f"Features with heteroskedastic signal (Levene p<0.05): {(levene_df['levene_p'] < 0.05).sum()}")
```

---

## LAYER 8 — Temporal Stability Analysis

### 8.1 Feature Drift Detection (Train Late vs Test)

```python
# Compare distribution of features in the LAST 20% of train (by ID)
# vs the test set — this approximates temporal proximity to test

n_late = int(0.2 * len(train))
train_late   = train.nlargest(n_late, 'ID')
train_early  = train.nsmallest(n_late, 'ID')

drift_results = []
for col in feature_cols:
    late_vals  = train_late[col].dropna().values
    test_vals  = test[col].dropna().values
    early_vals = train_early[col].dropna().values
    if len(late_vals) < 100 or len(test_vals) < 100:
        continue
    ks_late_test, p_late_test   = ks_2samp(late_vals, test_vals)
    ks_early_test, p_early_test = ks_2samp(early_vals, test_vals)
    # If late train is MORE similar to test than early train, feature is temporally smooth
    drift_results.append({
        'feature'          : col,
        'ks_late_vs_test'  : ks_late_test,
        'ks_early_vs_test' : ks_early_test,
        'late_closer'      : ks_late_test < ks_early_test
    })

drift_df = pd.DataFrame(drift_results)
pct_late_closer = drift_df['late_closer'].mean() * 100
print(f"\n{pct_late_closer:.1f}% of features have late-train closer to test than early-train")
print("(>50% confirms test is temporally closer to recent training data)")
drift_df.to_csv('outputs/eda/summaries/temporal_drift.csv', index=False)
```

### 8.2 Regime Detection via Clustering

```python
# Use top 20 features by Spearman to detect latent regimes
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

top20_cols = corr_df.dropna(subset=['spearman']).head(20)['feature'].tolist()
X_regime   = train[top20_cols].copy()
for col in top20_cols:
    X_regime[col] = X_regime[col].fillna(X_regime[col].median())

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_regime.values)

# Elbow method for k selection
inertias = []
K_range  = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=5)
    km.fit(X_scaled[:50000])  # sample for speed
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)'); plt.ylabel('Inertia')
plt.title('Elbow Method for Regime Clustering')
plt.savefig('outputs/eda/plots/07_regime_elbow.png', dpi=150)
plt.show()

# Fit with k=3 (typical: bull/bear/neutral)
k_best = 3
km_final = KMeans(n_clusters=k_best, random_state=RANDOM_SEED, n_init=10)
train['regime'] = km_final.fit_predict(X_scaled)

print("\nMean TARGET by regime:")
regime_stats = train.groupby('regime')['TARGET'].agg(['mean','std','count'])
print(regime_stats)
regime_stats.to_csv('outputs/eda/summaries/regime_stats.csv')
```

---

## LAYER 9 — Outlier & Anomaly Analysis

### 9.1 Isolation Forest Anomaly Scores

```python
# Isolation Forest: anomaly score based on average path length in random trees
# Score close to 1 = anomaly, close to 0.5 = normal

print("Fitting Isolation Forest for anomaly detection (sample of 50k rows)...")
sample_idx = np.random.choice(len(train), size=50_000, replace=False)
X_iso = train[top20_cols].iloc[sample_idx].copy()
for col in top20_cols:
    X_iso[col] = X_iso[col].fillna(X_iso[col].median())

iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=RANDOM_SEED, n_jobs=-1)
anomaly_scores = iso.fit_predict(X_iso.values)  # -1 = anomaly, 1 = normal

n_anomalies = (anomaly_scores == -1).sum()
print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(sample_idx)*100:.2f}%) in sample")

# Do anomalies have extreme TARGET values?
target_sample = train['TARGET'].iloc[sample_idx].values
mean_target_anomaly = target_sample[anomaly_scores == -1].mean()
mean_target_normal  = target_sample[anomaly_scores == 1].mean()
print(f"Mean |TARGET| for anomalies: {np.abs(target_sample[anomaly_scores == -1]).mean():.4f}")
print(f"Mean |TARGET| for normals:   {np.abs(target_sample[anomaly_scores == 1]).mean():.4f}")
```

### 9.2 Per-Feature Outlier Counts

```python
# Count rows beyond ±3σ and ±5σ for each feature
outlier_counts = []
for col in feature_cols:
    vals = train[col].dropna().values
    mu   = vals.mean()
    sig  = vals.std()
    n3   = np.mean(np.abs(vals - mu) > 3 * sig) * 100  # % beyond 3σ
    n5   = np.mean(np.abs(vals - mu) > 5 * sig) * 100  # % beyond 5σ
    outlier_counts.append({'feature': col, 'pct_beyond_3sigma': n3, 'pct_beyond_5sigma': n5})

outlier_df = pd.DataFrame(outlier_counts).sort_values('pct_beyond_3sigma', ascending=False)
extreme_outlier_cols = outlier_df[outlier_df['pct_beyond_3sigma'] > 5]['feature'].tolist()
print(f"\nFeatures with >5% of values beyond ±3σ: {len(extreme_outlier_cols)}")
print("  -> These need Winsorizing or log-transform before MLP/linear models")
outlier_df.to_csv('outputs/eda/summaries/outlier_counts.csv', index=False)
```

---

## LAYER 10 — SO3_T & Price Special Analysis

### 10.1 SO3_T as Regime Indicator

```python
# SO3_T is the only named covariate — treat it as a market condition variable
so3_col = 'SO3_T'
so3_vals = train[so3_col].dropna()

print(f"SO3_T stats:")
print(f"  Range   : {so3_vals.min():.4f} — {so3_vals.max():.4f}")
print(f"  Missing : {train[so3_col].isnull().mean()*100:.2f}%")

# Does SO3_T predict TARGET?
r_so3, p_so3 = pearsonr(
    train[so3_col].fillna(train[so3_col].median()).values,
    train['TARGET'].values
)
print(f"  Pearson r with TARGET: {r_so3:.4f} (p={p_so3:.4e})")

# Split into SO3_T quintiles and check TARGET stats per quintile
so3_quintiles = pd.qcut(train[so3_col].fillna(train[so3_col].median()), q=5, labels=False)
print("\nMean TARGET by SO3_T quintile:")
for q in range(5):
    mask = so3_quintiles == q
    print(f"  Q{q}: mean={train.loc[mask,'TARGET'].mean():.4f}, "
          f"std={train.loc[mask,'TARGET'].std():.4f}, "
          f"n={mask.sum():,}")

# Does SO3_T moderate other feature-TARGET correlations?
# Pick top 5 features and compute their correlation with TARGET at different SO3_T levels
top5 = corr_df.dropna(subset=['spearman']).head(5)['feature'].tolist()
print("\nFeature-TARGET correlations conditioned on SO3_T quintile:")
for col in top5:
    qcorrs = []
    for q in range(5):
        mask = so3_quintiles == q
        r, _ = pearsonr(
            train.loc[mask, col].fillna(train[col].median()).values,
            train.loc[mask, 'TARGET'].values
        )
        qcorrs.append(round(r, 3))
    print(f"  {col[:35]}: {qcorrs}")
```

### 10.2 Price Feature Analysis

```python
price_vals = train['Price'].values
log_price  = np.log1p(np.abs(price_vals)) * np.sign(price_vals)

r_price,     p_price     = pearsonr(train['Price'].fillna(train['Price'].median()).values, target_vals)
r_log_price, p_log_price = pearsonr(
    pd.Series(log_price).fillna(0).values, target_vals
)
print(f"Price correlation with TARGET      : r={r_price:.4f}, p={p_price:.4e}")
print(f"log(Price) correlation with TARGET : r={r_log_price:.4f}, p={p_log_price:.4e}")

# Rolling volatility of Price (using ID-sorted chunks as proxy)
price_sorted = train.sort_values('ID')['Price'].values
chunk_cv = []
for i in range(0, len(price_sorted)-1000, 1000):
    chunk = price_sorted[i:i+1000]
    valid = chunk[~np.isnan(chunk)]
    if len(valid) > 10:
        chunk_cv.append(np.std(valid) / (np.abs(np.mean(valid)) + 1e-8))

plt.figure(figsize=(12, 4))
plt.plot(chunk_cv, color='steelblue', lw=1)
plt.title('Price Coefficient of Variation Over ID-sorted Chunks (volatility proxy)')
plt.xlabel('Chunk'); plt.ylabel('CV = std/mean')
plt.tight_layout()
plt.savefig('outputs/eda/plots/08_price_volatility.png', dpi=150)
plt.show()
```

---

## LAYER 11 — PCA per Feature Family

```python
# For large families, check how many PCA components explain 95% variance
# This informs whether we should compress families into PCA features

pca_summary = []
for fam in fam_list:
    fam_cols = [c for c in feature_cols if extract_family(c) == fam]
    if len(fam_cols) < 4:
        continue
    X_fam = train[fam_cols].copy()
    for col in fam_cols:
        X_fam[col] = X_fam[col].fillna(X_fam[col].median())
    X_fam_std = (X_fam - X_fam.mean()) / (X_fam.std() + 1e-8)
    pca = PCA(random_state=RANDOM_SEED)
    pca.fit(X_fam_std.values[:50000])  # sample for speed
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.argmax(cum_var >= 0.95) + 1
    n_99 = np.argmax(cum_var >= 0.99) + 1
    pca_summary.append({
        'family'         : fam,
        'n_features'     : len(fam_cols),
        'n_components_95': n_95,
        'n_components_99': n_99,
        'compression_95' : round(n_95 / len(fam_cols), 2),
        'top1_var_ratio' : pca.explained_variance_ratio_[0]
    })

pca_df = pd.DataFrame(pca_summary).sort_values('compression_95')
print("\nPCA compression by family (compression_95 = n_comp_95 / n_features):")
print(pca_df.to_string())
pca_df.to_csv('outputs/eda/summaries/pca_family_summary.csv', index=False)
print("\nFamilies with compression_95 < 0.3 are highly redundant (prime for PCA)")
print(pca_df[pca_df['compression_95'] < 0.3][['family','n_features','n_components_95']].to_string())
```

---

## LAYER 12 — Per-Lag Predictive Power Comparison

**The most important lag-specific analysis.** For each base feature, which horizon
(T1, T2, T3) is most predictive of TARGET, and does the relationship flip sign
across horizons? A sign flip means T1 is a mean-reversion signal but T3 is a
momentum signal — they need separate treatment.

```python
# For each base feature with all 3 lag versions:
# Compute Spearman correlation of each lag with TARGET independently
# Then compute: best_lag, sign_flip flag, and IC ratio

print("Computing per-lag predictive power for all base features...")
print("This identifies which horizon (T1/T2/T3) dominates each signal family.")

lag_predictive_power = []

for base_col in base_features:
    l1 = base_col + '_LagT1'
    l2 = base_col + '_LagT2'
    l3 = base_col + '_LagT3'

    results = {}
    for lag_name, lag_col in [('T1', l1), ('T2', l2), ('T3', l3)]:
        if lag_col not in feature_cols:
            results[lag_name] = {'spearman': np.nan, 'pearson': np.nan}
            continue
        vals = train[lag_col].values
        valid = ~np.isnan(vals)
        if valid.sum() < 1000:
            results[lag_name] = {'spearman': np.nan, 'pearson': np.nan}
            continue
        r_s, _ = spearmanr(vals[valid], target_vals[valid])
        r_p, _ = pearsonr(vals[valid],  target_vals[valid])
        results[lag_name] = {'spearman': r_s, 'pearson': r_p}

    s1 = results['T1']['spearman']
    s2 = results['T2']['spearman']
    s3 = results['T3']['spearman']

    # Determine best lag by absolute Spearman
    lag_vals = {'T1': s1, 'T2': s2, 'T3': s3}
    valid_lags = {k: v for k, v in lag_vals.items() if not np.isnan(v)}
    if not valid_lags:
        continue
    best_lag = max(valid_lags, key=lambda k: abs(valid_lags[k]))

    # Sign flip: does any pair of lags have opposite signs?
    signs = [np.sign(v) for v in valid_lags.values() if v != 0]
    sign_flip = len(set(signs)) > 1

    # Monotonic decay: |T1| > |T2| > |T3| = mean reversion decaying
    # Monotonic increase: |T1| < |T2| < |T3| = momentum building
    abs_vals = [abs(valid_lags.get(k, np.nan)) for k in ['T1','T2','T3']]
    abs_valid = [v for v in abs_vals if not np.isnan(v)]
    if len(abs_valid) == 3:
        monotone_decay    = abs_valid[0] >= abs_valid[1] >= abs_valid[2]
        monotone_increase = abs_valid[0] <= abs_valid[1] <= abs_valid[2]
    else:
        monotone_decay = monotone_increase = False

    lag_predictive_power.append({
        'base_feature'     : base_col,
        'spearman_T1'      : s1,
        'spearman_T2'      : s2,
        'spearman_T3'      : s3,
        'best_lag'         : best_lag,
        'best_lag_spearman': valid_lags[best_lag],
        'sign_flip'        : sign_flip,
        'monotone_decay'   : monotone_decay,    # mean reversion pattern
        'monotone_increase': monotone_increase, # momentum pattern
        'family'           : extract_family(base_col)
    })

lag_power_df = pd.DataFrame(lag_predictive_power)
lag_power_df = lag_power_df.sort_values('best_lag_spearman', key=abs, ascending=False)
lag_power_df.to_csv('outputs/eda/summaries/lag_predictive_power.csv', index=False)

# Summary statistics
print(f"\nTotal base features analysed: {len(lag_power_df)}")
print(f"Features with sign flip across lags: {lag_power_df['sign_flip'].sum()}")
print(f"  -> These need separate T1 and T3 features in engineering")
print(f"Features showing mean-reversion pattern (|T1|>|T2|>|T3|): {lag_power_df['monotone_decay'].sum()}")
print(f"Features showing momentum pattern (|T1|<|T2|<|T3|):       {lag_power_df['monotone_increase'].sum()}")
print(f"\nBest lag distribution:")
print(lag_power_df['best_lag'].value_counts())
print(f"\nTop 20 features by best lag Spearman:")
print(lag_power_df[['base_feature','spearman_T1','spearman_T2','spearman_T3',
                     'best_lag','sign_flip']].head(20).to_string())

# Visualise sign flip features
sign_flip_features = lag_power_df[lag_power_df['sign_flip']]['base_feature'].tolist()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution of best lag
lag_power_df['best_lag'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Which Lag Horizon Is Most Predictive?\n(per base feature)')
axes[0].set_xlabel('Best Lag'); axes[0].set_ylabel('Count of Base Features')

# Scatter: T1 Spearman vs T3 Spearman (quadrant = signal type)
axes[1].scatter(lag_power_df['spearman_T1'], lag_power_df['spearman_T3'],
                alpha=0.4, s=15, color='purple')
axes[1].axhline(0, color='black', lw=1); axes[1].axvline(0, color='black', lw=1)
axes[1].set_xlabel('Spearman r (LagT1 vs TARGET)')
axes[1].set_ylabel('Spearman r (LagT3 vs TARGET)')
axes[1].set_title('T1 vs T3 Spearman\n(off-diagonal quadrants = sign flip features)')

plt.tight_layout()
plt.savefig('outputs/eda/plots/09_lag_predictive_power.png', dpi=150)
plt.show()
```

---

## LAYER 13 — Autocorrelation of Lag Difference Features

**Measures signal persistence.** Even in a shuffled dataset, the autocorrelation
of lag features (computed on ID-sorted rows) tells you how long a deviation tends
to persist. High ACF at lag-1 means the signal is slow-moving and smoothing helps.
Low ACF means the signal is noisy — smoothing will wash it out.

**Mathematical definition:**
```
ACF(k) = Cov(X_t, X_{t-k}) / Var(X_t)
       = E[(X_t - μ)(X_{t-k} - μ)] / σ²

Ljung-Box Q statistic tests H0: ACF(1)=...=ACF(m)=0 (no autocorrelation)
Q = n(n+2) * Σ_{k=1}^{m} [ρ̂²(k) / (n-k)]
```

```python
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

print("Computing ACF for lag difference features (ID-sorted)...")
print("High ACF(1) → smooth signal, benefits from temporal averaging")
print("Low ACF(1)  → noisy signal, smoothing will degrade it")

train_sorted = train.sort_values('ID').reset_index(drop=True)

acf_results = []
# Focus on LagT1 features (most immediate horizon, highest noise)
sample_lag_cols = working_lag1[:60] if 'working_lag1' in dir() else lag_t1_features[:60]

for col in sample_lag_cols:
    if col not in train_sorted.columns:
        continue
    vals = train_sorted[col].fillna(train_sorted[col].median()).values
    try:
        acf_vals = acf(vals[:20000], nlags=5, fft=True)
        # Ljung-Box test for any autocorrelation up to lag 5
        lb_result = acorr_ljungbox(vals[:20000], lags=[5], return_df=True)
        lb_p = lb_result['lb_pvalue'].values[0]
        acf_results.append({
            'feature'   : col,
            'acf_lag1'  : acf_vals[1],
            'acf_lag2'  : acf_vals[2],
            'acf_lag3'  : acf_vals[3],
            'acf_lag5'  : acf_vals[5],
            'lb_pvalue' : lb_p,
            'persistent': abs(acf_vals[1]) > 0.1,  # meaningful autocorrelation
            'family'    : extract_family(col)
        })
    except Exception:
        pass

acf_df = pd.DataFrame(acf_results).sort_values('acf_lag1', key=abs, ascending=False)
acf_df.to_csv('outputs/eda/summaries/lag_acf.csv', index=False)

print(f"\nFeatures with |ACF(lag=1)| > 0.1 (persistent signal): {acf_df['persistent'].sum()}")
print(f"  -> Smoothing (ts_mean over 3 rows) will improve these signals")
print(f"  -> From alpha research: smoothed mean reversion outperforms raw lags")
print(f"\nFeatures with Ljung-Box p < 0.05 (significant autocorrelation): "
      f"{(acf_df['lb_pvalue'] < 0.05).sum()}")
print(f"\nTop 20 by |ACF(1)|:")
print(acf_df[['feature','acf_lag1','acf_lag2','acf_lag5','persistent']].head(20).to_string())

# Plot ACF distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(acf_df['acf_lag1'], bins=50, color='steelblue', alpha=0.7)
axes[0].axvline(0.1,  color='red',   lw=2, linestyle='--', label='|0.1| threshold')
axes[0].axvline(-0.1, color='red',   lw=2, linestyle='--')
axes[0].set_title('Distribution of ACF(lag=1) for LagT1 Features')
axes[0].set_xlabel('ACF at lag 1'); axes[0].legend()

axes[1].scatter(acf_df['acf_lag1'], acf_df['acf_lag2'], alpha=0.5, s=15, color='purple')
axes[1].set_xlabel('ACF lag 1'); axes[1].set_ylabel('ACF lag 2')
axes[1].set_title('ACF(1) vs ACF(2)\n(top-right quadrant = persistent momentum signal)')
plt.tight_layout()
plt.savefig('outputs/eda/plots/10_lag_acf.png', dpi=150)
plt.show()
```

---

## LAYER 14 — Regime-Conditional Feature Correlations

**Answers the critical question:** For each top feature, does its relationship
with TARGET change — or flip sign — across regimes? If yes, that feature needs
a regime interaction term in the feature engineering phase.

**This uses the regime labels from Layer 8.2 (KMeans clustering).**

```python
# Must have train['regime'] column from Layer 8.2
assert 'regime' in train.columns, "Run Layer 8.2 (KMeans regime clustering) first"

print("Computing feature-TARGET correlations conditioned on regime...")
print("A sign flip across regimes = feature needs regime interaction term")

regimes = sorted(train['regime'].unique())
n_regimes = len(regimes)

# Focus on top 80 features by overall Spearman
top80_cols = corr_df.dropna(subset=['spearman']).head(80)['feature'].tolist()

regime_corr_results = []
for col in top80_cols:
    if col not in train.columns:
        continue
    row = {'feature': col}
    regime_corrs = []
    for r in regimes:
        mask = train['regime'] == r
        vals = train.loc[mask, col].fillna(train[col].median()).values
        tgt  = train.loc[mask, 'TARGET'].values
        if len(vals) < 500:
            row[f'r_regime_{r}'] = np.nan
            continue
        rc, _ = pearsonr(vals, tgt)
        row[f'r_regime_{r}'] = rc
        regime_corrs.append(rc)

    # Check for sign flip across regimes
    signs = [np.sign(v) for v in regime_corrs if v != 0 and not np.isnan(v)]
    row['regime_sign_flip'] = len(set(signs)) > 1
    # Magnitude of variation across regimes (std of regime correlations)
    row['regime_corr_std']  = np.std(regime_corrs) if len(regime_corrs) > 1 else np.nan
    # Overall unconditional Spearman (for comparison)
    row['overall_spearman'] = corr_df.loc[corr_df['feature'] == col, 'spearman'].values[0] \
                              if len(corr_df.loc[corr_df['feature'] == col]) > 0 else np.nan
    regime_corr_results.append(row)

regime_corr_df = pd.DataFrame(regime_corr_results)
regime_corr_df = regime_corr_df.sort_values('regime_corr_std', ascending=False)
regime_corr_df.to_csv('outputs/eda/summaries/regime_conditional_correlations.csv', index=False)

# Features that need regime interaction terms
needs_regime_interaction = regime_corr_df[
    regime_corr_df['regime_sign_flip'] == True
]['feature'].tolist()

high_regime_variation = regime_corr_df[
    regime_corr_df['regime_corr_std'] > 0.05
]['feature'].tolist()

print(f"\nFeatures with sign flip across regimes: {len(needs_regime_interaction)}")
print(f"  -> MUST add regime interaction term for these in feature engineering")
print(f"Features with high regime variation (std>0.05): {len(high_regime_variation)}")
print(f"\nTop 20 most regime-dependent features:")
r_cols = ['feature','overall_spearman','regime_sign_flip','regime_corr_std'] + \
         [f'r_regime_{r}' for r in regimes]
print(regime_corr_df[r_cols].head(20).to_string())

# Heatmap of feature-regime correlation matrix
pivot_cols = [f'r_regime_{r}' for r in regimes]
regime_pivot = regime_corr_df.set_index('feature')[pivot_cols].head(40)
plt.figure(figsize=(8, 14))
sns.heatmap(regime_pivot, cmap='RdBu_r', center=0, vmin=-0.2, vmax=0.2,
            annot=True, fmt='.3f', linewidths=0.5)
plt.title('Feature-TARGET Correlation Per Regime\n(red=positive, blue=negative; sign flip=green border)')
plt.tight_layout()
plt.savefig('outputs/eda/plots/11_regime_conditional_corr.png', dpi=150)
plt.show()
```

---

## LAYER 15 — Multiple Hypothesis Correction

**Critical for honest significance.** We run hundreds of statistical tests
(KS, KW, Pearson, Spearman, Levene, Mann-Whitney). Without correction, many
"significant" results are false positives purely by chance.

**Benjamini-Hochberg FDR procedure:**
```
Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
Find k* = max{k : p_(k) ≤ (k/m) * α}
Reject H0 for all tests with p_(i) ≤ p_(k*)

This controls the False Discovery Rate at level α
(expected proportion of false positives among rejections ≤ α)
Unlike Bonferroni, it is not overly conservative for large m.
```

```python
from statsmodels.stats.multitest import multipletests

print("Applying Benjamini-Hochberg FDR correction to all statistical tests...")
print("This gives HONEST significance thresholds after correcting for multiple comparisons")

# 1. Correct KS train-test shift tests
ks_df_loaded = pd.read_csv('outputs/eda/summaries/ks_train_test.csv')
ks_pvals = ks_df_loaded['ks_p'].fillna(1.0).values
reject_ks, pvals_corrected_ks, _, _ = multipletests(ks_pvals, alpha=0.05, method='fdr_bh')
ks_df_loaded['ks_p_corrected'] = pvals_corrected_ks
ks_df_loaded['significant_after_correction'] = reject_ks
ks_df_loaded.to_csv('outputs/eda/summaries/ks_train_test.csv', index=False)
print(f"\nKS tests — significant after BH correction: {reject_ks.sum()} / {len(reject_ks)}")
print(f"  (was {(ks_pvals < 0.05).sum()} before correction)")

# 2. Correct Spearman p-values
corr_df_loaded = pd.read_csv('outputs/eda/summaries/target_correlations.csv')
spearman_pvals = corr_df_loaded['spearman_p'].fillna(1.0).values
reject_sp, pvals_corrected_sp, _, _ = multipletests(spearman_pvals, alpha=0.05, method='fdr_bh')
corr_df_loaded['spearman_p_corrected'] = pvals_corrected_sp
corr_df_loaded['significant_after_correction'] = reject_sp
corr_df_loaded.to_csv('outputs/eda/summaries/target_correlations.csv', index=False)
print(f"\nSpearman tests — significant after BH correction: {reject_sp.sum()} / {len(reject_sp)}")

# 3. Correct KW tests
kw_df_loaded = pd.read_csv('outputs/eda/summaries/kruskal_wallis.csv')
kw_pvals = kw_df_loaded['kw_p'].fillna(1.0).values
reject_kw, pvals_corrected_kw, _, _ = multipletests(kw_pvals, alpha=0.05, method='fdr_bh')
kw_df_loaded['kw_p_corrected'] = pvals_corrected_kw
kw_df_loaded['significant_after_correction'] = reject_kw
kw_df_loaded.to_csv('outputs/eda/summaries/kruskal_wallis.csv', index=False)
print(f"\nKruskal-Wallis tests — significant after BH correction: {reject_kw.sum()} / {len(reject_kw)}")

# 4. Correct missingness signal tests
miss_df_loaded = pd.read_csv('outputs/eda/summaries/missingness_signal.csv')
miss_pvals = miss_df_loaded['mwu_pvalue'].fillna(1.0).values
reject_m, pvals_corrected_m, _, _ = multipletests(miss_pvals, alpha=0.05, method='fdr_bh')
miss_df_loaded['mwu_p_corrected'] = pvals_corrected_m
miss_df_loaded['significant_after_correction'] = reject_m
miss_df_loaded.to_csv('outputs/eda/summaries/missingness_signal.csv', index=False)
print(f"\nMissingness signal tests — significant after BH correction: {reject_m.sum()} / {len(reject_m)}")

print("\n[IMPORTANT] Use '_corrected' p-values for all feature selection decisions.")
print("Features significant only before correction but not after are likely false positives.")
```

---

## LAYER 16 — Pairwise Interaction Screening

**Identify candidate interaction terms before modelling.**
For each pair of top features, compute `corr(feat_A * feat_B, TARGET)`.
A pair whose product correlates with TARGET but neither alone does is
a pure interaction effect — a second-order signal.

**Mathematical definition:**
```
interaction_signal(A, B) = (A - μ_A)/σ_A  *  (B - μ_B)/σ_B

We standardise first to prevent magnitude differences from dominating.

IC_interaction = Spearman(interaction_signal, TARGET)

If |IC_interaction| >> max(|IC_A|, |IC_B|):
  → Pure interaction effect — create A*B feature in engineering
```

```python
print("Screening pairwise interactions among top features...")
print("Looking for feature pairs whose product predicts TARGET better than either alone")

# Use top 40 features to keep pair count manageable (40*39/2 = 780 pairs)
top40 = corr_df.dropna(subset=['spearman']).head(40)['feature'].tolist()
top40 = [c for c in top40 if c in train.columns]

# Standardise top features
X_top = train[top40].copy()
for col in top40:
    mu  = X_top[col].mean()
    sig = X_top[col].std() + 1e-8
    X_top[col] = ((X_top[col] - mu) / sig).clip(-5, 5)

interaction_results = []
for i in range(len(top40)):
    for j in range(i+1, len(top40)):
        col_a = top40[i]
        col_b = top40[j]
        prod  = (X_top[col_a] * X_top[col_b]).values

        # Skip if too many NaN in product
        valid = ~np.isnan(prod)
        if valid.sum() < 5000:
            continue

        ic_prod, _ = spearmanr(prod[valid], target_vals[valid])

        # Individual ICs for comparison
        ic_a = corr_df.loc[corr_df['feature'] == col_a, 'spearman'].values[0]
        ic_b = corr_df.loc[corr_df['feature'] == col_b, 'spearman'].values[0]

        # Interaction gain: how much does the product add beyond the best individual?
        interaction_gain = abs(ic_prod) - max(abs(ic_a), abs(ic_b))

        interaction_results.append({
            'feature_a'        : col_a,
            'feature_b'        : col_b,
            'ic_product'       : ic_prod,
            'ic_a'             : ic_a,
            'ic_b'             : ic_b,
            'interaction_gain' : interaction_gain,
            'same_family'      : extract_family(col_a) == extract_family(col_b)
        })

interaction_df = pd.DataFrame(interaction_results)
interaction_df = interaction_df.sort_values('interaction_gain', ascending=False)
interaction_df.to_csv('outputs/eda/summaries/pairwise_interactions.csv', index=False)

# Top interactions — positive gain means product > both individuals
top_interactions = interaction_df[interaction_df['interaction_gain'] > 0.01]
cross_family_interactions = top_interactions[~top_interactions['same_family']]

print(f"\nTotal pairs screened: {len(interaction_df)}")
print(f"Pairs with interaction gain > 0.01: {len(top_interactions)}")
print(f"  -> Cross-family interactions (most novel): {len(cross_family_interactions)}")
print(f"\nTop 20 candidate interaction pairs:")
print(interaction_df[['feature_a','feature_b','ic_product','ic_a','ic_b',
                       'interaction_gain','same_family']].head(20).to_string())

plt.figure(figsize=(10, 5))
plt.scatter(interaction_df['interaction_gain'],
            interaction_df['ic_product'].abs(),
            alpha=0.3, s=10, color='steelblue')
plt.axvline(0.01, color='red', lw=2, linestyle='--', label='Gain > 0.01 threshold')
plt.xlabel('Interaction Gain (|IC_product| - max(|IC_a|, |IC_b|))')
plt.ylabel('|IC of product|')
plt.title('Pairwise Interaction Screening\n(right of red line = add A*B as feature)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/eda/plots/12_interaction_screening.png', dpi=150)
plt.show()
```

---

## LAYER 17 — Information Coefficient (IC) and ICIR Analysis

**Directly from quantitative finance / alpha research.**
IC and ICIR are the standard metrics for evaluating signal quality in systematic trading
and are more informative than a single global Spearman correlation.

**Mathematical definitions:**
```
IC_t   = Spearman(feature_t, TARGET_t)      computed per time chunk t
Mean IC = (1/T) * Σ_t IC_t
IC Std  = std({IC_t})
ICIR   = Mean IC / IC Std                   (analogous to Sharpe ratio for signals)

Interpretation:
  |Mean IC| > 0.02  → signal has meaningful predictive power
  ICIR > 0.5        → signal is consistent enough to trade on
  ICIR > 1.0        → strong, institutional-grade signal
  Mean IC > 0 but ICIR low → signal works on average but is regime-dependent
```

This is the **most practically useful** metric for deciding which features to
prioritise in the model, because it measures both strength AND consistency.

```python
print("Computing IC and ICIR for all features across temporal chunks...")
print("IC = Spearman(feature, TARGET) per chunk | ICIR = mean(IC) / std(IC)")
print("(Higher ICIR = more consistent signal = safer to rely on in production)")

N_CHUNKS = 20
train_sorted_ic = train.sort_values('ID').reset_index(drop=True)
chunk_size_ic   = len(train_sorted_ic) // N_CHUNKS

ic_results = []
for col in feature_cols:
    if col not in train_sorted_ic.columns:
        continue
    chunk_ics = []
    for i in range(N_CHUNKS):
        chunk = train_sorted_ic.iloc[i*chunk_size_ic : (i+1)*chunk_size_ic]
        vals  = chunk[col].fillna(chunk[col].median()).values
        tgt   = chunk['TARGET'].values
        valid = ~np.isnan(vals)
        if valid.sum() < 200:
            chunk_ics.append(np.nan)
            continue
        ic, _ = spearmanr(vals[valid], tgt[valid])
        chunk_ics.append(ic)

    chunk_ics_valid = [v for v in chunk_ics if not np.isnan(v)]
    if len(chunk_ics_valid) < 5:
        continue

    mean_ic = np.mean(chunk_ics_valid)
    std_ic  = np.std(chunk_ics_valid) + 1e-8
    icir    = mean_ic / std_ic
    # IC > 0 fraction (what % of time periods is the signal in the right direction)
    ic_pos_frac = np.mean([v > 0 for v in chunk_ics_valid])

    ic_results.append({
        'feature'    : col,
        'mean_ic'    : mean_ic,
        'std_ic'     : std_ic,
        'icir'       : icir,
        'abs_icir'   : abs(icir),
        'ic_pos_frac': ic_pos_frac,
        'n_chunks'   : len(chunk_ics_valid),
        'family'     : extract_family(col),
        'lag_type'   : ('LagT1' if '_LagT1' in col else
                        'LagT2' if '_LagT2' in col else
                        'LagT3' if '_LagT3' in col else 'base')
    })

ic_df = pd.DataFrame(ic_results).sort_values('abs_icir', ascending=False)
ic_df.to_csv('outputs/eda/summaries/ic_icir.csv', index=False)

print(f"\nFeatures with |Mean IC| > 0.02:  {(ic_df['mean_ic'].abs() > 0.02).sum()}")
print(f"Features with |ICIR| > 0.5:      {(ic_df['abs_icir'] > 0.5).sum()}")
print(f"Features with |ICIR| > 1.0:      {(ic_df['abs_icir'] > 1.0).sum()}")
print(f"\nTop 25 features by |ICIR|:")
print(ic_df[['feature','mean_ic','std_ic','icir','ic_pos_frac']].head(25).to_string())

# Compare ICIR ranking vs Spearman ranking
ic_df_merged = ic_df.merge(
    corr_df[['feature','spearman']].rename(columns={'spearman':'global_spearman'}),
    on='feature', how='left'
)
# Features that rank highly by ICIR but not by global Spearman = hidden consistent signals
ic_df_merged['icir_rank']     = ic_df_merged['abs_icir'].rank(ascending=False)
ic_df_merged['spearman_rank'] = ic_df_merged['global_spearman'].abs().rank(ascending=False)
ic_df_merged['rank_diff']     = ic_df_merged['spearman_rank'] - ic_df_merged['icir_rank']
hidden_signals = ic_df_merged[ic_df_merged['rank_diff'] > 50].sort_values('rank_diff', ascending=False)
print(f"\nFeatures ranking much higher by ICIR than by global Spearman: {len(hidden_signals)}")
print("  -> These are consistent but modest signals — very valuable for ensemble stability")
print(hidden_signals[['feature','mean_ic','icir','global_spearman','rank_diff']].head(15).to_string())
ic_df_merged.to_csv('outputs/eda/summaries/ic_icir_full.csv', index=False)

# Visualise IC distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].hist(ic_df['mean_ic'], bins=60, color='steelblue', alpha=0.7)
axes[0,0].axvline(0.02,  color='red', lw=2, linestyle='--', label='IC=0.02 threshold')
axes[0,0].axvline(-0.02, color='red', lw=2, linestyle='--')
axes[0,0].set_title('Distribution of Mean IC Across Features')
axes[0,0].legend()

axes[0,1].hist(ic_df['icir'], bins=60, color='coral', alpha=0.7)
axes[0,1].axvline(0.5,  color='red', lw=2, linestyle='--', label='ICIR=0.5')
axes[0,1].axvline(-0.5, color='red', lw=2, linestyle='--')
axes[0,1].set_title('Distribution of ICIR Across Features')
axes[0,1].legend()

axes[1,0].scatter(ic_df['mean_ic'], ic_df['icir'], alpha=0.3, s=10, color='purple')
axes[1,0].axhline(0.5, color='red', lw=1, linestyle='--')
axes[1,0].axvline(0.02, color='red', lw=1, linestyle='--')
axes[1,0].set_title('Mean IC vs ICIR\n(top-right quadrant = strong + consistent signals)')
axes[1,0].set_xlabel('Mean IC'); axes[1,0].set_ylabel('ICIR')

ic_df_lag = ic_df.groupby('lag_type')['abs_icir'].mean()
axes[1,1].bar(ic_df_lag.index, ic_df_lag.values, color='green', alpha=0.7)
axes[1,1].set_title('Mean |ICIR| by Feature Type\n(base vs LagT1/T2/T3)')
axes[1,1].set_xlabel('Feature type'); axes[1,1].set_ylabel('Mean |ICIR|')

plt.suptitle('IC and ICIR Analysis — Signal Quality Across Time', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda/plots/13_ic_icir_analysis.png', dpi=150)
plt.show()
```

---

## LAYER 18 — Final EDA Summary Report & Recommendations

```python
# ================================================================
# COMPILE ALL FINDINGS INTO A SINGLE DECISION-READY REPORT
# This is the master output consumed by 02_feature_engineering.ipynb
# ================================================================

print("\n" + "="*70)
print("EDA COMPLETE — COMPREHENSIVE DECISION SUMMARY")
print("="*70)

# ── FEATURES TO DROP ──────────────────────────────────────────────
drop_decisions = list(set(
    drop_zero_var +
    miss_tiers['very_high_miss'] +
    unstable_features
))
print(f"\n[DROP]  Zero-variance + >80% missing + sign-flip unstable: {len(drop_decisions)} features")

# ── FEATURES NEEDING TRANSFORMATION ───────────────────────────────
needs_transform = list(set(high_skew_cols + extreme_outlier_cols))
print(f"[TRANSFORM] High skew / extreme outliers: {len(needs_transform)} features → Yeo-Johnson")

# ── TOP SIGNALS (honoured Spearman after BH correction) ───────────
corr_df_final = pd.read_csv('outputs/eda/summaries/target_correlations.csv')
top_signals = corr_df_final[
    (corr_df_final['spearman_abs'] > 0.03) &
    (corr_df_final['significant_after_correction'] == True) &
    (~corr_df_final['feature'].isin(drop_decisions))
].head(100)['feature'].tolist()
print(f"[KEEP]  Top-signal features (|Spearman|>0.03, BH-significant): {len(top_signals)}")

# ── TOP FEATURES BY ICIR (consistent signals) ─────────────────────
ic_df_final    = pd.read_csv('outputs/eda/summaries/ic_icir.csv')
top_icir_features = ic_df_final[ic_df_final['abs_icir'] > 0.5]['feature'].tolist()
print(f"[ICIR]  Features with |ICIR| > 0.5 (consistent across time): {len(top_icir_features)}")

# ── MISSINGNESS INDICATORS NEEDED ─────────────────────────────────
miss_df_final  = pd.read_csv('outputs/eda/summaries/missingness_signal.csv')
needs_indicator = miss_df_final[
    miss_df_final['significant_after_correction'] == True
]['feature'].tolist()
print(f"[INDICATOR] NaN is predictive of TARGET (BH-corrected): {len(needs_indicator)} → add binary flag")

# ── COVARIATE SHIFT RISK ───────────────────────────────────────────
ks_df_final    = pd.read_csv('outputs/eda/summaries/ks_train_test.csv')
high_shift     = ks_df_final[
    ks_df_final['significant_after_correction'] == True
]['feature'].tolist()
print(f"[RISK]  High train-test distribution shift (BH-corrected): {len(high_shift)} → quantile norm")

# ── REGIME INTERACTION FEATURES ───────────────────────────────────
regime_corr_final = pd.read_csv('outputs/eda/summaries/regime_conditional_correlations.csv')
needs_regime_interact = regime_corr_final[
    regime_corr_final['regime_sign_flip'] == True
]['feature'].tolist()
print(f"[REGIME] Features with sign flip across regimes: {len(needs_regime_interact)} → add regime*feature")

# ── SIGN-FLIP LAG FEATURES ────────────────────────────────────────
lag_power_final = pd.read_csv('outputs/eda/summaries/lag_predictive_power.csv')
sign_flip_lags  = lag_power_final[lag_power_final['sign_flip'] == True]['base_feature'].tolist()
print(f"[LAG]   Features with T1 vs T3 sign flip: {len(sign_flip_lags)} → keep T1 and T3 separately")

# ── BEST LAG PER FEATURE ──────────────────────────────────────────
best_lag_map = dict(zip(lag_power_final['base_feature'], lag_power_final['best_lag']))
print(f"[LAG]   Best lag distribution: {lag_power_final['best_lag'].value_counts().to_dict()}")

# ── TOP INTERACTION PAIRS ─────────────────────────────────────────
interact_final = pd.read_csv('outputs/eda/summaries/pairwise_interactions.csv')
top_interaction_pairs = interact_final[
    interact_final['interaction_gain'] > 0.01
][['feature_a','feature_b','interaction_gain']].head(20).to_dict('records')
print(f"[INTERACT] Top interaction pairs (gain>0.01): {len(top_interaction_pairs)} pairs")

# ── SMOOTHING CANDIDATES ──────────────────────────────────────────
acf_final = pd.read_csv('outputs/eda/summaries/lag_acf.csv')
needs_smoothing = acf_final[acf_final['persistent'] == True]['feature'].tolist()
print(f"[SMOOTH] Features with |ACF(1)| > 0.1 (benefit from smoothing): {len(needs_smoothing)}")

# ── FEATURES TO DEFINITELY INCLUDE IN MLP ─────────────────────────
# Intersection of: top ICIR + low KS shift + not highly skewed
mlp_priority = list(set(top_icir_features) - set(high_shift) - set(high_skew_cols))
print(f"[MLP]   Priority features for MLP (high ICIR + low shift + low skew): {len(mlp_priority)}")

# ── SAVE MASTER DECISIONS FILE ────────────────────────────────────
decisions = {
    'drop'                   : drop_decisions,
    'needs_transform'        : needs_transform,
    'top_signals'            : top_signals,
    'top_icir_features'      : top_icir_features,
    'needs_indicator'        : needs_indicator,
    'high_ks_shift'          : high_shift,
    'unstable_corr'          : unstable_features,
    'needs_regime_interaction': needs_regime_interact,
    'sign_flip_lags'         : sign_flip_lags,
    'best_lag_map'           : best_lag_map,
    'top_interaction_pairs'  : top_interaction_pairs,
    'needs_smoothing'        : needs_smoothing,
    'mlp_priority_features'  : mlp_priority,
    # Metadata for downstream notebooks
    'n_regimes'              : len(regimes) if 'regimes' in dir() else 3,
}
with open('outputs/eda/summaries/eda_decisions.pkl', 'wb') as f:
    pickle.dump(decisions, f)

# ── PRINT FULL RECOMMENDATION SUMMARY ────────────────────────────
print("\n" + "="*70)
print("RECOMMENDATIONS FOR FEATURE_ENGINEERING_INSTRUCTIONS.md")
print("="*70)
print(f"""
1. DROP {len(drop_decisions)} features immediately (zero variance, >80% missing, sign-flip unstable)

2. CREATE missingness indicators for {len(needs_indicator)} features
   (NaN is statistically predictive of TARGET after BH correction)

3. VOLATILITY-NORMALISE all lag features
   Primary signal type — validated by alpha research and ICIR analysis

4. SMOOTH (average T1+T2+T3) for {len(needs_smoothing)} features
   These have |ACF(1)| > 0.1 — temporal averaging improves signal quality

5. KEEP T1 AND T3 SEPARATE for {len(sign_flip_lags)} base features
   Sign flips mean T1 = mean reversion, T3 = momentum — opposite directions

6. ADD REGIME INTERACTION for {len(needs_regime_interact)} features
   Correlation with TARGET changes sign across market regimes

7. CREATE {len(top_interaction_pairs)} pairwise interaction features
   Product of two features predicts TARGET better than either alone

8. APPLY QUANTILE TRANSFORM to {len(high_shift)} high-drift features
   Significant train-test distribution shift (after BH correction)

9. YEO-JOHNSON TRANSFORM for {len(needs_transform)} skewed features
   Improves MLP and linear model inputs

10. PRIORITISE {len(top_icir_features)} features with |ICIR| > 0.5
    These are the most consistent signals across time — safest to rely on
""")

print("All EDA outputs saved to outputs/eda/")
print("Load eda_decisions.pkl in 02_feature_engineering.ipynb to proceed.")
```

---

## Output Files Checklist

After running all cells, verify these files exist in `outputs/eda/`:

**Summaries (`.csv` / `.pkl` / `.parquet`):**
- `taxonomy.pkl` — feature groupings
- `missingness_tiers.pkl` — miss rate tiers
- `missingness_comparison.csv` — train vs test missingness
- `missingness_signal.csv` — NaN predictive of TARGET (with BH correction)
- `feature_moments.csv` — skew, kurtosis, etc. for all features
- `lag_variance_consistency.csv` — lag monotonicity check
- `target_correlations.csv` — Pearson, Spearman, MI with TARGET (with BH correction)
- `mutual_information.csv` — MI scores
- `correlation_stability.csv` — per-chunk correlations
- `partial_correlations.csv` — partial r controlling for Price
- `full_corr_matrix.parquet` — 445×445 matrix
- `high_corr_pairs.csv` — pairs with |r|>0.90
- `intra_family_correlation.csv` — within-family redundancy
- `vif_top50.csv` — VIF for top 50 features
- `ks_train_test.csv` — train-test KS statistics (with BH correction)
- `adf_stationarity.csv` — ADF test results
- `kruskal_wallis.csv` — KW across TARGET quintiles (with BH correction)
- `levene_test.csv` — variance homogeneity
- `temporal_drift.csv` — early vs late train proximity to test
- `regime_stats.csv` — KMeans regime analysis
- `outlier_counts.csv` — outlier fractions per feature
- `pca_family_summary.csv` — PCA compression ratios
- `lag_predictive_power.csv` — per-lag Spearman, sign flips, momentum/reversion pattern
- `lag_acf.csv` — autocorrelation of lag features, smoothing candidates
- `regime_conditional_correlations.csv` — feature-TARGET corr per regime
- `pairwise_interactions.csv` — top interaction pairs by IC gain
- `ic_icir.csv` — IC and ICIR per feature (primary signal quality metric)
- `ic_icir_full.csv` — IC/ICIR merged with Spearman ranking comparison
- `eda_decisions.pkl` — **master decision file for Phase 2** (expanded with all new analyses)

**Plots (`.png`):**
- `01_id_distribution.png`
- `02_target_analysis.png`
- `03_missingness_family_corr.png`
- `04_distribution_shapes.png`
- `05_correlation_stability.png`
- `06_clustered_corr_heatmap.png`
- `07_regime_elbow.png`
- `08_price_volatility.png`
- `09_lag_predictive_power.png`
- `10_lag_acf.png`
- `11_regime_conditional_corr.png`
- `12_interaction_screening.png`
- `13_ic_icir_analysis.png`
