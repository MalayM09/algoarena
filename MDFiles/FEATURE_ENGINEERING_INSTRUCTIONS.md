# Feature Engineering Instructions — Short-Horizon Return Prediction
## For Claude Code Agent

---

## EDA-Driven Changes vs Previous Version

The following changes were made based on completed EDA findings.
Everything not listed here is **identical to the previous version**.

| What changed | Why |
|---|---|
| STEP 9C (lag smoothing) — **REMOVED** | ACF = 0.011 max across all lag features. Smoothing degrades signal. |
| STEP 10 (PCA compression) — **REMOVED** | No family has compression ratio < 0.3. Lowest is S03 at 0.34. PCA destroys signal here. |
| Pairwise interactions — **NEVER ADDED** | All 780 pairs screened — zero gain > 0.01. Interactions cancel, not amplify. |
| STEP 6 rank transforms — **EXPANDED** | Now covers ALL 444 high-KS-drift features, not just top_signals. 99.8% drift is universal. |
| STEP 11 — **NEW: regime × feature interactions** | 61 sign-flip features from EDA Layer 14 get `feature * regime_r` products. |
| STEP 12 quantile transform — **EXPANDED** | Now covers all high_ks_shift features (not capped at 30). Fitted in batches of 50. |
| STEP 14 feature sets — **UPDATED** | `tree_features` primary set is now `top_icir_features` (279 features with ICIR>0.5). |
| STEP 0 drops — **EXPANDED** | Now also drops adversarial validation features and r=1.000 duplicate pairs. |
| SO3_T composites (Step 9A) — **KEPT but flagged** | SO3_T r=0.0008 with TARGET (insignificant). Kept as low-priority candidates only. |

---

## Prerequisites

**Run AFTER both of the following are complete:**
1. `notebooks/01_eda.ipynb` — produces `outputs/eda/summaries/eda_decisions.pkl`
2. Adversarial validation — produces `outputs/eda/summaries/adversarial_drop_list.pkl`

**Location:** `notebooks/02_feature_engineering.ipynb`

**Inputs from EDA:**
```python
import pickle

with open('outputs/eda/summaries/eda_decisions.pkl', 'rb') as f:
    decisions = pickle.load(f)

# decisions contains:
# decisions['drop']                     — features to remove
# decisions['needs_transform']          — features needing Yeo-Johnson
# decisions['top_signals']              — BH-corrected Spearman significant features
# decisions['top_icir_features']        — 279 features with |ICIR| > 0.5 (PRIMARY SET)
# decisions['needs_indicator']          — features where NaN is predictive
# decisions['high_ks_shift']            — ALL 444 high train-test distribution shift features
# decisions['unstable_corr']            — correlation sign-flip features
# decisions['needs_regime_interaction'] — 61 features that flip sign across regimes
# decisions['sign_flip_lags']           — 22 base features with T1 vs T3 sign flip
# decisions['best_lag_map']             — dict: base_feature -> best lag horizon
# decisions['n_regimes']                — number of KMeans regimes (3)
```

**All engineered features must be saved to:**
- `data/processed/train_engineered.parquet`
- `data/processed/test_engineered.parquet`
- `outputs/feature_engineering/feature_registry.pkl`

---

## Environment Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import rankdata
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler
from sklearn.cluster import KMeans
import pickle
import os
import gc
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

os.makedirs('data/processed', exist_ok=True)
os.makedirs('outputs/feature_engineering', exist_ok=True)
os.makedirs('outputs/feature_engineering/plots', exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Auto-detect actual filenames — do not assume fixed names
raw_files   = os.listdir('data/raw/')
train_file  = [f for f in raw_files if 'train' in f.lower() and f.endswith('.parquet')][0]
test_file   = [f for f in raw_files if 'test'  in f.lower() and f.endswith('.parquet')][0]
print(f"Using train: {train_file}, test: {test_file}")

train = pd.read_parquet(f'data/raw/{train_file}')
test  = pd.read_parquet(f'data/raw/{test_file}')

# Load EDA outputs
with open('outputs/eda/summaries/eda_decisions.pkl', 'rb') as f:
    decisions = pickle.load(f)
with open('outputs/eda/summaries/taxonomy.pkl', 'rb') as f:
    taxonomy = pickle.load(f)

# Load adversarial drop list if it exists
adversarial_drop = []
adv_path = 'outputs/eda/summaries/adversarial_drop_list.pkl'
if os.path.exists(adv_path):
    with open(adv_path, 'rb') as f:
        adversarial_drop = pickle.load(f)
    print(f"Loaded {len(adversarial_drop)} adversarially-identified drop features")
else:
    print("WARNING: No adversarial drop list found. Run adversarial validation first.")

feature_cols    = [c for c in train.columns if c not in ['ID', 'TARGET']]
base_features   = taxonomy['base_features']
lag_t1_features = taxonomy['lag_t1_features']
lag_t2_features = taxonomy['lag_t2_features']
lag_t3_features = taxonomy['lag_t3_features']
family_map      = taxonomy['family_map']

def extract_family(col):
    col_clean = col.replace('_LagT1','').replace('_LagT2','').replace('_LagT3','')
    parts = col_clean.split('_')
    return parts[0] if len(parts) >= 1 else 'UNKNOWN'

# Downcast for M1 memory management
for col in feature_cols:
    train[col] = train[col].astype(np.float32)
for col in [c for c in test.columns if c != 'ID']:
    test[col] = test[col].astype(np.float32)

train_eng = train.copy()
test_eng  = test.copy()
feature_registry = {}

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Starting feature count: {len(feature_cols)}")
gc.collect()
```

---

## STEP 0 — Drop Useless Features

**Expanded vs previous version: now also drops adversarial validation features
and r=1.000 perfectly correlated duplicate pairs.**

```python
# Combine all drop sources
all_drops = list(set(
    decisions['drop'] +   # zero-variance, >80% missing, sign-flip unstable (EDA)
    adversarial_drop      # features driving train-test distinguishability
))

cols_to_drop = [c for c in all_drops if c in train_eng.columns]
train_eng.drop(columns=cols_to_drop, inplace=True, errors='ignore')
test_eng.drop(columns=cols_to_drop,  inplace=True, errors='ignore')

print(f"EDA drops          : {len(decisions['drop'])}")
print(f"Adversarial drops  : {len(adversarial_drop)}")
print(f"Total dropped      : {len(cols_to_drop)}")

# Drop one of each perfectly correlated pair (r=1.000 from EDA Layer 6)
# Keep feature_a, drop feature_b — they carry identical information
high_corr = pd.read_csv('outputs/eda/summaries/high_corr_pairs.csv')
perfect_drop = list(set(
    row['feature_b'] for _, row in high_corr.iterrows()
    if abs(row['correlation']) >= 0.9999 and row['feature_b'] in train_eng.columns
))
train_eng.drop(columns=perfect_drop, inplace=True, errors='ignore')
test_eng.drop(columns=perfect_drop,  inplace=True, errors='ignore')
print(f"Perfect-corr drops : {len(perfect_drop)} (r=1.000 duplicates)")
print(f"Remaining features : {train_eng.shape[1]}")

# Update working lists
working_features = [c for c in feature_cols if c in train_eng.columns]
working_base     = [c for c in base_features    if c in train_eng.columns]
working_lag1     = [c for c in lag_t1_features   if c in train_eng.columns]
working_lag2     = [c for c in lag_t2_features   if c in train_eng.columns]
working_lag3     = [c for c in lag_t3_features   if c in train_eng.columns]
gc.collect()
```

---

## STEP 1 — Missingness Indicator Features

**EDA finding: dataset has 0 missing values. This step creates 0 indicator features.
Kept for pipeline completeness and future compatibility.**

```python
# If E[TARGET | X=NaN] != E[TARGET | X observed], the missingness carries signal.
# BH-corrected result from EDA: 0 features with significant missingness signal.

miss_indicator_cols = []
for col in decisions['needs_indicator']:
    if col not in train_eng.columns:
        continue
    flag_col = f"{col}_was_missing"
    train_eng[flag_col] = train_eng[col].isnull().astype(np.float32)
    test_eng[flag_col]  = test_eng[col].isnull().astype(np.float32) \
                          if col in test_eng.columns else 0.0
    miss_indicator_cols.append(flag_col)
    feature_registry[flag_col] = f"Binary: 1 if {col} is NaN, 0 otherwise"

print(f"Created {len(miss_indicator_cols)} missingness indicator features.")
```

---

## STEP 2 — Median Imputation

Compute medians on training data ONLY. Apply to both train and test.

```python
imputation_medians = {}
for col in working_features:
    med = train_eng[col].median()
    imputation_medians[col] = med
    train_eng[col] = train_eng[col].fillna(med)
    if col in test_eng.columns:
        test_eng[col] = test_eng[col].fillna(med)

with open('outputs/feature_engineering/imputation_medians.pkl', 'wb') as f:
    pickle.dump(imputation_medians, f)

print("Imputation complete.")
assert train_eng[working_features].isnull().sum().sum() == 0, "NaN remain in train!"
print("All clear — no NaN in working features.")
gc.collect()
```

---

## STEP 3 — Volatility-Normalised Lag Features

**Most important engineering step.** Directly from alpha research:
`rank((delay - close) / ts_std_dev(returns, N))` outperforms raw lags.

**EDA confirmation:** Top 5 ICIR features are all LagT1/T2. LagT2 has the highest
mean ICIR (1.69) > LagT1 (1.64) > LagT3 (1.30) > base (0.65).

**Mathematical definition:**
```
vol_norm_lag(X, T) = LagT / (σ_cross + ε)

  LagT    = pre-computed lag-difference feature in dataset
  σ_cross = cross-sectional std of base feature across all rows
  ε       = 1e-8 numerical stability

Interpretation: How large is this deviation relative to typical
magnitude of this feature across the entire universe?
```

```python
vol_norm_features = []

for base_col in working_base:
    l1_col = base_col + '_LagT1'
    l2_col = base_col + '_LagT2'
    l3_col = base_col + '_LagT3'

    if base_col not in train_eng.columns:
        continue

    base_std = train_eng[base_col].std() + 1e-8

    for lag_col in [l1_col, l2_col, l3_col]:
        if lag_col not in train_eng.columns:
            continue
        new_col = f"{lag_col}_volnorm"
        train_eng[new_col] = (train_eng[lag_col] / base_std).astype(np.float32)
        if lag_col in test_eng.columns:
            test_eng[new_col] = (test_eng[lag_col] / base_std).astype(np.float32)
        vol_norm_features.append(new_col)
        feature_registry[new_col] = f"LagT_volnorm: {lag_col} / std({base_col})"

print(f"Created {len(vol_norm_features)} volatility-normalised lag features.")
gc.collect()
```

---

## STEP 4 — Lag Ratio Features (Momentum Acceleration)

**Mathematical definition:**
```
lag_ratio(X, T2, T1) = LagT2 / (|LagT1| + ε)

  ratio >> 1  → longer-horizon deviation much larger (momentum continuing)
  ratio ≈ 1   → consistent drift across horizons
  ratio << 1  → longer horizon already reverted (mean reversion imminent)
  ratio < 0   → lags point in opposite directions (oscillation)
```

```python
lag_ratio_features = []

for base_col in working_base:
    l1 = base_col + '_LagT1'
    l2 = base_col + '_LagT2'
    l3 = base_col + '_LagT3'

    if l1 not in train_eng.columns or l2 not in train_eng.columns:
        continue

    col_21 = f"{base_col}_lagrat_T2_T1"
    train_eng[col_21] = (train_eng[l2] / (train_eng[l1].abs() + 1e-8)).clip(-10, 10).astype(np.float32)
    if l1 in test_eng.columns and l2 in test_eng.columns:
        test_eng[col_21] = (test_eng[l2] / (test_eng[l1].abs() + 1e-8)).clip(-10, 10).astype(np.float32)
    lag_ratio_features.append(col_21)
    feature_registry[col_21] = f"Lag ratio T2/|T1| for {base_col}"

    if l3 not in train_eng.columns:
        continue

    col_32 = f"{base_col}_lagrat_T3_T2"
    train_eng[col_32] = (train_eng[l3] / (train_eng[l2].abs() + 1e-8)).clip(-10, 10).astype(np.float32)
    if l2 in test_eng.columns and l3 in test_eng.columns:
        test_eng[col_32] = (test_eng[l3] / (test_eng[l2].abs() + 1e-8)).clip(-10, 10).astype(np.float32)
    lag_ratio_features.append(col_32)
    feature_registry[col_32] = f"Lag ratio T3/|T2| for {base_col}"

print(f"Created {len(lag_ratio_features)} lag ratio features.")
```

---

## STEP 5 — Lag Convergence / Divergence Features

**Mathematical definition:**
```
lag_convergence(X) = LagT1 - LagT2/2
  > 0 → reverting faster than expected (strong mean reversion signal)
  < 0 → slow to revert (momentum continuation)

sign_agreement(X) = sign(LagT1) * sign(LagT2) * sign(LagT3)
  = +1 → all three lags same direction (persistent trend)
  = -1 → lags alternate signs (oscillation / noise)
```

```python
convergence_features = []

for base_col in working_base:
    l1 = base_col + '_LagT1'
    l2 = base_col + '_LagT2'
    l3 = base_col + '_LagT3'

    if l1 not in train_eng.columns or l2 not in train_eng.columns:
        continue

    conv_col = f"{base_col}_lag_convergence"
    train_eng[conv_col] = (train_eng[l1] - train_eng[l2] / 2).astype(np.float32)
    if l1 in test_eng.columns and l2 in test_eng.columns:
        test_eng[conv_col] = (test_eng[l1] - test_eng[l2] / 2).astype(np.float32)
    convergence_features.append(conv_col)
    feature_registry[conv_col] = f"Lag convergence: LagT1 - LagT2/2 for {base_col}"

    if l3 not in train_eng.columns:
        continue

    sign_col = f"{base_col}_lag_sign_agree"
    train_eng[sign_col] = (np.sign(train_eng[l1]) * np.sign(train_eng[l2]) * np.sign(train_eng[l3])).astype(np.float32)
    if all(c in test_eng.columns for c in [l1, l2, l3]):
        test_eng[sign_col] = (np.sign(test_eng[l1]) * np.sign(test_eng[l2]) * np.sign(test_eng[l3])).astype(np.float32)
    convergence_features.append(sign_col)
    feature_registry[sign_col] = f"Sign agreement across all 3 lags for {base_col}"

print(f"Created {len(convergence_features)} convergence/sign features.")
```

---

## STEP 6 — Cross-Sectional Rank Transforms

**EDA change: now covers ALL 444 high-KS-shift features (previously only top_signals).**

**Reason:** 99.8% of features have significant KS train-test drift. Rank transform
maps any distribution to [0,1] uniformly — train and test become identically
distributed after ranking regardless of raw value differences. This is the
most important single fix for the universal covariate shift problem.

**Mathematical definition:**
```
rank_transform(X) = rank(X) / N  in [0, 1]

Equivalent to empirical CDF: F_N(x) = (1/N) * #{i : X_i <= x}

Key property: rank(X_train) and rank(X_test) BOTH have uniform [0,1]
distributions regardless of raw distributional differences.
```

```python
rank_features = []

# EXPANDED: rank ALL high-KS-shift features — mandatory for 99.8% drift
rank_candidates = list(set(
    decisions['high_ks_shift'] +                            # all 444 drift features
    decisions.get('top_icir_features', decisions['top_signals'])  # ICIR primary signals
))
rank_candidates = [c for c in rank_candidates if c in train_eng.columns]
print(f"Applying rank transform to {len(rank_candidates)} features...")

n_train = len(train_eng)
n_test  = len(test_eng)

for col in rank_candidates:
    rank_col = f"{col}_rank"
    train_eng[rank_col] = (rankdata(train_eng[col].values) / n_train).astype(np.float32)
    test_eng[rank_col]  = (rankdata(
        test_eng[col].values if col in test_eng.columns else np.zeros(n_test)
    ) / n_test).astype(np.float32)
    rank_features.append(rank_col)
    feature_registry[rank_col] = f"Cross-sectional rank / N of {col} (KS-drift corrected)"

print(f"Created {len(rank_features)} rank-transformed features.")
gc.collect()
```

---

## STEP 7 — Z-Score Normalisation (Cross-Sectional)

**Mathematical definition:**
```
zscore(X) = (X - mu_X) / (sigma_X + epsilon)

mu_X and sigma_X computed over ALL training rows (cross-sectional).
Fitted on train only, applied to test using training statistics.
Critical for MLP — neural networks require zero-centered, bounded inputs.
```

```python
zscore_features = []

# Use ICIR-ranked features as primary candidates (EDA-driven)
zscore_candidates = list(set(
    decisions.get('top_icir_features', decisions['top_signals'])[:80] +
    working_base[:40]
))
zscore_candidates = [c for c in zscore_candidates if c in train_eng.columns]

zscore_params = {}
for col in zscore_candidates:
    mu  = train_eng[col].mean()
    sig = train_eng[col].std() + 1e-8
    zscore_params[col] = (mu, sig)

    z_col = f"{col}_zscore"
    train_eng[z_col] = ((train_eng[col] - mu) / sig).clip(-10, 10).astype(np.float32)
    if col in test_eng.columns:
        test_eng[z_col] = ((test_eng[col] - mu) / sig).clip(-10, 10).astype(np.float32)
    zscore_features.append(z_col)
    feature_registry[z_col] = f"Z-score of {col}: (x - {mu:.4f}) / {sig:.4f}"

with open('outputs/feature_engineering/zscore_params.pkl', 'wb') as f:
    pickle.dump(zscore_params, f)

print(f"Created {len(zscore_features)} z-score normalised features.")
```

---

## STEP 8 — Power Transforms for Skewed Features

**EDA finding: 200 features have |skew| > 2. Apply Yeo-Johnson to the top 50.**

**Yeo-Johnson transform (handles negative values):**
```
y = [(x+1)^lambda - 1] / lambda        if lambda != 0, x >= 0
y = log(x + 1)                         if lambda = 0,  x >= 0
y = -[(-x+1)^(2-lambda) - 1] / (2-lambda)  if lambda != 2, x < 0
y = -log(-x + 1)                       if lambda = 2,  x < 0

lambda estimated by MLE on training data only.
```

```python
from sklearn.preprocessing import PowerTransformer

power_transform_features = []
power_transformers = {}

moments_df = pd.read_csv('outputs/eda/summaries/feature_moments.csv')
high_skew_cols = moments_df[
    (moments_df['skewness'].abs() > 2) &
    (moments_df['feature'].isin(working_features))
]['feature'].tolist()[:50]

for col in high_skew_cols:
    if col not in train_eng.columns:
        continue
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    vals_train = train_eng[col].values.reshape(-1, 1)
    try:
        pt.fit(vals_train)
        pt_col = f"{col}_yeojohnson"
        train_eng[pt_col] = pt.transform(vals_train).flatten().astype(np.float32)
        if col in test_eng.columns:
            test_eng[pt_col] = pt.transform(test_eng[col].values.reshape(-1,1)).flatten().astype(np.float32)
        power_transformers[col] = pt
        power_transform_features.append(pt_col)
        feature_registry[pt_col] = f"Yeo-Johnson power transform of {col}"
    except Exception as e:
        print(f"  Skipped {col}: {e}")

with open('outputs/feature_engineering/power_transformers.pkl', 'wb') as f:
    pickle.dump(power_transformers, f)

print(f"Created {len(power_transform_features)} power-transformed features.")
```

---

## STEP 9 — Alpha-Inspired Composite Features

### 9A — SO3_T-Weighted Lag Signal

**EDA note:** SO3_T Pearson r with TARGET = 0.0008 (p=0.50 — statistically
insignificant). These features are **low priority** and will likely be pruned
by null importances at the modelling stage. Created for completeness.

```python
so3_col = 'SO3_T'
alpha_composite_features = []

if so3_col in train_eng.columns:
    so3_mean = train_eng[so3_col].mean()
    so3_std  = train_eng[so3_col].std() + 1e-8
    so3_norm_train = ((train_eng[so3_col] - so3_mean) / so3_std).clip(-3, 3)
    so3_norm_test  = ((test_eng[so3_col] - so3_mean) / so3_std).clip(-3, 3) \
                     if so3_col in test_eng.columns else 0.0

    for base_col in working_base[:30]:
        l1 = base_col + '_LagT1'
        if l1 not in train_eng.columns:
            continue
        new_col = f"{base_col}_so3_weighted_lag"
        train_eng[new_col] = (train_eng[l1] * (1 + so3_norm_train)).astype(np.float32)
        test_eng[new_col]  = (test_eng[l1] * (1 + so3_norm_test)).astype(np.float32) \
                              if l1 in test_eng.columns else 0.0
        alpha_composite_features.append(new_col)
        feature_registry[new_col] = f"SO3_T-weighted LagT1 of {base_col} (low priority)"

    print(f"Created {len(alpha_composite_features)} SO3_T-weighted composite features.")
else:
    alpha_composite_features = []
    print("SO3_T not available — skipping.")
```

### 9B — Inter-Family Ratio Features

```python
corr_df_fe = pd.read_csv('outputs/eda/summaries/target_correlations.csv')
top_positive = corr_df_fe[corr_df_fe['spearman'] > 0.02].head(20)['feature'].tolist()
top_negative = corr_df_fe[corr_df_fe['spearman'] < -0.02].head(20)['feature'].tolist()

ratio_features = []

for p_col in top_positive[:10]:
    for n_col in top_negative[:10]:
        if p_col not in train_eng.columns or n_col not in train_eng.columns:
            continue
        if family_map.get(p_col,'A') == family_map.get(n_col,'B'):
            continue
        ratio_col = f"ratio_{p_col[:15]}_{n_col[:15]}"
        denom_train = (train_eng[n_col].abs() + 1e-6)
        denom_test  = (test_eng[n_col].abs() + 1e-6) if n_col in test_eng.columns else 1.0
        train_eng[ratio_col] = (train_eng[p_col] / denom_train).clip(-100, 100).astype(np.float32)
        test_eng[ratio_col]  = (test_eng[p_col] / denom_test).clip(-100, 100).astype(np.float32) \
                                if p_col in test_eng.columns else 0.0
        ratio_features.append(ratio_col)
        feature_registry[ratio_col] = f"Ratio: {p_col} / |{n_col}|"

print(f"Created {len(ratio_features)} inter-family ratio features.")
```

> **STEP 9C (Lag Smoothing) — PERMANENTLY REMOVED**
> EDA Layer 13: max |ACF(1)| = 0.011, mean ACF(1) = -0.0011 across all lag features.
> There is no temporal persistence to smooth. Averaging lags reduces signal strength.

---

## STEP 10 — Regime Assignment

**STEP 10 (PCA Compression) — PERMANENTLY REMOVED**
EDA Layer 11: no family has compression ratio < 0.30. Lowest is S03 at 0.34.
PCA would discard real signal. All raw features are retained.

**This step now loads or fits the KMeans regime model and assigns regime labels.**

```python
# Load regime model from EDA output if it exists, otherwise fit fresh
regime_model_path = 'outputs/feature_engineering/regime_model.pkl'

if os.path.exists(regime_model_path):
    with open(regime_model_path, 'rb') as f:
        regime_data = pickle.load(f)
    km         = regime_data['km']
    scaler_reg = regime_data['scaler']
    top20_cols = regime_data['top20_cols']
    print(f"Loaded regime model from disk.")
else:
    # Fit using top ICIR features
    top20_cols = decisions.get('top_icir_features', decisions['top_signals'])[:20]
    top20_cols = [c for c in top20_cols if c in train_eng.columns]
    scaler_reg = StandardScaler()
    X_scaled   = scaler_reg.fit_transform(train_eng[top20_cols].values)
    n_regimes  = decisions.get('n_regimes', 3)
    km         = KMeans(n_clusters=n_regimes, random_state=RANDOM_SEED, n_init=10)
    km.fit(X_scaled[:50000])
    regime_data = {'km': km, 'scaler': scaler_reg, 'top20_cols': top20_cols}
    with open(regime_model_path, 'wb') as f:
        pickle.dump(regime_data, f)
    print(f"Fitted and saved new regime model (k={n_regimes}).")

n_regimes      = decisions.get('n_regimes', 3)
top20_avail_tr = [c for c in top20_cols if c in train_eng.columns]
top20_avail_te = [c for c in top20_cols if c in test_eng.columns]

train_eng['regime'] = km.predict(scaler_reg.transform(train_eng[top20_avail_tr].values)).astype(np.float32)
if len(top20_avail_te) == len(top20_cols):
    test_eng['regime'] = km.predict(scaler_reg.transform(test_eng[top20_avail_te].values)).astype(np.float32)
else:
    test_eng['regime'] = 0.0

# One-hot encode regime
regime_features = []
for r in range(n_regimes):
    col_name = f"regime_{r}"
    train_eng[col_name] = (train_eng['regime'] == r).astype(np.float32)
    test_eng[col_name]  = (test_eng['regime'] == r).astype(np.float32) \
                           if 'regime' in test_eng.columns else 0.0
    regime_features.append(col_name)
    feature_registry[col_name] = f"Binary: row belongs to KMeans regime {r}"

print(f"Regime distribution (train): {train_eng['regime'].value_counts().sort_index().to_dict()}")
print(f"Created {len(regime_features)} regime indicator features.")
gc.collect()
```

---

## STEP 11 — Regime × Feature Interaction Terms

**NEW STEP — addresses EDA Layer 14 finding of 61 sign-flip features.**

**EDA finding:** 61 features flip their correlation sign across regimes.
Feature X has positive ICIR in Regime 0 but negative ICIR in Regime 1.
Without regime conditioning, a model averages these out to near-zero signal.
Regime interaction terms allow the model to learn opposite coefficients per regime.

**Mathematical definition:**
```
regime_interaction(X, r) = X * I(regime == r)

This creates regime-conditional linear responses.
For a feature with IC_regime0 = +0.05 and IC_regime1 = -0.05:
  Without interaction: learned coefficient near 0 (signs cancel)
  With interaction:    correct direction extracted in each regime
```

```python
regime_interaction_features = []

# Features that flip sign across regimes — from EDA Layer 14
sign_flip_feats = decisions.get('needs_regime_interaction', [])
sign_flip_feats = [c for c in sign_flip_feats if c in train_eng.columns]

print(f"Creating regime interactions for {len(sign_flip_feats)} sign-flip features...")
print(f"Total interaction features to create: {len(sign_flip_feats)} x {n_regimes} = "
      f"{len(sign_flip_feats) * n_regimes}")

for feat_col in sign_flip_feats:
    for r in range(n_regimes):
        regime_ind = f"regime_{r}"
        if regime_ind not in train_eng.columns:
            continue
        new_col = f"{feat_col}_x_regime{r}"
        train_eng[new_col] = (train_eng[feat_col] * train_eng[regime_ind]).astype(np.float32)
        if feat_col in test_eng.columns and regime_ind in test_eng.columns:
            test_eng[new_col] = (test_eng[feat_col] * test_eng[regime_ind]).astype(np.float32)
        else:
            test_eng[new_col] = 0.0
        regime_interaction_features.append(new_col)
        feature_registry[new_col] = (
            f"Regime interaction: {feat_col} * regime_{r} — sign-flip feature from EDA Layer 14"
        )

print(f"Created {len(regime_interaction_features)} regime interaction features.")
gc.collect()
```

---

## STEP 12 — Quantile Transform for High-Shift Features

**EDA change: covers ALL high_ks_shift features (previously capped at 30).**
**EDA finding: 99.8% of features have significant KS drift — universal problem.**

```
quantile_transform(x) = Phi^{-1}(F_train(x))

  F_train = empirical CDF of training data
  Phi^{-1} = inverse normal CDF (probit)

Maps any distribution to N(0,1).
Fitted on training data ONLY — applied to test using training mapping.
Processed in batches of 50 for M1 memory management.
```

```python
qt_features   = []
qt_models_all = {}

high_ks_cols = [c for c in decisions['high_ks_shift'] if c in train_eng.columns]
print(f"Applying quantile transform to {len(high_ks_cols)} high-KS-shift features (batched)...")

BATCH_SIZE = 50
for batch_start in range(0, len(high_ks_cols), BATCH_SIZE):
    batch_cols      = high_ks_cols[batch_start : batch_start + BATCH_SIZE]
    batch_cols_test = [c for c in batch_cols if c in test_eng.columns]

    if not batch_cols:
        continue

    qt = QuantileTransformer(
        n_quantiles=min(1000, len(train_eng)),
        output_distribution='normal',
        random_state=RANDOM_SEED
    )

    X_qt_train = train_eng[batch_cols].values
    qt.fit(X_qt_train)
    X_qt_train_t = qt.transform(X_qt_train)

    if len(batch_cols_test) == len(batch_cols):
        X_qt_test_t = qt.transform(test_eng[batch_cols_test].values)
    else:
        X_qt_test_t = np.zeros((len(test_eng), len(batch_cols)))

    for i, col in enumerate(batch_cols):
        qt_col = f"{col}_qtrans"
        train_eng[qt_col] = X_qt_train_t[:, i].astype(np.float32)
        test_eng[qt_col]  = X_qt_test_t[:, i].astype(np.float32)
        qt_features.append(qt_col)
        feature_registry[qt_col] = f"Quantile→Normal transform of {col} (KS-drift feature)"

    qt_models_all[f'batch_{batch_start}'] = {'qt': qt, 'cols': batch_cols}

    batch_num = batch_start // BATCH_SIZE
    if batch_num % 5 == 0:
        done = min(batch_start + BATCH_SIZE, len(high_ks_cols))
        print(f"  Progress: {done}/{len(high_ks_cols)}")
        gc.collect()

with open('outputs/feature_engineering/quantile_transformers.pkl', 'wb') as f:
    pickle.dump(qt_models_all, f)

print(f"Created {len(qt_features)} quantile-transformed features.")
gc.collect()
```

---

## STEP 13 — Winsorize TARGET for Training

**EDA finding: TARGET kurtosis = 48.09. Winsorize at ±5σ (0.302% rows affected).**
±3σ clips 1.4% of rows which is excessive given the dataset size. ±5σ is correct.

```
y_wins = clip(y, q_low, q_high)
  q_low  = percentile(y, 0.5)
  q_high = percentile(y, 99.5)

Winsorize TRAINING data ONLY.
Never winsorize test predictions — submit raw model output.
```

```python
target_raw = train_eng['TARGET'].values
q_low  = np.percentile(target_raw, 0.5)
q_high = np.percentile(target_raw, 99.5)

train_eng['TARGET_raw']  = train_eng['TARGET'].copy()
train_eng['TARGET_wins'] = train_eng['TARGET'].clip(q_low, q_high)

pct_clipped = np.mean((target_raw < q_low) | (target_raw > q_high)) * 100
print(f"Winsorized TARGET at [{q_low:.4f}, {q_high:.4f}]")
print(f"Rows clipped: {pct_clipped:.3f}%  (expected ~0.3% for ±5sigma)")
print(f"TARGET std before: {target_raw.std():.4f}")
print(f"TARGET std after:  {train_eng['TARGET_wins'].std():.4f}")

with open('outputs/feature_engineering/target_winsorize_bounds.pkl', 'wb') as f:
    pickle.dump({'q_low': q_low, 'q_high': q_high}, f)
```

---

## STEP 14 — Build Final Feature Sets

**EDA change: primary tree feature set is `top_icir_features` (279 features
with |ICIR|>0.5), NOT `top_signals` (Spearman threshold).**

**Reason:** Only 5 features have |Spearman|>0.03. But 279 features have
|ICIR|>0.5 with mean ICIR=1.317. ICIR measures consistency over time —
the correct selection metric for a shuffled temporal dataset.

```python
# NOTE: smoothed_features and pca_new_features are removed from this list
engineered_features = (
    miss_indicator_cols          +
    vol_norm_features            +
    lag_ratio_features           +
    convergence_features         +
    rank_features                +
    zscore_features              +
    power_transform_features     +
    alpha_composite_features     +
    ratio_features               +
    # smoothed_features  REMOVED: ACF = 0 across all lags
    # pca_new_features   REMOVED: no family compressible below 0.3 ratio
    regime_features              +
    regime_interaction_features  +
    qt_features
)

print(f"\nOriginal features retained     : {len(working_features)}")
print(f"New engineered features        : {len(engineered_features)}")
print(f"  of which regime interactions : {len(regime_interaction_features)}")
print(f"  of which rank transforms     : {len(rank_features)}")
print(f"  of which quantile transforms : {len(qt_features)}")
print(f"Total features                 : {len(working_features) + len(engineered_features)}")

# Load ICIR-ranked primary features
top_icir = decisions.get('top_icir_features', decisions['top_signals'])
top_icir = [c for c in top_icir if c in train_eng.columns]

# ── TREE MODEL FEATURE SET ─────────────────────────────────────────────────
# Primary signals: top_icir_features + all regime/drift-corrected engineered features
tree_features = list(set(
    top_icir                      +  # 279 ICIR-ranked primary signals
    rank_features                 +  # rank transforms of all KS-drift features
    qt_features                   +  # quantile transforms
    regime_features               +  # regime indicators
    regime_interaction_features   +  # regime x sign-flip feature products (NEW)
    vol_norm_features             +  # volatility-normalised lags
    lag_ratio_features            +  # momentum acceleration
    convergence_features             # convergence / sign agreement
))
tree_features = [c for c in tree_features
                 if c in train_eng.columns
                 and c not in ['TARGET', 'TARGET_raw', 'TARGET_wins', 'ID', 'regime']]

# ── MLP FEATURE SET ────────────────────────────────────────────────────────
# Prefer normalised, bounded features for neural networks
mlp_features = [c for c in engineered_features
                if any(tag in c for tag in ['_zscore', '_rank', '_qtrans',
                                            '_volnorm', 'regime_', '_x_regime'])
                and c in train_eng.columns]
# Add base non-lag features from ICIR set
mlp_features += [c for c in top_icir
                 if c in train_eng.columns and '_LagT' not in c]
mlp_features = list(set(mlp_features))

# ── LINEAR BASELINE FEATURE SET ───────────────────────────────────────────
# Low-VIF only to avoid multicollinearity
vif_df = pd.read_csv('outputs/eda/summaries/vif_top50.csv')
low_vif_features = vif_df[vif_df['VIF'] < 10]['feature'].tolist()
linear_features  = [c for c in low_vif_features if c in train_eng.columns]

print(f"\nTree feature set     : {len(tree_features)} features")
print(f"MLP feature set      : {len(mlp_features)} features")
print(f"Linear feature set   : {len(linear_features)} features")

feature_sets = {
    'tree_features'              : tree_features,
    'mlp_features'               : mlp_features,
    'linear_features'            : linear_features,
    'all_engineered'             : engineered_features,
    'working_original'           : working_features,
    'top_icir_features'          : top_icir,
    'regime_interaction_features': regime_interaction_features,
}
with open('outputs/feature_engineering/feature_sets.pkl', 'wb') as f:
    pickle.dump(feature_sets, f)

with open('outputs/feature_engineering/feature_registry.pkl', 'wb') as f:
    pickle.dump(feature_registry, f)

print("Feature sets saved.")
gc.collect()
```

---

## STEP 15 — Save Engineered Datasets

```python
print("Final shapes before save:")
print(f"  train_eng: {train_eng.shape}")
print(f"  test_eng:  {test_eng.shape}")

# Clean all tree features: no inf/-inf, no NaN
for col in tree_features:
    if col in train_eng.columns:
        train_eng[col] = train_eng[col].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    if col in test_eng.columns:
        test_eng[col]  = test_eng[col].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

train_eng.to_parquet('data/processed/train_engineered.parquet', index=False)
test_eng.to_parquet('data/processed/test_engineered.parquet',   index=False)
print("SAVED: data/processed/train_engineered.parquet")
print("SAVED: data/processed/test_engineered.parquet")
gc.collect()
```

---

## STEP 16 — Feature Importance Quick Validation

Uses GroupKFold on regime labels to prevent same-regime leakage in CV folds.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

print("Quick validation: raw ICIR features vs full engineered set...")
print("Using GroupKFold on regime labels (prevents regime leakage)...")

groups = train_eng['regime'].astype(int).values
X_orig = train_eng[top_icir[:50]].fillna(0).values
X_eng  = train_eng[tree_features[:150]].fillna(0).values
y      = train_eng['TARGET_wins'].values

rf  = RandomForestRegressor(n_estimators=50, max_depth=6,
                             random_state=RANDOM_SEED, n_jobs=-1)
gkf = GroupKFold(n_splits=3)

scores_orig, scores_eng = [], []
for train_idx, val_idx in gkf.split(X_orig, y, groups):
    rf.fit(X_orig[train_idx], y[train_idx])
    scores_orig.append(rf.score(X_orig[val_idx], y[val_idx]))
    rf.fit(X_eng[train_idx], y[train_idx])
    scores_eng.append(rf.score(X_eng[val_idx], y[val_idx]))

import numpy as np
print(f"\nRaw ICIR features (top 50) — GroupKFold R²: "
      f"{np.mean(scores_orig):.4f} ± {np.std(scores_orig):.4f}")
print(f"Full engineered (top 150)  — GroupKFold R²: "
      f"{np.mean(scores_eng):.4f} ± {np.std(scores_eng):.4f}")
print(f"Improvement: {(np.mean(scores_eng) - np.mean(scores_orig))*100:.2f}% absolute R²")
print("\nGroupKFold on regime prevents inflated CV from regime leakage.")
```

---

## Output Files Checklist

After running all cells, verify these files exist:

**Processed data:**
- `data/processed/train_engineered.parquet`
- `data/processed/test_engineered.parquet`

**Feature engineering artifacts:**
- `outputs/feature_engineering/feature_sets.pkl` — tree / MLP / linear / ICIR lists
- `outputs/feature_engineering/feature_registry.pkl` — feature construction log
- `outputs/feature_engineering/imputation_medians.pkl` — for test imputation
- `outputs/feature_engineering/zscore_params.pkl` — mean/std for z-score features
- `outputs/feature_engineering/power_transformers.pkl` — fitted Yeo-Johnson transformers
- `outputs/feature_engineering/regime_model.pkl` — KMeans + scaler for regime assignment
- `outputs/feature_engineering/quantile_transformers.pkl` — fitted quantile transformers (batched)
- `outputs/feature_engineering/target_winsorize_bounds.pkl` — winsorization thresholds

**Removed vs previous version:**
- `outputs/feature_engineering/pca_models.pkl` — PCA step removed, no compressible families

**Next step:** Load `data/processed/train_engineered.parquet` and
`outputs/feature_engineering/feature_sets.pkl` in `notebooks/03_modelling.ipynb`.
