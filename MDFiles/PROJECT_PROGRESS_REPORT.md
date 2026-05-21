# Project Progress Report — Short-Horizon Return Prediction
## Kaggle Regression Competition

---

## Table of Contents

1. [Competition Overview](#1-competition-overview)
2. [Dataset Facts](#2-dataset-facts)
3. [Stage 1 — Exploratory Data Analysis (EDA)](#3-stage-1--exploratory-data-analysis-eda)
4. [Stage 2 — Feature Engineering](#4-stage-2--feature-engineering)
5. [Stage 3 — Modelling](#5-stage-3--modelling)
6. [Submission Results & Leaderboard Scores](#6-submission-results--leaderboard-scores)
7. [Root Cause Analysis — Why the Engineered Model Failed](#7-root-cause-analysis--why-the-engineered-model-failed)
8. [Current State & Next Steps](#8-current-state--next-steps)

---

## 1. Competition Overview

**Task:** Predict short-horizon price returns from a de-identified tabular dataset.

The target variable is:
```
TARGET = 100 * (Price[t+H] - Price[t]) / Price[t]
```
This is the percentage price return over a forward horizon `H`, where `H` is unknown. The competition is evaluated on **R² score** (coefficient of determination). A score of 0.0 means the model predicts no better than always guessing the mean. A negative R² means the model actively makes predictions in the wrong direction — worse than doing nothing.

The dataset is de-identified: feature names like `S01_F01_U01` are opaque codes with no semantic meaning. This is typical of quantitative finance competitions where the data provider deliberately obscures the underlying instruments and signals.

---

## 2. Dataset Facts

| Property | Value |
|---|---|
| Train rows | 661,574 |
| Test rows | 410,139 |
| Raw features | 445 |
| Base features | 112 |
| LagT1 features | 111 |
| LagT2 features | 111 |
| LagT3 features | 111 |
| Special covariate | SO3_T (continuous, no lag versions) |
| Target column | TARGET (regression, percentage return) |
| Evaluation metric | R² |

**Critical structural facts:**
- Rows are **shuffled** — temporal order is NOT preserved in the released files. You cannot sort by ID to recover time order.
- Lag features are pre-computed: `feat_LagT1 = feat[t] - feat[t-T1]`, and similarly for T2, T3. The exact lag horizons T1, T2, T3 are unknown, but T3 > T2 > T1.
- `SO3_T` is a standalone continuous covariate that serves as a regime indicator. It has 657,000+ unique values (essentially continuous). The KMeans clustering analysis identified 3 distinct regimes in the data.
- The test set covers a different time period than training. This means any signal that is regime-specific may not generalize.

---

## 3. Stage 1 — Exploratory Data Analysis (EDA)

**Location:** `notebooks/01_eda.ipynb`

The EDA was conducted across 14 analytical layers, each producing decisions that fed into the feature engineering pipeline.

### 3.1 Target Analysis

The TARGET distribution is approximately normal with slight fat tails (excess kurtosis ≈ 3–5), consistent with financial return distributions. Key statistics:
- The distribution is close to zero-mean (mean ≈ 0.0002)
- Heavy tails: extreme values go well beyond ±3σ
- Winsorization at ±5σ was applied during training to clip these outliers, producing `TARGET_wins`

**Winsorization bounds:** approximately [-0.1227, +0.1230]

### 3.2 Feature Distribution Analysis

- **99.8% of the 445 features** showed significant train-test distribution shift under the Kolmogorov-Smirnov test. This means the test set distribution is meaningfully different from training for almost every feature — a major challenge for generalization.
- Features show varying degrees of skewness and kurtosis. A subset required Yeo-Johnson power transforms to normalize their distributions.
- Several features had NaN values where the NaN itself was predictive of the target (i.e., the presence of a missing value correlates with the outcome). Indicator flags were created for these.

### 3.3 Regime Detection

KMeans clustering on `SO3_T` identified **3 distinct market regimes**:

| Regime | Row Count | Share |
|---|---|---|
| Regime 0 | 450,372 | 68.1% |
| Regime 1 | 116,560 | 17.6% |
| Regime 2 | 94,642 | 14.3% |

Regime 0 dominates the training set (68% of rows). The regime label was used as the grouping variable for cross-validation to prevent regime-leakage.

### 3.4 Signal Strength Analysis

- Spearman correlation was computed between each feature and TARGET, with Benjamini-Hochberg multiple-testing correction applied. A subset of features survived as statistically significant at the 5% FDR level.
- **Information Coefficient (IC)** and **IC Information Ratio (ICIR)** were computed per feature. ICIR measures the consistency of a feature's predictive power across different subsets of the data. 223 features had |ICIR| > 0.5 and were flagged as top-quality signals.
- **61 features** showed sign-flip behaviour across regimes — i.e., a feature that positively correlates with TARGET in one regime negatively correlates in another. These required regime interaction terms.

### 3.5 Autocorrelation & Lag Analysis

Maximum autocorrelation across all lag features was 0.011. This confirmed that smoothed lag signals (averaging T1/T2/T3) would degrade performance rather than help. Smoothing was removed from the feature engineering plan.

### 3.6 Adversarial Validation

An adversarial validation model was trained to distinguish train rows from test rows. Features where the adversarial model achieved very high accuracy were identified as "distribution-shift features" — the model could tell them apart too easily. These features were flagged for removal since a model trained on them would learn patterns that don't transfer to the test set.

### 3.7 Pairwise Interaction Screening

All 780 pairwise feature interactions (products of feature pairs) were screened for incremental R² gain. Zero pairs showed gain > 0.01. Pairwise interactions were abandoned entirely.

### 3.8 EDA Outputs

```
outputs/eda/summaries/eda_decisions.pkl   — all decisions in a single dict
outputs/eda/summaries/adversarial_drop_list.pkl
outputs/eda/plots/                        — all visualisation plots
```

The `eda_decisions.pkl` file contained:
- `drop` — features to remove (adversarial validation + duplicate pairs with r=1.000)
- `needs_transform` — features requiring Yeo-Johnson normalization
- `top_signals` — BH-corrected Spearman significant features
- `top_icir_features` — 223 features with |ICIR| > 0.5
- `high_ks_shift` — 444 features with high train-test distribution shift
- `needs_regime_interaction` — 61 sign-flip features requiring regime × feature products
- `sign_flip_lags` — 22 base features with T1 vs T3 sign flip
- `best_lag_map` — dict mapping base_feature → best lag horizon

---

## 4. Stage 2 — Feature Engineering

**Location:** `notebooks/02_feature_engineering.ipynb`

The feature engineering pipeline transformed the raw 445 features into a richer set of 1638+ features based on the EDA findings. All transformers were fitted on training data only and applied to both train and test.

### 4.1 Steps Executed

| Step | Description | Features Produced |
|---|---|---|
| Step 0 | Drop adversarial validation flagged features, remove r=1.000 duplicate pairs | −66 features |
| Step 1 | Yeo-Johnson power transform on non-normal features | In-place normalization |
| Step 2 | NaN indicator flags for features where NaN is predictive | +N binary flags |
| Step 3 | Lag difference ratios — `LagT1 / (abs(LagT1) + abs(LagT2) + abs(LagT3) + ε)` | Per-row, no cross-row dependency |
| Step 4 | Sign agreement signal — `sign(LagT1) * sign(LagT2) * sign(LagT3)` | Captures momentum consistency |
| Step 5 | Mean lag signal — `mean(LagT1, LagT2, LagT3)` | Denoised momentum direction |
| Step 6 | Cross-sectional rank transforms on all 444 high-KS-drift features | 444 rank features |
| Step 7 | Volatility normalization — `LagT / rolling_std(base_feature)` | Signal-to-noise ratio features |
| Step 8 | Target winsorization — clip TARGET at ±5σ → `TARGET_wins` | Training label only |
| Step 9A | SO3_T composite features | Low priority candidates |
| Step 11 | Regime × feature interactions for 61 sign-flip features — `feature * regime_r` | 61 regime-conditioned signals |
| Step 12 | Quantile transform (uniform) on all 444 high-KS-shift features | 444 additional features |
| Step 14 | Feature set assembly and registration | `feature_sets.pkl` |

### 4.2 Final Feature Sets

```
outputs/feature_engineering/feature_sets.pkl
```

| Feature Set | Count | Usage |
|---|---|---|
| `tree_features` | 1,638 | LightGBM, XGBoost, CatBoost |
| `mlp_features` | 1,214 | Neural network (MLP) |
| `linear_features` | 6 | Ridge regression (low VIF) |
| `all_engineered` | 1,569 | Full engineered set |
| `working_original` | 379 | Cleaned original features |
| `top_icir_features` | 223 | Features with |ICIR| > 0.5 |
| `regime_interaction_features` | 78 | Regime × feature products |

### 4.3 Engineered Parquet Outputs

```
data/processed/train_engineered.parquet   — 661,574 rows × 1,953 columns
data/processed/test_engineered.parquet    — 410,139 rows × 1,950 columns
```

The extra columns (1,953 vs 447 raw) include all engineered features, `TARGET_wins`, and the `regime` label column.

---

## 5. Stage 3 — Modelling

**Location:** `run_modelling.py` (v1), `run_modelling_v2.py` (v2 — XGBoost/CatBoost/Ridge only)

### 5.1 Cross-Validation Strategy

**Method:** 3-fold GroupKFold on `regime` label (0, 1, 2)

This is the strictest possible leakage prevention strategy for this dataset — each fold holds out exactly one market regime entirely. The model never sees the held-out regime during training.

| Fold | Training Regimes | Validation Regime | Train Rows | Val Rows |
|---|---|---|---|---|
| Fold 1 | Regime 1 + 2 | Regime 0 | 211,202 | 450,372 |
| Fold 2 | Regime 0 + 2 | Regime 1 | 545,014 | 116,560 |
| Fold 3 | Regime 0 + 1 | Regime 2 | 566,932 | 94,642 |

Note: Fold 1 has an inverted train/val size ratio because Regime 0 contains 68% of all rows. The model trains on 31% of the data and validates on 68% in that fold.

### 5.2 Loss Function — Fair Loss

Instead of standard MSE, a **Fair loss** (also called Cauchy loss or pseudo-Huber) was used for all tree models:

```
L(r) = c² * (|r|/c - log(1 + |r|/c))
grad = r / (1 + |r|/c)
hess = c² / (c + |r|)²
```

With `c = 1.0`. Fair loss is robust to outliers in financial return distributions — it down-weights large residuals rather than squaring them (like MSE does). This prevents the model from chasing extreme target values.

### 5.3 Model 1 — LightGBM (Completed ✓)

**Trained locally. All 3 folds completed.**

Parameters:
```python
num_leaves        = 255
learning_rate     = 0.03
n_estimators      = 3000
feature_fraction  = 0.6
bagging_fraction  = 0.8
bagging_freq      = 1
min_child_samples = 100
reg_alpha         = 0.1
reg_lambda        = 1.0
```

No early stopping was triggered — the model kept improving through all 3000 iterations in every fold. This means the learning rate of 0.03 was still finding gradients at iteration 3000.

**Fold-by-fold OOF results:**

| Fold | Val Regime | Best Iter | Fold R² |
|---|---|---|---|
| 1 | Regime 0 | 3000 | 0.081603 |
| 2 | Regime 1 | 3000 | 0.131562 |
| 3 | Regime 2 | 3000 | 0.131938 |
| **OOF (all)** | — | — | **0.101804** |

Fold 1 is weaker (R²=0.082) because the model trains on only 211k rows (regimes 1+2) and predicts on 450k rows (regime 0) — the largest, most heterogeneous regime.

**Saved outputs:**
```
outputs/oof_predictions/lgbm_oof.npy         — 661,574 OOF predictions
outputs/oof_predictions/lgbm_test_preds.npy  — 410,139 test predictions (avg of 3 folds)
models/checkpoints/lgbm_fold1.txt            — 83 MB model file
models/checkpoints/lgbm_fold2.txt            — 84 MB model file
models/checkpoints/lgbm_fold3.txt            — 84 MB model file
```

### 5.4 Model 2 — XGBoost (Incomplete — OOM killed)

**Training was not completed due to memory exhaustion on the local machine (8 GB RAM).**

The XGBoost training pipeline required:
- `X_train_tree`: 661,574 rows × 1,638 features × 4 bytes = **~4.3 GB**
- `X_test_tree`: 410,139 rows × 1,638 features × 4 bytes = **~2.7 GB**
- XGBoost DMatrix internal copies added another **~3–4 GB**

Total memory requirement exceeded 10 GB, forcing the OS to heavily compress and swap memory pages. The process ran at only ~20% CPU (rest spent swapping) for 14 hours without completing fold 2. It was killed (exit code 137 = OOM).

Only `models/checkpoints/xgb_fold1.json` was saved. No usable OOF or test predictions from XGBoost.

**Decision: move XGBoost, CatBoost, and MLP training to Kaggle** (16–30 GB RAM, GPU available).

### 5.5 Model 3 — CatBoost (Not Started)

Not started locally due to memory constraints. Planned for Kaggle.

### 5.6 Model 4 — Ridge (Not Started)

Not started. Would have been fast (6 linear features only). Can be run locally once XGBoost/CatBoost are complete on Kaggle.

### 5.7 Baseline LightGBM — Raw Features Only (Completed ✓)

To diagnose the leaderboard performance gap, a clean baseline model was trained using only the **original 445 raw features** with no engineering. This completed in under 2 minutes locally.

Parameters (slightly relaxed for speed):
```python
num_leaves        = 127
learning_rate     = 0.05
n_estimators      = 2000
feature_fraction  = 0.6
bagging_fraction  = 0.8
min_child_samples = 100
```

GroupKFold groups were created by binning the continuous `SO3_T` column into 3 quantile buckets using `pd.qcut`.

**Fold-by-fold OOF results:**

| Fold | Fold R² |
|---|---|
| 1 | ≈ 0.000 |
| 2 | ≈ 0.000 |
| 3 | ≈ 0.000 |
| **OOF (all)** | **-0.000496** |

The baseline OOF R² is essentially zero — the model is predicting near the mean for every row. Prediction standard deviation was only 0.000433 (vs 0.005337 for the engineered model).

**Saved outputs:**
```
outputs/oof_predictions/lgbm_baseline_oof.npy
outputs/oof_predictions/lgbm_baseline_test_preds.npy
models/checkpoints/lgbm_baseline_fold1.txt
models/checkpoints/lgbm_baseline_fold2.txt
models/checkpoints/lgbm_baseline_fold3.txt
outputs/submissions/lgbm_baseline_v1.csv
```

---

## 6. Submission Results & Leaderboard Scores

### 6.1 Submission Files

| File | Model | Description |
|---|---|---|
| `outputs/submissions/lgbm_v1.csv` | Engineered LightGBM | 1638 features, winsorized target |
| `outputs/submissions/lgbm_baseline_v1.csv` | Baseline LightGBM | 445 raw features, raw target |

All submissions are in the required format: `ID, TARGET` with 410,139 rows aligned to `sample_submission.csv`.

### 6.2 Leaderboard Results

| Model | OOF R² (local CV) | Kaggle Leaderboard R² | Gap |
|---|---|---|---|
| Engineered LightGBM (1638 features) | +0.1018 | **-0.04044** | -0.142 |
| Baseline LightGBM (445 raw features) | -0.0005 | **-0.00002** | -0.0005 |

### 6.3 Prediction Distribution

| Metric | Engineered LightGBM | Baseline LightGBM |
|---|---|---|
| Mean | 0.000146 | 0.000118 |
| Std | 0.005337 | 0.000433 |
| Min | -0.052421 | -0.003803 |
| Max | 0.060389 | 0.003592 |

The engineered model makes confident (high variance) predictions. The baseline predicts almost everything near zero (low variance). Both score near zero on the leaderboard, but the engineered model is actively worse.

---

## 7. Root Cause Analysis — Why the Engineered Model Failed

This section explains the -0.14 gap between OOF R² (+0.10) and leaderboard R² (-0.04) in detail.

### 7.1 The Core Problem: Cross-Sectional Feature Leakage

The feature engineering pipeline computed cross-sectional features on the **entire training set** before any cross-validation split. The most affected feature types were:

- **Rank transforms (Step 6):** `rank(feature)` was computed across all 661,574 training rows. This means the rank of any single row's feature value was computed relative to ALL other rows, including the rows that would later appear in the validation fold.
- **Quantile transforms (Step 12):** The QuantileTransformer was fitted on the full training set, encoding full-training-set distributional statistics into each transformed value.
- **Volatility normalization (Step 7):** Rolling standard deviations computed on the full (shuffled) dataset — these are meaningless noise when row order is random, but the model could still exploit spurious correlations in training.

**How this inflates OOF R²:**

```
Wrong pipeline:
  Full train set → compute rank(feature) → split into folds → train/eval

  During CV evaluation:
    Val fold rows' rank values were computed using training fold rows.
    The model learns: "row X has rank 0.73 globally → it tends to have high TARGET"
    This works IN the CV because the ranks are consistent across folds.

Correct pipeline:
  Split into folds → for each fold: fit rank on train fold → transform both folds

  During CV evaluation:
    Val fold rows' rank values computed without training fold rows.
    No information leakage. Harder to achieve high CV R².
```

On the actual test set, the rank features are computed using train-fitted quantile boundaries — not relative to the test set rows. The model tries to apply the pattern it learned ("global rank = 0.73 means high TARGET") but the ranks now encode different information. The predictions become systematically wrong.

### 7.2 Why the Leaderboard Score is Negative

A negative R² (-0.04) means the model's predictions are *further* from the true values than simply predicting the mean would be. This happens when:

1. The model learned spurious patterns (leakage-induced correlations) that have the **wrong sign** on the test set.
2. The regime-conditioned features (regime × feature products) encode regime 0/1/2 specific patterns that don't apply to the test set's regime distribution.

### 7.3 Why the Baseline Scores Near Zero

The baseline model (raw 445 features, no engineering) scores approximately zero on the leaderboard. This tells us:

- **There IS genuine signal in the raw features** — the model is not predicting nonsense. It's just predicting the mean, which is the safe fallback when no strong regime-invariant signal is found.
- The 3-fold GroupKFold on SO3_T quantiles is a very strict CV: training on 2 quantile buckets and generalizing to the 3rd is a genuine distribution shift. The raw features don't have strong enough regime-invariant predictive power to overcome this.
- The engineered model should have done AT LEAST as well as the baseline. It didn't — confirming the engineering introduced harmful leakage.

### 7.4 Summary of Mistakes

| Mistake | Impact | Fix |
|---|---|---|
| Cross-sectional rank features computed on full train set before CV | Inflated OOF R² by ~0.10. Caused -0.04 LB score. | Recompute ranks within each CV fold |
| Quantile transforms fitted on full train set | Same leakage mechanism as ranks | Fit QuantileTransformer on train fold only in each CV iteration |
| Rolling stats on shuffled rows | Noisy pseudo-features that overfit training | Either remove or compute with temporal awareness |
| 1638 features on 8 GB RAM machine | OOM, couldn't complete XGBoost/CatBoost | Use Kaggle for training (16–30 GB RAM) |
| OOF R² validated as ground truth | LB revealed +0.10 CV ≠ real signal | Always sanity-check with a leakage-free baseline first |

---

## 8. Current State & Next Steps

### 8.1 What Is Saved and Working

| Asset | Status | Path |
|---|---|---|
| Raw train/test parquet | ✓ | `data/raw/train-001.parquet`, `data/raw/test.parquet` |
| Engineered train/test parquet | ✓ (but with leakage) | `data/processed/train_engineered.parquet` |
| EDA decisions | ✓ | `outputs/eda/summaries/eda_decisions.pkl` |
| LightGBM engineered OOF + test preds | ✓ (OOF inflated by leakage) | `outputs/oof_predictions/lgbm_oof.npy` |
| LightGBM baseline OOF + test preds | ✓ (clean) | `outputs/oof_predictions/lgbm_baseline_oof.npy` |
| Submission — engineered LightGBM | ✓ submitted | `outputs/submissions/lgbm_v1.csv` |
| Submission — baseline LightGBM | ✓ submitted | `outputs/submissions/lgbm_baseline_v1.csv` |

### 8.2 Plan Going Forward

The work is being moved to **Kaggle** (more RAM, GPU available) for the heavy training. The local machine handles data prep and submission post-processing.

**Priority 1 — Fix the leakage and get a positive LB score:**
- Rebuild feature engineering to be fold-safe: all cross-sectional transforms (ranks, quantile normalizations) must be fitted inside each CV fold, not on the full dataset
- Start with a smaller, cleaner feature set (~200–300 fold-safe features) rather than 1638
- Any feature that requires seeing multiple rows simultaneously must be carefully evaluated for leakage

**Priority 2 — Expand the model zoo on Kaggle:**
- XGBoost (Fair loss, neg_r2 early stopping, 3-fold GroupKFold)
- CatBoost (RMSE loss, R2 eval metric, 3-fold GroupKFold)
- MLP / ResNet-style neural network on normalized features

**Priority 3 — Hyperparameter tuning (notebook 04):**
- Optuna search once clean models have a positive baseline LB score
- Tune: `num_leaves`, `min_child_samples`, `feature_fraction`, `subsample`, `reg_alpha`, `reg_lambda`

**Priority 4 — Ensemble (notebook 05):**
- Stacking / weighted average blend of OOF predictions
- Optimize blend weights using Nelder-Mead or scipy.optimize on OOF arrays

### 8.3 Leaderboard Target

| Milestone | Target LB R² |
|---|---|
| Baseline (current) | -0.00002 |
| Clean engineered features (fold-safe) | > 0.01 |
| Tuned single model | > 0.03 |
| Full ensemble | > 0.05 |

A leaderboard R² of 0.05+ in this type of financial return prediction competition is considered strong. The signal-to-noise ratio in financial data is inherently low.

---

*Report generated: March 24, 2026*
*Working directory: `/Users/malaymishra/Desktop/quant_ml_project`*
