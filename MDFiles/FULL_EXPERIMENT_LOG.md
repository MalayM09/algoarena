# Full Experiment Log — Short-Horizon Return Prediction
## Everything We Tried, What Worked, What Didn't, and Why

*Generated: March 25, 2026*
*Competition: Kaggle Short-Horizon Financial Return Prediction*
*Metric: R² (coefficient of determination)*

---

## Table of Contents

1. [Competition Setup & Dataset Facts](#1-competition-setup--dataset-facts)
2. [Stage 1 — EDA (14 Analytical Layers)](#2-stage-1--eda)
3. [Stage 2 — Feature Engineering Pipeline v1 (The Catastrophic Failure)](#3-stage-2--feature-engineering-pipeline-v1)
4. [Stage 3 — fold_safe_v1 (First Real Positive LB Score)](#4-stage-3--fold_safe_v1)
5. [Stage 4 — Ensemble + Regime Neutralization (Made Things Worse)](#5-stage-4--ensemble--regime-neutralization)
6. [Stage 5 — Target Z-Score + Ridge Ensemble](#6-stage-5--target-z-score--ridge-ensemble)
7. [Stage 6 — LightGBM + MLP Heterogeneous Ensemble](#7-stage-6--lightgbm--mlp-heterogeneous-ensemble)
8. [Complete Submission History & Leaderboard Scores](#8-complete-submission-history--leaderboard-scores)
9. [The Core Problem: OOF→LB Efficiency Collapse](#9-the-core-problem-oofib-efficiency-collapse)
10. [What The Data Has Proven (Non-Negotiable Facts)](#10-what-the-data-has-proven)
11. [Hypotheses For Next Steps](#11-hypotheses-for-next-steps)

---

## 1. Competition Setup & Dataset Facts

**Task:** Predict `TARGET = 100 * (Price[t+H] - Price[t]) / Price[t]`

This is the percentage price return over an unknown forward horizon H. De-identified tabular dataset — no semantic meaning to feature names.

| Property | Value |
|---|---|
| Train rows | 661,574 |
| Test rows | 410,139 |
| Raw features | 445 (112 base + 111 LagT1 + 111 LagT2 + 111 LagT3 + SO3_T) |
| Evaluation metric | R² |
| Rows are shuffled | YES — temporal order NOT preserved |
| Train-test distribution shift | 99.8% of features show significant KS shift |
| Special covariate | SO3_T — continuous regime indicator, 657K+ unique values |

**Critical structural note:** The test set is a strictly future time period. Any pattern that is regime-specific or time-specific in training may be absent or reversed in test.

---

## 2. Stage 1 — EDA (14 Analytical Layers)

**File:** `notebooks/01_eda.ipynb` (run locally)

### Key EDA Findings (with numbers)

| Analysis | Finding | Impact on Modelling |
|---|---|---|
| Target distribution | Normal with fat tails, kurtosis ≈ 3–5 | Use Fair Loss, not MSE |
| KS train-test shift | 444/445 features show significant shift (99.8%) | Cannot trust CV that ignores distribution shift |
| IC/ICIR analysis | 279 features with \|ICIR\| > 0.5 (166 noise features) | Feature selection candidates |
| Always-negative IC | Top LagT1 features have ic_pos_frac = 0.0 | Mean reversion signal, not momentum |
| ICIR top feature | ICIR up to 6.37 for best features | Strong relative signal |
| Regime detection | SO3_T partitions into 2 dominant regimes by IC flip | GroupKFold on SO3_T quintiles |
| 61 sign-flip features | Features flip IC sign across SO3_T regimes | Caused Fold 4 failures |
| 22 lag sign-flip families | LagT1 and LagT3 point opposite directions | LagT1–LagT3 difference feature |
| 5 MI U-shape features | High MI but U-shaped alpha profile | Need abs() transform |
| Feature importance flat | Top 15 features = only 11.7% of LGB gain | Signal is diffuse across ~436 features |
| Decile analysis | Signal almost entirely in d1 and d10 | Extreme indicators could help MLP |
| Pairwise interactions | Zero pairs showed gain > 0.01 | Do NOT engineer pairwise interactions |
| Row-wise stats | Row_mean of LagT1: Spearman = -0.009 vs best individual = -0.032 | 3x weaker — SKIP |
| Acceleration features | LagT1–2×LagT2+LagT3: 10–50x weaker than raw lags | SKIP |
| Regime-relative Z-scores | Zero/negative impact | SKIP |

**EDA outputs saved:**
```
outputs/eda/summaries/ic_icir.csv
outputs/eda/summaries/ic_icir_full.csv
outputs/eda/summaries/mutual_information.csv
outputs/eda/summaries/regime_conditional_correlations.csv
outputs/eda/summaries/target_correlations.csv
outputs/feature_selection/top150_lgb_gain.csv
outputs/feature_selection/top150_spearman.csv
outputs/feature_selection/union_lgb_spearman.csv       (239 features)
outputs/feature_selection/intersection_lgb_spearman.csv (61 features)
```

---

## 3. Stage 2 — Feature Engineering Pipeline v1 (The Catastrophic Failure)

**File:** `notebooks/02_feature_engineering.ipynb` (run locally)
**Script:** `run_feature_engineering.py`

### What Was Built

| Step | Transform | Features Produced |
|---|---|---|
| Step 0 | Drop adversarial-flagged features, r=1.000 duplicates | −66 features |
| Step 1 | Yeo-Johnson power transform | In-place normalization |
| Step 2 | NaN indicator flags | Binary flags |
| Step 3 | Lag difference ratios: LagT1/(abs(T1)+abs(T2)+abs(T3)+ε) | Per-row |
| Step 4 | Sign agreement: sign(T1)×sign(T2)×sign(T3) | Momentum consistency |
| Step 5 | Mean lag signal: mean(T1, T2, T3) | Denoised direction |
| **Step 6** | **Cross-sectional rank transform on 444 features** | **444 rank features** |
| Step 7 | Volatility normalization: LagT/rolling_std | SNR features |
| Step 8 | Target winsorization at ±5σ | Training label only |
| Step 11 | Regime × feature for 61 sign-flip features | 61 interaction features |
| **Step 12** | **Quantile transform (uniform) on 444 features** | **444 quantile features** |

**Total engineered features: 1,638 (tree), 1,214 (MLP)**
**Parquet output: `data/processed/train_engineered.parquet` (661,574 × 1,953 cols)**

### LightGBM on Engineered Features

```
Parameters:
  num_leaves=255, lr=0.03, n_estimators=3000
  feature_fraction=0.6, bagging_fraction=0.8
  min_child_samples=100, reg_alpha=0.1, reg_lambda=1.0

OOF Results:
  Fold 1 (val=Regime 0): R² = +0.0816
  Fold 2 (val=Regime 1): R² = +0.1316
  Fold 3 (val=Regime 2): R² = +0.1319
  OOF R² = +0.1018
```

**LB Score: −0.04044**

**Gap: +0.102 OOF → −0.040 LB = −0.142 gap.**

### Root Cause of Failure

**Cross-sectional leakage** in Steps 6 and 12:
- Rank transforms (Step 6): computed on the full 661K training set before any fold split. The rank of a validation row was computed using training rows → leakage.
- QuantileTransformer (Step 12): fitted on full training set — same problem.
- When applied to test: the test set has a different distribution (99.8% KS shift). Rank values that were predictive in training become meaningless or actively wrong on test.

**The "fingerprint scanner" effect:** With 255 leaves, the tree memorized combinations of leaked rank features that uniquely identified cross-sectional "timestamp twins" in the validation fold. On the test set (future timestamps, no twins), the fingerprints don't match → predictions go in the wrong direction → negative LB.

**Decision:** Abandon all cross-sectional feature engineering. Move to fold-safe pipeline with raw features only.

---

## 4. Stage 3 — fold_safe_v1 (First Real Positive LB Score)

**File:** `src/models/04_fold_safe_cross_sectional.py`
**Machine:** Kaggle (16 GB RAM)

### Architecture

- **Features:** ICIR top 279 (hardcoded from EDA) + raw features
- **CV:** GroupKFold(n_splits=5) on SO3_T quantile buckets
- **Scaling:** StandardScaler(copy=False) fitted ONLY on X_tr per fold → no leakage
- **Target:** Winsorized at train fold p1/p99 per fold
- **Loss:** Fair Loss (c=1.0) — robust to fat financial tails
- **X_test:** Scaled per fold in-place, inverse-transformed after predictions, reset from X_test_static

### Results

| Fold | Val Regime (SO3_T bucket) | R² |
|---|---|---|
| 1 | bucket 4 | +0.000610 |
| 2 | bucket 3 | +0.000522 |
| 3 | bucket 2 | +0.000580 |
| 4 | bucket 0 | **−0.000195** ← Fold 4 failure |
| 5 | bucket 1 | +0.000809 |
| **OOF** | | **+0.000544** |

**LB Score: +0.00005** ← First positive LB score.

**OOF → LB efficiency ratio: 0.00005 / 0.000544 = 9.2%**

### Key Diagnosis

**Fold 4 was the regime-flip fold.** The 61 sign-flip features have IC = +0.05 in Regime A and IC = −0.05 in Regime B. When the model trained on Regimes B,C,D,E and validated on Regime A, it learned the wrong direction for those 61 features → Fold 4 R² = −0.000195.

**Submission CSV stats:**
```
std=0.000624, skew=−0.339, pct_pos=49.1% — near-zero amplitude, near-balanced
```

---

## 5. Stage 4 — Ensemble + Regime Neutralization (Made Things Worse)

**File:** Kaggle notebook (LightGBM + CatBoost + Ranking + Regime Neutralization)
**LB Score: −0.00017**

### What Was Attempted

After fold_safe_v1 scored +0.00005, the next attempt added:
1. CatBoost model alongside LightGBM
2. Rank-based prediction outputs
3. Regime neutralization: subtract the within-regime mean from predictions to remove market-wide bias

### Why It Failed

Regime neutralization introduced **systematic negative bias:**
- Prediction skew: −0.822 (very heavy negative tail)
- % positive predictions: only 31.1% (vs 49.1% for fold_safe_v1)
- The regime mean subtraction made the predictions directionally wrong on the test set

**LB: −0.00017 — the only model that has scored BELOW the regime neutralization failure.**

---

## 6. Stage 5 — Target Z-Score Normalization + Ridge Ensemble

**File:** `src/models/05_target_norm_ridge_ensemble.py`
**Machine:** Local (macOS, 8 GB RAM)

### What Was New

- **Target Z-score normalization inside fold loop:** `y_tr_n = (y_tr − mean_y) / std_y`, same transform applied to y_va. Forces model to learn directional patterns immune to scale differences across regimes.
- **Ridge regression (α=10000)** on top 150 Spearman features, ensembled with LGB via scipy.optimize.

### Results

| Model | OOF R² |
|---|---|
| LGB only | +0.000532 |
| Ridge only | **−0.026814** |
| Ensemble (96% LGB + 4% Ridge) | +0.000579 |

**Fold 4 improvement:** −0.000195 (fold_safe_v1) → −0.000032 (tgt_norm). Target Z-score partially fixed the Fold 4 regime mismatch.

### Key Finding

**Ridge completely failed (OOF = −0.026814).** This proved the signal is non-linear, not diffuse-linear. Ridge learns coefficients that have the right sign globally but wrong sign in specific regimes. This validated the decision to not use any linear model.

**Saved:** `outputs/submissions/lgbm_v1.csv`, `outputs/left_to_submit/tgt_norm_lgb_only_v1.csv`, `outputs/left_to_submit/target_norm_ridge_v1.csv`

---

## 7. Stage 6 — LightGBM + MLP Heterogeneous Ensemble

**File:** `notebooks/kaggle_lgbm_mlp_ensemble.py`
**Machine:** Kaggle

### Architecture Decisions (All EDA-Backed)

| Feature Engineering Decision | Evidence | Applied |
|---|---|---|
| ICIR top 279 features | 166 features have \|ICIR\| < 0.5 = noise | ✓ |
| +22 LagT1−LagT3 difference features | 22 lag families show sign-flip between T1 and T3 | ✓ |
| +5 abs(MI U-shape features) | High MI but U-shaped alpha profile in decile analysis | ✓ |
| Remove regime×SO3_T interaction columns from LGB | LGB finds regime splits natively; ordinal multiplication wrong | ✓ (fixed after user review) |
| Binary extreme indicators for MLP only (top 10) | Decile analysis: signal in d1/d10; trees find thresholds natively | ✓ (fixed after user review) |
| Target Z-score normalization | Fold 4 fix from Stage 5 | ✓ |
| GroupKFold(n=5) on SO3_T quintiles | Regime-safe CV | ✓ |
| SO3_T as native feature for LGB | LGB splits natively on SO3_T > threshold | ✓ |

### Bugs Identified and Fixed During Code Review

| Bug | Impact | Fix |
|---|---|---|
| `train_mlp()` used unscaled `X_test_static` instead of fold-scaled `X_test_work` | Test predictions used raw features while training/val used scaled → silent corruption of all test predictions | Pass `X_test_arr` parameter |
| `del scaler` before section 8.9 tried to use `scaler.inverse_transform()` | NameError crash | Moved `del scaler` to after X_test reset |
| regime×SO3_T bucket multiplication | Multiplying by Z-scored SO3_T is mathematically valid but redundant for trees | Removed interaction columns from LGB |
| Binary indicators added to LGB | Trees find thresholds natively — redundant + memory overhead | Removed from LGB, kept for MLP (top 10) |
| state_dict not saved → test predictions use final overfit epoch weights | OOF uses best checkpoint preds; test uses final epoch → systematic OOF/LB gap | Added state_dict save/restore (in MLP v2 cells) |

### Run Results

| Fold | LGB R² | MLP R² |
|---|---|---|
| 1 | +0.001910 | +0.000377 |
| 2 | +0.001328 | −0.004773 |
| 3 | +0.001123 | +0.000308 |
| 4 | +0.000048 | −0.001751 |
| 5 | +0.000391 | −0.003161 |
| **OOF** | **+0.000659** | **−0.000618** |

**Ensemble OOF:** +0.000830 (scipy: 74% LGB + 26% MLP)

### MLP Failure Analysis

The MLP early-stopped at epoch 24–35 in every fold. Two failure modes:
- **Folds 1 & 3 (positive):** Model found a peak at ~epoch 10–15 then regressed. Test predictions used final overfit epoch weights, not best checkpoint.
- **Folds 2, 4, 5 (diverged):** `val_r2=−0.023` at epoch 25 = active gradient divergence. Root cause: lr=3e-4 too high for near-zero signal.

**Fix implemented (not yet run):** MLP_EPOCHS=500, MLP_PATIENCE=50, MLP_LR=1e-4, MLP_BATCH=2048, state_dict restore.

---

## 8. Complete Submission History & Leaderboard Scores

| # | CSV | OOF R² | LB Score | Notes |
|---|---|---|---|---|
| 1 | `lgbm_baseline_v1.csv` | −0.000496 | **−0.00002** | Raw 445 features, no engineering |
| 2 | `lgbm_v1.csv` (engineered) | +0.1018 | **−0.04044** | Cross-sectional leakage — catastrophic |
| 3 | `clean_lgbm_baseline.csv` | ~0 | ~0 | Clean baseline confirmation |
| 4 | `fold_safe_v1.csv` | +0.000544 | **+0.00005** | First positive LB. Calibration anchor. |
| 5 | `ensemble_rank_neutral_v1.csv` | N/A | **−0.00017** | Regime neutralization bias |
| 6 | `lgb_only_v1.csv` | +0.000659 | **−0.00068** | ← REGRESSION despite OOF improvement |
| 7 | `lgbm_mlp_ensemble_v1.csv` | +0.000830 | **−0.00155** | ← SEVERE REGRESSION despite best OOF |

**Not yet submitted:** `tgt_norm_lgb_only_v1.csv`, `target_norm_ridge_v1.csv`

---

## 9. The Core Problem: OOF→LB Efficiency Collapse

This is the most important section.

### The Numbers

| Submission | OOF R² | LB R² | Efficiency |
|---|---|---|---|
| fold_safe_v1 | +0.000544 | +0.00005 | **+9.2%** |
| lgb_only_v1 | +0.000659 | −0.00068 | **−103%** |
| lgbm_mlp_ensemble_v1 | +0.000830 | −0.00155 | **−187%** |

The OOF score went UP (+21%) while the LB score went DOWN (from +0.00005 to −0.00068). The ensemble made it even worse (OOF +0.000830 → LB −0.00155).

**This is not a small gap — it is a sign reversal.** The model is not just failing to generalize; it is actively learning the wrong direction on the test set.

### What Changed Between fold_safe_v1 and lgb_only_v1

fold_safe_v1 used all 445 features with minimal processing.

lgb_only_v1 added:
1. **ICIR top-279 feature selection** — selected from full training set EDA
2. **22 LagT1−LagT3 difference features** — engineered from sign-flip analysis on full training set
3. **5 abs(MI U-shape) features** — derived from full training set decile analysis
4. **Target Z-score normalization** — fold-safe (no leakage)
5. **Upgraded LGB hyperparameters** (num_leaves 63→127, min_child_samples 250→100)

### Root Cause Hypothesis

The OOF improvement was driven by items 1–3, all of which are **feature selection/engineering based on global training set statistics.** This is a subtle but real form of data leakage:

- Out of 445 features, the ICIR analysis selected the 279 that happened to correlate most consistently with the target across the training set.
- Some of those 279 have real, generalizable signal.
- Some have spurious correlation with the training distribution that reverses in the test set.
- By selecting and emphasizing them, we amplified both real signal AND regime-specific noise.

The LagT1−LagT3 difference features are particularly suspect. If LagT1 and LagT3 flip sign in the training set, that means this specific regime combination appeared in training. The test set may have a completely different regime structure where this flip doesn't exist — and now we've given the model a feature that has the wrong sign.

### Evidence for Regime Structural Break

```
fold_safe_v1:      std=0.000624  →  LB=+0.00005  (conservative, near-zero predictions)
lgb_only_v1:       std=0.001046  →  LB=−0.00068  (more confident, actively wrong)
lgbm_mlp_ensemble: std=0.001395  →  LB=−0.00155  (most confident, most wrong)
```

The pattern is clear: the more confident (higher std) the predictions, the worse the LB. The test set appears to be in a regime where the training signal direction is reversed or absent. **Higher confidence = more wrong.**

### The Deeper Issue

fold_safe_v1 (with LGB best_iter = 83 in the new run) is essentially predicting the mean plus tiny perturbations. The LB score of +0.00005 is barely above zero. All the prediction amplitude was noise.

The OOF signal we measured (+0.000544) was regime-specific — it was real within the training distribution but did not transfer to the test regime. When we made the model more confident (ICIR selection, lag diffs), we amplified regime-specific patterns that are inverted on the test set.

---

## 10. What The Data Has Proven (Non-Negotiable Facts)

1. **The test set is in a different regime from training.** Evidence: any prediction with high amplitude scores worse than near-zero predictions. The signal direction that is valid in training is partially reversed in test.

2. **99.8% train-test distribution shift is real and devastating.** The KS test found it in EDA. The LB scores confirm it.

3. **Cross-sectional feature engineering causes catastrophic leakage** (Stage 2 failure, LB = −0.04).

4. **Global EDA-derived feature selection also hurts** (Stage 6: ICIR top-279 selection drove LB from +0.00005 to −0.00068).

5. **Ridge regression has no signal** (OOF = −0.026814 with 150 Spearman features). Signal is non-linear.

6. **LGB with minimal processing is the only model that has ever scored positive** (fold_safe_v1, LB = +0.00005).

7. **MLP in current form does not generalize.** Even with positive OOF in 2/5 folds, the ensemble made LB worse.

8. **OOF R² improvements ≠ LB improvements** in this competition. The train-test regime break means OOF can go up while LB goes down.

9. **Prediction amplitude (std) is inversely correlated with LB score.** Lower amplitude = closer to zero = safer.

10. **The LGB early-stopping at iteration 21–83** (in the latest run) means the model is stopping very early. This is appropriate given the near-zero signal. Allowing more iterations makes overfitting worse.

---

## 11. Hypotheses For Next Steps

### What We Know Works

- fold_safe_v1 architecture (raw 445 features, GroupKFold on SO3_T, Fair Loss, StandardScaler inside fold)
- Conservative predictions (low amplitude, near-balanced pos/neg)
- LB = +0.00005 is the ONLY positive score from 7 submissions

### What Needs Fundamental Rethinking

The central question is: **why is the test set in a different regime, and how do we predict in that regime?**

#### Hypothesis A: Submission Format / Target Misunderstanding

- The sample_submission.csv maps test rows to IDs. Are we computing the correct target?
- `TARGET = 100 * (Price[t+H] - Price[t]) / Price[t]` — is this the correct formula?
- Are the test IDs in the same order as the test file? (Alignment bug could cause perfectly good predictions to be assigned to the wrong rows)

#### Hypothesis B: The Test Set Has a Regime We've Never Seen

- The training set has 3 regimes (SO3_T low/mid/high). The test set may be entirely in one extreme.
- The signal direction may be completely inverted in that regime.
- A model that predicts near zero (fold_safe_v1) is safer because it doesn't commit to a wrong direction.

#### Hypothesis C: We Are Using Too Many Features (Including Noise Features with Wrong Direction)

- fold_safe_v1 used 445 features with LGB stopping at low iterations → mostly predicts near zero
- lgb_only_v1 used 307 features that were EDA-selected for high signal → but signal direction in test is wrong
- Counterintuitive fix: use FEWER features, not more. Force the model to near-zero predictions.

#### Hypothesis D: Time-Based Target Horizon Mismatch

- The TARGET is `H`-forward return where `H` is unknown
- If the test set uses a different horizon H than training, all predictions are wrong
- Cannot fix this directly — but it would explain systematic sign reversal

#### Hypothesis E: The Signal is in SO3_T Alone

- SO3_T is the only feature without any train-test distribution shift (it's the regime indicator)
- A model that predicts TARGET from SO3_T alone might generalize better than any complex model
- Worth testing: Ridge(α=1) with only SO3_T as feature

### Priority Actions

1. **Validate submission alignment** — confirm IDs in submission match test set correctly
2. **Test pure SO3_T prediction** — if SO3_T alone scores positive, regime-conditional models are the path
3. **Try extreme regularization** — LGB with num_leaves=7, min_child_samples=5000, reg_lambda=100 on raw 445 features, expecting near-zero predictions
4. **Look at other competitors' public scores** — understand where +0.00005 ranks and what the top scores are
5. **Consider rank-based target** — predict rank(TARGET) instead of raw return; reduces sensitivity to regime scale differences

---

## Appendix: File Registry

### Source Code
```
src/models/04_fold_safe_cross_sectional.py     — fold_safe_v1 pipeline
src/models/05_target_norm_ridge_ensemble.py    — target norm + Ridge
notebooks/kaggle_lgbm_mlp_ensemble.py          — LGB + MLP heterogeneous ensemble
```

### Submitted CSVs
```
outputs/submissions/lgbm_baseline_v1.csv       — LB: −0.00002
outputs/submissions/lgbm_v1.csv                — LB: −0.04044 (leakage)
outputs/submissions/clean_lgbm_baseline.csv    — LB: ~0
outputs/submissions/fold_safe_v1.csv           — LB: +0.00005 ← best ever
outputs/submissions/ensemble_rank_neutral_v1.csv — LB: −0.00017
outputs/submissions/lgbm_mlp_ensemble_v1.csv   — LB: −0.00155
```

### Not Yet Submitted
```
outputs/left_to_submit/lgb_only_v1.csv         — LB: −0.00068 (submitted Mar 25)
outputs/left_to_submit/tgt_norm_lgb_only_v1.csv
outputs/left_to_submit/target_norm_ridge_v1.csv
```

### EDA Summaries
```
outputs/eda/summaries/ic_icir_full.csv
outputs/eda/summaries/regime_conditional_correlations.csv
outputs/feature_selection/top150_lgb_gain.csv
outputs/feature_selection/top150_spearman.csv
outputs/feature_selection/union_lgb_spearman.csv    (239 features)
outputs/feature_selection/intersection_lgb_spearman.csv (61 features)
```
