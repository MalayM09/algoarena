# Project README — Short-Horizon Return Prediction
## For Claude Code Agent

---

## What This Project Is

Kaggle regression competition. Predict `TARGET = 100 * (Price[t+H] - Price[t]) / Price[t]`,
the H-window-forward percentage return of `Price`, from a de-identified 445-feature dataset.
Evaluation metric: **R² score**. Submissions via reproducible Kaggle notebook.

---

## Directory Structure

```
quant_ml_project/
│
├── README.md                          ← you are here
├── EDA_INSTRUCTIONS.md                ← full EDA spec for Claude Code
├── FEATURE_ENGINEERING_INSTRUCTIONS.md ← full feature engineering spec
│
├── data/
│   ├── raw/
│   │   ├── train.parquet              ← 661,574 rows × 447 cols (features + ID + TARGET)
│   │   ├── test.parquet               ← 410,139 rows × 446 cols (features + ID)
│   │   └── sample_submission.csv      ← format: ID, TARGET
│   └── processed/
│       ├── train_engineered.parquet   ← output of 02_feature_engineering.ipynb
│       └── test_engineered.parquet    ← output of 02_feature_engineering.ipynb
│
├── notebooks/
│   ├── 01_eda.ipynb                   ← ALL EDA (see EDA_INSTRUCTIONS.md)
│   ├── 02_feature_engineering.ipynb   ← ALL feature work (see FEATURE_ENGINEERING_INSTRUCTIONS.md)
│   ├── 03_modelling.ipynb             ← LightGBM / XGBoost / CatBoost / MLP
│   ├── 04_hyperparameter_tuning.ipynb ← Optuna search
│   ├── 05_ensemble.ipynb              ← OOF blending and ensemble weights
│   └── submission/
│       └── final_submission.ipynb     ← clean reproducible notebook for Kaggle upload
│
├── src/
│   ├── __init__.py
│   ├── features.py                    ← all feature engineering functions (refactored from 02)
│   ├── models.py                      ← LightGBM / XGBoost / CatBoost / MLP wrappers
│   ├── validation.py                  ← KFold CV, OOF prediction, R² tracking
│   └── utils.py                       ← memory reduction, logging, reproducibility
│
├── models/
│   └── checkpoints/                   ← saved .pkl / .pt / .txt model files
│
├── outputs/
│   ├── eda/
│   │   ├── plots/                     ← all EDA plots (.png)
│   │   └── summaries/                 ← all EDA summary files (.csv / .pkl / .parquet)
│   ├── feature_engineering/           ← fitted transformers, feature lists, registry
│   ├── oof_predictions/               ← out-of-fold prediction arrays per model
│   └── submissions/                   ← versioned submission CSVs
│
└── configs/
    └── lgbm_params.json               ← best hyperparameter configs after tuning
```

---

## Execution Order

**Always run notebooks in this order. Each notebook depends on outputs from the previous.**

```
01_eda.ipynb
    ↓ outputs/eda/summaries/eda_decisions.pkl
02_feature_engineering.ipynb
    ↓ data/processed/train_engineered.parquet
    ↓ outputs/feature_engineering/feature_sets.pkl
03_modelling.ipynb
    ↓ outputs/oof_predictions/lgbm_oof.npy
    ↓ outputs/oof_predictions/xgb_oof.npy
    ↓ outputs/oof_predictions/mlp_oof.npy
04_hyperparameter_tuning.ipynb
    ↓ configs/lgbm_params.json
05_ensemble.ipynb
    ↓ outputs/submissions/ensemble_v1.csv
final_submission.ipynb
    ↓ submission.csv  (upload to Kaggle)
```

---

## Dataset Facts

| Property | Value |
|---|---|
| Train rows | 661,574 |
| Test rows | 410,139 |
| Total features | 445 |
| Base features | 112 |
| LagT1 features | 111 |
| LagT2 features | 111 |
| LagT3 features | 111 |
| Special covariate | SO3_T (no lag versions) |
| Target feature | Price (also used to define TARGET) |

**Feature naming:**
- `S01_F01_U01` style — de-identified, treat as opaque codes
- `feat_LagT1 = feat[t] - feat[t-T1]`
- `feat_LagT2 = feat[t] - feat[t-T2]`
- `feat_LagT3 = feat[t] - feat[t-T3]`
- `T3 > T2 > T1` (exact values unknown, infer from lag variance)
- Rows are **shuffled** — temporal order NOT preserved

**Target:**
```
TARGET = 100 * (Price[t+H] - Price[t]) / Price[t]
```

**Evaluation:**
```
R² = 1 - Σ(y_true - y_pred)² / Σ(y_true - mean(y_true))²
```

---

## Key Domain Knowledge (from WorldQuant Alpha Research)

The following signal types have been empirically validated on similar financial data
and should be prioritised in both feature engineering and model interpretation:

1. **Mean reversion signals** — lag features capture deviation from historical mean.
   Stocks/assets that have moved far from their recent average tend to revert.
   Signal: `LagT / volatility_norm`

2. **Volatility normalisation** — dividing lag signals by recent volatility
   improves Sharpe ratio. Raw price moves mean more in low-vol regimes.
   Signal: `LagT / rolling_std(base_feature, N)`

3. **Cross-sectional ranking** — applying rank() to signals reduces outlier
   sensitivity and converts absolute values to relative standing across the universe.
   Signal: `rank(feature) / N`

4. **Smoothed lag signals** — averaging lag signals over multiple horizons
   reduces noise while preserving direction.
   Signal: `mean(LagT1, LagT2, LagT3)`

5. **Sign agreement across lags** — when LagT1, LagT2, LagT3 all point the same
   direction, the signal is more persistent (momentum). Mixed signs = noise.
   Signal: `sign(LagT1) * sign(LagT2) * sign(LagT3)`

6. **Regime conditioning** — signal strength varies by market regime.
   SO3_T is the primary regime indicator in this dataset.

---

## Modelling Strategy Summary

| Model | Feature Set | Notes |
|---|---|---|
| Ridge (baseline) | linear_features (low VIF) | Floor R² benchmark |
| LightGBM | tree_features (all) | Primary workhorse |
| XGBoost | tree_features (all) | Diversity for ensemble |
| CatBoost | tree_features (all) | Best on categorical regime features |
| MLP (ResNet-style) | mlp_features (normalised) | Captures nonlinear interactions |
| Ensemble | OOF blend of all | Final submission |

---

## Reproducibility Rules

1. All random seeds set to `42`
2. All transformers (PCA, QuantileTransformer, scalers) fitted on **train only**
3. Test data transformed using **train-fitted** transformers (no test data leakage)
4. TARGET winsorization applied to **training only** — raw predictions submitted
5. All fitted objects saved as `.pkl` so final_submission.ipynb can load them
   without re-running the full pipeline

---

## Claude Code Usage Guide

When working with this project via Claude Code:

**For EDA:** `"Follow EDA_INSTRUCTIONS.md and run all layers in 01_eda.ipynb"`

**For feature engineering:** `"Follow FEATURE_ENGINEERING_INSTRUCTIONS.md and run 02_feature_engineering.ipynb"`

**For debugging:** Paste fold R² values and ask: `"My fold R² values are [x,y,z] with high variance — diagnose potential causes"`

**For modelling:** `"Implement 5-fold LightGBM with early stopping using tree_features from feature_sets.pkl, track OOF R²"`

**For Optuna tuning:** `"Write an Optuna objective for LightGBM that tunes num_leaves, min_child_samples, feature_fraction, subsample, reg_alpha, reg_lambda with 100 trials"`
