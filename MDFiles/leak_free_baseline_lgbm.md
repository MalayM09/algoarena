# Context: Clean Minimal Baseline Strategy

## The Objective
We are resetting our modelling pipeline to establish a true, leak-free baseline for a short-horizon financial return prediction competition. Previous iterations suffered from cross-sectional data leakage, resulting in inflated local CV scores and negative leaderboard scores. Our goal is to achieve an honest Out-Of-Fold (OOF) R² of ~0.001 to ~0.005 that translates to a positive score on the public leaderboard.

## Strict Anti-Leakage Rules
1. **NO Cross-Sectional Transforms:** Do not use `StandardScaler`, `QuantileTransformer`, rank transforms, or cross-sectional Z-scores across the entire dataset.
2. **Per-Row Isolation Only:** Feature engineering must be strictly limited to row-wise arithmetic (e.g., lag differences, lag ratios, sign agreement). A row must be processed entirely independently of any other row.
3. **Target Handling:** Predict the raw `TARGET`. Do not apply winsorization or clipping to the target globally before splitting.

## Cross-Validation Strategy
Because explicit timestamps are missing, we use `SO3_T` (a global market state proxy) to group our folds. 
* Use `pd.qcut` on `SO3_T` to create 5 quantile buckets.
* Use `GroupKFold(n_splits=5)` using these buckets as the grouping variable to ensure different market states are isolated between train and validation.

## Modelling Strategy
* **Algorithm:** LightGBM Regressor.
* **Hyperparameters:** Heavy regularization is mandatory. Shallow trees (`num_leaves` ~ 63), high minimum child samples (`min_child_samples` >= 500), and strong L1/L2 penalties.