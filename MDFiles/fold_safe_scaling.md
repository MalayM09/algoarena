# Context: Memory-Safe In-Fold Cross-Sectional Pipeline

## The Objective
We are building `04_fold_safe_cross_sectional.py` to run locally on an 8GB RAM machine. Our previous baseline proved that absolute, per-row features have zero predictive power (Leaderboard R² ≈ 0). The true alpha lies in **cross-sectional relative features** (e.g., Z-Scores). 
However, to prevent Data Leakage, we cannot scale or rank the dataset before splitting. We MUST compute the mean/std strictly on the training fold, and apply it to the validation and test sets.

## The Memory Constraint (8GB RAM Limit)
We cannot duplicate our feature matrices. The agent must strictly adhere to **In-Place Mutation**:
1. Use `sklearn.preprocessing.StandardScaler(copy=False)`.
2. When scaling `X_test` inside the CV loop, you must scale it in-place, generate predictions, and then immediately `.inverse_transform()` it in-place so it is ready for the next fold.
3. Aggressive use of `del` and `gc.collect()` at the end of every fold iteration is mandatory.

## Strict Anti-Leakage Math
1. **Target Winsorization:** Inside the loop, calculate the 1st and 99th percentile of `y_tr` (Training targets). Use `np.clip` to apply these exact boundaries to BOTH `y_tr` and `y_va`. Do not calculate percentiles on the validation set.
2. **Feature Scaling:** `StandardScaler` must only call `.fit()` on `X_tr`. It calls `.transform()` on `X_tr`, `X_va`, and `X_test`.
3. **Loss Function:** We must use a custom Fair Loss function to protect the trees from heavy financial tails.

## Cross-Validation
Use `GroupKFold(n_splits=5)` using 5 quantile buckets of `SO3_T` as the grouping variable. This ensures the model learns to generalize across different market regimes.