# ================================================================
# IC-WEIGHTED PREDICTION — SIGN-STABLE FEATURES ONLY
# ================================================================
# Hypothesis: use only features whose IC sign is consistent across
# ALL 5 SO3_T regimes as direct weights. No model fitting.
# Zero overfitting possible. If this scores positive, stable
# cross-sectional signals transfer to the test set.
#
# Key difference from fold_safe_v1:
#   - No StandardScaler (raw feature values, not Z-scores)
#   - No tree model (direct weighted sum)
#   - Only sign-stable features used as weights
# ================================================================

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')

print("Loading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)

y        = train['TARGET'].values
feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET'}]

# ── Step 1: Compute IC per feature on full training set ──────────────────────
print("Computing ICs and regime stability...")

so3 = train['SO3_T'].fillna(train['SO3_T'].median()).values
groups = pd.qcut(pd.Series(so3), q=5, labels=False, duplicates='drop').values
unique_groups = sorted(set(groups))

ics = {}
stable = {}

for col in feat_cols:
    vals = train[col].fillna(0).values
    global_ic = np.corrcoef(vals, y)[0, 1]
    if np.isnan(global_ic):
        continue

    regime_ics = []
    for g in unique_groups:
        idx = groups == g
        if idx.sum() < 100:
            continue
        r = np.corrcoef(vals[idx], y[idx])[0, 1]
        if not np.isnan(r):
            regime_ics.append(r)

    if len(regime_ics) < len(unique_groups):
        continue

    signs = set(np.sign(r) for r in regime_ics)
    is_stable = len(signs) == 1  # same sign in ALL regimes

    ics[col]    = global_ic
    stable[col] = is_stable

ic_series = pd.Series(ics)
stable_series = pd.Series(stable)

stable_features = stable_series[stable_series].index.tolist()
print(f"  Total features          : {len(ics)}")
print(f"  Sign-stable features    : {len(stable_features)}")

# ── Step 2: Rank by |IC| within stable features ─────────────────────────────
stable_ics = ic_series[stable_features].sort_values(key=abs, ascending=False)
print(f"\nTop stable features by |IC|:")
for col, ic in stable_ics.head(20).items():
    print(f"  {col:<50}  IC={ic:+.5f}")

# ── Step 3: IC-weighted prediction ──────────────────────────────────────────
# Variants: top 10, top 20, top 50, all stable

def ic_weighted_pred(test_df, feature_ics, top_n=None):
    """Weighted sum: prediction = Σ (IC_i / Σ|IC_i|) * feature_i"""
    sel = feature_ics.head(top_n) if top_n else feature_ics
    total_abs_ic = sel.abs().sum()
    preds = np.zeros(len(test_df))
    for col, ic in sel.items():
        if col in test_df.columns:
            vals = test_df[col].fillna(0).values
            preds += (ic / total_abs_ic) * vals
    return preds

results = {}
for top_n in [5, 10, 20, 50, None]:
    label = f'top{top_n}' if top_n else 'all_stable'
    preds = ic_weighted_pred(test, stable_ics, top_n=top_n)

    # Scale to match fold_safe_v1 amplitude (std=0.000624)
    if preds.std() > 0:
        preds_scaled = preds * (0.000624 / preds.std())
    else:
        preds_scaled = preds

    results[label] = preds_scaled
    print(f"\n  {label}:  "
          f"raw_std={preds.std():.6f}  "
          f"scaled_std={preds_scaled.std():.6f}  "
          f"pct_pos={(preds_scaled > 0).mean()*100:.1f}%  "
          f"skew={pd.Series(preds_scaled).skew():+.3f}")

# ── Step 4: OOF proxy — compute IC-weighted train predictions ────────────────
# Can't do proper CV (no model to retrain), so compute global train correlation
print("\n\nTrain-set IC-weighted correlation (proxy for OOF):")
for label, _ in results.items():
    top_n_val = int(label.replace('top', '')) if 'top' in label else None
    train_preds = ic_weighted_pred(train, stable_ics, top_n=top_n_val)
    corr = np.corrcoef(train_preds, y)[0, 1]
    r2   = r2_score(y, train_preds * (train['TARGET'].std() * 0.1 / (train_preds.std() + 1e-8)))
    print(f"  {label:<15}: corr={corr:+.6f}  (train R² meaningless — no CV)")

# ── Step 5: Save submissions ─────────────────────────────────────────────────
sample_sub_path = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
id_col = pd.read_csv(sample_sub_path)[['ID']] if os.path.exists(sample_sub_path) \
         else pd.DataFrame({'ID': test['ID'].values})

print("\n\nSaving CSVs:")
for label, preds in results.items():
    sub = pd.DataFrame({'ID': test['ID'].values, 'TARGET': preds})
    sub = id_col.merge(sub, on='ID', how='left').fillna(0.0)
    path = os.path.join(OUT_DIR, f'icw_{label}.csv')
    sub.to_csv(path, index=False)
    t = sub['TARGET']
    print(f"  icw_{label}.csv  "
          f"std={t.std():.6f}  mean={t.mean():+.7f}  "
          f"pct_pos={(t>0).mean()*100:.1f}%")

print("\nPriority: submit icw_top10.csv first — best balance of stable signal vs noise.")
print("If positive LB: icw_top20 next to check if more features help.")
print("If negative LB: the stable IC signal does not transfer to test period.")
