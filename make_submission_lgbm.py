import numpy as np
import pandas as pd
import os

os.chdir('/Users/malaymishra/Desktop/quant_ml_project')
os.makedirs('outputs/submissions', exist_ok=True)

# Load predictions
lgbm_test_preds = np.load('outputs/oof_predictions/lgbm_test_preds.npy')
print(f"Predictions loaded: {lgbm_test_preds.shape}, dtype={lgbm_test_preds.dtype}")
print(f"  min={lgbm_test_preds.min():.6f}  max={lgbm_test_preds.max():.6f}  mean={lgbm_test_preds.mean():.6f}")

# Load test IDs
test = pd.read_parquet('data/raw/test.parquet', columns=['ID'])
print(f"Test IDs loaded: {len(test):,} rows")

# Load sample submission (for correct ordering)
sample_sub = pd.read_csv('data/raw/sample_submission.csv')
print(f"Sample submission: {len(sample_sub):,} rows")

# Build submission aligned to sample ordering
sub = pd.DataFrame({'ID': test['ID'].values, 'TARGET': lgbm_test_preds})
sub = sample_sub[['ID']].merge(sub, on='ID', how='left')

null_count = sub['TARGET'].isnull().sum()
if null_count > 0:
    print(f"WARNING: {null_count} missing predictions — filling with 0.0")
    sub['TARGET'] = sub['TARGET'].fillna(0.0)

out_path = 'outputs/submissions/lgbm_v1.csv'
sub.to_csv(out_path, index=False)
print(f"\nSAVED: {out_path}  ({len(sub):,} rows)")
print(sub.head())
