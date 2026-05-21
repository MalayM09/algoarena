"""
Apply shrinkA_p5 confidence weighting to top v6 pd blends.
shrinkA_p5 was the LB winner in v5 (0.00084). Apply it to the new better blends.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm
import os

DATA_DIR = '/Users/malaymishra/Desktop/quant_ml_project/data/raw'
OUT_DIR  = '/Users/malaymishra/Desktop/quant_ml_project/outputs/submissions'

print('Loading train day counts...', flush=True)
train = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'), columns=['SO3_T'])
test  = pd.read_parquet(os.path.join(DATA_DIR, 'test.parquet'),  columns=['ID', 'SO3_T'])

train_day = train['SO3_T'].round(5).astype(str).values
test_day  = test['SO3_T'].round(5).astype(str).values
test_ids  = test['ID'].values

train_day_counts = {}
for d in np.unique(train_day):
    train_day_counts[d] = int((train_day == d).sum())

median_n = np.median(list(train_day_counts.values()))
print(f'Train days: {len(train_day_counts)}, median samples/day: {median_n:.0f}', flush=True)

def compute_conf(power=5.0):
    conf = np.zeros(len(test_ids), dtype=np.float64)
    for d in np.unique(test_day):
        m = test_day == d
        n_tr = train_day_counts.get(d, 0)
        if n_tr == 0:
            conf[m] = 0.0
        else:
            conf[m] = min(1.0, n_tr / median_n) ** (1.0 / power)
    return conf

conf5 = compute_conf(power=5.0)
print(f'ShrinkA p=5: mean={conf5.mean():.4f}, min={conf5.min():.4f}, zero_frac={np.mean(conf5==0):.3f}', flush=True)

def finalize(pred, target_std=0.000948):
    p = pred.astype(np.float64).copy()
    p -= p.mean()
    s = p.std()
    return p * (target_std / s) if s > 1e-10 else p

TARGET_STDS = [0.000700, 0.000948, 0.001200, 0.001500]

# Top blends to apply shrinkage to
top_blends = [
    'v6_ff60_40_rpd_60',
    'v6_ff60_50_rpd_50',
    'v6_ff50_40_rpd_60',
    'v6_ff50_50_rpd_50',
    'v6_ff60_60_rpd_40',
    'v6_ff50_60_rpd_40',
]

written = 0
for base_name in top_blends:
    for ts in TARGET_STDS:
        src = os.path.join(OUT_DIR, f'{base_name}_s{int(ts*1e6)}.csv')
        if not os.path.exists(src):
            print(f'  MISSING: {src}', flush=True)
            continue
        df = pd.read_csv(src).sort_values('ID')
        pred = df['TARGET'].values.astype(np.float64)
        # Apply confidence shrinkage (row-level, batch-independent)
        pred_sh = pred * conf5
        pred_sh = finalize(pred_sh, ts)
        dst = os.path.join(OUT_DIR, f'{base_name}_shA5_s{int(ts*1e6)}.csv')
        pd.DataFrame({'ID': test_ids, 'TARGET': pred_sh}).sort_values('ID').to_csv(dst, index=False)
        written += 1

print(f'Written {written} shA5 CSVs.', flush=True)
