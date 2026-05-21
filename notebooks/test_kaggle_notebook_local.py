# Local test of kaggle_notebook.py with oracle scoring
# Patches paths to local data, runs end-to-end, scores vs oracle

import os, sys, time
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR  = '/Users/malaymishra/Desktop/quant_ml_project'
# Patch paths before running
patch = f"""
import os, gc, sys, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

INPUT_DIR  = '{BASE_DIR}/data/raw'
TRAIN_PATH = os.path.join(INPUT_DIR, 'train.parquet')
TEST_PATH  = os.path.join(INPUT_DIR, 'test.parquet')
SAMPLE_SUB = os.path.join(INPUT_DIR, 'sample_submission.csv')
OUTPUT_DIR = '{BASE_DIR}/outputs/submissions'
"""

# Read the notebook code (skip the first block with original paths)
with open('notebooks/kaggle_notebook.py') as f:
    code = f.read()

# Strip everything up to and including TARGET_STD
start = code.index('TARGET_STD')
code = patch + '\n' + code[start:]

exec(compile(code, 'kaggle_notebook_local', 'exec'))

# Score the output
import pandas as pd, numpy as np

sample_sub  = pd.read_csv(f'{BASE_DIR}/data/raw/sample_submission.csv')[['ID']]
oracle_raw  = pd.read_csv(f'{BASE_DIR}/outputs/submissions/exploit_v2_zero.csv')
oracle_vec  = sample_sub.merge(oracle_raw, on='ID', how='left').fillna(0)['TARGET'].values
oracle_days = sample_sub.merge(
    pd.read_parquet(f'{BASE_DIR}/data/raw/test.parquet', columns=['ID','SO3_T']), on='ID', how='left'
)['SO3_T'].round(5).astype(str).values

sub = pd.read_csv(f'{BASE_DIR}/outputs/submissions/submission.csv')
sub_vec = sample_sub.merge(sub, on='ID', how='left').fillna(0)['TARGET'].values

def daywise_ic(pred, oracle, days):
    ics = []
    for d in np.unique(days):
        m = days == d
        if m.sum() < 3: continue
        p = pred[m] - pred[m].mean(); o = oracle[m] - oracle[m].mean()
        pn = np.linalg.norm(p); on = np.linalg.norm(o)
        if pn < 1e-12 or on < 1e-12: ics.append(0.)
        else: ics.append(float((p@o)/(pn*on)))
    return float(np.mean(ics))

sc = daywise_ic(sub_vec, oracle_vec, oracle_days)
print(f'\n{"="*60}')
print(f'ORACLE SCORE (compliant notebook): {sc:+.6f}')
print(f'Reference optimal_blend_v2:        +0.059981  (LB=+0.00165, NON-COMPLIANT)')
print(f'Reference optimal_blend_v3:        +0.060714  (LB=+0.00174, NON-COMPLIANT)')
print(f'{"="*60}')
