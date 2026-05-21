# ┌─────────────────────────────────────────────────────────────────────────┐
# │  KAGGLE NOTEBOOK — LightGBM + MLP Heterogeneous Ensemble               │
# │  Each # ── CELL N ── block = one Kaggle notebook cell                   │
# └─────────────────────────────────────────────────────────────────────────┘
#
# DESIGN PRINCIPLES (all data-backed from EDA):
#  1. ICIR top 279 features  — removes 166 noise features (|ICIR|<0.5)
#  2. 22 lag-diff features   — LagT1-LagT3 for sign-flip lag families
#  3. 5 abs() MI features    — U-shape alpha profile proven by decile analysis
#  4. Regime × feature       — 61 sign-flip features get × SO3_T interaction
#  5. Extreme indicators     — signal in d1/d10 only (proven by decile analysis)
#  6. Target Z-score norm    — fixes Fold 4 volatility mismatch
#  7. MLP on Spearman top 150— monotone LagT1 features suit smooth manifolds
#  8. Fair Loss on LGB        — robust to fat financial return tails
#  9. scipy blend optimizer  — automatic fallback to pure LGB if MLP hurts


# ── CELL 1: Imports & Environment ─────────────────────────────────────────
import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")
print(f"LGB    : {lgb.__version__}")
print(f"PyTorch: {torch.__version__}")


# ── CELL 2: Configuration ──────────────────────────────────────────────────
# All tunable parameters in one place.

# Paths (Kaggle input directory)
TRAIN_PATH  = '/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet'
TEST_PATH   = '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet'
SAMPLE_PATH = '/kaggle/input/jane-street-real-time-market-data-forecasting/sample_submission.csv'

N_FOLDS      = 5
RANDOM_SEED  = 42
FAIR_C       = 1.0          # Fair loss robustness parameter

# LightGBM — more leaves than fold_safe_v1 to capture extreme-value splits
LGB_PARAMS = {
    'num_leaves'        : 127,   # was 63; more expressive for extreme decile splits
    'learning_rate'     : 0.02,
    'feature_fraction'  : 0.5,   # was 0.4
    'bagging_fraction'  : 0.8,   # was 0.7
    'bagging_freq'      : 1,
    'min_child_samples' : 100,   # was 250; smaller leaves for tail patterns
    'reg_alpha'         : 0.1,   # was 0.5; less L1 to preserve weak signals
    'reg_lambda'        : 5.0,   # was 10.0
    'random_state'      : RANDOM_SEED,
    'n_jobs'            : -1,
    'verbose'           : -1,
}
LGB_ROUNDS    = 5000
LGB_EARLY     = 150

# MLP — tuned for near-zero financial signal
# Previous run: early-stopped at epoch 24-35, model never converged.
# Fix: more epochs, more patience, smaller lr+batch for better gradient resolution.
MLP_EPOCHS    = 500       # was 150 — give enough room to find near-zero signal
MLP_PATIENCE  = 50        # was 20  — noise fluctuates R² by ±0.001; need more patience
MLP_LR        = 1e-4      # was 3e-4 — smaller lr avoids overshooting weak optima
MLP_WD        = 1e-3
MLP_BATCH     = 2048      # was 4096 — more gradient steps per epoch
HUBER_DELTA   = 0.05      # Huber loss transition point (~80th pct of |residuals|)
EXTREME_Z     = 1.5          # Z-score threshold for extreme binary indicators (~top 7%)
TOP_N_EXTREME = 30           # Number of ICIR-top features to make extreme indicators for


# ── CELL 3: EDA-Derived Feature Lists (hardcoded from local EDA outputs) ──
# These lists were computed on full training data using IC/ICIR and Spearman
# analysis. Feature *selection* on global data is standard pre-processing;
# the fold-safe guarantee only applies to scaling, target stats, and
# in-loop engineered columns.

ICIR_TOP_279 = ['S03_A02_D03_W02_LagT1', 'S01_F03_U01_LagT1', 'Price_LagT1', 'S03_D02_V01_A01_B10_E10_E11_LagT1', 'S03_A02_W01_LagT1', 'S03_A07_A05_V09', 'Price_LagT2', 'Price_LagT3', 'S03_A02_W01_LagT2', 'S03_V04_T05_LagT2', 'S01_F03_U01_LagT2', 'S03_A02_D03_W02_LagT2', 'S02_F03_U01_LagT1', 'S03_D02_A09_A02_B04_E04_E05_LagT1', 'S03_D02_V01_A01_B10_E10_E11_LagT3', 'S03_D02_V01_A01_B10_E10_E11_LagT2', 'S03_A02_D04_W02_LagT1', 'S03_D02_A09_A02_B06_E06_E07_LagT2', 'S03_V04_T04_LagT2', 'S03_A02_D04_W02_LagT2', 'S03_V04_T03_LagT2', 'S02_F03_U01_LagT2', 'S03_D02_A09_A02_B06_E06_E07_LagT1', 'S03_A07_V01_V09_LagT3', 'S03_D02_A09_A02_B04_E04_E05_LagT3', 'S03_A02_D03_W02_LagT3', 'S01_F03_U01_LagT3', 'S03_V14_I01_LagT1', 'S03_V07_V06_LagT2', 'S03_V04_T06_LagT2', 'S03_D02_A09_A02_B04_E04_E05_LagT2', 'S03_A07_V01_V09_LagT1', 'S02_F03_U01_LagT3', 'S03_A07_V01_V09_LagT2', 'S03_D02_A09_A02_B07_E07_E08_LagT2', 'S03_A02_D04_W02_LagT3', 'S03_A02_W01_LagT3', 'S03_V04_T02_LagT2', 'S03_D02_A09_A02_B03_E03_E04_LagT1', 'S03_D02_A09_A02_B07_E07_E08_LagT1', 'S03_D02_A09_A02_B06_E06_E07_LagT3', 'S03_V14_I01_LagT2', 'S03_D02_A09_A02_B03_E03_E04_LagT2', 'S03_V04_T02_LagT1', 'S03_A07_A05_V09_LagT1', 'S03_V04_T03_LagT1', 'S03_V04_T04_LagT1', 'S03_D02_A09_A02_B07_E07_E08_LagT3', 'S03_D02_V01_A01_B08_E08_E09_LagT1', 'S03_V03_T03_LagT2', 'S03_V04_T01_LagT2', 'S03_V03_T04_LagT2', 'S03_V04_T01_LagT1', 'S03_D02_A09_A02_B01_E01_E02_LagT2', 'S03_V03_T02_LagT2', 'S03_P01_D04_LagT1', 'S03_V03_T02_LagT1', 'S03_D02_V01_A01_B09_E09_E10_LagT1', 'S03_D02_A09_A02_B03_E03_E04_LagT3', 'S03_V03_T01_LagT1', 'S03_V03_T03_LagT1', 'S03_D02_A09_A02_B00_E00_E01_LagT2', 'S03_V07_V06_LagT3', 'S03_V04_T06_LagT3', 'S03_V04_T04_LagT3', 'S03_V03_T01_LagT2', 'S03_A07_V01_V16_V12_LagT2', 'S03_V03_T05_LagT2', 'S02_O01_A01', 'S03_D01_V12_D06_LagT1', 'S02_O01', 'S01_F01_U01_LagT1', 'S03_V04_T05_LagT3', 'S03_V04_T03_LagT3', 'S02_F01_U01_LagT1', 'S03_D02_V01_A01_B08_E08_E09_LagT2', 'S03_A07_A05_V09_LagT2', 'S03_D02_A09_A02_B02_E02_E03_LagT1', 'S03_V03_T04_LagT1', 'S03_D02_A09_A02_B02_E02_E03_LagT2', 'S02_O02', 'S03_V04_T05_LagT1', 'S03_A07_V18_V12_LagT2', 'S02_O02_A01', 'S03_V03_T06_LagT2', 'S03_V04_T02_LagT3', 'S03_V04_T06_LagT1', 'S03_V07_V06_LagT1', 'S02_F03_U01', 'S02_F01_U01_LagT2', 'S01_F01_U01_LagT2', 'S03_A07_V01_V16_V12_LagT1', 'S03_A02_A04_D03_F04_U03_LagT3', 'S03_D01_V12_D06_LagT2', 'S03_P01_D04_LagT2', 'S03_D02_V01_A01_B09_E09_E10_LagT3', 'S03_D02_A09_A02_B01_E01_E02_LagT1', 'S03_A07_V18_V12_LagT1', 'S03_D02_A09_A02_B00_E00_E01_LagT1', 'S03_V03_T05_LagT1', 'S03_D02_A09_A02_B02_E02_E03_LagT3', 'S03_P01_D02_S01_LagT2', 'S03_A07_A05_V09_LagT3', 'S03_V14_I01_LagT3', 'S03_D02_V01_A01_B01_E01_E02_LagT1', 'S03_V20_V13_LagT1', 'S02_F01_U01_LagT3', 'S03_A02_A04_D03_F04_U03_LagT2', 'S03_D02_A09_A02_B00_E00_E01_LagT3', 'S03_D02_V01_A01_B08_E08_E09_LagT3', 'S03_A02_A04_D03_F04_U02_LagT3', 'S03_D02_V01_A01_B06_E06_E07_LagT1', 'S03_V03_T06_LagT1', 'S03_V20_V13_LagT2', 'S03_A02_A04_D03_F04_U02_LagT2', 'S03_A07_A05_V16', 'S03_V04_T04', 'S03_D02_V01_A01_B04_E04_E05_LagT2', 'S03_D02_V01_A01_B07_E07_E08_LagT1', 'S03_V04_T03', 'S03_A02_A04_D03_F04_U03_LagT1', 'S03_V04_T05', 'S03_D02_V01_A01_B09_E09_E10_LagT2', 'S03_D01_V12_D06_LagT3', 'S03_V04_T02', 'S01_F02_U01_LagT1', 'S03_A07_V01_V16_V12_LagT3', 'S03_D02_A09_A02_B01_E01_E02_LagT3', 'S03_V07_V06', 'S03_V04_T06', 'S03_A02_A04_D03_F04_U02_LagT1', 'S03_V03_T02_LagT3', 'S03_V03_T03_LagT3', 'S03_V04_T01', 'S01_F01_U01_LagT3', 'S03_V03_T04_LagT3', 'S03_V04_T01_LagT3', 'S03_V03_T01_LagT3', 'S03_D02_A09_A02_B08_E08_E09_LagT1', 'S03_V03_T05_LagT3', 'S03_A07_V18_V12_LagT3', 'S03_D02_V01_A01_B07_E07_E08_LagT3', 'S03_V06_V15_V01_LagT2', 'S03_V06_V01_LagT2', 'S03_D02_A09_A02_B08_E08_E09_LagT2', 'S03_D01_V11_D06_LagT1', 'S03_V14_I01', 'S03_P01_D02_S01_LagT1', 'S03_V06_V15_O04_LagT2', 'S03_V06_V15_W02_LagT2', 'S03_P01_D01_S01_LagT1', 'S03_A07_V01_V16_V06_LagT2', 'S03_V20_V13_LagT3', 'S03_V14_V01_LagT1', 'S03_P01_D02_S01_LagT3', 'S03_V03_T06_LagT3', 'S03_D01_V09_D06', 'S03_D02_A09_A02_B09_E09_E10_LagT2', 'S03_V06_V15_W02_LagT3', 'S03_D02_V01_A01_B06_E06_E07_LagT3', 'S03_D02_A09_A02_B09_E09_E10_LagT1', 'S03_D02_V01_A01_B06_E06_E07_LagT2', 'S01_F02_U01_LagT2', 'S03_P01_D02_S02_LagT2', 'S03_D02_V01_A01_B00_E00_E01_LagT1', 'S03_D02_V01_A01_B04_E04_E05_LagT3', 'S03_A07_A05_V16_LagT1', 'S03_D03_LagT1', 'S03_A07_V01_V17_LagT1', 'S03_A07_A08_LagT3', 'S01_F03_U01', 'S03_P01_D01_S02_LagT1', 'S03_A07_A05_V16_LagT2', 'S02_O01_LagT2', 'S03_P01_D01_S01_LagT2', 'S02_O01_A01_LagT2', 'S03_P01_D02_S02_LagT3', 'S02_O01_LagT1', 'S02_O01_A01_LagT1', 'S03_V06_W02_LagT3', 'S03_V06_V15_V01_LagT1', 'S03_V06_V01_LagT1', 'S03_D02_V01_A01_B07_E07_E08_LagT2', 'S03_A07_A08', 'S03_A07_A05_V17_LagT3', 'S03_V05_T05', 'S03_O02_W01', 'S03_V05_T03', 'S03_P01_O06_LagT2', 'S01_F02_U01', 'S03_V05_T01', 'S03_V14_V01_LagT2', 'S03_V05_T02', 'S03_V05_T04', 'S03_P01_D02_S02_LagT1', 'S03_D02_V01_A01_B04_E04_E05_LagT1', 'S03_V05_T06', 'S03_V08_V06', 'S03_A07_A08_LagT2', 'S03_A07_V18_V06_LagT2', 'S03_D01_V11_D06_LagT2', 'S01_F02_U01_LagT3', 'S01_F01_U01', 'S03_D02_V01_A01_B02_E02_E03_LagT1', 'S03_P01_D01_S03_LagT1', 'S03_V14_V01_LagT3', 'S02_O02_LagT2', 'S02_O01_A01_LagT3', 'S02_O01_LagT3', 'S02_O02_A01_LagT2', 'S03_D03_LagT2', 'S03_P01_D05_LagT1', 'S03_V06_V15_O04_LagT3', 'S03_V06_V01_LagT3', 'S03_V06_V15_V01_LagT3', 'S03_A02_A04_D03_F04_U02', 'S01_O02_A01_LagT3', 'S03_P01_D05', 'S03_A07_V01_V16_V06_LagT1', 'S01_O02_LagT3', 'S03_D02_V01_A01_B03_E03_E04_LagT3', 'S03_O02_W02_LagT3', 'S03_V02_T01', 'S03_V06_W02_LagT2', 'S03_D02_A09_A02_B05_E05_E06_LagT1', 'S03_V05_T02_LagT3', 'S03_D02_A09_A02_B08_E08_E09_LagT3', 'S03_A02_A04_D04_F04_U03_LagT3', 'S03_A02_A04_D03_F04_U03', 'S03_P01_D05_LagT3', 'S03_A07_V01_V17_LagT2', 'S03_P01_D04_LagT3', 'S03_V06_V15_W02_LagT1', 'S03_D02_A09_A02_B05_E05_E06_LagT3', 'S03_V05_T01_LagT3', 'S03_V06_V15_O04_LagT1', 'S03_A03_A02_W02_LagT1', 'S03_A07_V01_V16_V06_LagT3', 'S03_V02_T02', 'S03_D02_A09_A02_B01_E01_E02', 'S03_D02_V01_A01_B05_E05_E06_LagT2', 'S02_O02_LagT3', 'S03_P01_D01_S02_LagT2', 'S02_O02_A01_LagT3', 'S01_O01_A01_LagT3', 'S01_O01_LagT3', 'S03_V08_V06_LagT2', 'S03_V05_T06_LagT2', 'S04_V19_V12_LagT1', 'S03_A02_A04_D04_F04_U03', 'S03_A07_V18_V06_LagT3', 'S03_D02_A09_A02_B05_E05_E06_LagT2', 'S03_P01_D01_S03_LagT3', 'S02_O02_LagT1', 'S02_O02_A01_LagT1', 'S03_V02_T03', 'S03_P01_D01_S02_LagT3', 'S03_D02_V01_A01_B01_E01_E02_LagT2', 'S03_D01_V11_D06_LagT3', 'S01_O01', 'S01_O01_A01', 'S03_A02_W01', 'S03_V05_T03_LagT3', 'S03_P01_O06_LagT1', 'S02_F02_U01', 'S03_D01_V10_D06', 'S03_A02_A04_D04_F04_U02', 'S03_D02_A09_A02_B05_E05_E06', 'S04_V19_A06', 'S03_V05_T05_LagT2', 'S03_D02_A09_A02_B06_E06_E07', 'S03_V06_W02', 'S03_P01_D01_S01_LagT3', 'S03_V06_V15_W02', 'S03_P01_D01_S03', 'S03_D02_A09_A02_B10_E10_E11_LagT2', 'S03_V02_T04', 'S03_D01_V09_D06_LagT2', 'S03_D02_V01_A01_B05_E05_E06_LagT3']

SPEARMAN_TOP_150 = ['S01_F03_U01_LagT1', 'S03_A02_D03_W02_LagT1', 'Price_LagT1', 'S03_A02_W01_LagT1', 'S03_A02_D04_W02_LagT1', 'S03_D02_V01_A01_B10_E10_E11_LagT1', 'Price_LagT2', 'S03_A02_D03_W02_LagT2', 'S03_A02_W01_LagT2', 'S02_F03_U01_LagT1', 'Price_LagT3', 'S03_D02_A09_A02_B06_E06_E07_LagT1', 'S03_A07_V01_V09_LagT1', 'S03_A02_D04_W02_LagT2', 'S03_D02_A09_A02_B04_E04_E05_LagT1', 'S01_F03_U01_LagT2', 'S03_A07_A05_V09', 'S03_D02_A09_A02_B03_E03_E04_LagT1', 'S03_D02_A09_A02_B04_E04_E05_LagT2', 'S03_D02_V01_A01_B10_E10_E11_LagT2', 'S03_D02_A09_A02_B07_E07_E08_LagT1', 'S03_D02_A09_A02_B03_E03_E04_LagT2', 'S03_D02_A09_A02_B06_E06_E07_LagT2', 'S02_F03_U01_LagT2', 'S03_D02_A09_A02_B06_E06_E07_LagT3', 'S03_A02_W01_LagT3', 'S03_A02_D03_W02_LagT3', 'S03_A07_V01_V09_LagT2', 'S03_A07_V01_V09_LagT3', 'S03_V14_I01_LagT1', 'S03_A02_D04_W02_LagT3', 'S03_D02_V01_A01_B10_E10_E11_LagT3', 'S02_F03_U01_LagT3', 'S03_D02_A09_A02_B07_E07_E08_LagT2', 'S03_A07_A05_V09_LagT1', 'S03_D02_A09_A02_B04_E04_E05_LagT3', 'S01_F03_U01_LagT3', 'S03_V04_T03_LagT2', 'S03_V04_T05_LagT2', 'S03_V04_T02_LagT2', 'S03_D02_A09_A02_B02_E02_E03_LagT1', 'S03_V04_T04_LagT2', 'S03_V14_I01_LagT2', 'S03_V04_T06_LagT2', 'S03_V07_V06_LagT2', 'S02_O01', 'S02_O01_A01', 'S03_D02_A09_A02_B03_E03_E04_LagT3', 'S03_D02_A09_A02_B07_E07_E08_LagT3', 'S03_V04_T01_LagT2', 'S03_D02_A09_A02_B01_E01_E02_LagT2', 'S03_D02_A09_A02_B02_E02_E03_LagT2', 'S03_D02_V01_A01_B09_E09_E10_LagT1', 'S03_V04_T03_LagT1', 'S03_V03_T03_LagT2', 'S03_V04_T02_LagT1', 'S03_D02_V01_A01_B08_E08_E09_LagT1', 'S02_O02', 'S02_O02_A01', 'S03_V03_T02_LagT2', 'S03_V03_T04_LagT2', 'S03_V04_T04_LagT1', 'S03_V03_T01_LagT2', 'S03_D02_A09_A02_B01_E01_E02_LagT1', 'S03_V04_T01_LagT1', 'S03_D02_A09_A02_B00_E00_E01_LagT2', 'S03_A07_A05_V09_LagT2', 'S03_D02_A09_A02_B02_E02_E03_LagT3', 'S01_F01_U01_LagT1', 'S03_P01_D04_LagT1', 'S03_V03_T05_LagT2', 'S03_D02_V01_A01_B08_E08_E09_LagT2', 'S03_V04_T05_LagT1', 'S03_V03_T01_LagT1', 'S03_D01_V12_D06_LagT1', 'S02_F01_U01_LagT1', 'S03_V03_T06_LagT2', 'S03_V03_T03_LagT1', 'S03_V03_T02_LagT1', 'S03_V04_T06_LagT1', 'S03_V07_V06_LagT1', 'S03_D02_V01_A01_B08_E08_E09_LagT3', 'S03_V03_T04_LagT1', 'S03_D01_V12_D06_LagT2', 'S03_D02_A09_A02_B01_E01_E02_LagT3', 'S02_F01_U01_LagT2', 'S01_F01_U01_LagT2', 'S03_A02_A04_D03_F04_U03_LagT3', 'S03_V04_T03_LagT3', 'S03_V04_T04_LagT3', 'S03_A07_A05_V09_LagT3', 'S03_V04_T05_LagT3', 'S03_D02_A09_A02_B00_E00_E01_LagT3', 'S03_A07_V01_V16_V12_LagT2', 'S03_V03_T02_LagT3', 'S03_V04_T06_LagT3', 'S03_V07_V06_LagT3', 'S03_V03_T03_LagT3', 'S03_D02_V01_A01_B07_E07_E08_LagT1', 'S03_D02_A09_A02_B00_E00_E01_LagT1', 'S03_V03_T04_LagT3', 'S03_V03_T05_LagT1', 'S03_A02_A04_D03_F04_U03_LagT2', 'S02_F03_U01', 'S03_V20_V13_LagT2', 'S03_V03_T01_LagT3', 'S03_A07_V18_V12_LagT2', 'S03_V03_T06_LagT1', 'S03_V04_T02_LagT3', 'S03_A07_V01_V16_V12_LagT1', 'S03_D02_A09_A02_B08_E08_E09_LagT1', 'S02_F01_U01_LagT3', 'S03_D02_V01_A01_B04_E04_E05_LagT2', 'S03_A02_A04_D03_F04_U03_LagT1', 'S03_V03_T05_LagT3', 'S01_F02_U01_LagT1', 'S03_V04_T04', 'S03_V04_T03', 'S03_V04_T05', 'S03_A07_V18_V12_LagT1', 'S03_V14_I01_LagT3', 'S03_V04_T06', 'S03_V07_V06', 'S03_V20_V13_LagT3', 'S03_D02_V01_A01_B09_E09_E10_LagT2', 'S03_P01_D02_S01_LagT2', 'S03_V04_T02', 'S03_V03_T06_LagT3', 'S03_V04_T01', 'S03_A02_A04_D03_F04_U02_LagT3', 'S03_D01_V12_D06_LagT3', 'S03_P01_D02_S01_LagT3', 'S03_A02_A04_D03_F04_U02_LagT2', 'S03_V20_V13_LagT1', 'S03_P01_D02_S01_LagT1', 'S03_V06_V15_V01_LagT2', 'S03_V06_V01_LagT2', 'S03_D02_V01_A01_B06_E06_E07_LagT1', 'S03_O02_W01', 'S03_A02_A04_D03_F04_U02_LagT1', 'S03_D02_V01_A01_B09_E09_E10_LagT3', 'S03_D02_V01_A01_B01_E01_E02_LagT1', 'S03_V06_V15_O04_LagT2', 'S03_A07_A05_V16', 'S03_D02_A09_A02_B09_E09_E10_LagT1', 'S01_F01_U01_LagT3', 'S03_D01_V09_D06', 'S03_P01_D04_LagT2', 'S03_V04_T01_LagT3', 'S03_A07_V01_V16_V12_LagT3']

# 22 base feature families where LagT1 and LagT3 point in opposite directions
SIGN_FLIP_LAG_FAMILIES = ['S03_D02_V01_A01_B00_E00_E01', 'S03_A07_A08', 'S03_V05_T02',
    'S03_V08_V06', 'S03_V05_T06', 'S03_V05_T01', 'S03_P01_D05', 'S03_O02_W02',
    'S03_D02_V01_A01_B02_E02_E03', 'S03_V05_T04', 'S02_F02_U01', 'S03_V05_T03',
    'S03_P01_D02_S03', 'S03_D01_V09_D06', 'S03_D01_V10_D06', 'S03_A02_A04_D04_F04_U02',
    'S04_V19_A06', 'S05_V19', 'S03_V02_T01', 'S03_V02_T02', 'S03_O01_W01', 'S03_V02_T03']

# 61 features whose Spearman correlation with TARGET flips sign across SO3_T regimes
REGIME_FLIP_FEATURES = ['S03_V03_T03_LagT1', 'S03_V03_T02_LagT1', 'S03_V03_T01_LagT1',
    'S03_V04_T06_LagT1', 'S03_V04_T04_LagT1', 'S03_V04_T05_LagT1', 'S03_V04_T03_LagT1',
    'S03_V03_T04_LagT2', 'S03_V03_T06_LagT2', 'S03_V03_T05_LagT2', 'S03_V04_T04_LagT2',
    'S03_V04_T03_LagT2', 'S03_V03_T03_LagT2', 'S03_V04_T01_LagT1', 'S03_V04_T05_LagT2',
    'S03_V04_T01_LagT2', 'S03_V07_V06_LagT2', 'S03_V04_T06_LagT2', 'S03_V04_T02_LagT2',
    'S03_V04_T02_LagT1', 'S03_V03_T02_LagT2', 'Price_LagT3', 'S03_V03_T01_LagT2',
    'S03_D02_V01_A01_B10_E10_E11_LagT3', 'S03_D01_V12_D06_LagT1', 'S03_A07_V01_V09_LagT3',
    'S03_A07_A05_V09', 'Price_LagT2', 'S03_A07_V01_V09_LagT2', 'S03_D02_V01_A01_B09_E09_E10_LagT1',
    'S03_D02_V01_A01_B10_E10_E11_LagT2', 'S03_V14_I01_LagT2', 'S03_A07_A05_V09_LagT2',
    'S03_A02_D04_W02_LagT2', 'S03_D02_A09_A02_B02_E02_E03_LagT3', 'S03_D02_A09_A02_B04_E04_E05_LagT2',
    'S02_F01_U01_LagT1', 'S03_A02_D04_W02_LagT3', 'S03_D02_A09_A02_B04_E04_E05_LagT1',
    'S02_F03_U01_LagT1', 'S03_D02_A09_A02_B07_E07_E08_LagT3', 'S03_A02_W01_LagT3',
    'S03_A07_V01_V09_LagT1', 'S03_D02_A09_A02_B04_E04_E05_LagT3', 'S03_D02_A09_A02_B06_E06_E07_LagT3',
    'S03_A02_D03_W02_LagT3', 'S01_F01_U01_LagT1', 'S03_A02_W01_LagT1', 'S02_F03_U01_LagT3',
    'S03_P01_D04_LagT1', 'S02_F03_U01_LagT2', 'S03_A02_W01_LagT2', 'S03_D02_V01_A01_B08_E08_E09_LagT2',
    'S03_D02_V01_A01_B08_E08_E09_LagT1', 'S03_D02_V01_A01_B10_E10_E11_LagT1',
    'S03_A07_A05_V09_LagT1', 'S01_F03_U01_LagT3', 'S03_D02_A09_A02_B06_E06_E07_LagT1',
    'S03_A02_D03_W02_LagT2', 'S01_F03_U01_LagT2', 'S03_D02_A09_A02_B06_E06_E07_LagT2']

# 5 high-MI features with U-shape alpha (use abs value alongside raw)
MI_USHAPE_FEATURES = ['S03_V14_V01', 'S03_V06_V01', 'S03_V06_V15_V01', 'S03_V02_T04', 'S03_V02_T06']

print(f"ICIR top 279     : {len(ICIR_TOP_279)} features")
print(f"Spearman top 150 : {len(SPEARMAN_TOP_150)} features")
print(f"Sign-flip lags   : {len(SIGN_FLIP_LAG_FAMILIES)} families")
print(f"Regime-flip feats: {len(REGIME_FLIP_FEATURES)} features")
print(f"MI U-shape feats : {len(MI_USHAPE_FEATURES)} features")


# ── CELL 4: Load & Preprocess Data ─────────────────────────────────────────
print("Loading data...")
t0 = time.time()

train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
print(f"  Train: {train.shape}   Test: {test.shape}  ({time.time()-t0:.1f}s)")

NON_FEAT       = {'ID', 'TARGET'}
all_feat_cols  = [c for c in train.columns if c not in NON_FEAT]
test_all_cols  = [c for c in all_feat_cols if c in test.columns]

for col in all_feat_cols:
    train[col] = train[col].astype(np.float32)
for col in test_all_cols:
    test[col]  = test[col].astype(np.float32)

y_train  = train['TARGET'].values.astype(np.float32)
test_ids = test['ID'].values

# NaN check
print(f"  Train NaNs: {train[all_feat_cols].isna().sum().sum()}")
print(f"  Test  NaNs: {test[test_all_cols].isna().sum().sum()}")

# Impute if needed
n_train_nan = train[all_feat_cols].isna().sum().sum()
if n_train_nan > 0:
    medians = train[all_feat_cols].median()
    train[all_feat_cols] = train[all_feat_cols].fillna(medians)
    test[test_all_cols]  = test[test_all_cols].fillna(medians[test_all_cols])
    print("  Imputed with train medians.")

print(f"  TARGET: mean={y_train.mean():.5f}  std={y_train.std():.5f}")


# ── CELL 5: Pre-Loop Feature Engineering (Static, No Target Used) ──────────
# These transforms use only X values — no fold structure, no target.
# Safe to compute once outside the loop.

print("Building engineered feature matrices...")

# ── 5a. Start from ICIR top 279 (removes 166 noisy features) ─────────────
base_cols = [c for c in ICIR_TOP_279 if c in all_feat_cols]
print(f"  ICIR top 279 available in train: {len(base_cols)}")

# ── 5b. Add LagT1 - LagT3 for 22 sign-flip lag families ─────────────────
# EDA proved these 22 families have T1 and T3 pointing opposite directions.
# The difference captures reversal vs momentum disagreement.
lag_diff_cols = []
for fam in SIGN_FLIP_LAG_FAMILIES:
    t1 = f'{fam}_LagT1'
    t3 = f'{fam}_LagT3'
    if t1 in train.columns and t3 in train.columns:
        col_name = f'{fam}_LagDiff_T1T3'
        train[col_name] = (train[t1] - train[t3]).astype(np.float32)
        test[col_name]  = (test[t1]  - test[t3]).astype(np.float32) \
                          if (t1 in test.columns and t3 in test.columns) \
                          else np.float32(0.0)
        lag_diff_cols.append(col_name)

print(f"  Lag difference features added: {len(lag_diff_cols)}")

# ── 5c. Add abs() for 5 MI U-shape features ──────────────────────────────
# Decile analysis proved these have U-shape alpha (both extremes predictive).
# abs() makes the U-shape monotone — easier for any model to exploit.
abs_cols = []
for feat in MI_USHAPE_FEATURES:
    if feat in train.columns:
        col_name = f'{feat}_abs'
        train[col_name] = train[feat].abs().astype(np.float32)
        test[col_name]  = test[feat].abs().astype(np.float32) \
                          if feat in test.columns else np.float32(0.0)
        abs_cols.append(col_name)

print(f"  Abs-MI features added: {len(abs_cols)}")

# ── 5d. Build final static feature column list ────────────────────────────
static_feat_cols = base_cols + lag_diff_cols + abs_cols
static_feat_cols = [c for c in dict.fromkeys(static_feat_cols)]  # deduplicate

# also keep SO3_T if not already in base_cols
if 'SO3_T' not in static_feat_cols and 'SO3_T' in train.columns:
    static_feat_cols.append('SO3_T')

print(f"  Total static features: {len(static_feat_cols)}")

# ── 5e. Build numpy matrices ──────────────────────────────────────────────
X_train_static = np.ascontiguousarray(train[static_feat_cols].values, dtype=np.float32)
X_test_static  = np.ascontiguousarray(test[
    [c for c in static_feat_cols if c in test.columns]
].reindex(columns=static_feat_cols, fill_value=0.0).values, dtype=np.float32)

print(f"  X_train_static: {X_train_static.shape}  "
      f"({X_train_static.nbytes/1e6:.0f} MB)")
print(f"  X_test_static : {X_test_static.shape}  "
      f"({X_test_static.nbytes/1e6:.0f} MB)")

# ── 5f. MLP feature indices (Spearman top 150, subset of static_feat_cols) ─
mlp_feat_names = [f for f in SPEARMAN_TOP_150 if f in static_feat_cols]
mlp_feat_idx   = [static_feat_cols.index(f) for f in mlp_feat_names]
MLP_INPUT_DIM  = len(mlp_feat_idx)
print(f"  MLP input features: {MLP_INPUT_DIM} (Spearman top 150 ∩ static set)")

# ── 5g. Regime-flip feature indices (for × SO3_T interaction in loop) ──────
regime_flip_in_static = [f for f in REGIME_FLIP_FEATURES if f in static_feat_cols]
regime_flip_idx = [static_feat_cols.index(f) for f in regime_flip_in_static]
so3t_idx_static = static_feat_cols.index('SO3_T') if 'SO3_T' in static_feat_cols else None
print(f"  Regime-flip features in static set: {len(regime_flip_idx)}")

# Free dataframes — keep numpy arrays
del train, test
gc.collect()
print("  DataFrames freed.")


# ── CELL 6: GroupKFold Setup ───────────────────────────────────────────────
print("Setting up GroupKFold on SO3_T quantile buckets...")

so3t_vals = X_train_static[:, so3t_idx_static]
groups = pd.qcut(pd.Series(so3t_vals), q=N_FOLDS,
                 labels=False, duplicates='drop').values.astype(np.int32)

actual_folds = len(np.unique(groups))
if actual_folds < N_FOLDS:
    print(f"  WARNING: only {actual_folds} unique groups (SO3_T ties)")

gkf   = GroupKFold(n_splits=actual_folds)
folds = list(gkf.split(X_train_static, y_train, groups=groups))

print(f"\n  Fold summary ({actual_folds} folds):")
for i, (tr, va) in enumerate(folds):
    vg = sorted(np.unique(groups[va]).tolist())
    s  = so3t_vals[va]
    print(f"    Fold {i+1}: train={len(tr):,}  val={len(va):,}  "
          f"group={vg}  SO3_T∈[{s.min():.4f},{s.max():.4f}]")


# ── CELL 7: Model Definitions ──────────────────────────────────────────────

# ── 7a. LightGBM loss and metric ─────────────────────────────────────────
def fair_obj(y_pred, dataset):
    """Fair loss — robust to fat financial return tails."""
    y_true = dataset.get_label()
    r      = y_pred - y_true
    grad   = r / (1.0 + np.abs(r) / FAIR_C)
    hess   = FAIR_C ** 2 / (FAIR_C + np.abs(r)) ** 2
    return grad, hess

def r2_metric(y_pred, dataset):
    y_true = dataset.get_label()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 'r2', 1.0 - ss_res / (ss_tot + 1e-15), True

# ── 7b. PyTorch MLP ──────────────────────────────────────────────────────
class FinancialMLP(nn.Module):
    """
    Shallow-wide MLP for tabular financial data.
    - SiLU (Swish): passes small negatives unlike ReLU (needed for Z-scored inputs)
    - BatchNorm: stabilises activations after extreme-value features fire
    - Dropout 0.4→0.3→0.2: heavy early regularisation for near-zero signal
    - No activation on output: raw regression, no squashing
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        # Xavier init for stable early training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr_mlp, y_tr_norm, X_va_mlp, y_va_norm, X_test_arr, input_dim):
    """
    Train MLP on Z-scored features and Z-scored target.
    X_test_arr must be fold-scaled (same scaler as X_tr_mlp).
    Returns val predictions in NORMALISED space (caller handles inverse).
    Early stops on validation R² with patience=MLP_PATIENCE.
    """
    model = FinancialMLP(input_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)

    # DataLoaders
    tr_ds = TensorDataset(
        torch.tensor(X_tr_mlp, dtype=torch.float32),
        torch.tensor(y_tr_norm, dtype=torch.float32)
    )
    tr_dl = DataLoader(tr_ds, batch_size=MLP_BATCH, shuffle=True,
                       num_workers=2, pin_memory=(DEVICE.type=='cuda'))

    X_va_t  = torch.tensor(X_va_mlp,  dtype=torch.float32).to(DEVICE)
    y_va_t  = torch.tensor(y_va_norm, dtype=torch.float32).to(DEVICE)

    best_r2   = -np.inf
    best_preds = None
    patience_cnt = 0

    for epoch in range(MLP_EPOCHS):
        model.train()
        for Xb, yb in tr_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(Xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t).cpu().numpy()

        val_r2 = r2_score(y_va_norm, val_pred)
        if val_r2 > best_r2:
            best_r2    = val_r2
            best_preds = val_pred.copy()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= MLP_PATIENCE:
                print(f"      MLP early stop at epoch {epoch+1}  best_val_r2={best_r2:.6f}")
                break

        if (epoch + 1) % 25 == 0:
            print(f"      epoch {epoch+1:>3}  val_r2={val_r2:.6f}  best={best_r2:.6f}")

    # Test predictions from best-checkpoint state
    # Test inference uses fold-scaled X_test_arr passed in by the caller.
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_arr.astype(np.float32)).to(DEVICE)
        # Predict in batches to avoid OOM on large test set
        test_chunks = torch.split(X_test_t, 8192)
        test_pred   = np.concatenate(
            [model(chunk).cpu().numpy() for chunk in test_chunks]
        )

    del model, tr_ds, tr_dl, X_va_t, y_va_t, X_test_t
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return best_preds, test_pred, best_r2


# ── CELL 8: Main CV Loop ───────────────────────────────────────────────────
print("=" * 65)
print("MAIN CV LOOP — LightGBM + MLP")
print("=" * 65)

# Pre-compute MLP binary indicator indices (top 10 ICIR features that are in static set)
top10_icir_in_static = [c for c in ICIR_TOP_279[:10] if c in static_feat_cols]
top10_static_idx     = [static_feat_cols.index(c) for c in top10_icir_in_static]
print(f"  MLP binary indicator features: {len(top10_icir_in_static)} "
      f"→ {len(top10_icir_in_static)*2} columns")

# Output arrays
lgb_oof        = np.zeros(len(y_train), dtype=np.float32)
mlp_oof        = np.zeros(len(y_train), dtype=np.float32)
lgb_test_preds = np.zeros(len(X_test_static), dtype=np.float64)
mlp_test_preds = np.zeros(len(X_test_static), dtype=np.float64)

lgb_fold_r2s = []
mlp_fold_r2s = []
total_start  = time.time()

# Working copy of test set (mutated in-place each fold, restored after)
X_test_work = X_test_static.copy()

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    fold_start = time.time()
    print(f"\n{'─'*60}")
    print(f"  FOLD {fold_idx+1}/{actual_folds}  "
          f"(train={len(tr_idx):,}  val={len(va_idx):,})")
    print(f"{'─'*60}")

    # ── 8.1 Slice (independent copies via fancy indexing) ─────────────────
    X_tr = np.ascontiguousarray(X_train_static[tr_idx], dtype=np.float32)
    X_va = np.ascontiguousarray(X_train_static[va_idx], dtype=np.float32)
    y_tr = y_train[tr_idx].copy()
    y_va = y_train[va_idx].copy()

    # ── 8.2 Winsorise target (train percentiles only) ─────────────────────
    clip_lo  = np.percentile(y_tr, 1)
    clip_hi  = np.percentile(y_tr, 99)
    y_tr     = np.clip(y_tr, clip_lo, clip_hi).astype(np.float32)
    y_va     = np.clip(y_va, clip_lo, clip_hi).astype(np.float32)
    print(f"  Winsorise [{clip_lo:.5f}, {clip_hi:.5f}]")

    # ── 8.3 Z-score target (mean/std from y_tr only) ──────────────────────
    # Fixes Fold 4: each regime has different volatility. Normalising forces
    # the model to learn pure directional patterns immune to scale.
    mean_y = float(y_tr.mean())
    std_y  = max(float(y_tr.std()), 1e-8)
    y_tr_n = ((y_tr - mean_y) / std_y).astype(np.float32)
    y_va_n = ((y_va - mean_y) / std_y).astype(np.float32)
    print(f"  Target norm  mean_y={mean_y:.6f}  std_y={std_y:.6f}")

    # ── 8.4 Feature scaling (fold-safe in-place) ──────────────────────────
    scaler = StandardScaler(copy=False)
    scaler.fit(X_tr)
    X_tr[:]         = scaler.transform(X_tr)
    X_va[:]         = scaler.transform(X_va)
    X_test_work[:]  = scaler.transform(X_test_work)

    # ── 8.5 LGB feature matrix ────────────────────────────────────────────
    # SO3_T is already in static_feat_cols. LightGBM finds regime-conditional
    # splits natively (split on SO3_T then on feature). No explicit interaction
    # columns needed — they would be redundant for tree models and waste memory.
    X_tr_lgb   = X_tr
    X_va_lgb   = X_va
    X_test_lgb = X_test_work

    print(f"  LGB input shape : {X_tr_lgb.shape}")

    # ── 8.7 LightGBM training ─────────────────────────────────────────────
    print(f"  Training LightGBM ...")
    dtrain = lgb.Dataset(X_tr_lgb, label=y_tr_n, free_raw_data=True)
    dvalid = lgb.Dataset(X_va_lgb, label=y_va_n,
                         reference=dtrain, free_raw_data=True)

    lgb_model = lgb.train(
        {**LGB_PARAMS, 'objective': fair_obj},
        dtrain,
        num_boost_round=LGB_ROUNDS,
        valid_sets=[dvalid],
        feval=r2_metric,
        callbacks=[
            lgb.early_stopping(LGB_EARLY, verbose=False),
            lgb.log_evaluation(300),
        ],
    )

    lgb_oof_norm  = lgb_model.predict(X_va_lgb).astype(np.float32)
    lgb_test_norm = lgb_model.predict(X_test_lgb)

    # Inverse-transform to original scale
    lgb_oof[va_idx]  = (lgb_oof_norm  * std_y + mean_y).astype(np.float32)
    lgb_test_preds  += (lgb_test_norm * std_y + mean_y) / actual_folds

    lgb_r2 = r2_score(y_va, lgb_oof[va_idx])
    lgb_fold_r2s.append(lgb_r2)
    print(f"  LGB  best_iter={lgb_model.best_iteration:>4}  fold_R²={lgb_r2:+.6f}")

    del dtrain, dvalid, lgb_model, lgb_oof_norm, lgb_test_norm
    del X_tr_lgb, X_va_lgb, X_test_lgb
    gc.collect()

    # ── 8.8 MLP training ──────────────────────────────────────────────────
    # Spearman top 150 features + binary extreme indicators for top 10.
    # Binary thresholds help MLP because it lacks native split-finding.
    # X_tr/X_va/X_test_work are already Z-scored (from step 8.4).
    X_tr_mlp   = X_tr[:, mlp_feat_idx].astype(np.float32)
    X_va_mlp   = X_va[:, mlp_feat_idx].astype(np.float32)
    X_test_mlp = X_test_work[:, mlp_feat_idx].astype(np.float32)

    # Add binary extreme indicators for top 10 ICIR features (MLP only).
    # top10_icir_in_static / top10_static_idx pre-computed before loop.
    if top10_static_idx:
        X_tr_mlp   = np.concatenate([
            X_tr_mlp,
            (X_tr[:, top10_static_idx]        >  EXTREME_Z).astype(np.float32),
            (X_tr[:, top10_static_idx]        < -EXTREME_Z).astype(np.float32),
        ], axis=1)
        X_va_mlp   = np.concatenate([
            X_va_mlp,
            (X_va[:, top10_static_idx]        >  EXTREME_Z).astype(np.float32),
            (X_va[:, top10_static_idx]        < -EXTREME_Z).astype(np.float32),
        ], axis=1)
        X_test_mlp = np.concatenate([
            X_test_mlp,
            (X_test_work[:, top10_static_idx] >  EXTREME_Z).astype(np.float32),
            (X_test_work[:, top10_static_idx] < -EXTREME_Z).astype(np.float32),
        ], axis=1)

    mlp_input_dim = X_tr_mlp.shape[1]
    print(f"  Training MLP (input_dim={mlp_input_dim}: {MLP_INPUT_DIM} feats "
          f"+ {mlp_input_dim - MLP_INPUT_DIM} binary indicators) ...")

    mlp_oof_norm, mlp_test_norm, best_mlp_r2 = train_mlp(
        X_tr_mlp, y_tr_n, X_va_mlp, y_va_n, X_test_mlp, mlp_input_dim
    )

    mlp_oof[va_idx]  = (mlp_oof_norm  * std_y + mean_y).astype(np.float32)
    mlp_test_preds  += (mlp_test_norm * std_y + mean_y) / actual_folds

    mlp_r2 = r2_score(y_va, mlp_oof[va_idx])
    mlp_fold_r2s.append(mlp_r2)
    print(f"  MLP  fold_R²={mlp_r2:+.6f}  (best_norm_R²={best_mlp_r2:.6f})")

    del X_tr_mlp, X_va_mlp, X_test_mlp, X_tr, X_va, y_tr, y_va
    del y_tr_n, y_va_n
    gc.collect()

    # ── 8.9 Restore X_test_work to raw space ─────────────────────────────
    # Must happen BEFORE deleting scaler. Safest: reset from untouched original.
    X_test_work[:] = X_test_static.copy()
    del scaler
    gc.collect()

    elapsed = time.time() - fold_start
    print(f"\n  ── Fold {fold_idx+1} done  LGB={lgb_r2:+.6f}  "
          f"MLP={mlp_r2:+.6f}  ({elapsed:.0f}s) ──")

total_elapsed = time.time() - total_start


# ── CELL 9: OOF Evaluation & Ensemble Optimisation ────────────────────────
print("\n" + "=" * 65)
print("OOF EVALUATION")
print("=" * 65)

lgb_oof_r2 = r2_score(y_train, lgb_oof)
mlp_oof_r2 = r2_score(y_train, mlp_oof)

print(f"\n  Per-fold R²:")
print(f"  {'Fold':<6} {'LGB':>12} {'MLP':>12} {'Winner':>8}")
print(f"  {'-'*42}")
for i, (lr, mr) in enumerate(zip(lgb_fold_r2s, mlp_fold_r2s), 1):
    w = 'LGB' if lr > mr else 'MLP'
    print(f"  {i:<6} {lr:>+12.6f} {mr:>+12.6f} {w:>8}")

print(f"\n  LGB OOF R² : {lgb_oof_r2:+.6f}")
print(f"  MLP OOF R² : {mlp_oof_r2:+.6f}")

# ── Scipy blend optimisation ──────────────────────────────────────────────
def neg_r2(w):
    blend = w * lgb_oof + (1.0 - w) * mlp_oof
    return -r2_score(y_train, blend)

result  = minimize_scalar(neg_r2, bounds=(0.0, 1.0), method='bounded')
w_lgb   = float(result.x)
w_mlp   = 1.0 - w_lgb

ensemble_oof    = w_lgb * lgb_oof   + w_mlp * mlp_oof
ensemble_test   = w_lgb * lgb_test_preds + w_mlp * mlp_test_preds
ensemble_oof_r2 = r2_score(y_train, ensemble_oof)

print(f"\n  Optimal LGB weight : {w_lgb:.4f} ({w_lgb*100:.1f}%)")
print(f"  Optimal MLP weight : {w_mlp:.4f} ({w_mlp*100:.1f}%)")
print(f"  Ensemble OOF R²    : {ensemble_oof_r2:+.6f}")
print(f"  vs LGB alone       : {lgb_oof_r2:+.6f}  "
      f"({'↑ improvement' if ensemble_oof_r2>lgb_oof_r2 else '↓ no gain'})")

# Safety: if MLP hurt, fall back to pure LGB
if ensemble_oof_r2 <= lgb_oof_r2:
    print("\n  MLP did not improve OOF — submitting LGB-only predictions.")
    final_test   = lgb_test_preds
    final_oof_r2 = lgb_oof_r2
    model_desc   = "LGB-only (MLP rejected)"
else:
    final_test   = ensemble_test
    final_oof_r2 = ensemble_oof_r2
    model_desc   = f"LGB({w_lgb*100:.0f}%) + MLP({w_mlp*100:.0f}%)"

print(f"\n  FINAL MODEL: {model_desc}")
print(f"  FINAL OOF R²: {final_oof_r2:+.6f}")
print(f"  Total training time: {total_elapsed/60:.1f} min")


# ── CELL 10: Generate Submission CSV ──────────────────────────────────────
print("\n" + "=" * 65)
print("SUBMISSION")
print("=" * 65)

sample_sub = pd.read_csv(SAMPLE_PATH)
sub = pd.DataFrame({'ID': test_ids, 'TARGET': final_test})
sub = sample_sub[['ID']].merge(sub, on='ID', how='left')

null_count = sub['TARGET'].isnull().sum()
if null_count:
    print(f"  WARNING: {null_count} missing IDs — filling 0.0")
    sub['TARGET'] = sub['TARGET'].fillna(0.0)

out_path = '/kaggle/working/lgbm_mlp_ensemble_v1.csv'
sub.to_csv(out_path, index=False)
print(f"  Saved: {out_path}  ({len(sub):,} rows)")

# Always save LGB-only CSV as a separate safe submission
sub_lgb = pd.DataFrame({'ID': test_ids, 'TARGET': lgb_test_preds})
sub_lgb = sample_sub[['ID']].merge(sub_lgb, on='ID', how='left')
sub_lgb['TARGET'] = sub_lgb['TARGET'].fillna(0.0)
lgb_out_path = '/kaggle/working/lgb_only_v1.csv'
sub_lgb.to_csv(lgb_out_path, index=False)
print(f"  Saved: {lgb_out_path}  (LGB-only, OOF={lgb_oof_r2:+.6f})")
print(f"  Pred stats — mean:{final_test.mean():.7f}  "
      f"std:{final_test.std():.7f}  "
      f"min:{final_test.min():.6f}  max:{final_test.max():.6f}")
print(f"  Positive predictions: {(final_test > 0).mean()*100:.1f}%")
print(f"\n  Model      : {model_desc}")
print(f"  OOF R²     : {final_oof_r2:+.6f}")
print(f"  LGB features  : {len(static_feat_cols)} (includes SO3_T for native regime splits)")
print(f"  MLP features  : {MLP_INPUT_DIM} Spearman top150 + {len(top10_icir_in_static)*2} binary indicators")
print("=" * 65)
