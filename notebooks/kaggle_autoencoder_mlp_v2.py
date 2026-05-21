# ================================================================
# SUPERVISED AUTOENCODER + MLP — v2 (GPU-Optimised)
# ================================================================
# Changes from v1:
#   1. Early stopping with break (was running all 50 epochs always)
#   2. BATCH_SIZE 1024 → 4096 (4× fewer steps/epoch)
#   3. N_SEEDS 5 → 3 (saves 40% time, still good ensemble)
#   4. Mixed precision (torch.cuda.amp) — 2× speedup on modern GPUs
#   5. pin_memory + num_workers for faster CPU→GPU transfer
#   6. Best model state_dict saved and restored before test prediction
#      (v1 used model at last epoch, not best val epoch)
#   7. X_test moved to GPU once outside the loop
#   8. Per-fold timing printed so you know progress
#   9. Per-day rank normalization replaces global QuantileTransformer
#      (QT was fitted on liquid training data and applied to illiquid
#       test data — with KS=0.37 gap this caused up to 37 percentile
#       points of systematic error per feature. Per-day ranks each
#       asset only against its same-day peers, so liquid vs illiquid
#       distribution shift cannot contaminate the normalization.)
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')
t0 = time.time()

# ── Detect environment ─────────────────────────────────────────────
IS_KAGGLE = os.path.exists('/kaggle/input')
if IS_KAGGLE:
    COMP_SLUG   = 'competitions/short-horizon-return-prediction-challenge-by-i-rage'
    BASE_DIR    = f'/kaggle/input/{COMP_SLUG}'
    TRAIN_PATH  = os.path.join(BASE_DIR, 'train.parquet')
    TEST_PATH   = os.path.join(BASE_DIR, 'test.parquet')
    SAMPLE_PATH = os.path.join(BASE_DIR, 'sample_submission.csv')
    OUT_DIR     = '/kaggle/working'
    THREEWAY_PATH = '/kaggle/input/quant-submissions/threeway_r30_k40_g29.csv'
    BEST_ENS_PATH = '/kaggle/input/quant-submissions/ens_tw35_hyb30_g35.csv'
else:
    BASE_DIR      = '/Users/malaymishra/Desktop/quant_ml_project'
    TRAIN_PATH    = os.path.join(BASE_DIR, 'data/raw/train.parquet')
    TEST_PATH     = os.path.join(BASE_DIR, 'data/raw/test.parquet')
    SAMPLE_PATH   = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
    OUT_DIR       = os.path.join(BASE_DIR, 'outputs/submissions')
    THREEWAY_PATH = os.path.join(OUT_DIR, 'threeway_r30_k40_g29.csv')
    BEST_ENS_PATH = os.path.join(OUT_DIR, 'ens_tw35_hyb30_g35.csv')

os.makedirs(OUT_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────
TARGET_STD = 0.000948
N_SEEDS    = 3        # 5 → 3: saves 40% time, still good ensemble
N_FOLDS    = 5
NOISE_STD  = 0.1
ALPHA      = 0.8      # fraction of loss on prediction vs reconstruction
BATCH_SIZE = 4096     # 1024 → 4096: 4× fewer gradient steps per epoch
LR         = 1e-3
N_EPOCHS   = 50
PATIENCE   = 5        # early stopping patience
LATENT_DIM = 64

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP    = DEVICE.type == 'cuda'   # mixed precision only on GPU
N_WORKERS  = 2 if DEVICE.type == 'cuda' else 0

print("=" * 65)
print("SUPERVISED AUTOENCODER + MLP v2 (GPU-OPTIMISED + PER-DAY RANKS)")
print("=" * 65)
print(f"  Device   : {DEVICE}")
print(f"  AMP      : {USE_AMP}")
print(f"  Seeds    : {N_SEEDS}  Folds: {N_FOLDS}  Total models: {N_SEEDS*N_FOLDS}")
print(f"  Batch    : {BATCH_SIZE}  Epochs max: {N_EPOCHS}  Patience: {PATIENCE}")

# ── Load data ──────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)

feature_cols = [c for c in train.columns
                if c not in ['ID', 'TARGET', 'CV_GROUP', 'SO3_T', 'day_id']]
print(f"  Train: {len(train):,}  Test: {len(test):,}  Features: {len(feature_cols)}")

# ── Per-day rank normalization ─────────────────────────────────────
# Why NOT global QuantileTransformer:
#   QT is fit on liquid training data then applied to illiquid test
#   data. With KS=0.37 distribution gap, an illiquid asset at the
#   75th percentile of its OWN distribution might be mapped to the
#   55th percentile of the liquid distribution — up to 37 pct-point
#   error per feature. The MLP learns the wrong cross-sectional
#   signal for every test asset.
#
# Fix: rank each asset against its same-day peers only.
#   Training day  → liquid assets ranked vs liquid   (correct)
#   Test day      → illiquid assets ranked vs illiquid (correct)
#   Result mapped to [-1, +1]: rank 0 → -1, rank n-1 → +1
print("\nApplying per-day rank normalization...")
y_train_raw = train['TARGET'].values.astype(np.float32)
day_train   = train['day_id'].values

def rank_by_day(df, feature_cols):
    """Rank each feature within each day, map to [-1, +1]."""
    n_rows = len(df)
    n_feat = len(feature_cols)
    X_out  = np.empty((n_rows, n_feat), dtype=np.float32)
    idx_out = np.empty(n_rows, dtype=np.int64)
    pos = 0
    for day, grp in df.groupby('day_id'):
        X = grp[feature_cols].fillna(0).values.astype(np.float64)
        n = X.shape[0]
        idx = grp.index.values
        if n < 2:
            X_r = np.zeros((n, n_feat), dtype=np.float32)
        else:
            # rank within day → [-1, +1]; average handles ties
            X_r = np.apply_along_axis(
                lambda col: (rankdata(col, method='average') - 1) / (n - 1) * 2 - 1,
                axis=0, arr=X
            ).astype(np.float32)
        X_out[pos:pos+n]  = X_r
        idx_out[pos:pos+n] = idx
        pos += n
    # reorder to match original df index
    result = np.empty((n_rows, n_feat), dtype=np.float32)
    result[idx_out] = X_out
    return result

X_train = rank_by_day(train, feature_cols)
print(f"  Train done  [{(time.time()-t0)/60:.1f}m]")
X_test  = rank_by_day(test, feature_cols)
print(f"  Test done   [{(time.time()-t0)/60:.1f}m]")

# Winsorise targets (kurtosis=48 → MSE explodes without this)
lo, hi  = np.percentile(y_train_raw, 1), np.percentile(y_train_raw, 99)
y_train = np.clip(y_train_raw, lo, hi).astype(np.float32)
print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}  [{(time.time()-t0)/60:.1f}m]")

# ── CV split via CV_GROUP ──────────────────────────────────────────
print("\nMapping CV Groups...")
day_cv = (train[['day_id', 'CV_GROUP']]
          .drop_duplicates('day_id')
          .set_index('day_id')['CV_GROUP']
          .to_dict())
cv_groups    = np.array([day_cv.get(d, 0) for d in day_train], dtype=int)
unique_groups = np.sort(np.unique(cv_groups))
fold_map     = {g: i % N_FOLDS for i, g in enumerate(unique_groups)}
fold_arr     = np.array([fold_map[g] for g in cv_groups], dtype=int)
print(f"  {len(unique_groups)} CV groups → {N_FOLDS} folds")

# ── Model ──────────────────────────────────────────────────────────
class SupervisedAutoencoderMLP(nn.Module):
    def __init__(self, n_in, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, latent_dim), nn.BatchNorm1d(latent_dim), nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.SiLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.SiLU(),
            nn.Linear(256, n_in),
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_in + latent_dim, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, training=False):
        x_noisy = x + torch.randn_like(x) * NOISE_STD if training else x
        latent   = self.encoder(x_noisy)
        recon    = self.decoder(latent)
        pred     = self.mlp(torch.cat([x, latent], dim=1))
        return pred.squeeze(1), recon


# ── Train one fold ─────────────────────────────────────────────────
def train_one_fold(X_tr, y_tr, X_va, y_va, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         drop_last=True, pin_memory=(DEVICE.type == 'cuda'),
                         num_workers=N_WORKERS)

    # Move validation to GPU once
    X_va_t = torch.from_numpy(X_va).to(DEVICE)
    y_va_t = torch.from_numpy(y_va).to(DEVICE)

    model    = SupervisedAutoencoderMLP(X_tr.shape[1]).to(DEVICE)
    opt      = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched    = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                     factor=0.5, patience=2)
    scaler   = GradScaler(enabled=USE_AMP)
    mse_loss = nn.MSELoss()

    best_val   = float('inf')
    best_state = None
    patience_c = 0

    for epoch in range(N_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                pred, recon = model(xb, training=True)
                loss = ALPHA * mse_loss(pred, yb) + (1 - ALPHA) * mse_loss(recon, xb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        model.eval()
        with torch.no_grad(), autocast(enabled=USE_AMP):
            val_pred, _ = model(X_va_t, training=False)
            val_loss    = mse_loss(val_pred.float(), y_va_t.float()).item()

        sched.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
            if patience_c >= PATIENCE:
                print(f"      early stop epoch {epoch+1}  best_val={best_val:.6f}")
                break

    # Restore best weights for test prediction
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad(), autocast(enabled=USE_AMP):
        val_preds_final = model(X_va_t, training=False)[0].float().cpu().numpy()

    return val_preds_final, model


# ── Training loop ──────────────────────────────────────────────────
n_features   = X_train.shape[1]
oof_preds    = np.zeros(len(X_train), dtype=np.float32)
test_preds   = np.zeros(len(X_test),  dtype=np.float32)

# Move test tensor to GPU once (reused across all folds/seeds)
X_te_t = torch.from_numpy(X_test).to(DEVICE)

print(f"\nTraining {N_SEEDS} seeds × {N_FOLDS} folds = {N_SEEDS*N_FOLDS} models...")
t_train = time.time()

for seed in range(N_SEEDS):
    print(f"\n  Seed {seed+1}/{N_SEEDS}  [{(time.time()-t0)/60:.1f}m elapsed]")
    seed_test = np.zeros(len(X_test), dtype=np.float32)

    for fold in range(N_FOLDS):
        t_fold = time.time()
        tr = fold_arr != fold
        va = fold_arr == fold
        print(f"    Fold {fold}  train={tr.sum():,}  val={va.sum():,}", end='  ')

        val_p, model = train_one_fold(
            X_train[tr], y_train[tr],
            X_train[va], y_train[va],
            seed=seed * 100 + fold
        )

        oof_preds[va] += val_p / N_SEEDS

        # Test predictions — model already has best weights restored
        with torch.no_grad(), autocast(enabled=USE_AMP):
            te_p = model(X_te_t, training=False)[0].float().cpu().numpy()
        seed_test += te_p / N_FOLDS

        print(f"[{(time.time()-t_fold)/60:.1f}m]")

    test_preds += seed_test / N_SEEDS

print(f"\n  Training complete  [{(time.time()-t_train)/60:.1f}m]")

# ── Helpers ────────────────────────────────────────────────────────
def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d > 1e-12 else 0.0

sample_sub = pd.read_csv(SAMPLE_PATH)[['ID']]

def save_sub(preds, name):
    ps  = auto_scale(preds)
    sub = pd.DataFrame({'ID': test['ID'].values, 'TARGET': ps})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    print(f"  {name}  std={sub['TARGET'].std():.6f}")
    return sub['TARGET'].values

# ── Save submissions ───────────────────────────────────────────────
print("\nBuilding submissions...")
mlp_s = save_sub(test_preds, 'autoenc_mlp_quantile_v2')

# Blend with best existing submission if available
for path, label in [(BEST_ENS_PATH, 'ens_tw35_hyb30_g35'),
                    (THREEWAY_PATH, 'threeway_r30_k40_g29')]:
    if os.path.exists(path):
        ref = (sample_sub
               .merge(pd.read_csv(path)[['ID','TARGET']].rename(columns={'TARGET':'r'}),
                      on='ID', how='left')
               .fillna(0.0)['r'].values)
        ref_s = auto_scale(ref)
        print(f"  MLP ↔ {label}: corr={pearson_r(mlp_s, ref_s):+.4f}")
        save_sub(0.30 * mlp_s + 0.70 * ref_s, f'mlp30_{label[:12]}70')
        save_sub(0.50 * mlp_s + 0.50 * ref_s, f'mlp50_{label[:12]}50')
    else:
        print(f"  Blend skipped — {path} not found")

print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
print("""
── SUBMISSION GUIDE ──────────────────────────────────────────
  autoenc_mlp_quantile_v2   → pure MLP result
  mlp30_ens_tw35_hy70       → 30% MLP + 70% best ensemble
  mlp50_ens_tw35_hy50       → 50/50 blend

  If MLP LB > 0: nonlinear signal exists. Blend aggressively.
  If MLP LB < 0: signal is linear — Grinold/kernel is optimal.

  RECOMMENDED: submit mlp30_*70 first (safest blend).
""")
