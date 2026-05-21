# ================================================================
# SUPERVISED AUTOENCODER + MLP — Jane Street 2021 1st Place Architecture
# ================================================================
# Research basis (Yirun Zhang, Jane Street Market Prediction, 2021):
#   Winner used a supervised denoising autoencoder to compress 130
#   anonymous features into a latent representation, then trained
#   an MLP on [original features + latent representation] jointly.
#
#   Key innovations:
#   1. Gaussian noise injection before encoder → forces robust features
#   2. Supervised autoencoder: reconstruction loss + prediction loss trained jointly
#   3. Skip connection: original features concat with encoded representation
#   4. Swish activations, BatchNorm, heavy Dropout
#   5. PurgedGroupTimeSeriesSplit on date groups (our CV_GROUP)
#   6. Ensemble: multiple seeds
#
# Adaptation for our problem:
#   - 445 features (vs 130 in Jane Street)
#   - TARGET = forward return (same structure)
#   - CV_GROUP in train.parquet for purged CV
#   - Full-data training at the end (no held-out val)
#
# Architecture:
#   Input (445) → Gaussian Noise → Encoder [256→128→64] → Bottleneck (64)
#   Decoder: Bottleneck → [128→256→445] → Reconstruction loss
#   MLP: concat(Input_445, Bottleneck_64) → [512→256→128→64→1]
#   Losses: alpha * MSE(prediction, target) + (1-alpha) * MSE(reconstruction, input)
#   alpha=0.8 (prediction-dominant)
#
# Run this on Kaggle (GPU available, 30GB RAM).
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
t0 = time.time()

# ── Detect environment ─────────────────────────────────────────────
IS_KAGGLE = os.path.exists('/kaggle/input')
if IS_KAGGLE:
    # Update COMP_SLUG to match the competition data folder name in /kaggle/input/
    COMP_SLUG   = 'your-competition-data'  # ← UPDATE THIS
    BASE_DIR    = f'/kaggle/input/{COMP_SLUG}'
    TRAIN_PATH  = os.path.join(BASE_DIR, 'train.parquet')
    TEST_PATH   = os.path.join(BASE_DIR, 'test.parquet')
    SAMPLE_PATH = os.path.join(BASE_DIR, 'sample_submission.csv')
    OUT_DIR     = '/kaggle/working'
    # Optional: paths to blend with existing submissions uploaded as datasets
    # Upload threeway_r30_k40_g29.csv as a Kaggle dataset named "quant-submissions"
    GRINOLD_PATH  = '/kaggle/input/quant-submissions/grinold_allday_top10_probe_005.csv'
    THREEWAY_PATH = '/kaggle/input/quant-submissions/threeway_r30_k40_g29.csv'
else:
    BASE_DIR      = '/Users/malaymishra/Desktop/quant_ml_project'
    TRAIN_PATH    = os.path.join(BASE_DIR, 'data/raw/train.parquet')
    TEST_PATH     = os.path.join(BASE_DIR, 'data/raw/test.parquet')
    SAMPLE_PATH   = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
    OUT_DIR       = os.path.join(BASE_DIR, 'outputs/submissions')
    GRINOLD_PATH  = os.path.join(BASE_DIR, 'outputs/submissions/grinold_allday_top10_probe_005.csv')
    THREEWAY_PATH = os.path.join(BASE_DIR, 'outputs/submissions/threeway_r30_k40_g29.csv')

os.makedirs(OUT_DIR, exist_ok=True)

TARGET_STD = 0.000948
N_SEEDS    = 5         # ensemble size (5 seeds × model)
NOISE_STD  = 0.1       # Gaussian noise std for encoder input
ALPHA      = 0.8       # fraction of loss on prediction (vs reconstruction)
BATCH_SIZE = 4096
LR         = 1e-3
N_EPOCHS   = 50
PATIENCE   = 5         # early stopping patience on val loss
CLIP_Z     = 5.0

print("=" * 65)
print("SUPERVISED AUTOENCODER + MLP (Jane Street 2021 1st Place)")
print("=" * 65)

# ── Load data ──────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
y_all  = train['TARGET'].values.astype(np.float32)
ids_te = test['ID'].values
print(f"  Train: {len(train):,}  Test: {len(test):,}")

# ── Features ────────────────────────────────────────────────────────
# Use ALL 445 features (not just gold top-10)
feature_cols = [c for c in train.columns
                if c not in ['ID','TARGET','CV_GROUP','SO3_T','day_id']]
print(f"  Features: {len(feature_cols)}")

# ── Per-day RANK NORMALIZATION ──────────────────────────────────────
# Why ranks instead of z-scores:
#   The KS=0.37 gap between liquid (train) and illiquid (test) asset
#   distributions means z-scores HIDE the distribution shift. A test
#   asset z-score of +2.0 gets mapped to "large positive volume" which
#   activates the same MLP neurons as liquid large-caps — but the test
#   asset is just a slightly-above-average micro-cap.
#
#   Cross-sectional rank normalization fixes this: each asset is scored
#   purely by its RELATIVE position within the same day's cross-section.
#   Ranks are scale-invariant and bounded → no distribution leakage,
#   and the MLP learns "top-tier vs bottom-tier" positioning rather
#   than absolute magnitudes.
#
#   Formula: map ranks to [-1, +1] within each day
#   rank 0 (lowest) → -1.0   rank n-1 (highest) → +1.0
from scipy.stats import rankdata

print("\nApplying per-day rank normalization to all features...")
train_X_parts = []
train_y_parts = []
train_day_parts = []

for day, grp in train.groupby('day_id'):
    if len(grp) < 5: continue
    X = grp[feature_cols].fillna(0).values.astype(np.float64)
    y = y_all[grp.index].astype(np.float32)
    n = X.shape[0]
    # Rank each column within the day, map to [-1, 1]
    # method='average' handles ties; subtract 1 so range is [0, n-1] → [−1, +1]
    X_r = np.apply_along_axis(
        lambda col: (rankdata(col, method='average') - 1) / max(n - 1, 1) * 2 - 1,
        axis=0, arr=X
    ).astype(np.float32)
    # Winsorise targets: per-day p01/p99 to handle kurtosis=48
    lo, hi = np.percentile(y, 1), np.percentile(y, 99)
    y_w = np.clip(y, lo, hi)
    train_X_parts.append(X_r)
    train_y_parts.append(y_w)
    train_day_parts.append(grp['day_id'].values)

X_train = np.vstack(train_X_parts).astype(np.float32)
y_train = np.concatenate(train_y_parts).astype(np.float32)
day_train = np.concatenate(train_day_parts)
print(f"  X_train: {X_train.shape}  [{(time.time()-t0)/60:.1f}m]")

# Test: same rank normalization within each test day
# This is valid: test assets are ranked relative to themselves (illiquid cross-section)
# The MLP sees "illiquid relative ordering" — same signal space as training
test_X_parts = []
test_day_idx = []

for day, grp in test.groupby('day_id'):
    X = grp[feature_cols].fillna(0).values.astype(np.float64)
    n = X.shape[0]
    X_r = np.apply_along_axis(
        lambda col: (rankdata(col, method='average') - 1) / max(n - 1, 1) * 2 - 1,
        axis=0, arr=X
    ).astype(np.float32)
    test_X_parts.append(X_r)
    test_day_idx.extend(grp.index.tolist())

X_test = np.vstack(test_X_parts).astype(np.float32)
# Re-align to original test order
X_test_ordered = np.zeros((len(test), len(feature_cols)), dtype=np.float32)
X_test_ordered[test_day_idx] = X_test
X_test = X_test_ordered
print(f"  X_test:  {X_test.shape}  [{(time.time()-t0)/60:.1f}m]")

# ── CV split using CV_GROUP ─────────────────────────────────────────
# Map back CV_GROUP to the processed rows
# (rows were processed in day-order; rebuild the row mapping)
cv_group_map = train[['day_id','CV_GROUP']].copy()
cv_group_map['processed_idx'] = np.arange(len(train))

# Build CV_GROUP array aligned with X_train rows (in day-order)
day_train_df = pd.DataFrame({'day_id': day_train})
day_train_df = day_train_df.merge(
    train[['day_id','CV_GROUP']].drop_duplicates('day_id'),
    on='day_id', how='left'
)
cv_groups = day_train_df['CV_GROUP'].fillna(0).values.astype(int)
unique_groups = np.sort(np.unique(cv_groups))
N_FOLDS = 5
fold_map = {g: i % N_FOLDS for i, g in enumerate(unique_groups)}
fold_arr = np.array([fold_map[g] for g in cv_groups])
print(f"  CV: {N_FOLDS}-fold via CV_GROUP  [{(time.time()-t0)/60:.1f}m]")

# ── Try importing PyTorch ──────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  PyTorch available. Device: {DEVICE}")
    HAS_TORCH = True
except ImportError:
    print("\n  PyTorch not available — falling back to sklearn MLP")
    HAS_TORCH = False

if HAS_TORCH:
    n_features = X_train.shape[1]

    # ── Model architecture ─────────────────────────────────────────
    class SupervisedAutoencoderMLP(nn.Module):
        def __init__(self, n_in, latent_dim=64, noise_std=NOISE_STD, alpha=ALPHA):
            super().__init__()
            self.noise_std = noise_std
            self.alpha = alpha

            # Encoder: n_in → 256 → 128 → latent
            self.encoder = nn.Sequential(
                nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.3),
                nn.Linear(128, latent_dim), nn.BatchNorm1d(latent_dim), nn.SiLU(),
            )

            # Decoder: latent → 128 → 256 → n_in (reconstruction)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.SiLU(),
                nn.Linear(128, 256), nn.BatchNorm1d(256), nn.SiLU(),
                nn.Linear(256, n_in),
            )

            # MLP: concat(n_in + latent) → prediction head
            self.mlp = nn.Sequential(
                nn.Linear(n_in + latent_dim, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.4),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.SiLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x, training=False):
            # Inject Gaussian noise during training (denoising)
            if training and self.noise_std > 0:
                x_noisy = x + torch.randn_like(x) * self.noise_std
            else:
                x_noisy = x

            # Encode noisy input
            latent = self.encoder(x_noisy)

            # Decode for reconstruction loss
            reconstructed = self.decoder(latent)

            # Predict from concat(clean_input, latent) — skip connection
            joint = torch.cat([x, latent], dim=1)
            pred = self.mlp(joint)

            return pred.squeeze(1), reconstructed

        def predict(self, x):
            self.eval()
            with torch.no_grad():
                pred, _ = self.forward(x, training=False)
            return pred.cpu().numpy()

    def train_one_fold(X_tr, y_tr, X_va, y_va, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Convert to tensors
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)
        X_va_t = torch.tensor(X_va, dtype=torch.float32).to(DEVICE)
        y_va_t = torch.tensor(y_va, dtype=torch.float32).to(DEVICE)

        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        model = SupervisedAutoencoderMLP(n_features).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

        best_val_loss = float('inf')
        best_state    = None
        patience_counter = 0

        for epoch in range(N_EPOCHS):
            model.train()
            total_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred, recon = model(X_batch, training=True)
                # Joint loss: alpha * prediction + (1-alpha) * reconstruction
                pred_loss  = nn.MSELoss()(pred, y_batch)
                recon_loss = nn.MSELoss()(recon, X_batch)
                loss = ALPHA * pred_loss + (1 - ALPHA) * recon_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            # Validation: joint loss on val fold (has true labels)
            model.eval()
            with torch.no_grad():
                val_pred, val_recon = model(X_va_t, training=False)
                val_pred_loss  = nn.MSELoss()(val_pred, y_va_t).item()
                val_recon_loss = nn.MSELoss()(val_recon, X_va_t).item()
                val_loss = ALPHA * val_pred_loss + (1 - ALPHA) * val_recon_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"    Early stop at epoch {epoch+1}  best_val={best_val_loss:.6f}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{N_EPOCHS}  tr={total_loss/len(loader):.6f}  val={val_loss:.6f}")

        # Restore best weights before predicting
        if best_state is not None:
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

        model.eval()
        with torch.no_grad():
            best_preds = model(X_va_t, training=False)[0].cpu().numpy()
        return best_preds, model

    # ── OOF + Test predictions ─────────────────────────────────────
    print("\nTraining Supervised Autoencoder + MLP...")
    oof_preds_all   = np.zeros(len(X_train))
    test_preds_all  = np.zeros(len(X_test))
    n_seed_models   = 0

    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed+1}/{N_SEEDS}...")
        seed_test_preds = np.zeros(len(X_test))
        seed_oof_preds  = np.zeros(len(X_train))

        for fold in range(N_FOLDS):
            tr_mask = fold_arr != fold
            va_mask = fold_arr == fold
            print(f"    Fold {fold}  (train={tr_mask.sum():,}  val={va_mask.sum():,})", end='  ')

            X_tr = X_train[tr_mask]
            y_tr = y_train[tr_mask]
            X_va = X_train[va_mask]
            y_va = y_train[va_mask]

            val_preds, model = train_one_fold(X_tr, y_tr, X_va, y_va, seed=seed*100+fold)
            seed_oof_preds[va_mask] = val_preds

            # Test predictions
            X_te_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            te_preds = model.predict(X_te_t)
            seed_test_preds += te_preds / N_FOLDS

        oof_preds_all  += seed_oof_preds
        test_preds_all += seed_test_preds
        n_seed_models  += 1

    oof_preds_all  /= N_SEEDS
    test_preds_all /= N_SEEDS
    print(f"\n  Training complete  [{(time.time()-t0)/60:.1f}m]")

else:
    # ── Fallback: sklearn MLP ──────────────────────────────────────
    from sklearn.neural_network import MLPRegressor
    print("\nFallback: sklearn MLP (small architecture, no GPU)")
    # Use a single fold for speed
    tr_mask = fold_arr != 0
    X_tr = X_train[tr_mask]; y_tr = y_train[tr_mask]
    model_sk = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu', batch_size=2048,
        learning_rate_init=1e-3, max_iter=50,
        verbose=True, random_state=42
    )
    model_sk.fit(X_tr, y_tr)
    oof_preds_all  = model_sk.predict(X_train)
    test_preds_all = model_sk.predict(X_test)
    print(f"  Done  [{(time.time()-t0)/60:.1f}m]")

# ── Per-day demeaning of predictions ──────────────────────────────
print("\nDemeaning predictions per day...")

def demean_by_day(preds, day_ids, test_df=None):
    """Subtract per-day mean from predictions (cross-sectional neutralisation)."""
    preds_dm = preds.copy()
    if test_df is not None:
        for day, grp in test_df.groupby('day_id'):
            idx = grp.index.values
            preds_dm[idx] -= preds[idx].mean()
    else:
        unique_days = np.unique(day_ids)
        for d in unique_days:
            mask = day_ids == d
            preds_dm[mask] -= preds[mask].mean()
    return preds_dm

test['_orig_idx'] = np.arange(len(test))
test_preds_dm = demean_by_day(test_preds_all, None, test)

# ── Auto-scale to target std ───────────────────────────────────────
def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return (a @ b) / d if d > 1e-10 else 0.0

# ── Save submissions ───────────────────────────────────────────────
print("\nBuilding submissions...")
sample_sub = pd.read_csv(SAMPLE_PATH)[['ID']]

def save_sub(preds, name):
    ps = auto_scale(preds)
    sub = pd.DataFrame({'ID': ids_te, 'TARGET': ps})
    sub = sample_sub.merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    print(f"  {name}  std={sub['TARGET'].std():.6f}")

# Pure autoencoder-MLP
save_sub(test_preds_dm, 'autoenc_mlp_pure')

# Blend with existing Grinold and/or current best threeway if available
# (Upload those CSVs as a Kaggle dataset named "quant-submissions" to enable blends)
m_s = auto_scale(test_preds_dm)

if os.path.exists(GRINOLD_PATH):
    gdf = pd.read_csv(GRINOLD_PATH)
    gdf = sample_sub.merge(gdf[['ID','TARGET']].rename(columns={'TARGET':'g'}),
                           on='ID', how='left').fillna(0)
    g_s = auto_scale(gdf['g'].values)
    save_sub(0.50 * m_s + 0.50 * g_s, 'autoenc_mlp50_grinold50')
    save_sub(0.70 * m_s + 0.30 * g_s, 'autoenc_mlp70_grinold30')
    print(f"  Grinold blend corr_with_mlp: {pearson_r(m_s, g_s):+.4f}")
else:
    print(f"  Grinold blend skipped (file not found: {GRINOLD_PATH})")

if os.path.exists(THREEWAY_PATH):
    tdf = pd.read_csv(THREEWAY_PATH)
    tdf = sample_sub.merge(tdf[['ID','TARGET']].rename(columns={'TARGET':'t'}),
                           on='ID', how='left').fillna(0)
    t_s = auto_scale(tdf['t'].values)
    save_sub(0.50 * m_s + 0.50 * t_s, 'autoenc_mlp50_threeway50')
    save_sub(0.30 * m_s + 0.70 * t_s, 'autoenc_mlp30_threeway70')
    print(f"  Threeway blend corr_with_mlp: {pearson_r(m_s, t_s):+.4f}")
else:
    print(f"  Threeway blend skipped (file not found: {THREEWAY_PATH})")

print(f"\n  Total elapsed: {(time.time()-t0)/60:.1f} min")
print("""
  ── SUBMISSION GUIDE ─────────────────────────────────────────
  autoenc_mlp_pure           → pure MLP prediction
  autoenc_mlp50_grinold50    → safe blend with Grinold
  autoenc_mlp50_threeway50   → safe blend with current best
  autoenc_mlp30_threeway70   → conservative (mostly threeway)

  RECOMMENDED ORDER:
    1. autoenc_mlp30_threeway70 — safest, small MLP exposure
    2. autoenc_mlp50_threeway50 — balanced
    3. autoenc_mlp_pure         — full MLP

  If MLP LB > 0: nonlinear signal exists beyond Grinold.
  If MLP LB < 0: signal is purely linear (Grinold is optimal).

  Upload to Kaggle notebook for GPU acceleration.
""")
