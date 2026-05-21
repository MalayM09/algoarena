# ================================================================
# ADVERSARIAL VALIDATION — FAST VERSION
# ================================================================
# Strategy: subsample to 150k rows (75k train + 75k test) for
# speed. Scores are calibrated — relative P(test) ordering is what
# matters, not absolute values. All 3 runs finish in ~5-10 minutes.
#
# Run 1: Global adversarial (all features)     — subsample 150k
# Run 2: Low-drift features only               — subsample 150k
# Run 3: Within-regime adversarial             — per regime subsample
#
# Outputs:
#   outputs/analysis/adversarial_scores.csv   — weights per train row
#   outputs/analysis/adversarial_features.csv — feature importances
# ================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os, warnings, time
warnings.filterwarnings('ignore')

t0 = time.time()
rng = np.random.default_rng(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
DRIFT_PATH = os.path.join(BASE_DIR, 'outputs/analysis/drift_report.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# Max rows per class for each adversarial run
MAX_PER_CLASS = 75_000

print("=" * 65)
print("ADVERSARIAL VALIDATION — FAST VERSION (subsample + 3-fold)")
print("=" * 65)

# ── Load ─────────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)

feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET'}]
n_train   = len(train)
n_test    = len(test)
print(f"  Train: {n_train:,}  |  Test: {n_test:,}  |  Features: {len(feat_cols)}")

drift_df  = pd.read_csv(DRIFT_PATH)
low_drift = drift_df[drift_df['ks_statistic'] <= 0.20]['feature'].tolist()
gold_feats = drift_df[
    (drift_df['ks_statistic'] <= 0.20) &
    (drift_df['is_stable_feature'] == True)
]['feature'].tolist()
print(f"  Low-drift features: {len(low_drift)}  |  Gold features: {len(gold_feats)}")

# ── Regimes ──────────────────────────────────────────────────────
so3_train = train['SO3_T'].fillna(train['SO3_T'].median()).values
so3_test  = test['SO3_T'].fillna(test['SO3_T'].median()).values
edges     = np.percentile(so3_train, [0, 20, 40, 60, 80, 100])
edges[0] -= 1e-6;  edges[-1] += 1e-6

regime_train = np.digitize(so3_train, edges) - 1
regime_test  = np.digitize(so3_test,  edges) - 1

test_regime_pct  = {r: (regime_test  == r).mean() for r in range(5)}
train_regime_pct = {r: (regime_train == r).mean() for r in range(5)}
regime_weight_map = {r: test_regime_pct[r] / (train_regime_pct[r] + 1e-9)
                     for r in range(5)}

print(f"\n  Regime weight map (test% / train%):")
for r in range(5):
    print(f"    Regime {r}: train={train_regime_pct[r]*100:.1f}%  "
          f"test={test_regime_pct[r]*100:.1f}%  "
          f"weight={regime_weight_map[r]:.3f}x")

# ── Feature matrices (full, for final scoring) ───────────────────
print("\nPreparing feature matrices...")
X_tr_all = train[feat_cols].fillna(0).values.astype(np.float32)
X_te_all = test[feat_cols].reindex(columns=feat_cols, fill_value=0).values.astype(np.float32)

ld_idx   = [feat_cols.index(f) for f in low_drift if f in feat_cols]
X_tr_ld  = X_tr_all[:, ld_idx]
X_te_ld  = X_te_all[:, ld_idx]
print(f"  Done. X_tr_all={X_tr_all.shape}  X_te_all={X_te_all.shape}")

# ── LightGBM config ───────────────────────────────────────────────
def make_adv_model():
    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,          # reduced from 64 for speed
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

N_FOLDS = 3
skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)


def subsample_adv(X_tr, X_te, max_per_class=MAX_PER_CLASS):
    """Subsample balanced adversarial dataset; return (X, y, tr_idx, te_idx)"""
    n_tr_sub = min(len(X_tr), max_per_class)
    n_te_sub = min(len(X_te), max_per_class)
    tr_idx = rng.choice(len(X_tr), n_tr_sub, replace=False)
    te_idx = rng.choice(len(X_te), n_te_sub, replace=False)
    X = np.concatenate([X_tr[tr_idx], X_te[te_idx]], axis=0)
    y = np.concatenate([np.zeros(n_tr_sub), np.ones(n_te_sub)]).astype(np.int8)
    return X, y, tr_idx, te_idx


def run_adversarial(X_tr, X_te, tag, feat_names=None):
    """
    Full adversarial run with subsampling.
    Returns p_score: P(test) for EVERY row in X_tr (not just subsample).
    """
    X_sub, y_sub, tr_sub_idx, te_sub_idx = subsample_adv(X_tr, X_te)
    n_tr_sub = (y_sub == 0).sum()

    oof_sub = np.zeros(len(X_sub))
    feat_imp_sum = np.zeros(X_tr.shape[1])

    print(f"\n  {tag}: subsample {n_tr_sub:,} train + {(y_sub==1).sum():,} test")

    for fold, (fi, vi) in enumerate(skf.split(X_sub, y_sub)):
        t_f = time.time()
        model = make_adv_model()
        model.fit(
            X_sub[fi], y_sub[fi],
            eval_set=[(X_sub[vi], y_sub[vi])],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        oof_sub[vi] = model.predict_proba(X_sub[vi])[:, 1]
        feat_imp_sum += model.feature_importances_
        auc_fold = roc_auc_score(y_sub[vi], oof_sub[vi])
        print(f"    Fold {fold+1}  AUC={auc_fold:.4f}  "
              f"iter={model.best_iteration_}  ({time.time()-t_f:.0f}s)")

    oof_auc = roc_auc_score(y_sub, oof_sub)
    print(f"  OOF AUC ({tag}): {oof_auc:.4f}")
    if   oof_auc > 0.95: print("  → EXTREME shift")
    elif oof_auc > 0.85: print("  → STRONG shift")
    elif oof_auc > 0.70: print("  → MODERATE shift")
    else:                print("  → MILD shift")

    # Score ALL train rows using the last fold's model (fast proxy)
    # Better: retrain on full subsample, score all
    model_full = make_adv_model()
    model_full.fit(X_sub, y_sub,
                   callbacks=[lgb.log_evaluation(-1)])
    p_all_train = model_full.predict_proba(X_tr)[:, 1]

    feat_imp_df = None
    if feat_names is not None:
        feat_imp_df = pd.DataFrame({
            'feature': feat_names,
            'importance': feat_imp_sum / N_FOLDS,
        }).sort_values('importance', ascending=False)

    return p_all_train, oof_auc, feat_imp_df


# ================================================================
# RUN 1 — GLOBAL ADVERSARIAL (ALL FEATURES)
# ================================================================
print("\n" + "=" * 65)
print("RUN 1: Global adversarial — all features")
print("=" * 65)
p_global, auc_global, feat_imp = run_adversarial(
    X_tr_all, X_te_all, "all_features", feat_names=feat_cols
)
print(f"\n  Elapsed: {(time.time()-t0)/60:.1f} min")

# ================================================================
# RUN 2 — LOW-DRIFT FEATURES ONLY
# ================================================================
print("\n" + "=" * 65)
print("RUN 2: Low-drift features only (KS≤0.2)")
print("=" * 65)
p_global_ld, auc_ld, _ = run_adversarial(
    X_tr_ld, X_te_ld, "low_drift"
)
print(f"  AUC drop vs all features: {auc_global - auc_ld:.4f}")
if auc_global - auc_ld > 0.10:
    print("  → Drifted features are PRIMARY driver of shift signal.")
elif auc_global - auc_ld > 0.05:
    print("  → Drifted features contribute materially.")
else:
    print("  → Stable features alone explain most of the shift.")
print(f"\n  Elapsed: {(time.time()-t0)/60:.1f} min")

# ================================================================
# RUN 3 — WITHIN-REGIME ADVERSARIAL
# ================================================================
# regime-r train vs regime-r test (not all test)
# ================================================================
print("\n" + "=" * 65)
print("RUN 3: Within-regime adversarial")
print("=" * 65)

p_regime = np.zeros(n_train)

for r in range(5):
    tr_mask = regime_train == r
    te_mask = regime_test  == r
    n_tr_r  = tr_mask.sum()
    n_te_r  = te_mask.sum()

    if n_tr_r < 500 or n_te_r < 200:
        print(f"\n  Regime {r}: too few rows (train={n_tr_r}, test={n_te_r}) — skip")
        continue

    X_tr_r = X_tr_all[tr_mask]
    X_te_r = X_te_all[te_mask]

    p_r, auc_r, _ = run_adversarial(X_tr_r, X_te_r, f"regime_{r}")
    p_regime[tr_mask] = p_r
    print(f"  Regime {r}: n_train={n_tr_r:,}  n_test={n_te_r:,}  AUC={auc_r:.4f}")

print(f"\n  Elapsed: {(time.time()-t0)/60:.1f} min")

# ================================================================
# COMBINE INTO FINAL WEIGHTS
# ================================================================
print("\n" + "=" * 65)
print("COMPUTING FINAL WEIGHTS")
print("=" * 65)

# Map regime weight to each train row
regime_w = np.array([regime_weight_map[r] for r in regime_train])

# Normalise p_global and p_regime to [0,1] rank (percentile)
p_global_rank = pd.Series(p_global).rank(pct=True).values
p_regime_rank = pd.Series(p_regime).rank(pct=True).values

# Raw weight: product of global, regime, and regime_weight
raw_weight = p_global * p_regime * regime_w

# Normalise to mean=1.0
final_weight    = raw_weight / (raw_weight.mean() + 1e-12)
final_weight_sq = final_weight ** 2
final_weight_sq = final_weight_sq / (final_weight_sq.mean() + 1e-12)

# Cubed (more aggressive)
final_weight_cub = final_weight ** 3
final_weight_cub = final_weight_cub / (final_weight_cub.mean() + 1e-12)

# ESS (effective sample size)
ess_sq  = (final_weight_sq.sum())**2  / (final_weight_sq**2).sum()
ess_cub = (final_weight_cub.sum())**2 / (final_weight_cub**2).sum()
print(f"  ESS (squared weights) : {ess_sq:,.0f} / {n_train:,} ({ess_sq/n_train*100:.1f}%)")
print(f"  ESS (cubed  weights)  : {ess_cub:,.0f} / {n_train:,} ({ess_cub/n_train*100:.1f}%)")

# Regime distribution check after weighting
print(f"\n  Regime distribution under final_weight_sq:")
for r in range(5):
    mask = regime_train == r
    w_sum = final_weight_sq[mask].sum()
    w_pct = w_sum / final_weight_sq.sum() * 100
    te_pct = test_regime_pct[r] * 100
    print(f"    Regime {r}: weighted={w_pct:.1f}%  test={te_pct:.1f}%  "
          f"diff={w_pct-te_pct:+.1f}pp")

# ── Thresholds ───────────────────────────────────────────────────
print(f"\n  Train rows by p_global threshold:")
for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
    n = (p_global > thr).sum()
    pct = n / n_train * 100
    print(f"    p_global > {thr}: {n:,} rows ({pct:.1f}%)")

# ================================================================
# SAVE OUTPUTS
# ================================================================
print("\n" + "=" * 65)
print("SAVING OUTPUTS")
print("=" * 65)

scores_df = pd.DataFrame({
    'ID':                   train['ID'].values,
    'regime':               regime_train,
    'regime_weight':        regime_w,
    'p_global':             p_global,
    'p_global_ld':          p_global_ld,
    'p_regime':             p_regime,
    'p_global_rank':        p_global_rank,
    'p_regime_rank':        p_regime_rank,
    'final_weight':         final_weight,
    'final_weight_sq':      final_weight_sq,
    'final_weight_cub':     final_weight_cub,
})

scores_path = os.path.join(OUT_DIR, 'adversarial_scores.csv')
scores_df.to_csv(scores_path, index=False)
print(f"  Saved: {scores_path}  ({len(scores_df):,} rows)")

if feat_imp is not None:
    fimp_path = os.path.join(OUT_DIR, 'adversarial_features.csv')
    feat_imp.to_csv(fimp_path, index=False)
    print(f"  Saved: {fimp_path}")
    print(f"\n  Top 15 features driving train/test separation:")
    for _, row in feat_imp.head(15).iterrows():
        print(f"    {row['feature']:<50}  imp={row['importance']:.0f}")

total_min = (time.time() - t0) / 60
print(f"\n{'='*65}")
print(f"DONE in {total_min:.1f} minutes")
print(f"{'='*65}")

# ── Next steps summary ───────────────────────────────────────────
print(f"""
EXPERIMENTS TO RUN (priority order):

  EXP1  [safest]  fold_safe_v1 + sample_weight=final_weight_sq
         → Re-weights training so Regime 4 gets more influence
         → ESS ≈ {ess_sq:,.0f} rows (still large dataset)

  EXP2  [moderate] Keep top 20% by p_global_rank + all features
         → Rows: {(p_global_rank >= 0.80).sum():,} train rows

  EXP3  [aggressive] Keep top 10% by p_global_rank + all features
         → Rows: {(p_global_rank >= 0.90).sum():,} train rows

  EXP4  top 10% rows + 352 low-drift features
         → {(p_global_rank >= 0.90).sum():,} rows × {len(low_drift)} features

  EXP5  top 10% rows + {len(gold_feats)} gold features (sign-stable + low-drift)
         → {(p_global_rank >= 0.90).sum():,} rows × {len(gold_feats)} features

  Submit order: EXP1 → EXP3 → EXP5 (if EXP1 improves LB)
""")
