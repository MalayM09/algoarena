# ================================================================
# CS-LGB GOLD — Cross-Sectional LGB, Gold Features Only
# ================================================================
# Motivation:
#   - cs_v1 uses all 445 features — 394 weak features add noise
#   - Gold = abs_icir >= 3, sign-consistent (ic_pos_frac=0 or 1)
#   - 51 gold features, all with stable IC direction
#   - Fewer features → less overfitting to liquid distribution
#   - Better generalisation to illiquid test assets
#
# Identical to cross_sectional_v1 except feature subset.
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
ORACLE     = os.path.join(BASE_DIR, 'outputs/submissions/exploit_v2_zero.csv')
SAMPLE_SUB = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/submissions')

TARGET_STD = 0.000948
N_FOLDS    = 5
N_EST      = 2000
LR         = 0.05
ES_ROUNDS  = 50
t0         = time.time()

def auto_scale(p):
    s = p.std()
    return p * (TARGET_STD / s) if s > 1e-10 else p

def daywise_oracle_score(pred_vec, oracle_vec, day_ids):
    ics = []
    for day in np.unique(day_ids):
        m = day_ids == day
        if m.sum() < 3: continue
        p = pred_vec[m] - pred_vec[m].mean()
        o = oracle_vec[m] - oracle_vec[m].mean()
        pn = np.linalg.norm(p); on = np.linalg.norm(o)
        if pn < 1e-12 or on < 1e-12: ics.append(0.)
        else: ics.append(float((p@o)/(pn*on)))
    return float(np.mean(ics))

# ── Load ──────────────────────────────────────────────────────────
print("=" * 60)
print("CS-LGB GOLD — Loading data")
print("=" * 60)
t1 = time.time()
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
icir  = pd.read_csv(ICIR_PATH)
print(f"  Load: {time.time()-t1:.1f}s")

# Gold features only
gold_mask  = (icir['abs_icir'] >= 3) & (icir['ic_pos_frac'].isin([0.0, 1.0]))
all_feats  = [c for c in train.columns if c not in {'ID','TARGET','CV_GROUP'}]
gold_feats = [f for f in icir[gold_mask]['feature'].tolist() if f in all_feats]
print(f"  Gold features: {len(gold_feats)} / {len(all_feats)} total")

y_train   = train['TARGET'].values.astype(np.float64)
test_ids  = test['ID'].values
train_day = train['SO3_T'].round(5).astype(str).values
test_day  = test['SO3_T'].round(5).astype(str).values

# ── CS z-score (gold features only) ──────────────────────────────
print(f"\nCS z-score normalisation ({len(gold_feats)} features)...")
t1 = time.time()
tr_raw = train[gold_feats].fillna(0).values.astype(np.float32)
te_raw = test.reindex(columns=gold_feats, fill_value=0).values.astype(np.float32)

X_train = np.zeros_like(tr_raw)
for d in np.unique(train_day):
    m = train_day == d
    x = tr_raw[m]; s = x.std(0); s[s<1e-8] = 1.0
    X_train[m] = (x - x.mean(0)) / s

X_test = np.zeros_like(te_raw)
for d in np.unique(test_day):
    m = test_day == d
    x = te_raw[m]; s = x.std(0); s[s<1e-8] = 1.0
    X_test[m] = (x - x.mean(0)) / s

del tr_raw, te_raw; gc.collect()
print(f"  Done in {time.time()-t1:.1f}s")

# ── GroupKFold on SO3_T quintiles ─────────────────────────────────
so3t_idx  = gold_feats.index('SO3_T') if 'SO3_T' in gold_feats else None
if so3t_idx is None:
    # SO3_T not in gold — use raw value for grouping
    so3t_raw = train['SO3_T'].values
else:
    so3t_raw = X_train[:, so3t_idx]

groups  = pd.qcut(pd.Series(so3t_raw), q=N_FOLDS, labels=False,
                  duplicates='drop').values.astype(np.int32)
n_folds = len(np.unique(groups))
gkf     = GroupKFold(n_splits=n_folds)
folds   = list(gkf.split(X_train, y_train, groups=groups))
print(f"\nGroupKFold: {n_folds} folds on SO3_T quintiles")

# ── Oracle ────────────────────────────────────────────────────────
sample_sub  = pd.read_csv(SAMPLE_SUB)[['ID']]
oracle_df   = sample_sub.merge(pd.read_csv(ORACLE), on='ID', how='left').fillna(0.0)
oracle_vec  = oracle_df['TARGET'].values
test_day_df = sample_sub.merge(
    pd.read_parquet(TEST_PATH, columns=['ID','SO3_T']), on='ID', how='left')
oracle_days = test_day_df['SO3_T'].round(5).astype(str).values

# ── Winsorize ─────────────────────────────────────────────────────
lo, hi  = np.percentile(y_train, 1), np.percentile(y_train, 99)
y_wins  = np.clip(y_train, lo, hi)

# ── LGB params ────────────────────────────────────────────────────
LGB_PARAMS = dict(
    objective         = 'regression',
    metric            = 'rmse',
    num_leaves        = 63,
    learning_rate     = LR,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 1,
    min_child_samples = 50,
    lambda_l1         = 0.1,
    lambda_l2         = 1.0,
    n_jobs            = -1,
    verbose           = -1,
    seed              = 42,
)

# ── Train ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TRAINING cs_v2_gold")
print("="*60)

oof_preds  = np.zeros(len(y_wins))
test_preds = np.zeros(len(X_test))
best_iters = []

for fold_idx, (tr_idx, va_idx) in enumerate(folds):
    tf = time.time()
    print(f"\n  Fold {fold_idx+1}/{n_folds}")

    dtrain = lgb.Dataset(X_train[tr_idx], label=y_wins[tr_idx], free_raw_data=True)
    dval   = lgb.Dataset(X_train[va_idx], label=y_wins[va_idx],
                         reference=dtrain, free_raw_data=True)

    model = lgb.train(
        LGB_PARAMS, dtrain,
        num_boost_round = N_EST,
        valid_sets      = [dval],
        callbacks       = [lgb.early_stopping(ES_ROUNDS, verbose=False),
                           lgb.log_evaluation(200)],
    )

    bi = model.best_iteration
    best_iters.append(bi)
    oof_preds[va_idx] = model.predict(X_train[va_idx], num_iteration=bi)
    test_preds       += model.predict(X_test, num_iteration=bi) / n_folds

    fold_r2 = r2_score(y_wins[va_idx], oof_preds[va_idx])
    print(f"  best_iter={bi}  fold_R²={fold_r2:+.6f}  ({time.time()-tf:.0f}s)")
    del dtrain, dval, model; gc.collect()

oof_r2 = r2_score(y_wins, oof_preds)
print(f"\n  OOF R²={oof_r2:+.6f}  best_iters={best_iters}")

# ── Save + score ──────────────────────────────────────────────────
scaled   = auto_scale(test_preds)
pred_df  = pd.DataFrame({'ID': test_ids, 'TARGET': scaled})
sub_df   = sample_sub.merge(pred_df, on='ID', how='left').fillna(0.0)
oracle_s = daywise_oracle_score(sub_df['TARGET'].values, oracle_vec, oracle_days)
out_path = os.path.join(OUT_DIR, 'cs_v2_gold.csv')
sub_df.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")
print(f"  oracle_score: {oracle_s:+.6f}")

# ── Blend with current best ───────────────────────────────────────
print("\n" + "="*60)
print("BLEND CHECK vs optimal_blend_v1")
print("="*60)
anchor = sample_sub.merge(
    pd.read_csv(os.path.join(OUT_DIR, 'optimal_blend_v1.csv')),
    on='ID').fillna(0)['TARGET'].values
gold_vec = sub_df['TARGET'].values

print(f"\n  optimal_blend_v1  oracle=+0.058349  LB=0.00145 (current best)")
print(f"  cs_v2_gold        oracle={oracle_s:+.6f}")
print(f"\n  Pairwise blends:")
for w in [0.10, 0.20, 0.30, 0.40, 0.50]:
    blend = w * gold_vec + (1-w) * anchor
    s = blend.std(); blend = blend*(TARGET_STD/s) if s>1e-10 else blend
    sc = daywise_oracle_score(blend, oracle_vec, oracle_days)
    flag = '  ← BEATS THRESHOLD' if sc > 0.060349 else ''
    print(f"    w_gold={w:.0%}  oracle={sc:+.6f}  delta={sc-0.058349:+.6f}{flag}")

print(f"\n  Submit threshold: +0.060349")
print(f"  Total elapsed:    {(time.time()-t0)/60:.1f} min")
print(f"\n  NOTE: OOF R² inversely correlated with LB. Use oracle_score.")
