# ================================================================
# ILLIQUID-TUNED GRINOLD — CV_GROUP-aware IC computation
# ================================================================
# Key insight from CV_GROUP analysis:
#   The 70 CV_GROUPs are asset-level partitions. Some groups have
#   BookShape close to test assets (illiquid-like), others are far.
#
#   Computing IC across ALL groups dilutes the signal for illiquid
#   assets. Features work DIFFERENTLY for liquid vs illiquid assets:
#
#   Price_LagT3:        IC_close=-0.026  IC_far=-0.001  ratio=28x
#   S03_A07_A05_V09:    IC_close=-0.015  IC_far=+0.009  REVERSES
#
# Fix: compute IC using ONLY CV_GROUPs whose BookShape is closest
#   to the test asset BookShape distribution. These ICs transfer
#   better to illiquid test assets.
#
# Also: exclude / correct features that reverse sign between
#   liquid and illiquid populations.
#
# Generates:
#   ilgrin_close20.csv   — IC from 20 groups closest to test
#   ilgrin_close30.csv   — IC from 30 groups closest to test
#   ilgrin_noreverse.csv — Top-10 IC but exclude S03_A07_A05_V09
#   ilgrin_allgold.csv   — All gold features with close-group IC
#   ilgrin_best_ens.csv  — Best variant blended with current best
# ================================================================

import os, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR    = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH  = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH   = os.path.join(BASE_DIR, 'data/raw/test.parquet')
SAMPLE_PATH = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
ICIR_PATH   = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
BEST_ENS    = os.path.join(BASE_DIR, 'outputs/submissions/oracle_weighted_top10.csv')
OUT_DIR     = os.path.join(BASE_DIR, 'outputs/submissions')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_STD = 0.000948
CLIP_Z     = 5.0

print("=" * 65)
print("ILLIQUID-TUNED GRINOLD — CV_GROUP-aware IC")
print("=" * 65)

# ── Load data ──────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH).reset_index(drop=True)
test  = pd.read_parquet(TEST_PATH).reset_index(drop=True)
train['day_id'] = train['SO3_T'].round(5).astype(str)
test['day_id']  = test['SO3_T'].round(5).astype(str)
train_days = set(train['day_id'].unique())
y_train    = train['TARGET'].values.astype(np.float64)
test_ids   = test['ID'].values
sample_sub = pd.read_csv(SAMPLE_PATH)[['ID']]
n_test     = len(test)
print(f"  Train: {len(train):,}  Test: {n_test:,}")

# ── All gold features ──────────────────────────────────────────────
icir_df   = pd.read_csv(ICIR_PATH)
gold_mask = ((icir_df['abs_icir'] >= 3) &
             ((icir_df['ic_pos_frac'] == 0.0) | (icir_df['ic_pos_frac'] == 1.0)))
gold_df   = icir_df[gold_mask].sort_values('abs_icir', ascending=False)
all_gold  = [f for f in gold_df['feature'].tolist() if f in train.columns]
top10     = all_gold[:10]
ic_longrun = np.array([gold_df.set_index('feature')['mean_ic'][f] for f in top10])
print(f"  Gold features: {len(all_gold)}  Top-10: {top10[:3]}...")

# ── BookShape proxy ────────────────────────────────────────────────
b_near = [c for c in train.columns if 'Lag' not in c and
          any(f'_B0{i}' in c for i in range(1, 6))]
b_far  = [c for c in train.columns if 'Lag' not in c and
          any(f'_B{i}' in c for i in ['06','07','08','09','10'])]
train['bookshape'] = (train[b_near].fillna(0).sum(1) -
                      train[b_far].fillna(0).sum(1)).astype(np.float64)
test['bookshape']  = (test[b_near].fillna(0).sum(1) -
                      test[b_far].fillna(0).sum(1)).astype(np.float64)

test_bs_median = test['bookshape'].median()
print(f"  Test BookShape median: {test_bs_median:.0f}")

# ── Identify CV_GROUPs by BookShape proximity to test ─────────────
group_bs = train.groupby('CV_GROUP')['bookshape'].median().to_dict()
sorted_groups = sorted(group_bs.items(), key=lambda x: abs(x[1] - test_bs_median))
print(f"\n  CV_GROUPs by BookShape distance to test:")
for g, bs in sorted_groups[:8]:
    print(f"    CV_GROUP={g:2d}: BS={bs:.0f}  dist={abs(bs-test_bs_median):.0f}")

# ── Helpers ────────────────────────────────────────────────────────
def zscore_fit(X, clip=CLIP_Z):
    m = X.mean(0); s = X.std(0)
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip), m, s

def zscore_apply(X, m, s, clip=CLIP_Z):
    s = np.where(s < 1e-8, 1.0, s)
    return np.clip((X - m) / s, -clip, clip)

def auto_scale(p, std=TARGET_STD):
    s = p.std()
    return p * (std / s) if s > 1e-10 else p

def pearson_r(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d > 1e-12 else 0.0

# ── Step 1: Compute per-feature IC for each n_close variant ───────
def compute_group_ic(features, n_close):
    """Compute mean IC for features using n_close CV_GROUPs closest to test."""
    close_groups = set(g for g, _ in sorted_groups[:n_close])
    subset = train[train['CV_GROUP'].isin(close_groups)]

    ic_per_feat = np.zeros(len(features))
    for j, feat in enumerate(features):
        x = subset[feat].fillna(0).values.astype(np.float64)
        y = y_train[subset.index]
        x = x - x.mean(); y = y - y.mean()
        d = np.linalg.norm(x) * np.linalg.norm(y)
        ic_per_feat[j] = float(x @ y / d) if d > 1e-12 else 0.0
    return ic_per_feat

print("\nComputing close-group ICs...")
ic_close20  = compute_group_ic(top10, 20)
ic_close30  = compute_group_ic(top10, 30)
ic_all_gold_close20 = compute_group_ic(all_gold[:20], 20)

# Remove feature that reverses sign between liquid/illiquid
# S03_A07_A05_V09 has IC_close=-0.015, IC_far=+0.009 → reverses
reversing_feat = 'S03_A07_A05_V09'
top10_noreverse = [f for f in top10 if f != reversing_feat]
ic_noreverse    = compute_group_ic(top10_noreverse, 20)

print(f"\n  {'Feature':<45}  {'IC_longrun':>10}  {'IC_close20':>11}  {'IC_close30':>11}")
print("  " + "-" * 80)
for j, feat in enumerate(top10):
    marker = " ← REVERSES" if feat == reversing_feat else ""
    print(f"  {feat:<45}  {ic_longrun[j]:+.5f}    {ic_close20[j]:+.5f}     {ic_close30[j]:+.5f}{marker}")

# Key diagnostic: sign consistency
print(f"\n  Sign consistent (longrun vs close20): "
      f"{sum(np.sign(ic_longrun)==np.sign(ic_close20))}/{len(top10)} features")
print(f"  Sign of reverser in close20: {np.sign(ic_close20[list(top10).index(reversing_feat)]):.0f} "
      f"  in longrun: {np.sign(ic_longrun[list(top10).index(reversing_feat)]):.0f}")

# ── Step 2: Generate predictions for each IC variant ──────────────
print("\nGenerating predictions...")

def score_grinold(feats, ic_vec, label=""):
    preds = np.zeros(n_test)
    for day, grp_te in test.groupby('day_id'):
        te_idx   = grp_te.index.values
        X_te_raw = grp_te[feats].fillna(0).values.astype(np.float64)
        if day in train_days:
            grp_tr   = train[train['day_id'] == day]
            X_tr_raw = grp_tr[feats].fillna(0).values.astype(np.float64)
            _, m, s  = zscore_fit(X_tr_raw)
            X_te_z   = zscore_apply(X_te_raw, m, s)
        else:
            X_te_z, _, _ = zscore_fit(X_te_raw)
        pred = X_te_z @ ic_vec
        pred -= pred.mean()
        preds[te_idx] = pred
    return preds

preds_longrun    = score_grinold(top10, ic_longrun,     "longrun")
preds_close20    = score_grinold(top10, ic_close20,     "close20")
preds_close30    = score_grinold(top10, ic_close30,     "close30")
preds_noreverse  = score_grinold(top10_noreverse, ic_noreverse, "noreverse")
preds_allgold20  = score_grinold(all_gold[:20], ic_all_gold_close20, "allgold_close20")
print("  Done.")

# ── Step 3: Load best ensemble ─────────────────────────────────────
best_raw = (sample_sub
            .merge(pd.read_csv(BEST_ENS)[['ID','TARGET']]
                   .rename(columns={'TARGET':'b'}), on='ID', how='left')
            .fillna(0.0)['b'].values)
best_s = auto_scale(best_raw)
grinold_s = auto_scale(preds_longrun)

# ── Step 4: Oracle day-wise scoring ───────────────────────────────
print("\nComputing oracle day-wise scores...")
test_for_days = pd.read_parquet(TEST_PATH)[['ID','SO3_T']].reset_index(drop=True)
test_for_days['day_id'] = test_for_days['SO3_T'].round(5).astype(str)
day_map = test_for_days.set_index('ID')['day_id']
sample_sub['day_id'] = sample_sub['ID'].map(day_map)
day_ids = sample_sub['day_id'].values

oracle_df = sample_sub.merge(
    pd.read_csv(os.path.join(BASE_DIR,'outputs/submissions/exploit_v2_zero.csv'))
    [['ID','TARGET']].rename(columns={'TARGET':'oracle'}), on='ID', how='left').fillna(0.0)
oracle_vec = oracle_df['oracle'].values.astype(np.float64)

def daywise_score(pred):
    p = auto_scale(pred)
    corrs = []
    for d in np.unique(day_ids):
        m = day_ids == d
        if m.sum() < 3: continue
        pi = p[m] - p[m].mean(); o = oracle_vec[m] - oracle_vec[m].mean()
        dp = np.linalg.norm(pi); do = np.linalg.norm(o)
        corrs.append(float(pi@o/(dp*do)) if dp>1e-12 and do>1e-12 else 0.0)
    return float(np.mean(corrs))

# ── Step 5: Save and report ────────────────────────────────────────
print(f"\n{'Variant':<22}  {'oracle_score':>13}  {'corr_vs_grinold':>16}  {'corr_vs_best':>13}")
print("-" * 72)

variants = {
    'ilgrin_longrun':    preds_longrun,
    'ilgrin_close20':    preds_close20,
    'ilgrin_close30':    preds_close30,
    'ilgrin_noreverse':  preds_noreverse,
    'ilgrin_allgold20':  preds_allgold20,
}
saved = {}
for vname, preds in variants.items():
    ps      = auto_scale(preds)
    osc     = daywise_score(preds)
    c_grin  = pearson_r(ps, grinold_s)
    c_best  = pearson_r(ps, best_s)
    sub     = pd.DataFrame({'ID': test_ids, 'TARGET': ps})
    sub     = sample_sub[['ID']].merge(sub, on='ID', how='left').fillna(0.0)
    sub.to_csv(os.path.join(OUT_DIR, f'{vname}.csv'), index=False)
    saved[vname] = ps
    print(f"  {vname:<22}  {osc:+.6f}      {c_grin:+.6f}      {c_best:+.6f}")

# Reference scores
print(f"\n  {'oracle_weighted_top10 (LB=0.00143)':<22}  "
      f"{daywise_score(best_raw):+.6f}  (current best)")
print(f"  {'cs_w20_ow80':<22}  "
      f"{daywise_score(pd.read_csv(os.path.join(BASE_DIR,'outputs/submissions/cs_w20_ow80.csv'))['TARGET'].values):+.6f}")

# ── Step 6: Blend best variant with best ensemble ─────────────────
print("\nBuilding blends...")
best_variant = max(saved, key=lambda k: daywise_score(saved[k]))
print(f"  Best variant: {best_variant}")

for vname in ['ilgrin_close20', 'ilgrin_noreverse', 'ilgrin_allgold20']:
    rs = saved[vname]
    for w in [0.20, 0.30, 0.40]:
        blend   = (1-w)*best_s + w*rs
        blend_s = auto_scale(blend)
        osc     = daywise_score(blend)
        name    = f'{vname}_w{int(w*100)}_ens{int((1-w)*100)}'
        sub     = pd.DataFrame({'ID': test_ids, 'TARGET': blend_s})
        sub     = sample_sub[['ID']].merge(sub, on='ID', how='left').fillna(0.0)
        sub.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
        print(f"  {name:<55}  oracle={osc:+.6f}")

print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
