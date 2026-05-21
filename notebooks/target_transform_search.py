# ================================================================
# TARGET TRANSFORM SEARCH
# ================================================================
# Goal: find the transformation f(features) such that
#   TARGET * f(features)
# has the MINIMUM variance / fewest unique values.
#
# Hypothesis (confirmed): TARGET = tick_move / price
#   → TARGET * price = tick_move  (discrete, few levels)
#   → TARGET * price / tick_normalizer = integer tick count
#
# Search space:
#   Phase 1 (single): TARGET * f,  TARGET / f         (all 445 features)
#   Phase 2 (pairs):  TARGET * f1 / f2,               (top-50 × all 445)
#                     TARGET * f1 * f2
#   Phase 3 (triples): top-20 pairs × best singles    (pruned)
#
# Scoring metrics (lower = better for variance reduction):
#   1. n_unique_r5  : unique values after rounding to 5 decimal places
#   2. std          : standard deviation of transformed target
#   3. grid_score   : how tightly values snap to a uniform grid
#                     (0=perfect grid, 1=random)
#   4. iqr_ratio    : IQR / std  (high = leptokurtic / discrete)
# ================================================================

import os
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train-001.parquet')
OUT_DIR    = os.path.join(BASE_DIR, 'outputs/analysis')
os.makedirs(OUT_DIR, exist_ok=True)

ROUND_DP   = 5          # decimal places for n_unique scoring
TOP_K_P1   = 50         # top single-transform features to carry into Phase 2
TOP_K_P2   = 20         # top pairs to carry into Phase 3
SAMPLE_N   = 200_000    # subsample for speed (Phase 1/2); full for Phase 3

print("=" * 65)
print("TARGET TRANSFORM SEARCH")
print("=" * 65)

# ── Load ─────────────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_parquet(TRAIN_PATH)
feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET'}]
y_full = train['TARGET'].values
n_full = len(y_full)

# Subsample for speed in Phases 1/2
rng = np.random.default_rng(42)
sub_idx = rng.choice(n_full, min(SAMPLE_N, n_full), replace=False)
y_sub = y_full[sub_idx]

print(f"  Full dataset : {n_full:,} rows")
print(f"  Subsample    : {len(y_sub):,} rows")
print(f"  TARGET std   : {y_full.std():.6f}")
print(f"  TARGET unique: {len(np.unique(np.round(y_full, ROUND_DP))):,}")

# ── Scoring function ─────────────────────────────────────────────
def score_transform(t):
    """
    Lower is better for all metrics.
    Returns dict: n_unique, std, grid_score, composite.

    grid_score: after estimating the best-fit grid step,
    measure the mean absolute deviation from the nearest grid point.
    Normalized to std → 0=perfect discrete, 1=continuous.
    """
    t = t[np.isfinite(t)]
    if len(t) < 100:
        return dict(n_unique=9e9, std=9e9, grid_score=9e9, composite=9e9)

    t_round = np.round(t, ROUND_DP)
    n_unique = len(np.unique(t_round))
    std_t    = float(np.std(t))

    # Grid score: estimate step as median gap between sorted unique values
    uniq = np.sort(np.unique(t_round))
    if len(uniq) < 2:
        grid_score = 0.0
    else:
        gaps = np.diff(uniq)
        step = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 1.0
        if step < 1e-12:
            grid_score = 0.0
        else:
            nearest = np.round(t / step) * step
            mad_grid = float(np.mean(np.abs(t - nearest)))
            grid_score = mad_grid / (std_t + 1e-12)

    # Composite: we want n_unique low and grid_score low
    # Normalise n_unique against baseline (TARGET itself)
    composite = n_unique / 1000.0 + grid_score * 10

    return dict(n_unique=n_unique, std=std_t, grid_score=grid_score,
                composite=composite)


# Baseline: raw TARGET
baseline = score_transform(y_sub)
print(f"\n  Baseline TARGET:  n_unique={baseline['n_unique']:,}  "
      f"std={baseline['std']:.6f}  grid={baseline['grid_score']:.4f}  "
      f"composite={baseline['composite']:.4f}")


# ── Helpers ──────────────────────────────────────────────────────
def get_feat_sub(col):
    """Get subsampled feature values, fillna with median."""
    v = train[col].values[sub_idx].astype(np.float64)
    med = np.nanmedian(v)
    v[~np.isfinite(v)] = med
    return v


def safe_multiply(y, f, eps=1e-10):
    result = y * f
    return result


def safe_divide(y, f, eps=1e-10):
    denom = f.copy()
    denom[np.abs(denom) < eps] = eps
    return y / denom


# ================================================================
# PHASE 1 — SINGLE FEATURE TRANSFORMS
# ================================================================
print("\n" + "=" * 65)
print("PHASE 1: Single-feature transforms  (all 445 features × 2 ops)")
print("=" * 65)

ops_p1 = {
    'mul':      lambda y, f: safe_multiply(y, f),
    'div':      lambda y, f: safe_divide(y, f),
    'mul_abs':  lambda y, f: safe_multiply(y, np.abs(f)),
    'div_abs':  lambda y, f: safe_divide(y, np.abs(f)),
    'mul_sq':   lambda y, f: safe_multiply(y, f ** 2),
    'mul_sqrt': lambda y, f: safe_multiply(y, np.sqrt(np.abs(f))),
    'div_sqrt': lambda y, f: safe_divide(y, np.sqrt(np.abs(f)) + 1e-10),
}

p1_rows = []
for col in feat_cols:
    f = get_feat_sub(col)
    for op_name, op_fn in ops_p1.items():
        try:
            t = op_fn(y_sub, f)
            s = score_transform(t)
            p1_rows.append({
                'transform': f'TARGET_{op_name}_{col}',
                'feat1': col, 'feat2': None,
                'op': op_name,
                **s
            })
        except Exception:
            pass

p1_df = pd.DataFrame(p1_rows).sort_values('composite')
print(f"\n  Evaluated {len(p1_df):,} single transforms")
print(f"\n  TOP 30 (by composite score):")
print(f"  {'Transform':<55}  {'n_unique':>8}  {'std':>10}  {'grid':>8}  {'comp':>8}")
print("  " + "-" * 95)
for _, row in p1_df.head(30).iterrows():
    name = row['transform'][:54]
    print(f"  {name:<55}  {row['n_unique']:>8,}  {row['std']:>10.6f}  "
          f"{row['grid_score']:>8.4f}  {row['composite']:>8.4f}")


# ================================================================
# PHASE 2 — PAIR TRANSFORMS (top-K1 from Phase 1)
# ================================================================
print("\n" + "=" * 65)
print(f"PHASE 2: Pair transforms  (top-{TOP_K_P1} single × all 445 features)")
print("=" * 65)

# Best single features (distinct) to use as "base" transformers
top_p1_feats = p1_df.drop_duplicates('feat1').head(TOP_K_P1)['feat1'].tolist()

ops_p2 = {
    'mul_div':  lambda y, f1, f2: safe_divide(safe_multiply(y, f1), f2),
    'div_mul':  lambda y, f1, f2: safe_multiply(safe_divide(y, f1), f2),
    'mul_mul':  lambda y, f1, f2: safe_multiply(safe_multiply(y, f1), f2),
    'div_div':  lambda y, f1, f2: safe_divide(safe_divide(y, f1), f2),
    'mul_div_abs': lambda y, f1, f2: safe_divide(safe_multiply(y, np.abs(f1)), np.abs(f2) + 1e-10),
}

p2_rows = []
for col1 in top_p1_feats:
    f1 = get_feat_sub(col1)
    for col2 in feat_cols:
        if col2 == col1:
            continue
        f2 = get_feat_sub(col2)
        for op_name, op_fn in ops_p2.items():
            try:
                t = op_fn(y_sub, f1, f2)
                s = score_transform(t)
                if s['composite'] < baseline['composite']:   # only keep if better than raw
                    p2_rows.append({
                        'transform': f'TARGET_{op_name}_{col1}_{col2}',
                        'feat1': col1, 'feat2': col2,
                        'op': op_name,
                        **s
                    })
            except Exception:
                pass

p2_df = pd.DataFrame(p2_rows).sort_values('composite') if p2_rows else pd.DataFrame()
print(f"\n  Evaluated {TOP_K_P1} × {len(feat_cols)} × {len(ops_p2)} = "
      f"{TOP_K_P1 * len(feat_cols) * len(ops_p2):,} pair transforms")
print(f"  Survivors (better than baseline): {len(p2_df):,}")

if len(p2_df) > 0:
    print(f"\n  TOP 30 PAIRS:")
    print(f"  {'Transform':<70}  {'n_unique':>8}  {'std':>10}  {'grid':>8}  {'comp':>8}")
    print("  " + "-" * 110)
    for _, row in p2_df.head(30).iterrows():
        name = row['transform'][:69]
        print(f"  {name:<70}  {row['n_unique']:>8,}  {row['std']:>10.6f}  "
              f"{row['grid_score']:>8.4f}  {row['composite']:>8.4f}")


# ================================================================
# PHASE 3 — TRIPLE TRANSFORMS (top-K2 pairs × top-K1 singles)
# ================================================================
print("\n" + "=" * 65)
print(f"PHASE 3: Triple transforms  (top-{TOP_K_P2} pairs × top-{TOP_K_P1} singles)")
print("=" * 65)

ops_p3 = {
    'mul_div_div': lambda y, f1, f2, f3: safe_divide(safe_divide(safe_multiply(y, f1), f2), f3),
    'mul_mul_div': lambda y, f1, f2, f3: safe_divide(safe_multiply(safe_multiply(y, f1), f2), f3),
    'div_div_mul': lambda y, f1, f2, f3: safe_multiply(safe_divide(safe_divide(y, f1), f2), f3),
    'div_mul_div': lambda y, f1, f2, f3: safe_divide(safe_multiply(safe_divide(y, f1), f2), f3),
}

p3_rows = []
if len(p2_df) > 0:
    top_p2 = p2_df.head(TOP_K_P2)
    for _, p2row in top_p2.iterrows():
        f1 = get_feat_sub(p2row['feat1'])
        f2 = get_feat_sub(p2row['feat2'])

        for col3 in top_p1_feats:
            if col3 in {p2row['feat1'], p2row['feat2']}:
                continue
            f3 = get_feat_sub(col3)
            for op_name, op_fn in ops_p3.items():
                try:
                    t = op_fn(y_sub, f1, f2, f3)
                    s = score_transform(t)
                    best_p2 = p2_df['composite'].min()
                    if s['composite'] < best_p2:   # only keep if better than best pair
                        p3_rows.append({
                            'transform': f'TARGET_{op_name}_{p2row["feat1"]}_{p2row["feat2"]}_{col3}',
                            'feat1': p2row['feat1'], 'feat2': p2row['feat2'], 'feat3': col3,
                            'op': op_name,
                            **s
                        })
                except Exception:
                    pass

p3_df = pd.DataFrame(p3_rows).sort_values('composite') if p3_rows else pd.DataFrame()
print(f"\n  Survivors (better than best pair): {len(p3_df):,}")
if len(p3_df) > 0:
    print(f"\n  TOP 20 TRIPLES:")
    print(f"  {'Transform':<80}  {'n_unique':>8}  {'std':>10}  {'grid':>8}  {'comp':>8}")
    print("  " + "-" * 120)
    for _, row in p3_df.head(20).iterrows():
        name = row['transform'][:79]
        print(f"  {name:<80}  {row['n_unique']:>8,}  {row['std']:>10.6f}  "
              f"{row['grid_score']:>8.4f}  {row['composite']:>8.4f}")


# ================================================================
# PHASE 4 — VERIFY BEST ON FULL DATASET
# ================================================================
print("\n" + "=" * 65)
print("PHASE 4: Verify best transforms on FULL dataset")
print("=" * 65)

# Collect the best from all phases
all_results = []
for df, phase in [(p1_df, 'P1'), (p2_df, 'P2'), (p3_df, 'P3')]:
    if len(df) > 0:
        for _, row in df.head(5).iterrows():
            all_results.append((phase, row))

def apply_transform_full(row):
    """Rebuild a transform from its row metadata, applied to full dataset."""
    y = y_full.copy()
    feat1 = row.get('feat1')
    feat2 = row.get('feat2')
    feat3 = row.get('feat3') if 'feat3' in row.index else None
    op    = row['op']

    def get_full(col):
        if col is None or pd.isna(col):
            return None
        v = train[col].values.astype(np.float64)
        v[~np.isfinite(v)] = np.nanmedian(v)
        return v

    f1 = get_full(feat1)
    f2 = get_full(feat2)
    f3 = get_full(feat3)

    if op == 'mul':             t = safe_multiply(y, f1)
    elif op == 'div':           t = safe_divide(y, f1)
    elif op == 'mul_abs':       t = safe_multiply(y, np.abs(f1))
    elif op == 'div_abs':       t = safe_divide(y, np.abs(f1))
    elif op == 'mul_sq':        t = safe_multiply(y, f1 ** 2)
    elif op == 'mul_sqrt':      t = safe_multiply(y, np.sqrt(np.abs(f1)))
    elif op == 'div_sqrt':      t = safe_divide(y, np.sqrt(np.abs(f1)) + 1e-10)
    elif op == 'mul_div':       t = safe_divide(safe_multiply(y, f1), f2)
    elif op == 'div_mul':       t = safe_multiply(safe_divide(y, f1), f2)
    elif op == 'mul_mul':       t = safe_multiply(safe_multiply(y, f1), f2)
    elif op == 'div_div':       t = safe_divide(safe_divide(y, f1), f2)
    elif op == 'mul_div_abs':   t = safe_divide(safe_multiply(y, np.abs(f1)), np.abs(f2) + 1e-10)
    elif op == 'mul_div_div':   t = safe_divide(safe_divide(safe_multiply(y, f1), f2), f3)
    elif op == 'mul_mul_div':   t = safe_divide(safe_multiply(safe_multiply(y, f1), f2), f3)
    elif op == 'div_div_mul':   t = safe_multiply(safe_divide(safe_divide(y, f1), f2), f3)
    elif op == 'div_mul_div':   t = safe_divide(safe_multiply(safe_divide(y, f1), f2), f3)
    else:
        return y
    return t


print(f"\n  {'Phase':<5} {'n_unique_full':>14} {'std_full':>12} {'grid_full':>10}  Transform")
print("  " + "-" * 90)

full_results = []
for phase, row in all_results:
    t = apply_transform_full(row)
    s = score_transform(t)
    full_results.append({**row.to_dict(), 'phase': phase, **{f'full_{k}': v for k, v in s.items()}})
    name = row['transform'][:60]
    print(f"  {phase:<5} {s['n_unique']:>14,} {s['std']:>12.6f} {s['grid_score']:>10.4f}  {name}")

# Baseline on full
print(f"\n  {'BASE':<5} {baseline['n_unique']:>14,} {y_full.std():>12.6f} "
      f"{'N/A':>10}  TARGET (raw)")

# ================================================================
# SAVE RESULTS
# ================================================================
all_df = pd.concat([p1_df.head(100), p2_df.head(100) if len(p2_df) > 0 else pd.DataFrame(),
                    p3_df.head(50)  if len(p3_df) > 0 else pd.DataFrame()],
                   ignore_index=True).sort_values('composite')

out_path = os.path.join(OUT_DIR, 'target_transform_results.csv')
all_df.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")

# ================================================================
# SUMMARY + RECOMMENDATION
# ================================================================
print("\n" + "=" * 65)
print("FINAL RECOMMENDATION")
print("=" * 65)

if full_results:
    best = min(full_results, key=lambda x: x['full_n_unique'])
    print(f"\n  BEST by n_unique  : {best['transform']}")
    print(f"    n_unique = {best['full_n_unique']:,}  (vs baseline {baseline['n_unique']:,})")
    print(f"    std      = {best['full_std']:.6f}  (vs baseline {y_full.std():.6f})")
    print(f"    grid     = {best['full_grid_score']:.4f}")
    print(f"    reduction: {(1 - best['full_n_unique']/baseline['n_unique'])*100:.1f}%")

    best_std = min(full_results, key=lambda x: x['full_std'])
    if best_std['transform'] != best['transform']:
        print(f"\n  BEST by std       : {best_std['transform']}")
        print(f"    std = {best_std['full_std']:.6f}  (vs baseline {y_full.std():.6f})")

print(f"""
  HOW TO USE THE BEST TRANSFORM IN TRAINING:
  ───────────────────────────────────────────
  # Suppose best transform is TARGET * f1 / f2:

  y_scaled = train['TARGET'] * train[f1] / train[f2]

  # Train model on y_scaled instead of TARGET
  model.fit(X_train, y_scaled, ...)

  # At test time, INVERT the transform:
  y_pred_scaled = model.predict(X_test)
  y_pred_target = y_pred_scaled * test[f2] / test[f1]

  # Why this works:
  #   The model learns to predict the STABLE underlying signal
  #   (discrete tick moves), not the price-diluted return.
  #   Even if price levels shift in test, the tick-move signal
  #   should transfer cleanly.
""")
