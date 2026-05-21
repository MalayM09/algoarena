# ================================================================
# FEATURE COMBINATION SEARCH — GPU OPTIMISED (KAGGLE, 12h)
# ================================================================
# Metrics computed for every combination that passes pruning:
#
#   PEARSON ICIR   : |mean_Pearson_IC| / std_IC  across 5 regimes
#   SPEARMAN ICIR  : same using rank-based correlation (outlier-robust)
#   SIGN_CONSISTENCY: fraction of regimes where IC sign agrees
#   REGIME P-VALUE : t-test on mean_IC across 5 regimes
#                    t = mean_IC * sqrt(5) / std_IC  (df=4)
#                    tells us "is this signal statistically real?"
#   MUTUAL INFO    : histogram MI between combo and TARGET
#                    computed for top-1000 survivors as post-processing
#                    captures non-linear relationships Pearson/Spearman miss
#
# Distance correlation excluded: O(n²) for n=660k → ~3.5 TB GPU memory.
#
# ADAPTIVE PRUNING:
#   Thresholds automatically loosen when >7 hours remain and tighten
#   as time runs low.  This maximises discovery when time is available.
#
#   time_left > 7h : very loose  → find as many combinations as possible
#   time_left 4-7h : standard
#   time_left 2-4h : moderate
#   time_left < 2h : tight      → save time, only high-confidence combos
#
# Operations:
#   Stage 1 : 8 transforms per feature
#   Stage 2 : ALL 445C2=99,010 pairs × 9 ops  (891k combos)
#   Stage 3 : ALL 445C3=14.7M triples × 5 ops
#   Stage 3B: Synergistic triples (top-50 × top-50 × all 445)
#   Stage 4 : Top survivors × 445 × 3 ops
#   Stage 5 : Top survivors × 445 × 2 ops
#   Post    : Mutual Information for top-1000 survivors (CPU)
# ================================================================

import numpy as np
import pandas as pd
import gc, time, os, warnings
from scipy.stats import t as t_dist
warnings.filterwarnings('ignore')

# ── GPU setup ────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    GPU = True
    print(f"CuPy {cp.__version__} found — running on GPU")
    mempool = cp.get_default_memory_pool()
    def free_gpu():
        mempool.free_all_blocks()
        gc.collect()
except ImportError:
    import numpy as cp
    GPU = False
    def free_gpu():
        gc.collect()
    print("CuPy not found — falling back to CPU (will be slower)")

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_PATH = '/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet'
OUT_DIR    = '/kaggle/working'
CHECKPOINT = os.path.join(OUT_DIR, 'combo_checkpoint.csv')
FINAL_OUT  = os.path.join(OUT_DIR, 'feature_combinations.csv')

SESSION_LIMIT_SEC = 11.5 * 3600
BEST_SINGLE       = 2.539   # Pearson ICIR of S03_V04_T04 — bar to beat
SAVE_EVERY        = 1800    # checkpoint every 30 min

# Batch size: 16 GB VRAM budget
#   F_batch float64 (660k × 500)     = 2.64 GB
#   F_ranked float32 (660k × 500)    = 1.32 GB
#   Per-regime slices + y arrays     = ~0.5 GB
#   X_gpu float32 (445 × 660k)       = 1.17 GB
#   Peak total                       ≈ 5.6 GB  (comfortable in 16 GB)
BSIZ = 500

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
t0 = time.time()

df        = pd.read_parquet(TRAIN_PATH)
y_np      = df['TARGET'].values.astype(np.float64)
feat_cols = [c for c in df.columns if c not in {'ID', 'TARGET'}]
n_feat    = len(feat_cols)
X_np      = df[feat_cols].values.astype(np.float32)
del df; gc.collect()

n_samples = len(y_np)
print(f"  Train: {X_np.shape}  |  Features: {n_feat}  |  Samples: {n_samples:,}")

# ── Regime groups (SO3_T quintiles) ──────────────────────────────────────────
so3_idx   = feat_cols.index('SO3_T')
so3       = X_np[:, so3_idx].astype(np.float64)
groups_np = pd.qcut(pd.Series(so3), q=5, labels=False,
                    duplicates='drop').values.astype(np.int32)
unique_g  = sorted(set(groups_np))
n_groups  = len(unique_g)
print(f"  Regimes: {unique_g}  |  sizes: {[int((groups_np==g).sum()) for g in unique_g]}")

# ── Transfer to GPU ───────────────────────────────────────────────────────────
X_gpu      = cp.array(X_np, dtype=cp.float32)
y_gpu      = cp.array(y_np, dtype=cp.float64)
groups_gpu = cp.array(groups_np, dtype=cp.int32)

# Pearson y stats — global + per regime
y_mean_gpu = y_gpu.mean()
y_c_gpu    = y_gpu - y_mean_gpu
y_std_gpu  = float(y_c_gpu.std()) + 1e-12

regime_masks_gpu = [groups_gpu == g for g in unique_g]
regime_y_c, regime_y_std = [], []
for mask in regime_masks_gpu:
    y_r  = y_gpu[mask];  yc_r = y_r - y_r.mean()
    regime_y_c.append(yc_r)
    regime_y_std.append(float(yc_r.std()) + 1e-12)

# Spearman y ranks — global + per regime
def gpu_rank_1d(x):
    return cp.argsort(cp.argsort(x)).astype(cp.float32)

y_rank_gpu  = gpu_rank_1d(y_gpu.astype(cp.float32))
y_rank_c    = y_rank_gpu - y_rank_gpu.mean()
y_rank_std  = float(y_rank_c.std()) + 1e-12

regime_y_rank_c, regime_y_rank_std = [], []
for mask in regime_masks_gpu:
    y_r = y_gpu[mask].astype(cp.float32)
    yr  = gpu_rank_1d(y_r);  yrc = yr - yr.mean()
    regime_y_rank_c.append(yrc)
    regime_y_rank_std.append(float(yrc.std()) + 1e-12)

if GPU:
    free = cp.cuda.Device().mem_info[0] / 1e9
    print(f"  GPU transfer done  |  X={X_gpu.nbytes/1e9:.2f}GB  free={free:.1f}GB")
else:
    print("  (CPU mode)")


# ================================================================
# ADAPTIVE THRESHOLD SYSTEM
# ================================================================
# Thresholds loosen when time is plentiful.  Checked before each
# stage and also inside filter_and_store (so mid-stage tightening
# happens automatically if we start running short on time).
# ================================================================

def time_left():
    return SESSION_LIMIT_SEC - (time.time() - t0)

def time_left_hours():
    return time_left() / 3600

def get_stage_params(stage):
    """
    Returns (thr_icir, thr_sc, top_k_pass) based on current time remaining.

    thr_icir  : minimum max(pearson_icir, spearman_icir) to keep a combo
    thr_sc    : minimum sign_consistency to keep a combo
    top_k_pass: how many survivors from this stage feed the next stage

    Logic: with >7h left use very loose thresholds to find as many
    combinations as possible; tighten as time runs out.
    """
    tl = time_left_hours()

    if tl > 7:
        # Very loose — maximize discovery
        thr = {1:(0.1,0.4), 2:(0.3,0.5), 3:(0.6,0.6), 4:(1.2,0.7), 5:(1.8,0.8)}
        top_k = 500
    elif tl > 4:
        # Standard
        thr = {1:(0.3,0.5), 2:(0.7,0.7), 3:(1.2,0.7), 4:(1.8,0.8), 5:(2.1,0.9)}
        top_k = 300
    elif tl > 2:
        # Moderate
        thr = {1:(0.5,0.6), 2:(1.0,0.8), 3:(1.5,0.8), 4:(2.0,0.8), 5:(2.3,1.0)}
        top_k = 200
    else:
        # Tight — save remaining time
        thr = {1:(0.8,0.7), 2:(1.5,0.9), 3:(2.0,0.9), 4:(2.5,0.9), 5:(2.7,1.0)}
        top_k = 100

    return thr[stage], top_k


# ================================================================
# REGIME P-VALUE
# ================================================================
# We have the IC measured in 5 separate market regimes.
# Treat these 5 IC values as a small sample and ask:
# "Is the mean IC significantly different from zero?"
#
# t_stat = mean_IC / (std_IC / sqrt(5))
# p_value = 2 * P(T(df=4) > |t_stat|)
#
# p < 0.05 means the signal is statistically significant across regimes.
# p < 0.01 means very strong evidence.
#
# Note: n_groups=5 gives df=4 — small sample, so only ICIR > ~2.5
# will achieve p < 0.05.  This correctly filters out weak signals.
# ================================================================

def regime_pvalue(mean_ic_val, std_ic_val):
    """Compute two-sided p-value for mean IC being non-zero across 5 regimes."""
    t_stat = abs(float(mean_ic_val)) * np.sqrt(n_groups) / (float(std_ic_val) + 1e-12)
    return float(t_dist.sf(t_stat, df=n_groups - 1) * 2)


# ================================================================
# MUTUAL INFORMATION — CPU histogram method
# ================================================================
# MI measures how much knowing feature X reduces uncertainty about y.
# Unlike Pearson/Spearman, MI detects NON-LINEAR relationships.
#
# We use a simple histogram approximation:
#   1. Bin both feature and target into n_bins quantile buckets
#   2. Build a joint frequency table (n_bins × n_bins)
#   3. MI = Σ p(x,y) * log2( p(x,y) / (p(x)*p(y)) )
#
# This is O(n) and fast (< 0.1s per feature).
# Computed as post-processing on top-1000 survivors only.
# ================================================================

# Pre-bin y into 50 quantile buckets (done once)
_N_BINS   = 50
_y_bins_np = pd.qcut(pd.Series(y_np), q=_N_BINS, labels=False,
                      duplicates='drop').values.astype(np.int32)
_n_y_bins  = _y_bins_np.max() + 1   # actual number of bins (may be < _N_BINS on ties)

def hist_mi(f_np, n_bins=_N_BINS):
    """
    Compute histogram mutual information between feature array f_np and TARGET.
    f_np : 1-D numpy array of length n_samples
    Returns MI in nats (float).
    """
    try:
        f_bins = pd.qcut(pd.Series(f_np), q=n_bins, labels=False,
                          duplicates='drop').values.astype(np.int32)
        n_f_bins = f_bins.max() + 1

        # Joint histogram via scatter add
        joint = np.zeros((n_f_bins, _n_y_bins), dtype=np.float64)
        np.add.at(joint, (f_bins, _y_bins_np), 1.0)
        joint /= len(f_np)

        p_f = joint.sum(axis=1, keepdims=True)   # (n_f_bins, 1)
        p_y = joint.sum(axis=0, keepdims=True)   # (1, n_y_bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            # MI = Σ p(x,y) * log(p(x,y) / (p(x)*p(y)))
            denom = p_f * p_y
            ratio = np.where(joint > 0, joint / (denom + 1e-300), 0.0)
            mi    = float(np.nansum(joint * np.log(ratio + 1e-300)))
        return max(0.0, mi)
    except Exception:
        return np.nan


# ================================================================
# BATCH GPU ANALYSIS — Pearson + Spearman in one GPU call
# ================================================================

def batch_rank(F):
    """Rank each column of (n, B) cupy array independently (float32)."""
    return cp.argsort(cp.argsort(F.astype(cp.float32), axis=0),
                      axis=0).astype(cp.float32)


def batch_analyse(F_batch):
    """
    Compute Pearson IC, Spearman IC, sign_consistency, ICIR for B combinations.

    F_batch : (n_samples, B) cupy array
    Returns : dict with numpy arrays of shape (B,)
              Keys: global_ic, mean_ic, std_ic, sign_consistency, icir
                    spear_global_ic, spear_mean_ic, spear_icir, spear_sign_cons
                    r0..r4 (Pearson per regime), sr0..sr4 (Spearman per regime)
    """
    if F_batch.shape[1] == 0:
        return None

    F64 = F_batch.astype(cp.float64)

    # ── Pearson global ───────────────────────────────────────────────────────
    F_c   = F64 - F64.mean(axis=0)
    std_F = F_c.std(axis=0)
    valid = std_F > 1e-12
    cov   = (F_c * y_c_gpu[:, None]).mean(axis=0)
    g_ic  = cp.where(valid, cov / (std_F * y_std_gpu), 0.0)

    # ── Pearson per-regime ───────────────────────────────────────────────────
    p_ics = []
    for mask, yc_r, ys_r in zip(regime_masks_gpu, regime_y_c, regime_y_std):
        Fg = F64[mask];  Fg_c = Fg - Fg.mean(axis=0)
        sf = Fg_c.std(axis=0);  val = sf > 1e-12
        cvr = (Fg_c * yc_r[:, None]).mean(axis=0)
        p_ics.append(cp.where(val, cvr / (sf * ys_r), 0.0))

    pics  = cp.stack(p_ics, axis=0)                 # (5, B)
    mean_ic   = pics.mean(axis=0)
    std_ic    = pics.std(axis=0)
    sign_mean = cp.sign(mean_ic)
    sign_cons = (cp.sign(pics) == sign_mean[None, :]).mean(axis=0)
    icir      = cp.abs(mean_ic) / (std_ic + 1e-8)

    # ── Spearman: rank each column, then Pearson on ranks ───────────────────
    F_r  = batch_rank(F64)                          # (n, B) float32
    Fr_c = F_r - F_r.mean(axis=0)
    std_Fr = Fr_c.std(axis=0);  valid_r = std_Fr > 1e-12
    cov_r  = (Fr_c * y_rank_c[:, None]).mean(axis=0)
    sp_g_ic = cp.where(valid_r, cov_r / (std_Fr * y_rank_std), 0.0)

    sp_ics = []
    for mask, yrc_r, yrs_r in zip(regime_masks_gpu, regime_y_rank_c, regime_y_rank_std):
        Fg_r   = F_r[mask];  Fg_r_c = Fg_r - Fg_r.mean(axis=0)
        sf_r   = Fg_r_c.std(axis=0);  val_r = sf_r > 1e-12
        cvr_r  = (Fg_r_c * yrc_r[:, None]).mean(axis=0)
        sp_ics.append(cp.where(val_r, cvr_r / (sf_r * yrs_r), 0.0))

    sp_st   = cp.stack(sp_ics, axis=0)              # (5, B)
    sp_mean = sp_st.mean(axis=0);  sp_std = sp_st.std(axis=0)
    sp_sm   = cp.sign(sp_mean)
    sp_sc   = (cp.sign(sp_st) == sp_sm[None, :]).mean(axis=0)
    sp_icir = cp.abs(sp_mean) / (sp_std + 1e-8)

    def cpu(a): return cp.asnumpy(a) if GPU else a

    return dict(
        global_ic=cpu(g_ic), mean_ic=cpu(mean_ic), std_ic=cpu(std_ic),
        sign_consistency=cpu(sign_cons), icir=cpu(icir),
        r0=cpu(p_ics[0]), r1=cpu(p_ics[1]), r2=cpu(p_ics[2]),
        r3=cpu(p_ics[3]), r4=cpu(p_ics[4]),
        spear_global_ic=cpu(sp_g_ic), spear_mean_ic=cpu(sp_mean),
        spear_icir=cpu(sp_icir), spear_sign_cons=cpu(sp_sc),
        sr0=cpu(sp_ics[0]), sr1=cpu(sp_ics[1]), sr2=cpu(sp_ics[2]),
        sr3=cpu(sp_ics[3]), sr4=cpu(sp_ics[4]),
    )


# ================================================================
# FILTER AND STORE
# ================================================================
# Called after every GPU batch.  For each of the B combinations:
#   1. Check adaptive threshold: max(pearson_icir, spearman_icir) ≥ thr_icir
#                                AND sign_consistency ≥ thr_sc
#   2. If passes: compute regime p-value (cheap, CPU-only)
#   3. Store full record.  MI added later as post-processing.
# ================================================================

all_results = []   # all survivors across all stages
last_save   = time.time()

def filter_and_store(stats, names, stage, feat1_list, feat2_list, op_list,
                     survivors_out):
    """
    Apply adaptive threshold, compute p-value for survivors, append records.
    Returns number of survivors from this batch.
    """
    (thr_icir, thr_sc), _ = get_stage_params(stage)
    B = len(names)
    kept = 0
    for b in range(B):
        p_ic  = float(stats['icir'][b])
        s_ic  = float(stats['spear_icir'][b])
        sc    = float(stats['sign_consistency'][b])
        best  = max(p_ic, s_ic)
        if best < thr_icir or sc < thr_sc:
            continue
        # Regime p-value: t-test on mean IC across the 5 regimes
        p_val = regime_pvalue(stats['mean_ic'][b], stats['std_ic'][b])
        sp_pval = regime_pvalue(stats['spear_mean_ic'][b], stats['spear_icir'][b])
        rec = {
            'combo':             names[b],
            'stage':             stage,
            'feat1':             feat1_list[b],
            'feat2':             feat2_list[b],
            'op':                op_list[b],
            # Pearson
            'global_ic':         float(stats['global_ic'][b]),
            'mean_ic':           float(stats['mean_ic'][b]),
            'std_ic':            float(stats['std_ic'][b]),
            'sign_consistency':  sc,
            'icir':              p_ic,
            'regime_pvalue':     p_val,
            'r0': float(stats['r0'][b]), 'r1': float(stats['r1'][b]),
            'r2': float(stats['r2'][b]), 'r3': float(stats['r3'][b]),
            'r4': float(stats['r4'][b]),
            # Spearman
            'spear_global_ic':   float(stats['spear_global_ic'][b]),
            'spear_mean_ic':     float(stats['spear_mean_ic'][b]),
            'spear_sign_cons':   float(stats['spear_sign_cons'][b]),
            'spear_icir':        s_ic,
            'spear_regime_pvalue': sp_pval,
            'sr0': float(stats['sr0'][b]), 'sr1': float(stats['sr1'][b]),
            'sr2': float(stats['sr2'][b]), 'sr3': float(stats['sr3'][b]),
            'sr4': float(stats['sr4'][b]),
            # Combined ranking score (used for sorting and TOP_K selection)
            'score': best * sc,
            # MI filled in post-processing
            'mutual_info': np.nan,
        }
        all_results.append(rec)
        survivors_out.append(rec)
        kept += 1
    return kept


def maybe_checkpoint():
    global last_save
    if time.time() - last_save > SAVE_EVERY and all_results:
        pd.DataFrame(all_results).to_csv(CHECKPOINT, index=False)
        elapsed = (time.time() - t0) / 60
        tl = time_left_hours()
        thr_info = get_stage_params(2)   # show Stage 2 threshold as reference
        print(f"  [ckpt] {len(all_results)} survivors | {elapsed:.1f}m elapsed"
              f" | {tl:.1f}h left | thr={thr_info[0]}")
        last_save = time.time()


# ================================================================
# OPERATIONS LIBRARY
# ================================================================

PAIR_OPS = {
    'product':    lambda a, b: a * b,
    'ratio_ab':   lambda a, b: a / (cp.abs(b) + 1e-6),
    'abs_diff':   lambda a, b: cp.abs(a - b),
    'sign_cross': lambda a, b: cp.sign(a) * cp.abs(b),
    'sum_ab':     lambda a, b: a + b,
    'diff_ab':    lambda a, b: a - b,
    'max_ab':     lambda a, b: cp.maximum(a, b),
    'min_ab':     lambda a, b: cp.minimum(a, b),
    'log_ratio':  lambda a, b: cp.log(cp.abs(a / (cp.abs(b) + 1e-6)) + 1e-8),
}

TRIPLE_OPS = {
    'product3': lambda p, s: p * s,
    'ratio3':   lambda p, s: p / (cp.abs(s) + 1e-6),
    'sum3':     lambda p, s: p + s,
    'diff3':    lambda p, s: p - s,
    'max3':     lambda p, s: cp.maximum(cp.abs(p), cp.abs(s)) * cp.sign(p),
}

QUAD_OPS = {
    'product4': lambda a, b: a * b,
    'ratio4':   lambda a, b: a / (cp.abs(b) + 1e-6),
    'sum4':     lambda a, b: a + b,
}

PENTA_OPS = {
    'product5': lambda a, b: a * b,
    'sum5':     lambda a, b: a + b,
}


# ================================================================
# STAGE 1 — Single-feature transforms (8 per feature)
# ================================================================
print(f"\n{'='*60}")
print(f"STAGE 1: {n_feat} features × 8 transforms")
thr1, top_k1 = get_stage_params(1)
print(f"  Threshold: ICIR≥{thr1[0]}  sign_cons≥{thr1[1]}")
print(f"{'='*60}")

def _rank_col(f_raw):
    f_np = cp.asnumpy(f_raw) if GPU else f_raw
    return cp.array(pd.Series(f_np.astype(np.float32)).rank(pct=True).values,
                    dtype=cp.float32) if GPU else pd.Series(f_np).rank(pct=True).values.astype(np.float32)

single_ops = {
    'raw':       lambda f: f.astype(cp.float32),
    'square':    lambda f: f.astype(cp.float32) ** 2,
    'abs':       lambda f: cp.abs(f.astype(cp.float32)),
    'log_abs':   lambda f: cp.log(cp.abs(f.astype(cp.float32)) + 1e-8),
    'rank':      lambda f: _rank_col(f),
    'sign':      lambda f: cp.sign(f.astype(cp.float32)),
    'reciprocal':lambda f: 1.0 / (cp.abs(f.astype(cp.float32)) + 1e-6),
    'signed_sq': lambda f: f.astype(cp.float32) * cp.abs(f.astype(cp.float32)),
}

stage1_surv = []
for i, col in enumerate(feat_cols):
    f_raw = X_gpu[:, i]
    batch = cp.stack([op(f_raw) for op in single_ops.values()], axis=1)
    stats = batch_analyse(batch)
    op_names = list(single_ops.keys())
    names    = [f'{col}__{o}' for o in op_names]
    filter_and_store(stats, names, 1,
                     [col]*8, ['']*8, op_names, stage1_surv)
    del batch; free_gpu()

stage1_surv.sort(key=lambda x: -x['score'])
_, top_k1 = get_stage_params(1)
print(f"Stage 1 done: {len(stage1_surv)} survivors")
if stage1_surv:
    t1 = stage1_surv[0]
    print(f"  Best: {t1['combo']}  ICIR={t1['icir']:.3f}  Spear={t1['spear_icir']:.3f}"
          f"  p={t1['regime_pvalue']:.4f}")
maybe_checkpoint()


# ================================================================
# STAGE 2 — ALL 445C2 pairs × 9 ops (GPU batched)
# ================================================================
n_pairs = n_feat * (n_feat - 1) // 2
thr2, top_k2 = get_stage_params(2)
print(f"\n{'='*60}")
print(f"STAGE 2: {n_pairs:,} pairs × {len(PAIR_OPS)} ops = {n_pairs*len(PAIR_OPS):,} combos")
print(f"  Threshold: ICIR≥{thr2[0]}  sign_cons≥{thr2[1]}")
print(f"{'='*60}")

stage2_surv                         = []
batch_F, batch_names                = [], []
batch_feat1, batch_feat2, batch_ops = [], [], []
stage2_tested = stage2_kept = 0

def flush_pair_batch():
    global stage2_kept, stage2_tested
    if not batch_F:
        return
    F_batch = cp.stack(batch_F, axis=1)
    stats   = batch_analyse(F_batch)
    stage2_tested += len(batch_names)
    stage2_kept   += filter_and_store(
        stats, batch_names, 2,
        batch_feat1, batch_feat2, batch_ops, stage2_surv)
    batch_F.clear(); batch_names.clear()
    batch_feat1.clear(); batch_feat2.clear(); batch_ops.clear()
    del F_batch; free_gpu()

for i in range(n_feat):
    f_i = X_gpu[:, i].astype(cp.float32)
    for j in range(i + 1, n_feat):
        f_j = X_gpu[:, j].astype(cp.float32)
        for op_name, op_fn in PAIR_OPS.items():
            try:
                batch_F.append(op_fn(f_i, f_j))
                batch_names.append(f'{feat_cols[i]}__{op_name}__{feat_cols[j]}')
                batch_feat1.append(feat_cols[i])
                batch_feat2.append(feat_cols[j])
                batch_ops.append(op_name)
                if len(batch_F) >= BSIZ:
                    flush_pair_batch()
            except Exception:
                pass

    maybe_checkpoint()
    if i % 50 == 0 and i > 0:
        elapsed = (time.time() - t0) / 60
        done = i * n_feat - i*(i+1)//2
        eta  = (n_pairs - done) / max(done / max(elapsed, 1), 1)
        thr2, _ = get_stage_params(2)
        print(f"  i={i:>4}  tested={stage2_tested:>9,}  kept={stage2_kept:>5,}"
              f"  elapsed={elapsed:.0f}m  eta={eta:.0f}m  thr={thr2[0]:.1f}")

flush_pair_batch()
maybe_checkpoint()

stage2_surv.sort(key=lambda x: -x['score'])
_, top_k2 = get_stage_params(2)
top_pairs = stage2_surv[:top_k2]
print(f"Stage 2 done: {stage2_tested:,} tested | {stage2_kept:,} kept | top_k={top_k2}")
if stage2_surv:
    t2 = stage2_surv[0]
    print(f"  Best: {t2['combo']}  ICIR={t2['icir']:.3f}  Spear={t2['spear_icir']:.3f}"
          f"  p={t2['regime_pvalue']:.4f}")


# ================================================================
# STAGE 3 — ALL 445C3 triples (one GPU batch per pair × triple op)
# ================================================================
thr3, _ = get_stage_params(3)
print(f"\n{'='*60}")
print(f"STAGE 3: ALL 445C3 triples × {len(TRIPLE_OPS)} ops")
print(f"  Threshold: ICIR≥{thr3[0]}  sign_cons≥{thr3[1]}")
print(f"{'='*60}")

stage3_surv                   = []
stage3_tested = stage3_kept   = 0

for i in range(n_feat):
    if time_left() < 2.5 * 3600:
        print(f"  Time guard — stopping Stage 3 at i={i}  ({time_left_hours():.1f}h left)")
        break

    f_i = X_gpu[:, i].astype(cp.float32)
    for j in range(i + 1, n_feat):
        f_j = X_gpu[:, j].astype(cp.float32)

        for pop, pop_fn in PAIR_OPS.items():
            try:
                f_pair = pop_fn(f_i, f_j)
            except Exception:
                continue

            for top, top_fn in TRIPLE_OPS.items():
                k_idx = [k for k in range(n_feat) if k != i and k != j]
                F_k, k_names, k_f2, k_ops = [], [], [], []
                for k in k_idx:
                    try:
                        F_k.append(top_fn(f_pair, X_gpu[:, k].astype(cp.float32)))
                        k_names.append(
                            f'({feat_cols[i]}__{pop}__{feat_cols[j]})'
                            f'__{top}__{feat_cols[k]}'
                        )
                        k_f2.append(feat_cols[k]); k_ops.append(top)
                    except Exception:
                        pass

                if not F_k:
                    continue
                F_batch_gpu = cp.stack(F_k, axis=1)
                stats = batch_analyse(F_batch_gpu)
                B = len(k_names)
                stage3_tested += B
                stage3_kept   += filter_and_store(
                    stats, k_names, 3,
                    [f'{feat_cols[i]}__{pop}__{feat_cols[j]}']*B,
                    k_f2, k_ops, stage3_surv)
                del F_batch_gpu, F_k; free_gpu()

    maybe_checkpoint()
    if i % 20 == 0 and i > 0:
        thr3, _ = get_stage_params(3)
        print(f"  i={i:>4}  tested={stage3_tested:>10,}  kept={stage3_kept:>5,}"
              f"  {time_left_hours():.1f}h left  thr={thr3[0]:.1f}")

maybe_checkpoint()
stage3_surv.sort(key=lambda x: -x['score'])
print(f"Stage 3 done: {stage3_tested:,} tested | {stage3_kept:,} kept")


# ================================================================
# STAGE 3B — Synergistic triples (top-50 singles × top-50 × all 445)
# ================================================================
# Catches signals where NO 2-feature sub-pair is individually strong
# (so they never enter Stage 3's pruned path), but the 3-way
# interaction of three specific features IS a strong signal.
# ================================================================
print(f"\n{'='*60}")
print(f"STAGE 3B: Synergistic triples (top-50 × top-50 × all 445)")
print(f"{'='*60}")

stage3b_surv                    = []
stage3b_tested = stage3b_kept   = 0

if time_left() > 1.5 * 3600 and stage1_surv:
    top50 = [r for r in stage1_surv if r['op'] == 'raw'][:50]
    if len(top50) < 10:
        top50 = stage1_surv[:50]
    print(f"  Seeds: {len(top50)} singles  |  {time_left_hours():.1f}h remaining")

    for si in range(len(top50)):
        if time_left() < 1.2 * 3600:
            print(f"  Time guard — stopping 3B at si={si}")
            break
        fi_rec  = top50[si]
        fi_name = fi_rec['feat1']; fi_op = fi_rec['op']
        fi      = single_ops[fi_op](X_gpu[:, feat_cols.index(fi_name)])

        for sj in range(si + 1, len(top50)):
            fj_rec  = top50[sj]
            fj_name = fj_rec['feat1']; fj_op = fj_rec['op']
            fj      = single_ops[fj_op](X_gpu[:, feat_cols.index(fj_name)])

            for pop, pop_fn in PAIR_OPS.items():
                try:
                    f_pair = pop_fn(fi, fj)
                except Exception:
                    continue

                for top, top_fn in TRIPLE_OPS.items():
                    k_idx = [k for k in range(n_feat)
                              if feat_cols[k] not in {fi_name, fj_name}]
                    F_k, k_names, k_f2, k_ops = [], [], [], []
                    for k in k_idx:
                        try:
                            F_k.append(top_fn(f_pair, X_gpu[:, k].astype(cp.float32)))
                            k_names.append(
                                f'({fi_name}__{fi_op}__{pop}'
                                f'__{fj_name}__{fj_op})'
                                f'__{top}__{feat_cols[k]}'
                            )
                            k_f2.append(feat_cols[k]); k_ops.append(top)
                        except Exception:
                            pass

                    if not F_k:
                        continue
                    F_batch_gpu = cp.stack(F_k, axis=1)
                    stats = batch_analyse(F_batch_gpu)
                    B = len(k_names)
                    stage3b_tested += B
                    stage3b_kept   += filter_and_store(
                        stats, k_names, 3,
                        [f'{fi_name}__{fi_op}__{pop}__{fj_name}__{fj_op}']*B,
                        k_f2, k_ops, stage3b_surv)
                    del F_batch_gpu, F_k; free_gpu()

        maybe_checkpoint()

    stage3b_surv.sort(key=lambda x: -x['score'])
    print(f"Stage 3B done: {stage3b_tested:,} tested | {stage3b_kept:,} kept")
else:
    print("  Skipped (not enough time remaining)")

# Merge Stage 3 + 3B survivors
all_triples = stage3_surv + stage3b_surv
all_triples.sort(key=lambda x: -x['score'])
_, top_k3 = get_stage_params(3)
top_triples = all_triples[:top_k3]
print(f"  Top triples pool: {len(top_triples)} (top_k={top_k3})")


# ================================================================
# STAGE 4 — Top triples × all singles (quads)
# ================================================================
thr4, top_k4 = get_stage_params(4)
print(f"\n{'='*60}")
print(f"STAGE 4: {len(top_triples)} triples × {n_feat} singles × {len(QUAD_OPS)} ops")
print(f"  Threshold: ICIR≥{thr4[0]}  sign_cons≥{thr4[1]}")
print(f"{'='*60}")

stage4_surv                   = []
stage4_tested = stage4_kept   = 0

for tri_rec in top_triples:
    if time_left() < 1.0 * 3600:
        print("  Time guard — stopping Stage 4")
        break

    try:
        pair_str              = tri_rec['feat1']
        fi_name, pop, fj_name = pair_str.split('__')[:3]
        fk_name               = tri_rec['feat2']
        top_name              = tri_rec['op']
        fi = X_gpu[:, feat_cols.index(fi_name)].astype(cp.float32)
        fj = X_gpu[:, feat_cols.index(fj_name)].astype(cp.float32)
        fk = X_gpu[:, feat_cols.index(fk_name)].astype(cp.float32)
        f_tri = TRIPLE_OPS[top_name](PAIR_OPS[pop](fi, fj), fk)
    except Exception:
        continue

    used = {fi_name, fj_name, fk_name}
    F_m, m_names, m_f2, m_ops = [], [], [], []
    for qop, qfn in QUAD_OPS.items():
        for m in range(n_feat):
            if feat_cols[m] in used:
                continue
            try:
                F_m.append(qfn(f_tri, X_gpu[:, m].astype(cp.float32)))
                m_names.append(f'({tri_rec["combo"]})'
                                f'__{qop}__{feat_cols[m]}')
                m_f2.append(feat_cols[m]); m_ops.append(qop)
            except Exception:
                pass

    for start in range(0, len(F_m), BSIZ):
        chunk = F_m[start:start+BSIZ]
        if not chunk:
            continue
        F_batch_gpu = cp.stack(chunk, axis=1)
        stats = batch_analyse(F_batch_gpu)
        B = len(m_names[start:start+BSIZ])
        stage4_tested += B
        stage4_kept   += filter_and_store(
            stats, m_names[start:start+BSIZ], 4,
            [tri_rec['combo']]*B,
            m_f2[start:start+BSIZ], m_ops[start:start+BSIZ], stage4_surv)
        del F_batch_gpu; free_gpu()

    maybe_checkpoint()

stage4_surv.sort(key=lambda x: -x['score'])
_, top_k4 = get_stage_params(4)
top_quads = stage4_surv[:top_k4]
print(f"Stage 4 done: {stage4_tested:,} tested | {stage4_kept:,} kept")


# ================================================================
# STAGE 5 — Top quads × all singles (5-feature)
# ================================================================
thr5, _ = get_stage_params(5)
print(f"\n{'='*60}")
print(f"STAGE 5: {len(top_quads)} quads × {n_feat} singles × {len(PENTA_OPS)} ops")
print(f"  Threshold: ICIR≥{thr5[0]}  sign_cons≥{thr5[1]}")
print(f"{'='*60}")

stage5_surv                   = []
stage5_tested = stage5_kept   = 0

for quad_rec in top_quads:
    if time_left() < 1800:
        print("  Time guard — stopping Stage 5")
        break

    try:
        pair_str  = quad_rec['feat1']
        inner     = pair_str.split(')__')[0].lstrip('(')
        parts     = inner.split('__')
        fi_name, pop, fj_name = parts[0], parts[1], parts[2]
        top_name  = pair_str.split(')__')[1]
        fk_name   = pair_str.split('__')[-1] if ')__' in pair_str else ''
        fm_name   = quad_rec['feat2']
        qop       = quad_rec['op']

        fi = X_gpu[:, feat_cols.index(fi_name)].astype(cp.float32)
        fj = X_gpu[:, feat_cols.index(fj_name)].astype(cp.float32)
        fk = X_gpu[:, feat_cols.index(fk_name)].astype(cp.float32)
        fm = X_gpu[:, feat_cols.index(fm_name)].astype(cp.float32)
        f_quad = QUAD_OPS[qop](
            TRIPLE_OPS[top_name](PAIR_OPS[pop](fi, fj), fk), fm)
    except Exception:
        continue

    used = {fi_name, fj_name, fk_name, fm_name}
    F_p, p_names, p_f2, p_ops = [], [], [], []
    for pop_name, pop_fn in PENTA_OPS.items():
        for p in range(n_feat):
            if feat_cols[p] in used:
                continue
            try:
                F_p.append(pop_fn(f_quad, X_gpu[:, p].astype(cp.float32)))
                p_names.append(f'({quad_rec["combo"]})'
                                f'__{pop_name}__{feat_cols[p]}')
                p_f2.append(feat_cols[p]); p_ops.append(pop_name)
            except Exception:
                pass

    for start in range(0, len(F_p), BSIZ):
        chunk = F_p[start:start+BSIZ]
        if not chunk:
            continue
        F_batch_gpu = cp.stack(chunk, axis=1)
        stats = batch_analyse(F_batch_gpu)
        B = len(p_names[start:start+BSIZ])
        stage5_tested += B
        stage5_kept   += filter_and_store(
            stats, p_names[start:start+BSIZ], 5,
            [quad_rec['combo']]*B,
            p_f2[start:start+BSIZ], p_ops[start:start+BSIZ], stage5_surv)
        del F_batch_gpu; free_gpu()

    maybe_checkpoint()

print(f"Stage 5 done: {stage5_tested:,} tested | {stage5_kept:,} kept")


# ================================================================
# POST-PROCESSING: MUTUAL INFORMATION for top-1000 survivors
# ================================================================
# MI is computed on CPU using the histogram method.
# This detects non-linear relationships that Pearson/Spearman miss.
# Only computed for top-1000 by score (not all survivors — could be many).
#
# MI > 0.05 nats is generally considered a meaningful signal.
# MI = 0 means completely independent; higher = more predictive.
# ================================================================
print(f"\n{'='*60}")
print(f"POST: Mutual Information for top-1000 survivors (CPU histogram)")
print(f"{'='*60}")

# Sort all survivors by score and take top 1000 for MI
all_results.sort(key=lambda x: -x['score'])
mi_candidates = all_results[:1000]
print(f"  Computing MI for {len(mi_candidates)} survivors...")

def reconstruct_feature(rec):
    """
    Attempt to reconstruct the feature array for a survivor record.
    Returns numpy array or None if recipe parsing fails.
    Only handles Stage 1 and Stage 2 (single and pair combos).
    Higher stages are skipped (recipe too complex to parse reliably).
    """
    try:
        stage = rec['stage']
        if stage == 1:
            col   = rec['feat1']
            op    = rec['op']
            f_raw = X_np[:, feat_cols.index(col)].astype(np.float32)
            ops_np = {
                'raw':       lambda f: f,
                'square':    lambda f: f ** 2,
                'abs':       lambda f: np.abs(f),
                'log_abs':   lambda f: np.log(np.abs(f) + 1e-8),
                'rank':      lambda f: pd.Series(f).rank(pct=True).values.astype(np.float32),
                'sign':      lambda f: np.sign(f),
                'reciprocal':lambda f: 1.0 / (np.abs(f) + 1e-6),
                'signed_sq': lambda f: f * np.abs(f),
            }
            return ops_np[op](f_raw)

        elif stage == 2:
            fi_name, pop, fj_name = rec['combo'].split('__')[:3]
            fi = X_np[:, feat_cols.index(fi_name)].astype(np.float32)
            fj = X_np[:, feat_cols.index(fj_name)].astype(np.float32)
            ops_np2 = {
                'product':    lambda a, b: a * b,
                'ratio_ab':   lambda a, b: a / (np.abs(b) + 1e-6),
                'abs_diff':   lambda a, b: np.abs(a - b),
                'sign_cross': lambda a, b: np.sign(a) * np.abs(b),
                'sum_ab':     lambda a, b: a + b,
                'diff_ab':    lambda a, b: a - b,
                'max_ab':     lambda a, b: np.maximum(a, b),
                'min_ab':     lambda a, b: np.minimum(a, b),
                'log_ratio':  lambda a, b: np.log(np.abs(a / (np.abs(b) + 1e-6)) + 1e-8),
            }
            return ops_np2[pop](fi, fj)
        else:
            return None   # Stage 3+ recipes too complex; skip MI
    except Exception:
        return None

mi_computed = 0
for rec in mi_candidates:
    f_arr = reconstruct_feature(rec)
    if f_arr is not None:
        rec['mutual_info'] = hist_mi(f_arr)
        mi_computed += 1

print(f"  MI computed for {mi_computed} / {len(mi_candidates)} candidates"
      f"  (Stage 3+ skipped — recipe complexity)")


# ================================================================
# FINAL OUTPUT
# ================================================================
print(f"\n{'='*60}")
print(f"FINAL SUMMARY")
print(f"{'='*60}")

df_out = pd.DataFrame(all_results)

if len(df_out) == 0:
    print("No survivors found — thresholds may be too tight for this dataset.")
else:
    df_out['score'] = df_out.apply(
        lambda r: max(r['icir'], r['spear_icir']) * r['sign_consistency'], axis=1)
    df_out = df_out.sort_values('score', ascending=False).reset_index(drop=True)

    # Stage counts
    for s in [1, 2, 3, 4, 5]:
        n = (df_out['stage'] == s).sum()
        if n: print(f"  Stage {s} survivors : {n:,}")

    # Significant by p-value (p < 0.05)
    sig_p = df_out[df_out['regime_pvalue'] < 0.05]
    print(f"\n  Statistically significant (Pearson p<0.05): {len(sig_p):,}")
    sig_sp = df_out[df_out['spear_regime_pvalue'] < 0.05]
    print(f"  Statistically significant (Spearman p<0.05): {len(sig_sp):,}")

    # Breakthroughs: beat best known single feature
    bt_p = df_out[(df_out['sign_consistency'] == 1.0) &
                  (df_out['icir'] > BEST_SINGLE) &
                  (df_out['stage'] > 1)]
    bt_s = df_out[(df_out['spear_sign_cons'] == 1.0) &
                  (df_out['spear_icir'] > BEST_SINGLE) &
                  (df_out['stage'] > 1)]

    print(f"\n  Breakthroughs (Pearson ICIR>{BEST_SINGLE}, sign_cons=1.0): {len(bt_p)}")
    if len(bt_p) > 0:
        cols = ['combo','stage','mean_ic','icir','sign_consistency','regime_pvalue','spear_icir']
        print(bt_p[cols].head(10).to_string(index=False))

    print(f"\n  Breakthroughs (Spearman ICIR>{BEST_SINGLE}, sign_cons=1.0): {len(bt_s)}")
    if len(bt_s) > 0:
        cols = ['combo','stage','spear_mean_ic','spear_icir','spear_sign_cons','spear_regime_pvalue','icir']
        print(bt_s[cols].head(10).to_string(index=False))

    if len(bt_p) == 0 and len(bt_s) == 0:
        print("  No combination beats the best single feature on either metric.")

    # Top 20 overall
    print(f"\nTop 20 by score (max(ICIR, SpearICIR) × sign_consistency):")
    cols = ['combo','stage','icir','spear_icir','sign_consistency','regime_pvalue','mutual_info','score']
    print(df_out[cols].head(20).to_string(index=False))

    # Top 10 by Pearson ICIR
    print(f"\nTop 10 by Pearson ICIR:")
    cols = ['combo','stage','mean_ic','icir','sign_consistency','regime_pvalue']
    print(df_out.sort_values('icir', ascending=False)[cols].head(10).to_string(index=False))

    # Top 10 by Spearman ICIR
    print(f"\nTop 10 by Spearman ICIR:")
    cols = ['combo','stage','spear_mean_ic','spear_icir','spear_sign_cons','spear_regime_pvalue']
    print(df_out.sort_values('spear_icir', ascending=False)[cols].head(10).to_string(index=False))

    # Top 10 by Mutual Information (Stage 1+2 only)
    mi_valid = df_out[df_out['mutual_info'].notna()].sort_values('mutual_info', ascending=False)
    if len(mi_valid) > 0:
        print(f"\nTop 10 by Mutual Information (Stage 1+2, higher = more non-linear signal):")
        cols = ['combo','stage','mutual_info','icir','spear_icir','sign_consistency']
        print(mi_valid[cols].head(10).to_string(index=False))

    # Save
    df_out.to_csv(FINAL_OUT, index=False)
    df_out.head(500).to_csv(
        os.path.join(OUT_DIR, 'feature_combinations_top500.csv'), index=False)

elapsed = (time.time() - t0) / 3600
print(f"\nTotal elapsed: {elapsed:.2f} hours")
print(f"Total survivors: {len(df_out):,}")
print(f"Saved → {FINAL_OUT}")
print(f"Saved → feature_combinations_top500.csv")
print(f"\nNote: Distance correlation excluded (O(n²) memory, infeasible for n={n_samples:,}).")
print(f"Note: MI computed only for Stage 1+2 survivors (Stage 3+ recipes too complex to reconstruct).")
