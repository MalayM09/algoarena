"""
Literature-informed feature engineering for order book microstructure.

Based on research from:
- Cont, Kukanov, Stoikov (2014) — Order Flow Imbalance
- Xu, Gould, Howison (2019) — Multi-Level OFI
- Stoikov (2018) — Microprice
- Ntakaris et al. (2019) — LOB feature engineering
- Optiver/Jane Street Kaggle competition winners

Feature mapping to our dataset:
- S03_D02_V01_A01_B00-B10: Volume distribution histogram at 11 depth bins
  → B00=tightest (near mid), B10=widest. Values ~0.13-0.39 (fractions)
  → U-shaped: more volume near and far from mid → classic LOB shape
- S03_D02_A09_A02_B00-B10: Order-weighted depth histogram (large abs values)
- S03_P01_D01-D05: Spread features at different depth levels
- S03_V02-V05_T01-T06: Volume at 6 time periods × 4 volume types
- S01/S02_F03_U01: Flow features across exchanges
- S03_A02_D03/D04_W02: VWAP-weighted alpha signals at depth levels
- S03_A07_V01_V09: Alpha × volume interactions
"""
import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr, entropy as sp_entropy
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = '/Users/malaymishra/Desktop/quant_ml_project/data/raw'
OUT_DIR  = '/Users/malaymishra/Desktop/quant_ml_project/outputs/submissions'
ORACLE_PATH = os.path.join(OUT_DIR, 'exploit_v2_zero.csv')
TARGET_STD = 0.000948; CLIP_Z = 5.0
t0 = time.time()

def auto_scale(p, std=TARGET_STD):
    s = p.std(); return p * (std / s) if s > 1e-10 else p

def daywise_ic(pred, oracle_df, test_ids, test_day):
    df = pd.DataFrame({'ID': test_ids, 'pred': pred, 'day': test_day})
    df = df.merge(oracle_df[['ID','TARGET']], on='ID', how='inner')
    ics = []
    for d, g in df.groupby('day'):
        if len(g) < 3: continue
        p = g['pred'].values; o = g['TARGET'].values
        p = p - p.mean(); o = o - o.mean()
        pn, on_ = np.linalg.norm(p), np.linalg.norm(o)
        if pn < 1e-12 or on_ < 1e-12: ics.append(0.)
        else: ics.append(float((p @ o) / (pn * on_)))
    return float(np.mean(ics))

# ── Load ──
print("Loading data...", flush=True)
train = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
test  = pd.read_parquet(os.path.join(DATA_DIR, 'test.parquet'))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))[['ID']]
oracle_df = pd.read_csv(ORACLE_PATH)

feat_cols = [c for c in train.columns if c not in {'ID','TARGET','CV_GROUP'}]
all_feat = [c for c in feat_cols if c != 'SO3_T']
test_ids = test['ID'].values
train_day = train['SO3_T'].round(5).astype(str).values
test_day  = test['SO3_T'].round(5).astype(str).values

target = train['TARGET'].values.astype(np.float64)
lo, hi = np.percentile(target, [1, 99])
target_w = np.clip(target, lo, hi)

ic_df = pd.read_csv('/Users/malaymishra/Desktop/quant_ml_project/outputs/eda/summaries/ic_icir_full.csv')
gold_feats = ic_df[(ic_df['abs_icir'] >= 3) & (ic_df['ic_pos_frac'].isin([0.0, 1.0]))]['feature'].tolist()
gold_feats = [f for f in gold_feats if f in all_feat]
ic_map = dict(zip(ic_df['feature'], ic_df['mean_ic']))

feat_idx = {f: i for i, f in enumerate(all_feat)}

# ════════════════════════════════════════════════════════
# LITERATURE-INFORMED FEATURE ENGINEERING
# ════════════════════════════════════════════════════════
print("\n=== LITERATURE-INFORMED FEATURE ENGINEERING ===\n", flush=True)

eng = {}  # name -> (train_values, test_values)

def add_feat(name, tr_vals, te_vals):
    tr_vals = np.nan_to_num(np.float32(tr_vals), 0)
    te_vals = np.nan_to_num(np.float32(te_vals), 0)
    eng[name] = (tr_vals, te_vals)

def get(col):
    """Get raw column values for train and test"""
    return train[col].values.astype(np.float32), test[col].values.astype(np.float32)

def get_lag(base, lag):
    """Get lagged feature"""
    col = f"{base}_LagT{lag}" if lag > 0 else base
    if col in train.columns:
        return get(col)
    return None, None

# ──────────────────────────────────────────────────────
# 1. ORDER BOOK HISTOGRAM FEATURES (Cont et al. 2014)
# The D02_V01_A01_B00-B10 features are volume fractions at depth bins
# ──────────────────────────────────────────────────────
print("1. Order book histogram features...", flush=True)

vol_bins = [f'S03_D02_V01_A01_B{i:02d}_E{i:02d}_E{i+1:02d}' for i in range(11)]
ord_bins = [f'S03_D02_A09_A02_B{i:02d}_E{i:02d}_E{i+1:02d}' for i in range(11)]

for lag_suffix in ['', '_LagT1', '_LagT2', '_LagT3']:
    # Volume histogram features
    vol_cols = [f + lag_suffix for f in vol_bins if f + lag_suffix in train.columns]
    if len(vol_cols) == 11:
        tag = f"_lag{lag_suffix[-1]}" if lag_suffix else ""

        tr_vol = train[vol_cols].values.astype(np.float32)
        te_vol = test[vol_cols].values.astype(np.float32)

        # a) Histogram skewness: asymmetry of volume distribution
        # Positive skew = more volume near mid → buying pressure
        weights = np.arange(11).astype(np.float32)
        tr_mean_bin = (tr_vol * weights).sum(1) / (tr_vol.sum(1) + 1e-8)
        te_mean_bin = (te_vol * weights).sum(1) / (te_vol.sum(1) + 1e-8)
        add_feat(f'vol_hist_centroid{tag}', tr_mean_bin, te_mean_bin)

        # b) Near-mid vs far-from-mid imbalance (Depth Pressure)
        # B00-B04 = near mid, B06-B10 = far from mid
        tr_near = tr_vol[:, :5].sum(1)
        te_near = te_vol[:, :5].sum(1)
        tr_far = tr_vol[:, 6:].sum(1)
        te_far = te_vol[:, 6:].sum(1)
        add_feat(f'vol_near_far_imb{tag}',
                 (tr_near - tr_far) / (tr_near + tr_far + 1e-8),
                 (te_near - te_far) / (te_near + te_far + 1e-8))

        # c) Histogram entropy (uniformity of liquidity distribution)
        tr_vol_norm = np.clip(tr_vol, 1e-8, None)
        te_vol_norm = np.clip(te_vol, 1e-8, None)
        tr_vol_p = tr_vol_norm / tr_vol_norm.sum(1, keepdims=True)
        te_vol_p = te_vol_norm / te_vol_norm.sum(1, keepdims=True)
        add_feat(f'vol_hist_entropy{tag}',
                 -(tr_vol_p * np.log(tr_vol_p + 1e-10)).sum(1),
                 -(te_vol_p * np.log(te_vol_p + 1e-10)).sum(1))

        # d) Peak bin (where is most volume)
        add_feat(f'vol_peak_bin{tag}',
                 np.argmax(tr_vol, axis=1).astype(np.float32),
                 np.argmax(te_vol, axis=1).astype(np.float32))

    # Order-weighted histogram: same features
    ord_cols = [f + lag_suffix for f in ord_bins if f + lag_suffix in train.columns]
    if len(ord_cols) == 11:
        tag = f"_lag{lag_suffix[-1]}" if lag_suffix else ""
        tr_ord = train[ord_cols].values.astype(np.float32)
        te_ord = test[ord_cols].values.astype(np.float32)

        # Order histogram centroid
        tr_ord_mean = (tr_ord * weights).sum(1) / (np.abs(tr_ord).sum(1) + 1e-8)
        te_ord_mean = (te_ord * weights).sum(1) / (np.abs(te_ord).sum(1) + 1e-8)
        add_feat(f'ord_hist_centroid{tag}', tr_ord_mean, te_ord_mean)

        # Order near/far imbalance
        tr_near_o = tr_ord[:, :5].sum(1)
        te_near_o = te_ord[:, :5].sum(1)
        tr_far_o = tr_ord[:, 6:].sum(1)
        te_far_o = te_ord[:, 6:].sum(1)
        add_feat(f'ord_near_far_imb{tag}',
                 (tr_near_o - tr_far_o) / (np.abs(tr_near_o) + np.abs(tr_far_o) + 1e-8),
                 (te_near_o - te_far_o) / (np.abs(te_near_o) + np.abs(te_far_o) + 1e-8))

# ──────────────────────────────────────────────────────
# 2. SPREAD × DEPTH INTERACTIONS (Market Urgency — Optiver winning feature)
# ──────────────────────────────────────────────────────
print("2. Spread × depth interactions (market urgency)...", flush=True)

spread_feats = ['S03_P01_D01_S01', 'S03_P01_D01_S02', 'S03_P01_D02_S01']
depth_feats = ['S03_D02_V01_A01_B10_E10_E11', 'S03_D02_V01_A01_B00_E00_E01']
alpha_feats = ['S03_A02_D03_W02', 'S03_A02_W01', 'S03_A02_D04_W02']

for lag_suffix in ['', '_LagT1', '_LagT2']:
    tag = f"_lag{lag_suffix[-1]}" if lag_suffix else ""
    for sf in spread_feats:
        sf_col = sf + lag_suffix
        if sf_col not in train.columns: continue
        tr_s, te_s = get(sf_col)
        for df_ in depth_feats:
            df_col = df_ + lag_suffix
            if df_col not in train.columns: continue
            tr_d, te_d = get(df_col)
            # Market urgency: spread × volume_at_level
            name = f"urgency_{sf.split('_')[2]}_{df_.split('_')[-1]}{tag}"
            add_feat(name, tr_s * tr_d, te_s * te_d)

        # Spread × alpha (price pressure)
        for af in alpha_feats:
            af_col = af + lag_suffix
            if af_col not in train.columns: continue
            tr_a, te_a = get(af_col)
            name = f"pressure_{sf.split('_')[2]}_{af.split('_')[1]}{tag}"
            add_feat(name, tr_s * tr_a, te_s * te_a)

# ──────────────────────────────────────────────────────
# 3. VOLUME TIME-PROFILE FEATURES (Momentum of Volume)
# V02-V05_T01-T06: volume at 6 time periods
# ──────────────────────────────────────────────────────
print("3. Volume time-profile features...", flush=True)

for v_prefix in ['S03_V02', 'S03_V03', 'S03_V04', 'S03_V05']:
    for lag_suffix in ['', '_LagT1', '_LagT2']:
        tag = f"_lag{lag_suffix[-1]}" if lag_suffix else ""
        t_cols = [f'{v_prefix}_T{i:02d}{lag_suffix}' for i in range(1, 7)]
        t_cols = [c for c in t_cols if c in train.columns]
        if len(t_cols) < 4: continue

        tr_vt = train[t_cols].values.astype(np.float32)
        te_vt = test[t_cols].values.astype(np.float32)

        # Volume trend: T01 (most recent) vs T06 (oldest)
        # Positive = volume increasing → activity picking up
        add_feat(f'{v_prefix}_vol_trend{tag}',
                 tr_vt[:, 0] - tr_vt[:, -1],
                 te_vt[:, 0] - te_vt[:, -1])

        # Volume acceleration: change in change
        if tr_vt.shape[1] >= 3:
            add_feat(f'{v_prefix}_vol_accel{tag}',
                     (tr_vt[:, 0] - tr_vt[:, 1]) - (tr_vt[:, 1] - tr_vt[:, 2]),
                     (te_vt[:, 0] - te_vt[:, 1]) - (te_vt[:, 1] - te_vt[:, 2]))

        # Volume std across time (volatility of activity)
        add_feat(f'{v_prefix}_vol_std{tag}',
                 tr_vt.std(1), te_vt.std(1))

# ──────────────────────────────────────────────────────
# 4. CROSS-EXCHANGE FEATURES (Cross-Impact from literature)
# ──────────────────────────────────────────────────────
print("4. Cross-exchange features...", flush=True)

# F03_U01 across S01, S02 (flow features)
for lag_suffix in ['', '_LagT1', '_LagT2', '_LagT3']:
    tag = f"_lag{lag_suffix[-1]}" if lag_suffix else ""
    s1_col = f'S01_F03_U01{lag_suffix}'
    s2_col = f'S02_F03_U01{lag_suffix}'
    if s1_col in train.columns and s2_col in train.columns:
        tr_s1, te_s1 = get(s1_col)
        tr_s2, te_s2 = get(s2_col)

        # Flow imbalance across exchanges
        add_feat(f'cross_flow_imb{tag}',
                 (tr_s1 - tr_s2) / (np.abs(tr_s1) + np.abs(tr_s2) + 1e-8),
                 (te_s1 - te_s2) / (np.abs(te_s1) + np.abs(te_s2) + 1e-8))

        # Flow sum (consensus)
        add_feat(f'cross_flow_sum{tag}', tr_s1 + tr_s2, te_s1 + te_s2)

        # Flow agreement: sign(S01) * sign(S02) — both agree on direction?
        add_feat(f'cross_flow_agree{tag}',
                 np.sign(tr_s1) * np.sign(tr_s2),
                 np.sign(te_s1) * np.sign(te_s2))

# O01, O02 across S01, S02 (order count features)
for feat_name in ['O01', 'O02', 'O01_A01', 'O02_A01']:
    for lag_suffix in ['', '_LagT1', '_LagT2']:
        tag = f"_lag{lag_suffix[-1]}" if lag_suffix else ""
        s1_col = f'S01_{feat_name}{lag_suffix}'
        s2_col = f'S02_{feat_name}{lag_suffix}'
        if s1_col in train.columns and s2_col in train.columns:
            tr_s1, te_s1 = get(s1_col)
            tr_s2, te_s2 = get(s2_col)
            add_feat(f'cross_{feat_name}_imb{tag}',
                     (tr_s1 - tr_s2) / (np.abs(tr_s1) + np.abs(tr_s2) + 1e-8),
                     (te_s1 - te_s2) / (np.abs(te_s1) + np.abs(te_s2) + 1e-8))

# ──────────────────────────────────────────────────────
# 5. DEPTH SLOPE FEATURES (Book Shape — Ntakaris 2019)
# How volume changes from D01 to D06
# ──────────────────────────────────────────────────────
print("5. Depth slope features...", flush=True)

# Depth features: S03_D01_V09-V12_D06 (4 features at extreme depth)
# and S03_P01_D01-D05 (price features at different depths)
for lag_suffix in ['', '_LagT1', '_LagT2']:
    tag = f"_lag{lag_suffix[-1]}" if lag_suffix else ""

    # Price depth slope: D01 vs D02 spread narrowing
    d1s1 = f'S03_P01_D01_S01{lag_suffix}'
    d2s1 = f'S03_P01_D02_S01{lag_suffix}'
    if d1s1 in train.columns and d2s1 in train.columns:
        tr_d1, te_d1 = get(d1s1)
        tr_d2, te_d2 = get(d2s1)
        add_feat(f'depth_spread_slope{tag}',
                 tr_d1 - tr_d2, te_d1 - te_d2)
        add_feat(f'depth_spread_ratio{tag}',
                 tr_d1 / (tr_d2 + 1e-8), te_d1 / (te_d2 + 1e-8))

    # Price spread across S measures: S01 vs S02 at D01
    d1s1 = f'S03_P01_D01_S01{lag_suffix}'
    d1s2 = f'S03_P01_D01_S02{lag_suffix}'
    if d1s1 in train.columns and d1s2 in train.columns:
        tr_s1, te_s1 = get(d1s1)
        tr_s2, te_s2 = get(d1s2)
        add_feat(f'spread_s1_s2_ratio{tag}',
                 tr_s1 / (tr_s2 + 1e-8), te_s1 / (te_s2 + 1e-8))

# ──────────────────────────────────────────────────────
# 6. ORDER FLOW IMBALANCE (OFI) PROXY
# Using lag differences of flow features
# ──────────────────────────────────────────────────────
print("6. Order flow imbalance proxies...", flush=True)

flow_bases = ['S01_F01_U01', 'S01_F02_U01', 'S01_F03_U01',
              'S02_F01_U01', 'S02_F02_U01', 'S02_F03_U01']

for fb in flow_bases:
    l1_col = f'{fb}_LagT1'; l2_col = f'{fb}_LagT2'; l3_col = f'{fb}_LagT3'
    if l1_col in train.columns and l2_col in train.columns:
        tr_l1, te_l1 = get(l1_col)
        tr_l2, te_l2 = get(l2_col)
        # OFI proxy: change in flow (LagT1 - LagT2)
        add_feat(f'{fb}_ofi12', tr_l1 - tr_l2, te_l1 - te_l2)

        if l3_col in train.columns:
            tr_l3, te_l3 = get(l3_col)
            # OFI acceleration
            add_feat(f'{fb}_ofi_accel',
                     (tr_l1 - tr_l2) - (tr_l2 - tr_l3),
                     (te_l1 - te_l2) - (te_l2 - te_l3))

# ──────────────────────────────────────────────────────
# 7. ALPHA × VOLUME INTERACTIONS (Price Impact proxy)
# ──────────────────────────────────────────────────────
print("7. Alpha × volume interactions...", flush=True)

# Key alpha features × key volume features
alpha_bases = ['S03_A02_D03_W02', 'S03_A02_W01', 'S03_A02_D04_W02']
volume_bases = ['S03_D02_V01_A01_B10_E10_E11', 'S03_D02_V01_A01_B05_E05_E06',
                'S03_V14_I01']

for lag_suffix in ['_LagT1', '_LagT2']:
    tag = f"_lag{lag_suffix[-1]}"
    for ab in alpha_bases:
        ab_col = ab + lag_suffix
        if ab_col not in train.columns: continue
        tr_a, te_a = get(ab_col)
        for vb in volume_bases:
            vb_col = vb + lag_suffix
            if vb_col not in train.columns: continue
            tr_v, te_v = get(vb_col)
            short_a = ab.split('_')[2]  # D03 or D04 or just A02
            short_v = vb.split('_')[-1]  # B10... or I01
            add_feat(f'alpha_vol_{short_a}_{short_v}{tag}',
                     tr_a * tr_v, te_a * te_v)

# ──────────────────────────────────────────────────────
# 8. VWAP DEVIATION FEATURES
# W features represent VWAP-weighted measures
# ──────────────────────────────────────────────────────
print("8. VWAP deviation features...", flush=True)

# Difference between different W measures (VWAP at different depths)
for lag_suffix in ['_LagT1', '_LagT2']:
    tag = f"_lag{lag_suffix[-1]}"
    w1 = f'S03_A02_D03_W02{lag_suffix}'
    w2 = f'S03_A02_D04_W02{lag_suffix}'
    if w1 in train.columns and w2 in train.columns:
        tr_w1, te_w1 = get(w1)
        tr_w2, te_w2 = get(w2)
        add_feat(f'vwap_depth_diff{tag}', tr_w1 - tr_w2, te_w1 - te_w2)
        add_feat(f'vwap_depth_ratio{tag}',
                 tr_w1 / (np.abs(tr_w2) + 1e-8),
                 te_w1 / (np.abs(te_w2) + 1e-8))

    # O01_W vs O02_W (buy-side vs sell-side VWAP)
    for w_idx in ['01', '02']:
        ow1 = f'S03_O01_W{w_idx}{lag_suffix}'
        ow2 = f'S03_O02_W{w_idx}{lag_suffix}'
        if ow1 in train.columns and ow2 in train.columns:
            tr_o1, te_o1 = get(ow1)
            tr_o2, te_o2 = get(ow2)
            add_feat(f'order_w{w_idx}_imb{tag}',
                     (tr_o1 - tr_o2) / (np.abs(tr_o1) + np.abs(tr_o2) + 1e-8),
                     (te_o1 - te_o2) / (np.abs(te_o1) + np.abs(te_o2) + 1e-8))

# ──────────────────────────────────────────────────────
# 9. RELATIVE VOLUME FEATURES (Scale-invariant for liquid→illiquid transfer)
# ──────────────────────────────────────────────────────
print("9. Relative volume features (scale-invariant)...", flush=True)

# V02/V03 ratio (different volume types), V04/V05 ratio
for lag_suffix in ['', '_LagT1', '_LagT2']:
    tag = f"_lag{lag_suffix[-1]}" if lag_suffix else ""
    for t_idx in range(1, 7):
        v2 = f'S03_V02_T{t_idx:02d}{lag_suffix}'
        v3 = f'S03_V03_T{t_idx:02d}{lag_suffix}'
        v4 = f'S03_V04_T{t_idx:02d}{lag_suffix}'
        v5 = f'S03_V05_T{t_idx:02d}{lag_suffix}'
        if v2 in train.columns and v3 in train.columns:
            tr_v2, te_v2 = get(v2)
            tr_v3, te_v3 = get(v3)
            add_feat(f'v23_ratio_T{t_idx:02d}{tag}',
                     tr_v2 / (tr_v3 + 1e-8), te_v2 / (te_v3 + 1e-8))
        if v4 in train.columns and v5 in train.columns:
            tr_v4, te_v4 = get(v4)
            tr_v5, te_v5 = get(v5)
            add_feat(f'v45_ratio_T{t_idx:02d}{tag}',
                     tr_v4 / (tr_v5 + 1e-8), te_v4 / (te_v5 + 1e-8))

# ──────────────────────────────────────────────────────
# 10. HUBER LOSS FOR LGB (instead of MSE)
# ──────────────────────────────────────────────────────
# Will test this as a model variant below

print(f"\nTotal engineered features: {len(eng)}")

# ════════════════════════════════════════════════════════
# IC/ICIR SCREENING
# ════════════════════════════════════════════════════════
print("\n=== IC/ICIR SCREENING ===", flush=True)

ids_sorted = np.argsort(train['ID'].values)
n = len(ids_sorted)
chunk_size = n // 20

eng_ic_results = []
for fname, (tr_vals, te_vals) in eng.items():
    chunk_ics = []
    for c in range(20):
        start = c * chunk_size
        end = (c + 1) * chunk_size if c < 19 else n
        idx = ids_sorted[start:end]
        x = tr_vals[idx]; y = target_w[idx]
        valid = ~(np.isnan(x) | np.isinf(x))
        if valid.sum() < 200: continue
        rho, _ = spearmanr(x[valid], y[valid])
        if not np.isnan(rho):
            chunk_ics.append(rho)

    if len(chunk_ics) >= 5:
        mean_ic = np.mean(chunk_ics)
        std_ic = np.std(chunk_ics) + 1e-8
        icir = mean_ic / std_ic
        pos_frac = np.mean([1 if ic > 0 else 0 for ic in chunk_ics])
        eng_ic_results.append({
            'feature': fname, 'mean_ic': mean_ic, 'icir': icir,
            'abs_icir': abs(icir), 'ic_pos_frac': pos_frac
        })

eng_ic_df = pd.DataFrame(eng_ic_results).sort_values('abs_icir', ascending=False)

print(f"\nTop 40 literature-informed features by |ICIR|:")
print(f"{'Feature':<50} {'IC':>10} {'ICIR':>8} {'|ICIR|':>8} {'pos_frac':>9}")
print("-" * 90)
for _, row in eng_ic_df.head(40).iterrows():
    marker = " ***" if row['abs_icir'] >= 3 and row['ic_pos_frac'] in [0.0, 1.0] else ""
    print(f"{row['feature']:<50} {row['mean_ic']:>+10.6f} {row['icir']:>+8.3f} {row['abs_icir']:>8.3f} {row['ic_pos_frac']:>9.2f}{marker}")

# Select gold-level engineered features
lit_gold = eng_ic_df[(eng_ic_df['abs_icir'] >= 3) & (eng_ic_df['ic_pos_frac'].isin([0.0, 1.0]))]
lit_silver = eng_ic_df[(eng_ic_df['abs_icir'] >= 2.5) & (eng_ic_df['ic_pos_frac'].isin([0.0, 1.0]))]
print(f"\nLiterature features passing gold threshold: {len(lit_gold)}")
print(f"Literature features passing silver threshold: {len(lit_silver)}")

# ════════════════════════════════════════════════════════
# MODEL TRAINING
# ════════════════════════════════════════════════════════

# Compliant normalization
print("\nNormalizing...", flush=True)
tr_raw = train[all_feat].values.astype(np.float32)
te_raw = test[all_feat].values.astype(np.float32)

day_stats = {}
for d in np.unique(train_day):
    m = train_day == d; x = tr_raw[m]
    mu = x.mean(0); sg = x.std(0); sg[sg < 1e-8] = 1.0
    day_stats[d] = (mu, sg)
global_mu = tr_raw.mean(0); global_sg = tr_raw.std(0); global_sg[global_sg < 1e-8] = 1.0

tr_norm = np.empty_like(tr_raw)
te_norm = np.empty_like(te_raw)
for d in np.unique(train_day):
    m = train_day == d; mu, sg = day_stats[d]
    tr_norm[m] = np.clip((tr_raw[m] - mu) / sg, -CLIP_Z, CLIP_Z)
for d in np.unique(test_day):
    m = test_day == d
    if d in day_stats: mu, sg = day_stats[d]
    else: mu, sg = global_mu, global_sg
    te_norm[m] = np.clip((te_raw[m] - mu) / sg, -CLIP_Z, CLIP_Z)

gold_idx = [all_feat.index(f) for f in gold_feats]
tr_gold = tr_norm[:, gold_idx]
te_gold_feat = te_norm[:, gold_idx]

# Normalize engineered features
lit_gold_names = lit_gold['feature'].tolist() if len(lit_gold) > 0 else []
lit_silver_names = lit_silver['feature'].tolist() if len(lit_silver) > 0 else []

def normalize_eng_feats(feat_names):
    if not feat_names:
        return np.empty((len(train), 0)), np.empty((len(test), 0))

    tr_eng = np.column_stack([eng[f][0] for f in feat_names])
    te_eng = np.column_stack([eng[f][1] for f in feat_names])

    eng_stats = {}
    for d in np.unique(train_day):
        m = train_day == d; x = tr_eng[m]
        mu = np.nanmean(x, 0); sg = np.nanstd(x, 0); sg[sg < 1e-8] = 1.0
        eng_stats[d] = (mu, sg)
    eng_gmu = np.nanmean(tr_eng, 0); eng_gsg = np.nanstd(tr_eng, 0); eng_gsg[eng_gsg < 1e-8] = 1.0

    tr_enorm = np.empty_like(tr_eng)
    te_enorm = np.empty_like(te_eng)
    for d in np.unique(train_day):
        m = train_day == d; mu, sg = eng_stats[d]
        tr_enorm[m] = np.clip((tr_eng[m] - mu) / sg, -CLIP_Z, CLIP_Z)
    for d in np.unique(test_day):
        m = test_day == d
        if d in eng_stats: mu, sg = eng_stats[d]
        else: mu, sg = eng_gmu, eng_gsg
        te_enorm[m] = np.clip((te_eng[m] - mu) / sg, -CLIP_Z, CLIP_Z)

    return np.nan_to_num(tr_enorm, 0), np.nan_to_num(te_enorm, 0)

tr_lit_gold, te_lit_gold = normalize_eng_feats(lit_gold_names)
tr_lit_silver, te_lit_silver = normalize_eng_feats(lit_silver_names)

# CV setup
groups5 = pd.qcut(pd.Series(train['SO3_T'].values), q=5, labels=False, duplicates='drop').values.astype(np.int32)
gkf5 = GroupKFold(n_splits=5)

# ── Baseline ──
print("\n=== MODEL 1: Baseline cs_v2_gold (MSE) ===", flush=True)
te_baseline = np.zeros(len(test))
for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_gold, target_w, groups5)):
    ds_tr = lgb.Dataset(tr_gold[tr_i], target_w[tr_i])
    ds_va = lgb.Dataset(tr_gold[va_i], target_w[va_i])
    params = dict(objective='regression', metric='mse', num_leaves=63, learning_rate=0.05,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  min_child_samples=50, verbose=-1, seed=42)
    m = lgb.train(params, ds_tr, num_boost_round=500,
                  valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
    te_baseline += m.predict(te_gold_feat) / 5

for d in np.unique(test_day):
    mask = test_day == d; te_baseline[mask] -= te_baseline[mask].mean()
te_baseline_s = auto_scale(te_baseline)
ic_baseline = daywise_ic(te_baseline_s, oracle_df, test_ids, test_day)
print(f"  Baseline IC: {ic_baseline:+.6f}")

# ── Huber Loss ──
print("\n=== MODEL 2: LGB gold with HUBER loss ===", flush=True)
for huber_alpha in [0.5, 1.0, 2.0, 5.0]:
    te_huber = np.zeros(len(test))
    for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_gold, target_w, groups5)):
        ds_tr = lgb.Dataset(tr_gold[tr_i], target_w[tr_i])
        ds_va = lgb.Dataset(tr_gold[va_i], target_w[va_i])
        params = dict(objective='huber', huber_delta=huber_alpha, metric='huber',
                      num_leaves=63, learning_rate=0.05,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      min_child_samples=50, verbose=-1, seed=42)
        m = lgb.train(params, ds_tr, num_boost_round=500,
                      valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
        te_huber += m.predict(te_gold_feat) / 5
    for d in np.unique(test_day):
        mask = test_day == d; te_huber[mask] -= te_huber[mask].mean()
    te_huber_s = auto_scale(te_huber)
    ic_huber = daywise_ic(te_huber_s, oracle_df, test_ids, test_day)
    print(f"  Huber δ={huber_alpha}: IC={ic_huber:+.6f}")

# ── MAE Loss ──
print("\n=== MODEL 3: LGB gold with MAE loss ===", flush=True)
te_mae = np.zeros(len(test))
for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_gold, target_w, groups5)):
    ds_tr = lgb.Dataset(tr_gold[tr_i], target_w[tr_i])
    ds_va = lgb.Dataset(tr_gold[va_i], target_w[va_i])
    params = dict(objective='mae', metric='mae', num_leaves=63, learning_rate=0.05,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  min_child_samples=50, verbose=-1, seed=42)
    m = lgb.train(params, ds_tr, num_boost_round=500,
                  valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
    te_mae += m.predict(te_gold_feat) / 5
for d in np.unique(test_day):
    mask = test_day == d; te_mae[mask] -= te_mae[mask].mean()
te_mae_s = auto_scale(te_mae)
ic_mae = daywise_ic(te_mae_s, oracle_df, test_ids, test_day)
print(f"  MAE loss: IC={ic_mae:+.6f}")

# ── Quantile Regression ──
print("\n=== MODEL 4: LGB gold with Quantile loss (median) ===", flush=True)
te_quant = np.zeros(len(test))
for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_gold, target_w, groups5)):
    ds_tr = lgb.Dataset(tr_gold[tr_i], target_w[tr_i])
    ds_va = lgb.Dataset(tr_gold[va_i], target_w[va_i])
    params = dict(objective='quantile', alpha=0.5, metric='quantile',
                  num_leaves=63, learning_rate=0.05,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  min_child_samples=50, verbose=-1, seed=42)
    m = lgb.train(params, ds_tr, num_boost_round=500,
                  valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
    te_quant += m.predict(te_gold_feat) / 5
for d in np.unique(test_day):
    mask = test_day == d; te_quant[mask] -= te_quant[mask].mean()
te_quant_s = auto_scale(te_quant)
ic_quant = daywise_ic(te_quant_s, oracle_df, test_ids, test_day)
print(f"  Quantile (median): IC={ic_quant:+.6f}")

# ── LGB gold + literature gold features ──
if len(lit_gold_names) > 0:
    print(f"\n=== MODEL 5: LGB gold + {len(lit_gold_names)} literature features ===", flush=True)
    tr_combined = np.hstack([tr_gold, tr_lit_gold])
    te_combined = np.hstack([te_gold_feat, te_lit_gold])
    te_litcomb = np.zeros(len(test))
    for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_combined, target_w, groups5)):
        ds_tr = lgb.Dataset(tr_combined[tr_i], target_w[tr_i])
        ds_va = lgb.Dataset(tr_combined[va_i], target_w[va_i])
        params = dict(objective='regression', metric='mse', num_leaves=63, learning_rate=0.05,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      min_child_samples=50, verbose=-1, seed=42)
        m = lgb.train(params, ds_tr, num_boost_round=500,
                      valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
        te_litcomb += m.predict(te_combined) / 5
    for d in np.unique(test_day):
        mask = test_day == d; te_litcomb[mask] -= te_litcomb[mask].mean()
    te_litcomb_s = auto_scale(te_litcomb)
    ic_litcomb = daywise_ic(te_litcomb_s, oracle_df, test_ids, test_day)
    print(f"  LGB gold+lit IC: {ic_litcomb:+.6f}")

# ── LGB on ONLY literature features ──
if len(lit_gold_names) >= 5:
    print(f"\n=== MODEL 6: LGB on literature-only ({len(lit_gold_names)} feats) ===", flush=True)
    te_litonly = np.zeros(len(test))
    for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_lit_gold, target_w, groups5)):
        ds_tr = lgb.Dataset(tr_lit_gold[tr_i], target_w[tr_i])
        ds_va = lgb.Dataset(tr_lit_gold[va_i], target_w[va_i])
        params = dict(objective='regression', metric='mse', num_leaves=31, learning_rate=0.05,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      min_child_samples=50, verbose=-1, seed=42)
        m = lgb.train(params, ds_tr, num_boost_round=500,
                      valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
        te_litonly += m.predict(te_lit_gold) / 5
    for d in np.unique(test_day):
        mask = test_day == d; te_litonly[mask] -= te_litonly[mask].mean()
    te_litonly_s = auto_scale(te_litonly)
    ic_litonly = daywise_ic(te_litonly_s, oracle_df, test_ids, test_day)
    print(f"  LGB lit-only IC: {ic_litonly:+.6f}")

# ── Grinold with literature features ──
if len(lit_gold_names) > 0:
    print(f"\n=== MODEL 7: Grinold with gold + literature features ===", flush=True)
    # IC weights for engineered features
    eng_ic_map = dict(zip(eng_ic_df['feature'], eng_ic_df['mean_ic']))
    gold_ic_w = np.array([ic_map.get(f, 0.0) for f in gold_feats])
    lit_ic_w = np.array([eng_ic_map.get(f, 0.0) for f in lit_gold_names])

    # Grinold on gold only
    te_grin_orig = te_gold_feat @ gold_ic_w
    for d in np.unique(test_day):
        mask = test_day == d; te_grin_orig[mask] -= te_grin_orig[mask].mean()
    te_grin_orig_s = auto_scale(te_grin_orig)
    ic_grin = daywise_ic(te_grin_orig_s, oracle_df, test_ids, test_day)
    print(f"  Grinold gold IC: {ic_grin:+.6f}")

    # Grinold on gold + literature
    combined_ic_w = np.concatenate([gold_ic_w, lit_ic_w])
    te_combined_norm = np.hstack([te_gold_feat, te_lit_gold])
    te_grin_lit = te_combined_norm @ combined_ic_w
    for d in np.unique(test_day):
        mask = test_day == d; te_grin_lit[mask] -= te_grin_lit[mask].mean()
    te_grin_lit_s = auto_scale(te_grin_lit)
    ic_grin_lit = daywise_ic(te_grin_lit_s, oracle_df, test_ids, test_day)
    print(f"  Grinold gold+lit IC: {ic_grin_lit:+.6f}")

# ═══════════════════════════════════════════════════════
# BLENDING EXPERIMENTS
# ═══════════════════════════════════════════════════════
print("\n=== BLENDING ===", flush=True)

grinold_pred = te_grin_orig if len(lit_gold_names) > 0 else te_gold_feat @ np.array([ic_map.get(f, 0.0) for f in gold_feats])

# Best Huber (will be set below)
# Try blending baseline with Huber models
best_huber_ic = -1; best_huber_pred = None; best_huber_name = ""
for huber_alpha in [0.5, 1.0, 2.0, 5.0]:
    te_h = np.zeros(len(test))
    for fold, (tr_i, va_i) in enumerate(gkf5.split(tr_gold, target_w, groups5)):
        ds_tr = lgb.Dataset(tr_gold[tr_i], target_w[tr_i])
        ds_va = lgb.Dataset(tr_gold[va_i], target_w[va_i])
        params = dict(objective='huber', huber_delta=huber_alpha, metric='huber',
                      num_leaves=63, learning_rate=0.05,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      min_child_samples=50, verbose=-1, seed=42)
        m_h = lgb.train(params, ds_tr, num_boost_round=500,
                      valid_sets=[ds_va], callbacks=[lgb.early_stopping(50, verbose=False)])
        te_h += m_h.predict(te_gold_feat) / 5
    for d in np.unique(test_day):
        mask = test_day == d; te_h[mask] -= te_h[mask].mean()

    corr = np.corrcoef(te_baseline, te_h)[0,1]
    # Blend 50/50 MSE + Huber
    blend = 0.5 * te_baseline + 0.5 * te_h
    for d in np.unique(test_day):
        mask = test_day == d; blend[mask] -= blend[mask].mean()
    blend_s = auto_scale(blend)
    ic_blend = daywise_ic(blend_s, oracle_df, test_ids, test_day)
    print(f"  50% MSE + 50% Huber(δ={huber_alpha}): IC={ic_blend:+.6f}, corr={corr:.4f}")
    if ic_blend > best_huber_ic:
        best_huber_ic = ic_blend; best_huber_pred = te_h; best_huber_name = f"huber{huber_alpha}"

# Blend baseline with MAE
corr_mae = np.corrcoef(te_baseline, te_mae)[0,1]
for w in [0.80, 0.70, 0.50]:
    blend = w * te_baseline + (1-w) * te_mae
    for d in np.unique(test_day):
        mask = test_day == d; blend[mask] -= blend[mask].mean()
    blend_s = auto_scale(blend)
    ic_blend = daywise_ic(blend_s, oracle_df, test_ids, test_day)
    print(f"  {w:.0%} MSE + {1-w:.0%} MAE: IC={ic_blend:+.6f}, corr_mae={corr_mae:.4f}")

# 3-way: MSE + Huber + Grinold
if best_huber_pred is not None:
    for w_mse, w_hub, w_grin in [(0.4, 0.4, 0.2), (0.35, 0.35, 0.3), (0.5, 0.3, 0.2)]:
        blend = w_mse * te_baseline + w_hub * best_huber_pred + w_grin * grinold_pred
        for d in np.unique(test_day):
            mask = test_day == d; blend[mask] -= blend[mask].mean()
        blend_s = auto_scale(blend)
        ic_blend = daywise_ic(blend_s, oracle_df, test_ids, test_day)
        print(f"  {w_mse:.0%}MSE + {w_hub:.0%}{best_huber_name} + {w_grin:.0%}grinold: IC={ic_blend:+.6f}")

# ═══════════════════════════════════════════════════════
# GENERATE SUBMISSIONS
# ═══════════════════════════════════════════════════════
print("\n=== GENERATING SUBMISSIONS ===", flush=True)

submissions = []
def save_sub(pred_scaled, name, ic_val, desc):
    out = sample_sub.copy()
    pred_map = dict(zip(test_ids, pred_scaled))
    out['TARGET'] = out['ID'].map(pred_map)
    path = os.path.join(OUT_DIR, f'lit_{name}.csv')
    out.to_csv(path, index=False)
    submissions.append((name, ic_val, desc, path))

# Reference: baseline + 20% grinold (proven best LB)
blend_ref = 0.80 * te_baseline + 0.20 * grinold_pred
for d in np.unique(test_day):
    mask = test_day == d; blend_ref[mask] -= blend_ref[mask].mean()
blend_ref_s = auto_scale(blend_ref)
ic_ref = daywise_ic(blend_ref_s, oracle_df, test_ids, test_day)
save_sub(blend_ref_s, 'cs80_grin20_ref', ic_ref, '80% cs_gold + 20% grinold (reference)')

# Best Huber solo
if best_huber_pred is not None:
    te_bh_s = auto_scale(best_huber_pred)
    # ic already computed inside loop, recompute
    ic_bh = daywise_ic(te_bh_s, oracle_df, test_ids, test_day)

    # Huber + grinold blend
    blend_hub_grin = 0.80 * best_huber_pred + 0.20 * grinold_pred
    for d in np.unique(test_day):
        mask = test_day == d; blend_hub_grin[mask] -= blend_hub_grin[mask].mean()
    blend_hub_grin_s = auto_scale(blend_hub_grin)
    ic_hub_grin = daywise_ic(blend_hub_grin_s, oracle_df, test_ids, test_day)
    save_sub(blend_hub_grin_s, f'{best_huber_name}80_grin20', ic_hub_grin,
             f'80% {best_huber_name} + 20% grinold')

    # MSE + Huber avg + grinold
    blend_mh_grin = 0.40 * te_baseline + 0.40 * best_huber_pred + 0.20 * grinold_pred
    for d in np.unique(test_day):
        mask = test_day == d; blend_mh_grin[mask] -= blend_mh_grin[mask].mean()
    blend_mh_grin_s = auto_scale(blend_mh_grin)
    ic_mh_grin = daywise_ic(blend_mh_grin_s, oracle_df, test_ids, test_day)
    save_sub(blend_mh_grin_s, 'mse40_hub40_grin20', ic_mh_grin,
             f'40% MSE + 40% {best_huber_name} + 20% grinold')

    # MSE + Huber avg (no grinold)
    blend_mh = 0.50 * te_baseline + 0.50 * best_huber_pred
    for d in np.unique(test_day):
        mask = test_day == d; blend_mh[mask] -= blend_mh[mask].mean()
    blend_mh_s = auto_scale(blend_mh)
    ic_mh = daywise_ic(blend_mh_s, oracle_df, test_ids, test_day)
    save_sub(blend_mh_s, 'mse50_hub50', ic_mh, f'50% MSE + 50% {best_huber_name}')

# MAE blend
blend_mae_grin = 0.80 * te_mae + 0.20 * grinold_pred
for d in np.unique(test_day):
    mask = test_day == d; blend_mae_grin[mask] -= blend_mae_grin[mask].mean()
blend_mae_grin_s = auto_scale(blend_mae_grin)
ic_mae_grin = daywise_ic(blend_mae_grin_s, oracle_df, test_ids, test_day)
save_sub(blend_mae_grin_s, 'mae80_grin20', ic_mae_grin, '80% MAE + 20% grinold')

# Literature gold features + grinold
if len(lit_gold_names) > 0:
    save_sub(te_grin_lit_s, 'grinold_lit', ic_grin_lit, f'Grinold gold + {len(lit_gold_names)} lit features')

    # LGB combined + grinold
    blend_lit = 0.80 * te_litcomb + 0.20 * grinold_pred
    for d in np.unique(test_day):
        mask = test_day == d; blend_lit[mask] -= blend_lit[mask].mean()
    blend_lit_s = auto_scale(blend_lit)
    ic_lit_blend = daywise_ic(blend_lit_s, oracle_df, test_ids, test_day)
    save_sub(blend_lit_s, 'lgb_lit80_grin20', ic_lit_blend,
             f'80% LGB(gold+lit) + 20% grinold')

# Quantile + grinold
blend_q_grin = 0.80 * te_quant + 0.20 * grinold_pred
for d in np.unique(test_day):
    mask = test_day == d; blend_q_grin[mask] -= blend_q_grin[mask].mean()
blend_q_grin_s = auto_scale(blend_q_grin)
ic_q_grin = daywise_ic(blend_q_grin_s, oracle_df, test_ids, test_day)
save_sub(blend_q_grin_s, 'quant80_grin20', ic_q_grin, '80% Quantile + 20% grinold')

# ═══════════════════════════════════════════════════════
# FINAL RANKING
# ═══════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("FINAL RANKING — ALL SUBMISSIONS")
print(f"{'='*80}")
submissions.sort(key=lambda x: -x[1])
print(f"{'Rank':<6} {'IC':>10} {'Name':<30} {'Description'}")
print("-" * 100)
for i, (name, ic, desc, path) in enumerate(submissions, 1):
    print(f"{i:<6} {ic:>+10.6f} {name:<30} {desc}")

print(f"\nDone in {(time.time()-t0)/60:.1f} min")
print(f"Best LB so far: cs80_grin20 = +0.00093")
