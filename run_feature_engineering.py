"""
Feature Engineering — notebooks/02_feature_engineering.ipynb equivalent
Full implementation per FEATURE_ENGINEERING_INSTRUCTIONS.md
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import rankdata
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
import pickle
import os
import gc
import warnings
warnings.filterwarnings('ignore')

os.chdir('/Users/malaymishra/Desktop/quant_ml_project')
os.makedirs('data/processed', exist_ok=True)
os.makedirs('outputs/feature_engineering', exist_ok=True)
os.makedirs('outputs/feature_engineering/plots', exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 70)
print("FEATURE ENGINEERING — notebooks/02_feature_engineering.ipynb")
print("=" * 70)

# ── Environment Setup ─────────────────────────────────────────────────────
print("\n--- Environment Setup ---")
raw_files  = os.listdir('data/raw/')
train_file = [f for f in raw_files if 'train' in f.lower() and f.endswith('.parquet')][0]
test_file  = [f for f in raw_files if 'test'  in f.lower() and f.endswith('.parquet')][0]
print(f"Using train: {train_file}, test: {test_file}")

train = pd.read_parquet(f'data/raw/{train_file}')
test  = pd.read_parquet(f'data/raw/{test_file}')

with open('outputs/eda/summaries/eda_decisions.pkl', 'rb') as f:
    decisions = pickle.load(f)
with open('outputs/eda/summaries/taxonomy.pkl', 'rb') as f:
    taxonomy = pickle.load(f)

adversarial_drop = []
adv_path = 'outputs/eda/summaries/adversarial_drop_list.pkl'
if os.path.exists(adv_path):
    with open(adv_path, 'rb') as f:
        adversarial_drop = pickle.load(f)
    print(f"Loaded {len(adversarial_drop)} adversarially-identified drop features")
else:
    print("WARNING: No adversarial drop list found.")

feature_cols    = [c for c in train.columns if c not in ['ID', 'TARGET']]
base_features   = taxonomy['base_features']
lag_t1_features = taxonomy['lag_t1_features']
lag_t2_features = taxonomy['lag_t2_features']
lag_t3_features = taxonomy['lag_t3_features']
family_map      = taxonomy['family_map']

def extract_family(col):
    col_clean = col.replace('_LagT1','').replace('_LagT2','').replace('_LagT3','')
    parts = col_clean.split('_')
    return parts[0] if len(parts) >= 1 else 'UNKNOWN'

for col in feature_cols:
    train[col] = train[col].astype(np.float32)
for col in [c for c in test.columns if c != 'ID']:
    test[col] = test[col].astype(np.float32)

train_eng = train.copy()
test_eng  = test.copy()
feature_registry = {}

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Starting feature count: {len(feature_cols)}")
gc.collect()

# ── STEP 0 — Drop Useless Features ──────────────────────────────────────────
print("\n--- STEP 0: Drop Useless Features ---")

all_drops = list(set(decisions['drop'] + adversarial_drop))
cols_to_drop = [c for c in all_drops if c in train_eng.columns]
train_eng.drop(columns=cols_to_drop, inplace=True, errors='ignore')
test_eng.drop(columns=cols_to_drop,  inplace=True, errors='ignore')

print(f"EDA drops          : {len(decisions['drop'])}")
print(f"Adversarial drops  : {len(adversarial_drop)}")
print(f"Total dropped      : {len(cols_to_drop)}")

# Drop perfectly correlated duplicates (r=1.000)
high_corr = pd.read_csv('outputs/eda/summaries/high_corr_pairs.csv')
perfect_drop = list(set(
    row['feature_b'] for _, row in high_corr.iterrows()
    if abs(row['correlation']) >= 0.9999 and row['feature_b'] in train_eng.columns
))
train_eng.drop(columns=perfect_drop, inplace=True, errors='ignore')
test_eng.drop(columns=perfect_drop,  inplace=True, errors='ignore')
print(f"Perfect-corr drops : {len(perfect_drop)} (r=1.000 duplicates)")
print(f"Remaining features : {train_eng.shape[1]}")

working_features = [c for c in feature_cols if c in train_eng.columns]
working_base     = [c for c in base_features    if c in train_eng.columns]
working_lag1     = [c for c in lag_t1_features   if c in train_eng.columns]
working_lag2     = [c for c in lag_t2_features   if c in train_eng.columns]
working_lag3     = [c for c in lag_t3_features   if c in train_eng.columns]
gc.collect()

# ── STEP 1 — Missingness Indicator Features ──────────────────────────────────
print("\n--- STEP 1: Missingness Indicator Features ---")
# EDA: 0 features with missingness signal
miss_indicator_cols = []
for col in decisions['needs_indicator']:
    if col not in train_eng.columns:
        continue
    flag_col = f"{col}_was_missing"
    train_eng[flag_col] = train_eng[col].isnull().astype(np.float32)
    test_eng[flag_col]  = test_eng[col].isnull().astype(np.float32) \
                          if col in test_eng.columns else 0.0
    miss_indicator_cols.append(flag_col)
    feature_registry[flag_col] = f"Binary: 1 if {col} is NaN, 0 otherwise"
print(f"Created {len(miss_indicator_cols)} missingness indicator features.")

# ── STEP 2 — Median Imputation ───────────────────────────────────────────────
print("\n--- STEP 2: Median Imputation ---")
imputation_medians = {}
for col in working_features:
    med = float(train_eng[col].median())
    imputation_medians[col] = med
    train_eng[col] = train_eng[col].fillna(med)
    if col in test_eng.columns:
        test_eng[col] = test_eng[col].fillna(med)

with open('outputs/feature_engineering/imputation_medians.pkl', 'wb') as f:
    pickle.dump(imputation_medians, f)
print("SAVED: outputs/feature_engineering/imputation_medians.pkl")

nan_remaining = train_eng[working_features].isnull().sum().sum()
assert nan_remaining == 0, f"NaN remain in train: {nan_remaining}"
print("All clear — no NaN in working features.")
gc.collect()

# ── STEP 3 — Volatility-Normalised Lag Features ──────────────────────────────
print("\n--- STEP 3: Volatility-Normalised Lag Features ---")
vol_norm_features = []

for base_col in working_base:
    if base_col not in train_eng.columns:
        continue
    base_std = float(train_eng[base_col].std()) + 1e-8

    for lag_col in [base_col + '_LagT1', base_col + '_LagT2', base_col + '_LagT3']:
        if lag_col not in train_eng.columns:
            continue
        new_col = f"{lag_col}_volnorm"
        train_eng[new_col] = (train_eng[lag_col] / base_std).astype(np.float32)
        if lag_col in test_eng.columns:
            test_eng[new_col] = (test_eng[lag_col] / base_std).astype(np.float32)
        vol_norm_features.append(new_col)
        feature_registry[new_col] = f"LagT_volnorm: {lag_col} / std({base_col})"

print(f"Created {len(vol_norm_features)} volatility-normalised lag features.")
gc.collect()

# ── STEP 4 — Lag Ratio Features ──────────────────────────────────────────────
print("\n--- STEP 4: Lag Ratio Features ---")
lag_ratio_features = []

for base_col in working_base:
    l1 = base_col + '_LagT1'
    l2 = base_col + '_LagT2'
    l3 = base_col + '_LagT3'

    if l1 not in train_eng.columns or l2 not in train_eng.columns:
        continue

    col_21 = f"{base_col}_lagrat_T2_T1"
    train_eng[col_21] = (train_eng[l2] / (train_eng[l1].abs() + 1e-8)).clip(-10, 10).astype(np.float32)
    if l1 in test_eng.columns and l2 in test_eng.columns:
        test_eng[col_21] = (test_eng[l2] / (test_eng[l1].abs() + 1e-8)).clip(-10, 10).astype(np.float32)
    lag_ratio_features.append(col_21)
    feature_registry[col_21] = f"Lag ratio T2/|T1| for {base_col}"

    if l3 not in train_eng.columns:
        continue

    col_32 = f"{base_col}_lagrat_T3_T2"
    train_eng[col_32] = (train_eng[l3] / (train_eng[l2].abs() + 1e-8)).clip(-10, 10).astype(np.float32)
    if l2 in test_eng.columns and l3 in test_eng.columns:
        test_eng[col_32] = (test_eng[l3] / (test_eng[l2].abs() + 1e-8)).clip(-10, 10).astype(np.float32)
    lag_ratio_features.append(col_32)
    feature_registry[col_32] = f"Lag ratio T3/|T2| for {base_col}"

print(f"Created {len(lag_ratio_features)} lag ratio features.")

# ── STEP 5 — Lag Convergence / Sign Agreement ────────────────────────────────
print("\n--- STEP 5: Lag Convergence / Sign Agreement ---")
convergence_features = []

for base_col in working_base:
    l1 = base_col + '_LagT1'
    l2 = base_col + '_LagT2'
    l3 = base_col + '_LagT3'

    if l1 not in train_eng.columns or l2 not in train_eng.columns:
        continue

    conv_col = f"{base_col}_lag_convergence"
    train_eng[conv_col] = (train_eng[l1] - train_eng[l2] / 2).astype(np.float32)
    if l1 in test_eng.columns and l2 in test_eng.columns:
        test_eng[conv_col] = (test_eng[l1] - test_eng[l2] / 2).astype(np.float32)
    convergence_features.append(conv_col)
    feature_registry[conv_col] = f"Lag convergence: LagT1 - LagT2/2 for {base_col}"

    if l3 not in train_eng.columns:
        continue

    sign_col = f"{base_col}_lag_sign_agree"
    train_eng[sign_col] = (np.sign(train_eng[l1]) * np.sign(train_eng[l2]) * np.sign(train_eng[l3])).astype(np.float32)
    if all(c in test_eng.columns for c in [l1, l2, l3]):
        test_eng[sign_col] = (np.sign(test_eng[l1]) * np.sign(test_eng[l2]) * np.sign(test_eng[l3])).astype(np.float32)
    convergence_features.append(sign_col)
    feature_registry[sign_col] = f"Sign agreement across all 3 lags for {base_col}"

print(f"Created {len(convergence_features)} convergence/sign features.")

# ── STEP 6 — Cross-Sectional Rank Transforms ─────────────────────────────────
print("\n--- STEP 6: Cross-Sectional Rank Transforms (ALL 444 KS-drift features) ---")
rank_features = []

rank_candidates = list(set(
    decisions['high_ks_shift'] +
    decisions.get('top_icir_features', decisions['top_signals'])
))
rank_candidates = [c for c in rank_candidates if c in train_eng.columns]
print(f"Applying rank transform to {len(rank_candidates)} features...")

n_train = len(train_eng)
n_test  = len(test_eng)

for col in rank_candidates:
    rank_col = f"{col}_rank"
    train_eng[rank_col] = (rankdata(train_eng[col].values) / n_train).astype(np.float32)
    test_eng[rank_col]  = (rankdata(
        test_eng[col].values if col in test_eng.columns else np.zeros(n_test)
    ) / n_test).astype(np.float32)
    rank_features.append(rank_col)
    feature_registry[rank_col] = f"Cross-sectional rank / N of {col} (KS-drift corrected)"

print(f"Created {len(rank_features)} rank-transformed features.")
gc.collect()

# ── STEP 7 — Z-Score Normalisation ───────────────────────────────────────────
print("\n--- STEP 7: Z-Score Normalisation ---")
zscore_features = []

zscore_candidates = list(set(
    decisions.get('top_icir_features', decisions['top_signals'])[:80] +
    working_base[:40]
))
zscore_candidates = [c for c in zscore_candidates if c in train_eng.columns]

zscore_params = {}
for col in zscore_candidates:
    mu  = float(train_eng[col].mean())
    sig = float(train_eng[col].std()) + 1e-8
    zscore_params[col] = (mu, sig)

    z_col = f"{col}_zscore"
    train_eng[z_col] = ((train_eng[col] - mu) / sig).clip(-10, 10).astype(np.float32)
    if col in test_eng.columns:
        test_eng[z_col] = ((test_eng[col] - mu) / sig).clip(-10, 10).astype(np.float32)
    zscore_features.append(z_col)
    feature_registry[z_col] = f"Z-score of {col}: (x - {mu:.4f}) / {sig:.4f}"

with open('outputs/feature_engineering/zscore_params.pkl', 'wb') as f:
    pickle.dump(zscore_params, f)
print("SAVED: outputs/feature_engineering/zscore_params.pkl")
print(f"Created {len(zscore_features)} z-score normalised features.")

# ── STEP 8 — Power Transforms (Yeo-Johnson) ──────────────────────────────────
print("\n--- STEP 8: Power Transforms for High-Skew Features ---")
power_transform_features = []
power_transformers = {}

moments_df = pd.read_csv('outputs/eda/summaries/feature_moments.csv')
high_skew_cols = moments_df[
    (moments_df['skewness'].abs() > 2) &
    (moments_df['feature'].isin(working_features))
]['feature'].tolist()[:50]

for col in high_skew_cols:
    if col not in train_eng.columns:
        continue
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    vals_train = train_eng[col].values.reshape(-1, 1).astype(np.float64)
    try:
        pt.fit(vals_train)
        pt_col = f"{col}_yeojohnson"
        train_eng[pt_col] = pt.transform(vals_train).flatten().astype(np.float32)
        if col in test_eng.columns:
            test_eng[pt_col] = pt.transform(
                test_eng[col].values.reshape(-1, 1).astype(np.float64)
            ).flatten().astype(np.float32)
        power_transformers[col] = pt
        power_transform_features.append(pt_col)
        feature_registry[pt_col] = f"Yeo-Johnson power transform of {col}"
    except Exception as e:
        print(f"  Skipped {col}: {e}")

with open('outputs/feature_engineering/power_transformers.pkl', 'wb') as f:
    pickle.dump(power_transformers, f)
print("SAVED: outputs/feature_engineering/power_transformers.pkl")
print(f"Created {len(power_transform_features)} power-transformed features.")
gc.collect()

# ── STEP 9A — SO3_T-Weighted Lag Signal ──────────────────────────────────────
print("\n--- STEP 9A: SO3_T-Weighted Composite Features (low priority) ---")
so3_col = 'SO3_T'
alpha_composite_features = []

if so3_col in train_eng.columns:
    so3_mean = float(train_eng[so3_col].mean())
    so3_std  = float(train_eng[so3_col].std()) + 1e-8
    so3_norm_train = ((train_eng[so3_col] - so3_mean) / so3_std).clip(-3, 3)
    so3_norm_test  = ((test_eng[so3_col] - so3_mean) / so3_std).clip(-3, 3) \
                     if so3_col in test_eng.columns else 0.0

    for base_col in working_base[:30]:
        l1 = base_col + '_LagT1'
        if l1 not in train_eng.columns:
            continue
        new_col = f"{base_col}_so3_weighted_lag"
        train_eng[new_col] = (train_eng[l1] * (1 + so3_norm_train)).astype(np.float32)
        if l1 in test_eng.columns:
            test_eng[new_col] = (test_eng[l1] * (1 + so3_norm_test)).astype(np.float32)
        else:
            test_eng[new_col] = 0.0
        alpha_composite_features.append(new_col)
        feature_registry[new_col] = f"SO3_T-weighted LagT1 of {base_col} (low priority)"

    print(f"Created {len(alpha_composite_features)} SO3_T-weighted composite features.")
else:
    print("SO3_T not available — skipping.")

# ── STEP 9B — Inter-Family Ratio Features ────────────────────────────────────
print("\n--- STEP 9B: Inter-Family Ratio Features ---")
corr_df_fe = pd.read_csv('outputs/eda/summaries/target_correlations.csv')
top_positive = corr_df_fe[corr_df_fe['spearman'] > 0.02].head(20)['feature'].tolist()
top_negative = corr_df_fe[corr_df_fe['spearman'] < -0.02].head(20)['feature'].tolist()

ratio_features = []

for p_col in top_positive[:10]:
    for n_col in top_negative[:10]:
        if p_col not in train_eng.columns or n_col not in train_eng.columns:
            continue
        if family_map.get(p_col, 'A') == family_map.get(n_col, 'B'):
            continue
        ratio_col = f"ratio_{p_col[:15]}_{n_col[:15]}"
        denom_train = (train_eng[n_col].abs() + 1e-6)
        denom_test  = (test_eng[n_col].abs() + 1e-6) if n_col in test_eng.columns else 1.0
        train_eng[ratio_col] = (train_eng[p_col] / denom_train).clip(-100, 100).astype(np.float32)
        test_eng[ratio_col]  = (test_eng[p_col] / denom_test).clip(-100, 100).astype(np.float32) \
                                if p_col in test_eng.columns else 0.0
        ratio_features.append(ratio_col)
        feature_registry[ratio_col] = f"Ratio: {p_col} / |{n_col}|"

print(f"Created {len(ratio_features)} inter-family ratio features.")
# NOTE: STEP 9C (Lag Smoothing) REMOVED — ACF max 0.011, no smoothing benefit
gc.collect()

# ── STEP 10 — Regime Assignment ───────────────────────────────────────────────
print("\n--- STEP 10: Regime Assignment ---")
# NOTE: PCA Compression REMOVED — no family compressible below 0.30

regime_model_path = 'outputs/feature_engineering/regime_model.pkl'

if os.path.exists(regime_model_path):
    with open(regime_model_path, 'rb') as f:
        regime_data = pickle.load(f)
    km         = regime_data['km']
    scaler_reg = regime_data['scaler']
    top20_cols = regime_data['top20_cols']
    print(f"Loaded regime model from disk.")
else:
    top20_cols = decisions.get('top_icir_features', decisions['top_signals'])[:20]
    top20_cols = [c for c in top20_cols if c in train_eng.columns]
    scaler_reg = StandardScaler()
    X_scaled   = scaler_reg.fit_transform(train_eng[top20_cols].values.astype(np.float64))
    n_regimes  = decisions.get('n_regimes', 3)
    km = KMeans(n_clusters=n_regimes, random_state=RANDOM_SEED, n_init=10)
    km.fit(X_scaled[:50000])
    regime_data = {'km': km, 'scaler': scaler_reg, 'top20_cols': top20_cols}
    with open(regime_model_path, 'wb') as f:
        pickle.dump(regime_data, f)
    print(f"Fitted and saved new regime model (k={n_regimes}).")

n_regimes      = decisions.get('n_regimes', 3)
top20_avail_tr = [c for c in top20_cols if c in train_eng.columns]
top20_avail_te = [c for c in top20_cols if c in test_eng.columns]

# Cast to float64 to avoid KMeans dtype mismatch
km.cluster_centers_ = km.cluster_centers_.astype(np.float64)
train_eng['regime'] = km.predict(
    scaler_reg.transform(train_eng[top20_avail_tr].values.astype(np.float64))
).astype(np.float32)

if len(top20_avail_te) == len(top20_cols):
    test_eng['regime'] = km.predict(
        scaler_reg.transform(test_eng[top20_avail_te].values.astype(np.float64))
    ).astype(np.float32)
else:
    test_eng['regime'] = 0.0

regime_features = []
for r in range(n_regimes):
    col_name = f"regime_{r}"
    train_eng[col_name] = (train_eng['regime'] == r).astype(np.float32)
    test_eng[col_name]  = (test_eng['regime'] == r).astype(np.float32) \
                           if 'regime' in test_eng.columns else 0.0
    regime_features.append(col_name)
    feature_registry[col_name] = f"Binary: row belongs to KMeans regime {r}"

print(f"Regime distribution (train): {train_eng['regime'].value_counts().sort_index().to_dict()}")
print(f"Created {len(regime_features)} regime indicator features.")
gc.collect()

# ── STEP 11 — Regime × Feature Interaction Terms ─────────────────────────────
print("\n--- STEP 11: Regime × Feature Interactions (61 sign-flip features) ---")
regime_interaction_features = []

sign_flip_feats = decisions.get('needs_regime_interaction', [])
sign_flip_feats = [c for c in sign_flip_feats if c in train_eng.columns]
print(f"Creating regime interactions for {len(sign_flip_feats)} sign-flip features...")
print(f"Total interactions: {len(sign_flip_feats)} × {n_regimes} = {len(sign_flip_feats) * n_regimes}")

for feat_col in sign_flip_feats:
    for r in range(n_regimes):
        regime_ind = f"regime_{r}"
        if regime_ind not in train_eng.columns:
            continue
        new_col = f"{feat_col}_x_regime{r}"
        train_eng[new_col] = (train_eng[feat_col] * train_eng[regime_ind]).astype(np.float32)
        if feat_col in test_eng.columns and regime_ind in test_eng.columns:
            test_eng[new_col] = (test_eng[feat_col] * test_eng[regime_ind]).astype(np.float32)
        else:
            test_eng[new_col] = 0.0
        regime_interaction_features.append(new_col)
        feature_registry[new_col] = (
            f"Regime interaction: {feat_col} * regime_{r} — sign-flip feature"
        )

print(f"Created {len(regime_interaction_features)} regime interaction features.")
gc.collect()

# ── STEP 12 — Quantile Transform for High-Shift Features ─────────────────────
print("\n--- STEP 12: Quantile Transform (ALL 444 KS-drift features, batched) ---")
qt_features   = []
qt_models_all = {}

high_ks_cols = [c for c in decisions['high_ks_shift'] if c in train_eng.columns]
print(f"Applying quantile transform to {len(high_ks_cols)} high-KS-shift features (batched by 50)...")

BATCH_SIZE = 50
for batch_start in range(0, len(high_ks_cols), BATCH_SIZE):
    batch_cols      = high_ks_cols[batch_start : batch_start + BATCH_SIZE]
    batch_cols_test = [c for c in batch_cols if c in test_eng.columns]

    if not batch_cols:
        continue

    qt = QuantileTransformer(
        n_quantiles=min(1000, len(train_eng)),
        output_distribution='normal',
        random_state=RANDOM_SEED
    )

    X_qt_train = train_eng[batch_cols].values.astype(np.float64)
    qt.fit(X_qt_train)
    X_qt_train_t = qt.transform(X_qt_train)

    if len(batch_cols_test) == len(batch_cols):
        X_qt_test_t = qt.transform(test_eng[batch_cols_test].values.astype(np.float64))
    else:
        X_qt_test_t = np.zeros((len(test_eng), len(batch_cols)))

    for i, col in enumerate(batch_cols):
        qt_col = f"{col}_qtrans"
        train_eng[qt_col] = X_qt_train_t[:, i].astype(np.float32)
        test_eng[qt_col]  = X_qt_test_t[:, i].astype(np.float32)
        qt_features.append(qt_col)
        feature_registry[qt_col] = f"Quantile→Normal transform of {col} (KS-drift feature)"

    qt_models_all[f'batch_{batch_start}'] = {'qt': qt, 'cols': batch_cols}

    batch_num = batch_start // BATCH_SIZE
    if batch_num % 5 == 0:
        done = min(batch_start + BATCH_SIZE, len(high_ks_cols))
        print(f"  Progress: {done}/{len(high_ks_cols)}")
        gc.collect()

with open('outputs/feature_engineering/quantile_transformers.pkl', 'wb') as f:
    pickle.dump(qt_models_all, f)
print("SAVED: outputs/feature_engineering/quantile_transformers.pkl")
print(f"Created {len(qt_features)} quantile-transformed features.")
gc.collect()

# ── STEP 13 — Winsorize TARGET ───────────────────────────────────────────────
print("\n--- STEP 13: Winsorize TARGET ---")
target_raw = train_eng['TARGET'].values
q_low  = float(np.percentile(target_raw, 0.5))
q_high = float(np.percentile(target_raw, 99.5))

train_eng['TARGET_raw']  = train_eng['TARGET'].copy()
train_eng['TARGET_wins'] = train_eng['TARGET'].clip(q_low, q_high)

pct_clipped = np.mean((target_raw < q_low) | (target_raw > q_high)) * 100
print(f"Winsorized TARGET at [{q_low:.4f}, {q_high:.4f}]")
print(f"Rows clipped: {pct_clipped:.3f}%  (expected ~0.3% for ±5sigma)")
print(f"TARGET std before: {target_raw.std():.4f}")
print(f"TARGET std after:  {train_eng['TARGET_wins'].std():.4f}")

with open('outputs/feature_engineering/target_winsorize_bounds.pkl', 'wb') as f:
    pickle.dump({'q_low': q_low, 'q_high': q_high}, f)
print("SAVED: outputs/feature_engineering/target_winsorize_bounds.pkl")

# ── STEP 14 — Build Final Feature Sets ───────────────────────────────────────
print("\n--- STEP 14: Build Final Feature Sets ---")

engineered_features = (
    miss_indicator_cols          +
    vol_norm_features            +
    lag_ratio_features           +
    convergence_features         +
    rank_features                +
    zscore_features              +
    power_transform_features     +
    alpha_composite_features     +
    ratio_features               +
    regime_features              +
    regime_interaction_features  +
    qt_features
)

print(f"\nOriginal features retained     : {len(working_features)}")
print(f"New engineered features        : {len(engineered_features)}")
print(f"  of which regime interactions : {len(regime_interaction_features)}")
print(f"  of which rank transforms     : {len(rank_features)}")
print(f"  of which quantile transforms : {len(qt_features)}")
print(f"Total features                 : {len(working_features) + len(engineered_features)}")

top_icir = decisions.get('top_icir_features', decisions['top_signals'])
top_icir = [c for c in top_icir if c in train_eng.columns]

# Tree feature set
tree_features = list(set(
    top_icir                      +
    rank_features                 +
    qt_features                   +
    regime_features               +
    regime_interaction_features   +
    vol_norm_features             +
    lag_ratio_features            +
    convergence_features
))
tree_features = [c for c in tree_features
                 if c in train_eng.columns
                 and c not in ['TARGET', 'TARGET_raw', 'TARGET_wins', 'ID', 'regime']]

# MLP feature set
mlp_features = [c for c in engineered_features
                if any(tag in c for tag in ['_zscore', '_rank', '_qtrans',
                                            '_volnorm', 'regime_', '_x_regime'])
                and c in train_eng.columns]
mlp_features += [c for c in top_icir
                 if c in train_eng.columns and '_LagT' not in c]
mlp_features = list(set(mlp_features))

# Linear feature set
vif_df = pd.read_csv('outputs/eda/summaries/vif_top50.csv')
low_vif_features = vif_df[vif_df['VIF'] < 10]['feature'].tolist()
linear_features  = [c for c in low_vif_features if c in train_eng.columns]

print(f"\nTree feature set     : {len(tree_features)} features")
print(f"MLP feature set      : {len(mlp_features)} features")
print(f"Linear feature set   : {len(linear_features)} features")

feature_sets = {
    'tree_features'              : tree_features,
    'mlp_features'               : mlp_features,
    'linear_features'            : linear_features,
    'all_engineered'             : engineered_features,
    'working_original'           : working_features,
    'top_icir_features'          : top_icir,
    'regime_interaction_features': regime_interaction_features,
}

with open('outputs/feature_engineering/feature_sets.pkl', 'wb') as f:
    pickle.dump(feature_sets, f)
print("SAVED: outputs/feature_engineering/feature_sets.pkl")

with open('outputs/feature_engineering/feature_registry.pkl', 'wb') as f:
    pickle.dump(feature_registry, f)
print("SAVED: outputs/feature_engineering/feature_registry.pkl")
gc.collect()

# ── STEP 15 — Save Engineered Datasets ───────────────────────────────────────
print("\n--- STEP 15: Save Engineered Datasets ---")
print(f"Final shapes before save:")
print(f"  train_eng: {train_eng.shape}")
print(f"  test_eng:  {test_eng.shape}")

# Clean all tree features: no inf/-inf, no NaN
print("Cleaning inf/-inf/NaN from tree features...")
for col in tree_features:
    if col in train_eng.columns:
        train_eng[col] = train_eng[col].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    if col in test_eng.columns:
        test_eng[col]  = test_eng[col].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

train_eng.to_parquet('data/processed/train_engineered.parquet', index=False)
test_eng.to_parquet('data/processed/test_engineered.parquet',   index=False)
print("SAVED: data/processed/train_engineered.parquet")
print("SAVED: data/processed/test_engineered.parquet")
gc.collect()

# ── STEP 16 — Quick Validation ────────────────────────────────────────────────
print("\n--- STEP 16: Feature Importance Quick Validation ---")
print("Quick validation: raw ICIR features vs full engineered set...")
print("Using GroupKFold on regime labels (prevents regime leakage)...")

groups = train_eng['regime'].astype(int).values
top50_icir = [c for c in top_icir[:50] if c in train_eng.columns]
top150_tree = [c for c in tree_features[:150] if c in train_eng.columns]

X_orig = train_eng[top50_icir].fillna(0).values
X_eng  = train_eng[top150_tree].fillna(0).values
y      = train_eng['TARGET_wins'].values

rf  = RandomForestRegressor(n_estimators=50, max_depth=6,
                             random_state=RANDOM_SEED, n_jobs=-1)
gkf = GroupKFold(n_splits=3)

scores_orig, scores_eng = [], []
for train_idx, val_idx in gkf.split(X_orig, y, groups):
    rf.fit(X_orig[train_idx], y[train_idx])
    scores_orig.append(rf.score(X_orig[val_idx], y[val_idx]))
    rf.fit(X_eng[train_idx], y[train_idx])
    scores_eng.append(rf.score(X_eng[val_idx], y[val_idx]))

print(f"\nRaw ICIR features (top 50) — GroupKFold R²: "
      f"{np.mean(scores_orig):.4f} ± {np.std(scores_orig):.4f}")
print(f"Full engineered (top 150)  — GroupKFold R²: "
      f"{np.mean(scores_eng):.4f} ± {np.std(scores_eng):.4f}")
print(f"Improvement: {(np.mean(scores_eng) - np.mean(scores_orig))*100:.2f}% absolute R²")
print("\nGroupKFold on regime prevents inflated CV from regime leakage.")

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)
print(f"  Original features            : {len(working_features)}")
print(f"  Dropped (EDA + adversarial)  : {len(cols_to_drop) + len(perfect_drop)}")
print(f"  Engineered features created  : {len(engineered_features)}")
print(f"  Tree feature set             : {len(tree_features)}")
print(f"  MLP feature set              : {len(mlp_features)}")
print(f"  Linear feature set           : {len(linear_features)}")
print()
print("  Files saved:")
print("    data/processed/train_engineered.parquet")
print("    data/processed/test_engineered.parquet")
print("    outputs/feature_engineering/feature_sets.pkl")
print("    outputs/feature_engineering/feature_registry.pkl")
print("    outputs/feature_engineering/imputation_medians.pkl")
print("    outputs/feature_engineering/zscore_params.pkl")
print("    outputs/feature_engineering/power_transformers.pkl")
print("    outputs/feature_engineering/regime_model.pkl")
print("    outputs/feature_engineering/quantile_transformers.pkl")
print("    outputs/feature_engineering/target_winsorize_bounds.pkl")
print()
print("  Next step: notebooks/03_modelling.ipynb")
print("=" * 70)

gc.collect()
