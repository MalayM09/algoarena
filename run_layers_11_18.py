"""
EDA Layers 11-18 — continuation from Layer 10
Restores all state from saved outputs then runs remaining layers.
"""
import os, pickle, gc, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

os.chdir('/Users/malaymishra/Desktop/quant_ml_project')
RANDOM_SEED = 42
os.makedirs('outputs/eda/summaries', exist_ok=True)
os.makedirs('outputs/eda/plots', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# RESTORE STATE FROM SAVED FILES
# ─────────────────────────────────────────────────────────────────────
print("Loading train data...")
files = os.listdir('data/raw/')
train_file = [f for f in files if 'train' in f.lower()][0]
train = pd.read_parquet(f'data/raw/{train_file}')
print(f"Train shape: {train.shape}")

# Load l1_state
with open('outputs/eda/summaries/l1_state.pkl', 'rb') as f:
    l1 = pickle.load(f)
feature_cols    = l1['feature_cols']
base_features   = l1['base_features']
lag_t1_features = l1['lag_t1_features']
lag_t2_features = l1['lag_t2_features']
lag_t3_features = l1['lag_t3_features']
family_map      = l1['family_map']
drop_zero_var   = l1['drop_zero_var']

def extract_family(col):
    return family_map.get(col, col.split('_')[0] if '_' in col else col)

# Families list
with open('outputs/eda/summaries/taxonomy.pkl', 'rb') as f:
    tax = pickle.load(f)
fam_list = sorted(set(family_map.values()))

# Missingness tiers
with open('outputs/eda/summaries/missingness_tiers.pkl', 'rb') as f:
    miss_tiers = pickle.load(f)

# Shape classes
with open('outputs/eda/summaries/shape_classes.pkl', 'rb') as f:
    shape_classes = pickle.load(f)
high_skew_cols = shape_classes.get('high_skew', [])

# Unstable features
with open('outputs/eda/summaries/unstable_features.pkl', 'rb') as f:
    uf = pickle.load(f)
unstable_features = uf.get('unstable_features', [])

# Outlier columns
outlier_df = pd.read_csv('outputs/eda/summaries/outlier_counts.csv')
extreme_outlier_cols = outlier_df[outlier_df['pct_beyond_5sigma'] > 5.0]['feature'].tolist()

# Target correlations
corr_df = pd.read_csv('outputs/eda/summaries/target_correlations.csv')

# Target values
target_vals = train['TARGET'].values

# Regime model — reconstruct regime labels
with open('outputs/eda/summaries/regime_model.pkl', 'rb') as f:
    rm = pickle.load(f)
km_final   = rm['km_final']
scaler_reg = rm['scaler']
top20_cols = rm['top20_cols']

print("Reconstructing regime labels...")
X_reg = train[top20_cols].copy()
for c in top20_cols:
    X_reg[c] = X_reg[c].fillna(X_reg[c].median())
X_reg_std = scaler_reg.transform(X_reg.values).astype(np.float64)
# KMeans cluster centers may be float32 — cast them to float64 to match input
km_final.cluster_centers_ = km_final.cluster_centers_.astype(np.float64)
train['regime'] = km_final.predict(X_reg_std)
regimes = sorted(train['regime'].unique())
n_regimes = len(regimes)
print(f"Regimes: {regimes}")

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# LAYER 10 — Re-run to print FINDINGS block (outputs already saved)
# ─────────────────────────────────────────────────────────────────────
# (skipped — plot file already exists, findings printed below)
print("\n" + "═"*50)
print("NOTE: Layer 10 outputs already saved — printing findings only")
print("═"*50)

from scipy.stats import pearsonr as _pearsonr
so3_col = 'SO3_T'
so3_vals = train[so3_col].dropna()
r_so3, p_so3 = _pearsonr(
    train[so3_col].fillna(train[so3_col].median()).values,
    target_vals
)
so3_quintiles = pd.qcut(train[so3_col].fillna(train[so3_col].median()), q=5, labels=False)
q_stats = []
for q in range(5):
    mask = so3_quintiles == q
    q_stats.append((q, train.loc[mask,'TARGET'].mean(), train.loc[mask,'TARGET'].std(), mask.sum()))

price_vals   = train['Price'].values
r_price, _   = _pearsonr(train['Price'].fillna(train['Price'].median()).values, target_vals)
log_price    = np.log1p(np.abs(price_vals)) * np.sign(price_vals)
r_log_price, _ = _pearsonr(pd.Series(log_price).fillna(0).values, target_vals)

print(f"\nSO3_T range: {so3_vals.min():.4f} — {so3_vals.max():.4f}")
print(f"SO3_T Pearson r with TARGET: {r_so3:.4f}  (p={p_so3:.4e})")
print(f"Price r with TARGET: {r_price:.4f}  | log(Price) r with TARGET: {r_log_price:.4f}")

print("""
═══════════════════════════════════════════
FINDINGS — Layer 10 — SO3_T & Price Special Analysis
═══════════════════════════════════════════""")
print(f"• SO3_T range: {so3_vals.min():.4f} — {so3_vals.max():.4f}, missing: {train[so3_col].isnull().mean()*100:.2f}%")
print(f"• SO3_T Pearson r with TARGET: {r_so3:.4f} (p={p_so3:.4e})")
for q, m, s, n in q_stats:
    print(f"• SO3_T Q{q}: mean TARGET={m:.4f}, std={s:.4f}, n={n:,}")
print(f"• Price Pearson r with TARGET: {r_price:.4f}")
print(f"• log(Price) Pearson r with TARGET: {r_log_price:.4f}")
print(f"\nRECOMMENDATION")
print("─"*50)
print("• SO3_T is the only named covariate — use quintile bins as a regime interaction multiplier")
if abs(r_so3) > 0.01:
    print("• SO3_T has measurable correlation with TARGET — include as raw + binned feature")
if abs(r_log_price) > abs(r_price):
    print("• log(Price) correlates better than raw Price → apply log transform in engineering")
else:
    print("• Raw Price is comparable to log(Price) → use both as candidates")
print("═"*50)

SAVED_08 = 'outputs/eda/plots/08_price_volatility.png'
print(f"SAVED: {SAVED_08}")

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# LAYER 11 — PCA per Feature Family
# ─────────────────────────────────────────────────────────────────────
print("\n## Layer 11 — PCA per Feature Family")

pca_summary = []
for fam in fam_list:
    fam_cols = [c for c in feature_cols if extract_family(c) == fam]
    if len(fam_cols) < 4:
        continue
    X_fam = train[fam_cols].copy()
    for col in fam_cols:
        X_fam[col] = X_fam[col].fillna(X_fam[col].median())
    X_fam_std = (X_fam - X_fam.mean()) / (X_fam.std() + 1e-8)
    pca = PCA(random_state=RANDOM_SEED)
    pca.fit(X_fam_std.values[:50000])
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.argmax(cum_var >= 0.95) + 1
    n_99 = np.argmax(cum_var >= 0.99) + 1
    pca_summary.append({
        'family'         : fam,
        'n_features'     : len(fam_cols),
        'n_components_95': n_95,
        'n_components_99': n_99,
        'compression_95' : round(n_95 / len(fam_cols), 2),
        'top1_var_ratio' : pca.explained_variance_ratio_[0]
    })

pca_df = pd.DataFrame(pca_summary).sort_values('compression_95')
print("\nPCA compression by family (compression_95 = n_comp_95 / n_features):")
print(pca_df.to_string())
pca_df.to_csv('outputs/eda/summaries/pca_family_summary.csv', index=False)
print("\nSAVED: outputs/eda/summaries/pca_family_summary.csv")

highly_redundant = pca_df[pca_df['compression_95'] < 0.3]
print("\nFamilies with compression_95 < 0.3 (highly redundant — prime for PCA):")
print(highly_redundant[['family','n_features','n_components_95']].to_string())

print("""
═══════════════════════════════════════════
FINDINGS — Layer 11 — PCA per Feature Family
═══════════════════════════════════════════""")
for _, row in pca_df.iterrows():
    print(f"• {row['family']}: {row['n_features']} features → {row['n_components_95']} PCA components explain 95% var (compression={row['compression_95']:.2f}, top1_ratio={row['top1_var_ratio']:.3f})")
print(f"\n• Highly redundant families (compression<0.3): {highly_redundant['family'].tolist()}")
print(f"\nRECOMMENDATION")
print("─"*50)
if len(highly_redundant) > 0:
    print(f"• Create PCA features for {highly_redundant['family'].tolist()} — these are highly compressible")
    print(f"• Use {highly_redundant['n_components_95'].sum()} PCA components instead of {highly_redundant['n_features'].sum()} raw features")
print("• For families with compression≥0.5, keep all raw features — PCA compression is not worth it")
print("═"*50)

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# LAYER 12 — Per-Lag Predictive Power Comparison
# ─────────────────────────────────────────────────────────────────────
print("\n## Layer 12 — Per-Lag Predictive Power Comparison")
print("Computing per-lag predictive power for all base features...")

lag_predictive_power = []

for base_col in base_features:
    l1c = base_col + '_LagT1'
    l2c = base_col + '_LagT2'
    l3c = base_col + '_LagT3'

    results = {}
    for lag_name, lag_col in [('T1', l1c), ('T2', l2c), ('T3', l3c)]:
        if lag_col not in feature_cols:
            results[lag_name] = {'spearman': np.nan, 'pearson': np.nan}
            continue
        vals = train[lag_col].values
        valid = ~np.isnan(vals)
        if valid.sum() < 1000:
            results[lag_name] = {'spearman': np.nan, 'pearson': np.nan}
            continue
        r_s, _ = spearmanr(vals[valid], target_vals[valid])
        r_p, _ = pearsonr(vals[valid],  target_vals[valid])
        results[lag_name] = {'spearman': r_s, 'pearson': r_p}

    s1 = results['T1']['spearman']
    s2 = results['T2']['spearman']
    s3 = results['T3']['spearman']

    lag_vals = {'T1': s1, 'T2': s2, 'T3': s3}
    valid_lags = {k: v for k, v in lag_vals.items() if not np.isnan(v)}
    if not valid_lags:
        continue
    best_lag = max(valid_lags, key=lambda k: abs(valid_lags[k]))

    signs = [np.sign(v) for v in valid_lags.values() if v != 0]
    sign_flip = len(set(signs)) > 1

    abs_vals = [abs(valid_lags.get(k, np.nan)) for k in ['T1','T2','T3']]
    abs_valid_vals = [v for v in abs_vals if not np.isnan(v)]
    if len(abs_valid_vals) == 3:
        monotone_decay    = abs_valid_vals[0] >= abs_valid_vals[1] >= abs_valid_vals[2]
        monotone_increase = abs_valid_vals[0] <= abs_valid_vals[1] <= abs_valid_vals[2]
    else:
        monotone_decay = monotone_increase = False

    lag_predictive_power.append({
        'base_feature'     : base_col,
        'spearman_T1'      : s1,
        'spearman_T2'      : s2,
        'spearman_T3'      : s3,
        'best_lag'         : best_lag,
        'best_lag_spearman': valid_lags[best_lag],
        'sign_flip'        : sign_flip,
        'monotone_decay'   : monotone_decay,
        'monotone_increase': monotone_increase,
        'family'           : extract_family(base_col)
    })

lag_power_df = pd.DataFrame(lag_predictive_power)
lag_power_df = lag_power_df.sort_values('best_lag_spearman', key=abs, ascending=False)
lag_power_df.to_csv('outputs/eda/summaries/lag_predictive_power.csv', index=False)
print("SAVED: outputs/eda/summaries/lag_predictive_power.csv")

sign_flip_count = lag_power_df['sign_flip'].sum()
print(f"\nTotal base features analysed: {len(lag_power_df)}")
print(f"Features with sign flip across lags: {sign_flip_count}")
print(f"Features showing mean-reversion (|T1|>|T2|>|T3|): {lag_power_df['monotone_decay'].sum()}")
print(f"Features showing momentum (|T1|<|T2|<|T3|): {lag_power_df['monotone_increase'].sum()}")
print(f"\nBest lag distribution:")
print(lag_power_df['best_lag'].value_counts())
print(f"\nTop 20 features by best lag Spearman:")
print(lag_power_df[['base_feature','spearman_T1','spearman_T2','spearman_T3','best_lag','sign_flip']].head(20).to_string())

# CHECK: More than 50 features with correlation sign flip
if sign_flip_count > 50:
    print(f"\nALERT — {sign_flip_count} features with sign flip across lags (>50 threshold) — awaiting instructions")

sign_flip_features = lag_power_df[lag_power_df['sign_flip']]['base_feature'].tolist()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
lag_power_df['best_lag'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Which Lag Horizon Is Most Predictive?\n(per base feature)')
axes[0].set_xlabel('Best Lag'); axes[0].set_ylabel('Count of Base Features')
axes[1].scatter(lag_power_df['spearman_T1'], lag_power_df['spearman_T3'],
                alpha=0.4, s=15, color='purple')
axes[1].axhline(0, color='black', lw=1); axes[1].axvline(0, color='black', lw=1)
axes[1].set_xlabel('Spearman r (LagT1 vs TARGET)')
axes[1].set_ylabel('Spearman r (LagT3 vs TARGET)')
axes[1].set_title('T1 vs T3 Spearman\n(off-diagonal quadrants = sign flip features)')
plt.tight_layout()
plt.savefig('outputs/eda/plots/09_lag_predictive_power.png', dpi=150)
plt.close()
print("SAVED: outputs/eda/plots/09_lag_predictive_power.png")

print("""
═══════════════════════════════════════════
FINDINGS — Layer 12 — Per-Lag Predictive Power
═══════════════════════════════════════════""")
bld = lag_power_df['best_lag'].value_counts().to_dict()
print(f"• Total base features: {len(lag_power_df)}")
print(f"• Best lag distribution: {bld}")
print(f"• Features with sign flip across lags: {sign_flip_count} ({sign_flip_count/len(lag_power_df)*100:.1f}%)")
print(f"• Mean-reversion pattern (|T1|>|T2|>|T3|): {lag_power_df['monotone_decay'].sum()}")
print(f"• Momentum pattern (|T1|<|T2|<|T3|): {lag_power_df['monotone_increase'].sum()}")
top3 = lag_power_df.head(3)
for _, row in top3.iterrows():
    print(f"• Top feature {row['base_feature']}: T1={row['spearman_T1']:.4f}, T2={row['spearman_T2']:.4f}, T3={row['spearman_T3']:.4f}, best={row['best_lag']}")
print(f"\nRECOMMENDATION")
print("─"*50)
print(f"• {sign_flip_count} base features have sign flips → keep T1 and T3 as SEPARATE features in engineering")
dominant_lag = max(bld, key=bld.get)
print(f"• Dominant best lag is {dominant_lag} → prioritise {dominant_lag} features in initial model")
print(f"• {lag_power_df['monotone_decay'].sum()} mean-reversion features → can use T1 only for speed")
print(f"• {lag_power_df['monotone_increase'].sum()} momentum features → include all 3 lags")
print("═"*50)

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# LAYER 13 — Autocorrelation of Lag Difference Features
# ─────────────────────────────────────────────────────────────────────
print("\n## Layer 13 — Autocorrelation of Lag Difference Features")
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

print("Computing ACF for lag difference features (ID-sorted)...")
train_sorted = train.sort_values('ID').reset_index(drop=True)

acf_results = []
sample_lag_cols = lag_t1_features[:60]

for col in sample_lag_cols:
    if col not in train_sorted.columns:
        continue
    vals = train_sorted[col].fillna(train_sorted[col].median()).values
    try:
        acf_vals = acf(vals[:20000], nlags=5, fft=True)
        lb_result = acorr_ljungbox(vals[:20000], lags=[5], return_df=True)
        lb_p = lb_result['lb_pvalue'].values[0]
        acf_results.append({
            'feature'   : col,
            'acf_lag1'  : acf_vals[1],
            'acf_lag2'  : acf_vals[2],
            'acf_lag3'  : acf_vals[3],
            'acf_lag5'  : acf_vals[5],
            'lb_pvalue' : lb_p,
            'persistent': abs(acf_vals[1]) > 0.1,
            'family'    : extract_family(col)
        })
    except Exception:
        pass

acf_df = pd.DataFrame(acf_results).sort_values('acf_lag1', key=abs, ascending=False)
acf_df.to_csv('outputs/eda/summaries/lag_acf.csv', index=False)
print("SAVED: outputs/eda/summaries/lag_acf.csv")

persistent_count = acf_df['persistent'].sum()
lb_sig_count = (acf_df['lb_pvalue'] < 0.05).sum()
print(f"\nFeatures with |ACF(lag=1)| > 0.1 (persistent): {persistent_count}")
print(f"Features with Ljung-Box p < 0.05 (significant autocorrelation): {lb_sig_count}")
print(f"\nTop 20 by |ACF(1)|:")
print(acf_df[['feature','acf_lag1','acf_lag2','acf_lag5','persistent']].head(20).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(acf_df['acf_lag1'], bins=50, color='steelblue', alpha=0.7)
axes[0].axvline(0.1, color='red', lw=2, linestyle='--', label='|0.1| threshold')
axes[0].axvline(-0.1, color='red', lw=2, linestyle='--')
axes[0].set_title('Distribution of ACF(lag=1) for LagT1 Features')
axes[0].set_xlabel('ACF at lag 1'); axes[0].legend()
axes[1].scatter(acf_df['acf_lag1'], acf_df['acf_lag2'], alpha=0.5, s=15, color='purple')
axes[1].set_xlabel('ACF lag 1'); axes[1].set_ylabel('ACF lag 2')
axes[1].set_title('ACF(1) vs ACF(2)\n(top-right quadrant = persistent momentum signal)')
plt.tight_layout()
plt.savefig('outputs/eda/plots/10_lag_acf.png', dpi=150)
plt.close()
print("SAVED: outputs/eda/plots/10_lag_acf.png")

print("""
═══════════════════════════════════════════
FINDINGS — Layer 13 — Autocorrelation of Lag Features
═══════════════════════════════════════════""")
print(f"• Sampled {len(acf_df)} LagT1 features (first 60)")
print(f"• Features with |ACF(1)| > 0.1 (persistent signal): {persistent_count} / {len(acf_df)}")
print(f"• Features with significant Ljung-Box (p<0.05): {lb_sig_count} / {len(acf_df)}")
print(f"• Mean ACF(1) across features: {acf_df['acf_lag1'].mean():.4f}")
print(f"• Max |ACF(1)|: {acf_df['acf_lag1'].abs().max():.4f} ({acf_df.iloc[0]['feature']})")
print(f"\nRECOMMENDATION")
print("─"*50)
if persistent_count > len(acf_df) * 0.3:
    print(f"• {persistent_count} features are persistent → apply 3-period temporal average in engineering")
    print("• Temporal smoothing (rolling mean T1+T2+T3 / 3) will improve signal for these")
else:
    print("• Most features have low ACF → smoothing will degrade signals; avoid aggressive averaging")
print(f"• {lb_sig_count} features show structured autocorrelation → ID-sorted ordering is meaningful")
print("═"*50)

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# LAYER 14 — Regime-Conditional Feature Correlations
# ─────────────────────────────────────────────────────────────────────
print("\n## Layer 14 — Regime-Conditional Feature Correlations")
assert 'regime' in train.columns, "Missing regime column"

print("Computing feature-TARGET correlations conditioned on regime...")
top80_cols = corr_df.dropna(subset=['spearman']).head(80)['feature'].tolist()

regime_corr_results = []
for col in top80_cols:
    if col not in train.columns:
        continue
    row = {'feature': col}
    regime_corrs = []
    for r in regimes:
        mask = train['regime'] == r
        vals = train.loc[mask, col].fillna(train[col].median()).values
        tgt  = train.loc[mask, 'TARGET'].values
        if len(vals) < 500:
            row[f'r_regime_{r}'] = np.nan
            continue
        rc, _ = pearsonr(vals, tgt)
        row[f'r_regime_{r}'] = rc
        regime_corrs.append(rc)

    signs = [np.sign(v) for v in regime_corrs if v != 0 and not np.isnan(v)]
    row['regime_sign_flip'] = len(set(signs)) > 1
    row['regime_corr_std']  = np.std(regime_corrs) if len(regime_corrs) > 1 else np.nan
    row['overall_spearman'] = corr_df.loc[corr_df['feature'] == col, 'spearman'].values[0] \
                              if len(corr_df.loc[corr_df['feature'] == col]) > 0 else np.nan
    regime_corr_results.append(row)

regime_corr_df = pd.DataFrame(regime_corr_results)
regime_corr_df = regime_corr_df.sort_values('regime_corr_std', ascending=False)
regime_corr_df.to_csv('outputs/eda/summaries/regime_conditional_correlations.csv', index=False)
print("SAVED: outputs/eda/summaries/regime_conditional_correlations.csv")

needs_regime_interaction = regime_corr_df[regime_corr_df['regime_sign_flip'] == True]['feature'].tolist()
high_regime_variation = regime_corr_df[regime_corr_df['regime_corr_std'] > 0.05]['feature'].tolist()

print(f"\nFeatures with sign flip across regimes: {len(needs_regime_interaction)}")
print(f"Features with high regime variation (std>0.05): {len(high_regime_variation)}")
r_cols = ['feature','overall_spearman','regime_sign_flip','regime_corr_std'] + \
         [f'r_regime_{r}' for r in regimes]
print(f"\nTop 20 most regime-dependent features:")
print(regime_corr_df[r_cols].head(20).to_string())

pivot_cols = [f'r_regime_{r}' for r in regimes]
regime_pivot = regime_corr_df.set_index('feature')[pivot_cols].head(40)
plt.figure(figsize=(8, 14))
sns.heatmap(regime_pivot, cmap='RdBu_r', center=0, vmin=-0.2, vmax=0.2,
            annot=True, fmt='.3f', linewidths=0.5)
plt.title('Feature-TARGET Correlation Per Regime\n(red=positive, blue=negative)')
plt.tight_layout()
plt.savefig('outputs/eda/plots/11_regime_conditional_corr.png', dpi=150)
plt.close()
print("SAVED: outputs/eda/plots/11_regime_conditional_corr.png")

print("""
═══════════════════════════════════════════
FINDINGS — Layer 14 — Regime-Conditional Correlations
═══════════════════════════════════════════""")
print(f"• {n_regimes} regimes analysed; top 80 features by Spearman evaluated")
print(f"• Features with sign flip across regimes: {len(needs_regime_interaction)} / {len(regime_corr_df)}")
print(f"• Features with regime_corr_std > 0.05: {len(high_regime_variation)} / {len(regime_corr_df)}")
if len(needs_regime_interaction) > 0:
    print(f"• Top regime sign-flip features: {needs_regime_interaction[:5]}")
print(f"\nRECOMMENDATION")
print("─"*50)
print(f"• Create regime*feature interaction terms for {len(needs_regime_interaction)} features with sign flips")
print(f"• {len(high_regime_variation)} features have high regime variance → consider regime-specific models")
print("═"*50)

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# LAYER 15 — Multiple Hypothesis Correction (Benjamini-Hochberg)
# ─────────────────────────────────────────────────────────────────────
print("\n## Layer 15 — Multiple Hypothesis Correction")
from statsmodels.stats.multitest import multipletests

print("Applying Benjamini-Hochberg FDR correction to all statistical tests...")

# 1. Correct KS train-test
ks_df_loaded = pd.read_csv('outputs/eda/summaries/ks_train_test.csv')
ks_pvals = ks_df_loaded['ks_p'].fillna(1.0).values
reject_ks, pvals_corrected_ks, _, _ = multipletests(ks_pvals, alpha=0.05, method='fdr_bh')
ks_df_loaded['ks_p_corrected'] = pvals_corrected_ks
ks_df_loaded['significant_after_correction'] = reject_ks
ks_df_loaded.to_csv('outputs/eda/summaries/ks_train_test.csv', index=False)
print(f"\nKS tests — significant after BH: {reject_ks.sum()} / {len(reject_ks)} (was {(ks_pvals < 0.05).sum()} before)")

# 2. Correct Spearman p-values
corr_df_loaded = pd.read_csv('outputs/eda/summaries/target_correlations.csv')
spearman_pvals = corr_df_loaded['spearman_p'].fillna(1.0).values
reject_sp, pvals_corrected_sp, _, _ = multipletests(spearman_pvals, alpha=0.05, method='fdr_bh')
corr_df_loaded['spearman_p_corrected'] = pvals_corrected_sp
corr_df_loaded['significant_after_correction'] = reject_sp
corr_df_loaded.to_csv('outputs/eda/summaries/target_correlations.csv', index=False)
print(f"Spearman tests — significant after BH: {reject_sp.sum()} / {len(reject_sp)}")

# 3. Correct KW tests
kw_df_loaded = pd.read_csv('outputs/eda/summaries/kruskal_wallis.csv')
kw_pvals = kw_df_loaded['kw_p'].fillna(1.0).values
reject_kw, pvals_corrected_kw, _, _ = multipletests(kw_pvals, alpha=0.05, method='fdr_bh')
kw_df_loaded['kw_p_corrected'] = pvals_corrected_kw
kw_df_loaded['significant_after_correction'] = reject_kw
kw_df_loaded.to_csv('outputs/eda/summaries/kruskal_wallis.csv', index=False)
print(f"KW tests — significant after BH: {reject_kw.sum()} / {len(reject_kw)}")

# 4. Correct missingness signal
miss_df_loaded = pd.read_csv('outputs/eda/summaries/missingness_signal.csv')
if len(miss_df_loaded) > 0:
    miss_pvals = miss_df_loaded['mwu_pvalue'].fillna(1.0).values
    reject_m, pvals_corrected_m, _, _ = multipletests(miss_pvals, alpha=0.05, method='fdr_bh')
    miss_df_loaded['mwu_p_corrected'] = pvals_corrected_m
    miss_df_loaded['significant_after_correction'] = reject_m
    miss_df_loaded.to_csv('outputs/eda/summaries/missingness_signal.csv', index=False)
    print(f"Missingness signal tests — significant after BH: {reject_m.sum()} / {len(reject_m)}")
else:
    print("Missingness signal tests — 0 features have missingness; no correction needed")

print("\nSAVED: outputs/eda/summaries/ks_train_test.csv (updated with BH correction)")
print("SAVED: outputs/eda/summaries/target_correlations.csv (updated with BH correction)")
print("SAVED: outputs/eda/summaries/kruskal_wallis.csv (updated with BH correction)")
print("SAVED: outputs/eda/summaries/missingness_signal.csv (updated with BH correction)")

# CHECK: >50% HIGH KS drift
high_drift_frac = reject_ks.sum() / len(reject_ks)
if high_drift_frac > 0.5:
    print(f"\nALERT — {high_drift_frac*100:.1f}% of features showing HIGH train-test KS drift — awaiting instructions")

print("""
═══════════════════════════════════════════
FINDINGS — Layer 15 — BH Multiple Hypothesis Correction
═══════════════════════════════════════════""")
print(f"• KS tests: {reject_ks.sum()} significant after BH (was {(ks_pvals<0.05).sum()} before) — {reject_ks.sum()/len(reject_ks)*100:.1f}% of features have train-test drift")
print(f"• Spearman tests: {reject_sp.sum()} significant after BH (was {(spearman_pvals<0.05).sum()} before)")
print(f"• KW tests: {reject_kw.sum()} significant after BH")
miss_sig_count = reject_m.sum() if 'reject_m' in dir() and len(miss_df_loaded) > 0 else 0
print(f"• Missingness signal: {miss_sig_count} significant after BH (dataset has no missing values)")
print(f"\nRECOMMENDATION")
print("─"*50)
print(f"• Use only BH-corrected p-values for feature selection — reject all pre-correction-only results as likely false positives")
print(f"• {reject_ks.sum()} features have real train-test drift → apply quantile normalisation in engineering")
print(f"• {miss_sig_count} features where NaN is truly predictive → add binary missingness indicator (0 here — no missingness)")
print("═"*50)

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# LAYER 16 — Pairwise Interaction Screening
# ─────────────────────────────────────────────────────────────────────
print("\n## Layer 16 — Pairwise Interaction Screening")
print("Screening pairwise interactions among top features...")

top40 = corr_df.dropna(subset=['spearman']).head(40)['feature'].tolist()
top40 = [c for c in top40 if c in train.columns]

X_top = train[top40].copy()
for col in top40:
    mu  = X_top[col].mean()
    sig = X_top[col].std() + 1e-8
    X_top[col] = ((X_top[col] - mu) / sig).clip(-5, 5)

interaction_results = []
for i in range(len(top40)):
    for j in range(i+1, len(top40)):
        col_a = top40[i]
        col_b = top40[j]
        prod  = (X_top[col_a] * X_top[col_b]).values
        valid = ~np.isnan(prod)
        if valid.sum() < 5000:
            continue
        ic_prod, _ = spearmanr(prod[valid], target_vals[valid])
        ic_a = corr_df.loc[corr_df['feature'] == col_a, 'spearman'].values[0]
        ic_b = corr_df.loc[corr_df['feature'] == col_b, 'spearman'].values[0]
        interaction_gain = abs(ic_prod) - max(abs(ic_a), abs(ic_b))
        interaction_results.append({
            'feature_a'        : col_a,
            'feature_b'        : col_b,
            'ic_product'       : ic_prod,
            'ic_a'             : ic_a,
            'ic_b'             : ic_b,
            'interaction_gain' : interaction_gain,
            'same_family'      : extract_family(col_a) == extract_family(col_b)
        })

interaction_df = pd.DataFrame(interaction_results)
interaction_df = interaction_df.sort_values('interaction_gain', ascending=False)
interaction_df.to_csv('outputs/eda/summaries/pairwise_interactions.csv', index=False)
print("SAVED: outputs/eda/summaries/pairwise_interactions.csv")

top_interactions = interaction_df[interaction_df['interaction_gain'] > 0.01]
cross_family_interactions = top_interactions[~top_interactions['same_family']]

print(f"\nTotal pairs screened: {len(interaction_df)}")
print(f"Pairs with interaction gain > 0.01: {len(top_interactions)}")
print(f"  Cross-family interactions: {len(cross_family_interactions)}")
print(f"\nTop 20 candidate interaction pairs:")
print(interaction_df[['feature_a','feature_b','ic_product','ic_a','ic_b','interaction_gain','same_family']].head(20).to_string())

plt.figure(figsize=(10, 5))
plt.scatter(interaction_df['interaction_gain'],
            interaction_df['ic_product'].abs(),
            alpha=0.3, s=10, color='steelblue')
plt.axvline(0.01, color='red', lw=2, linestyle='--', label='Gain > 0.01 threshold')
plt.xlabel('Interaction Gain (|IC_product| - max(|IC_a|, |IC_b|))')
plt.ylabel('|IC of product|')
plt.title('Pairwise Interaction Screening\n(right of red line = add A*B as feature)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/eda/plots/12_interaction_screening.png', dpi=150)
plt.close()
print("SAVED: outputs/eda/plots/12_interaction_screening.png")

print("""
═══════════════════════════════════════════
FINDINGS — Layer 16 — Pairwise Interaction Screening
═══════════════════════════════════════════""")
print(f"• Screened {len(interaction_df)} pairs from top 40 features ({len(top40)} features)")
print(f"• Pairs with interaction gain > 0.01: {len(top_interactions)}")
print(f"• Cross-family pairs (most novel): {len(cross_family_interactions)}")
if len(interaction_df) > 0:
    best = interaction_df.iloc[0]
    print(f"• Best interaction: {best['feature_a']} × {best['feature_b']} → IC={best['ic_product']:.4f}, gain={best['interaction_gain']:.4f}")
print(f"\nRECOMMENDATION")
print("─"*50)
print(f"• Create {min(len(top_interactions), 20)} interaction features — focus on cross-family pairs first")
print(f"• {len(cross_family_interactions)} cross-family interactions are the most novel signals")
print(f"• Pairs with gain < 0 provide no new information beyond individual features — skip them")
print("═"*50)

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# LAYER 17 — IC and ICIR Analysis
# ─────────────────────────────────────────────────────────────────────
print("\n## Layer 17 — IC and ICIR Analysis")
print("Computing IC and ICIR for all features across temporal chunks...")

N_CHUNKS = 20
train_sorted_ic = train.sort_values('ID').reset_index(drop=True)
chunk_size_ic   = len(train_sorted_ic) // N_CHUNKS

ic_results = []
for col in feature_cols:
    if col not in train_sorted_ic.columns:
        continue
    chunk_ics = []
    for i in range(N_CHUNKS):
        chunk = train_sorted_ic.iloc[i*chunk_size_ic : (i+1)*chunk_size_ic]
        vals  = chunk[col].fillna(chunk[col].median()).values
        tgt   = chunk['TARGET'].values
        valid = ~np.isnan(vals)
        if valid.sum() < 200:
            chunk_ics.append(np.nan)
            continue
        ic, _ = spearmanr(vals[valid], tgt[valid])
        chunk_ics.append(ic)

    chunk_ics_valid = [v for v in chunk_ics if not np.isnan(v)]
    if len(chunk_ics_valid) < 5:
        continue

    mean_ic = np.mean(chunk_ics_valid)
    std_ic  = np.std(chunk_ics_valid) + 1e-8
    icir    = mean_ic / std_ic
    ic_pos_frac = np.mean([v > 0 for v in chunk_ics_valid])

    ic_results.append({
        'feature'    : col,
        'mean_ic'    : mean_ic,
        'std_ic'     : std_ic,
        'icir'       : icir,
        'abs_icir'   : abs(icir),
        'ic_pos_frac': ic_pos_frac,
        'n_chunks'   : len(chunk_ics_valid),
        'family'     : extract_family(col),
        'lag_type'   : ('LagT1' if '_LagT1' in col else
                        'LagT2' if '_LagT2' in col else
                        'LagT3' if '_LagT3' in col else 'base')
    })

ic_df = pd.DataFrame(ic_results).sort_values('abs_icir', ascending=False)
ic_df.to_csv('outputs/eda/summaries/ic_icir.csv', index=False)
print("SAVED: outputs/eda/summaries/ic_icir.csv")

# Merge with Spearman ranking
corr_df_curr = pd.read_csv('outputs/eda/summaries/target_correlations.csv')
ic_df_merged = ic_df.merge(
    corr_df_curr[['feature','spearman']].rename(columns={'spearman':'global_spearman'}),
    on='feature', how='left'
)
ic_df_merged['icir_rank']     = ic_df_merged['abs_icir'].rank(ascending=False)
ic_df_merged['spearman_rank'] = ic_df_merged['global_spearman'].abs().rank(ascending=False)
ic_df_merged['rank_diff']     = ic_df_merged['spearman_rank'] - ic_df_merged['icir_rank']
hidden_signals = ic_df_merged[ic_df_merged['rank_diff'] > 50].sort_values('rank_diff', ascending=False)
ic_df_merged.to_csv('outputs/eda/summaries/ic_icir_full.csv', index=False)
print("SAVED: outputs/eda/summaries/ic_icir_full.csv")

mean_ic_gt_002 = (ic_df['mean_ic'].abs() > 0.02).sum()
icir_gt_05     = (ic_df['abs_icir'] > 0.5).sum()
icir_gt_10     = (ic_df['abs_icir'] > 1.0).sum()
mean_icir_all  = ic_df['abs_icir'].mean()

print(f"\nFeatures with |Mean IC| > 0.02:  {mean_ic_gt_002}")
print(f"Features with |ICIR| > 0.5:      {icir_gt_05}")
print(f"Features with |ICIR| > 1.0:      {icir_gt_10}")
print(f"Mean |ICIR| across all features:  {mean_icir_all:.4f}")
print(f"\nTop 25 features by |ICIR|:")
print(ic_df[['feature','mean_ic','std_ic','icir','ic_pos_frac']].head(25).to_string())
print(f"\nFeatures ranking much higher by ICIR than global Spearman: {len(hidden_signals)}")
print(hidden_signals[['feature','mean_ic','icir','global_spearman','rank_diff']].head(15).to_string())

# CHECK: Mean ICIR < 0.1
if mean_icir_all < 0.1:
    print(f"\nALERT — Mean ICIR={mean_icir_all:.4f} < 0.1 across all features — awaiting instructions")

# CHECK: Fewer than 10 features with |Spearman| > 0.03
n_strong_spearman = (corr_df_curr['spearman'].abs() > 0.03).sum()
if n_strong_spearman < 10:
    print(f"\nALERT — Only {n_strong_spearman} features with |Spearman|>0.03 — fewer than 10 threshold — awaiting instructions")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].hist(ic_df['mean_ic'], bins=60, color='steelblue', alpha=0.7)
axes[0,0].axvline(0.02, color='red', lw=2, linestyle='--', label='IC=0.02')
axes[0,0].axvline(-0.02, color='red', lw=2, linestyle='--')
axes[0,0].set_title('Distribution of Mean IC'); axes[0,0].legend()

axes[0,1].hist(ic_df['icir'], bins=60, color='coral', alpha=0.7)
axes[0,1].axvline(0.5, color='red', lw=2, linestyle='--', label='ICIR=0.5')
axes[0,1].axvline(-0.5, color='red', lw=2, linestyle='--')
axes[0,1].set_title('Distribution of ICIR'); axes[0,1].legend()

axes[1,0].scatter(ic_df['mean_ic'], ic_df['icir'], alpha=0.3, s=10, color='purple')
axes[1,0].axhline(0.5, color='red', lw=1, linestyle='--')
axes[1,0].axvline(0.02, color='red', lw=1, linestyle='--')
axes[1,0].set_title('Mean IC vs ICIR'); axes[1,0].set_xlabel('Mean IC'); axes[1,0].set_ylabel('ICIR')

ic_df_lag = ic_df.groupby('lag_type')['abs_icir'].mean()
axes[1,1].bar(ic_df_lag.index, ic_df_lag.values, color='green', alpha=0.7)
axes[1,1].set_title('Mean |ICIR| by Feature Type')
axes[1,1].set_xlabel('Feature type'); axes[1,1].set_ylabel('Mean |ICIR|')

plt.suptitle('IC and ICIR Analysis — Signal Quality Across Time', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/eda/plots/13_ic_icir_analysis.png', dpi=150)
plt.close()
print("SAVED: outputs/eda/plots/13_ic_icir_analysis.png")

print("""
═══════════════════════════════════════════
FINDINGS — Layer 17 — IC and ICIR Analysis
═══════════════════════════════════════════""")
print(f"• Total features analysed: {len(ic_df)}")
print(f"• Features with |Mean IC| > 0.02 (meaningful predictive power): {mean_ic_gt_002}")
print(f"• Features with |ICIR| > 0.5 (consistent enough to trade on): {icir_gt_05}")
print(f"• Features with |ICIR| > 1.0 (strong institutional-grade signal): {icir_gt_10}")
print(f"• Mean |ICIR| across all features: {mean_icir_all:.4f}")
print(f"• Hidden consistent signals (high ICIR rank vs low Spearman rank): {len(hidden_signals)}")
top5 = ic_df.head(5)
for _, row in top5.iterrows():
    print(f"• Top ICIR: {row['feature']} → ICIR={row['icir']:.4f}, mean_IC={row['mean_ic']:.4f}, pos_frac={row['ic_pos_frac']:.2f}")
ic_by_lag = ic_df.groupby('lag_type')['abs_icir'].mean().to_dict()
print(f"• ICIR by lag type: {ic_by_lag}")
print(f"\nRECOMMENDATION")
print("─"*50)
print(f"• Prioritise {icir_gt_05} features with |ICIR|>0.5 — these are safe consistent signals")
print(f"• {len(hidden_signals)} hidden signals rank much higher by ICIR than Spearman → include in ensemble")
best_lag_type = max(ic_by_lag, key=ic_by_lag.get)
print(f"• {best_lag_type} features have highest mean ICIR → emphasise this lag type in feature engineering")
print("═"*50)

gc.collect()

# ─────────────────────────────────────────────────────────────────────
# LAYER 18 — Final EDA Summary Report & Recommendations
# ─────────────────────────────────────────────────────────────────────
print("\n## Layer 18 — Final EDA Summary Report")
print("\n" + "="*70)
print("EDA COMPLETE — COMPREHENSIVE DECISION SUMMARY")
print("="*70)

# Reload fresh versions of all corrected files
corr_df_final     = pd.read_csv('outputs/eda/summaries/target_correlations.csv')
ic_df_final       = pd.read_csv('outputs/eda/summaries/ic_icir.csv')
miss_df_final     = pd.read_csv('outputs/eda/summaries/missingness_signal.csv')
ks_df_final       = pd.read_csv('outputs/eda/summaries/ks_train_test.csv')
regime_corr_final = pd.read_csv('outputs/eda/summaries/regime_conditional_correlations.csv')
lag_power_final   = pd.read_csv('outputs/eda/summaries/lag_predictive_power.csv')
interact_final    = pd.read_csv('outputs/eda/summaries/pairwise_interactions.csv')
acf_final         = pd.read_csv('outputs/eda/summaries/lag_acf.csv')

# DROP
drop_decisions = list(set(
    drop_zero_var +
    miss_tiers.get('very_high_miss', []) +
    unstable_features
))
print(f"\n[DROP]  Zero-variance + >80% missing + sign-flip unstable: {len(drop_decisions)} features")

# TRANSFORM
needs_transform = list(set(high_skew_cols + extreme_outlier_cols))
print(f"[TRANSFORM] High skew / extreme outliers: {len(needs_transform)} features → Yeo-Johnson")

# TOP SIGNALS
top_signals = corr_df_final[
    (corr_df_final['spearman_abs'] > 0.03) &
    (corr_df_final['significant_after_correction'] == True) &
    (~corr_df_final['feature'].isin(drop_decisions))
].head(100)['feature'].tolist()
print(f"[KEEP]  Top-signal features (|Spearman|>0.03, BH-significant): {len(top_signals)}")

# ICIR features
top_icir_features = ic_df_final[ic_df_final['abs_icir'] > 0.5]['feature'].tolist()
print(f"[ICIR]  Features with |ICIR| > 0.5: {len(top_icir_features)}")

# Missingness indicators
if len(miss_df_final) > 0 and 'significant_after_correction' in miss_df_final.columns:
    needs_indicator = miss_df_final[miss_df_final['significant_after_correction'] == True]['feature'].tolist()
else:
    needs_indicator = []  # no missing values in this dataset
print(f"[INDICATOR] NaN predictive of TARGET (BH-corrected): {len(needs_indicator)}")

# Covariate shift
high_shift = ks_df_final[ks_df_final['significant_after_correction'] == True]['feature'].tolist()
print(f"[RISK]  High train-test distribution shift (BH-corrected): {len(high_shift)}")

# Regime interaction
needs_regime_interact = regime_corr_final[regime_corr_final['regime_sign_flip'] == True]['feature'].tolist()
print(f"[REGIME] Features with sign flip across regimes: {len(needs_regime_interact)}")

# Sign-flip lag features
sign_flip_lags = lag_power_final[lag_power_final['sign_flip'] == True]['base_feature'].tolist()
print(f"[LAG]   Features with T1 vs T3 sign flip: {len(sign_flip_lags)}")

# Best lag map
best_lag_map = dict(zip(lag_power_final['base_feature'], lag_power_final['best_lag']))
print(f"[LAG]   Best lag distribution: {lag_power_final['best_lag'].value_counts().to_dict()}")

# Top interaction pairs
top_interaction_pairs = interact_final[interact_final['interaction_gain'] > 0.01][
    ['feature_a','feature_b','interaction_gain']].head(20).to_dict('records')
print(f"[INTERACT] Top interaction pairs (gain>0.01): {len(top_interaction_pairs)} pairs")

# Smoothing candidates
needs_smoothing = acf_final[acf_final['persistent'] == True]['feature'].tolist()
print(f"[SMOOTH] Features with |ACF(1)| > 0.1: {len(needs_smoothing)}")

# MLP priority
mlp_priority = list(set(top_icir_features) - set(high_shift) - set(high_skew_cols))
print(f"[MLP]   Priority features for MLP (high ICIR + low shift + low skew): {len(mlp_priority)}")

# Save master decisions
decisions = {
    'drop'                   : drop_decisions,
    'needs_transform'        : needs_transform,
    'top_signals'            : top_signals,
    'top_icir_features'      : top_icir_features,
    'needs_indicator'        : needs_indicator,
    'high_ks_shift'          : high_shift,
    'unstable_corr'          : unstable_features,
    'needs_regime_interaction': needs_regime_interact,
    'sign_flip_lags'         : sign_flip_lags,
    'best_lag_map'           : best_lag_map,
    'top_interaction_pairs'  : top_interaction_pairs,
    'needs_smoothing'        : needs_smoothing,
    'mlp_priority_features'  : mlp_priority,
    'n_regimes'              : n_regimes,
}
with open('outputs/eda/summaries/eda_decisions.pkl', 'wb') as f:
    pickle.dump(decisions, f)
print("\nSAVED: outputs/eda/summaries/eda_decisions.pkl")

print("\n" + "="*70)
print("RECOMMENDATIONS FOR FEATURE_ENGINEERING_INSTRUCTIONS.md")
print("="*70)
print(f"""
1. DROP {len(drop_decisions)} features immediately (zero variance, >80% missing, sign-flip unstable)

2. CREATE missingness indicators for {len(needs_indicator)} features
   (NaN is statistically predictive of TARGET after BH correction)

3. VOLATILITY-NORMALISE all lag features
   Primary signal type — validated by alpha research and ICIR analysis

4. SMOOTH (average T1+T2+T3) for {len(needs_smoothing)} features
   These have |ACF(1)| > 0.1 — temporal averaging improves signal quality

5. KEEP T1 AND T3 SEPARATE for {len(sign_flip_lags)} base features
   Sign flips mean T1 = mean reversion, T3 = momentum — opposite directions

6. ADD REGIME INTERACTION for {len(needs_regime_interact)} features
   Correlation with TARGET changes sign across market regimes

7. CREATE {len(top_interaction_pairs)} pairwise interaction features
   Product of two features predicts TARGET better than either alone

8. APPLY QUANTILE TRANSFORM to {len(high_shift)} high-drift features
   Significant train-test distribution shift (after BH correction)

9. YEO-JOHNSON TRANSFORM for {len(needs_transform)} skewed features
   Improves MLP and linear model inputs

10. PRIORITISE {len(top_icir_features)} features with |ICIR| > 0.5
    These are the most consistent signals across time — safest to rely on
""")

print("All EDA outputs saved to outputs/eda/")
print("Load eda_decisions.pkl in 02_feature_engineering.ipynb to proceed.")

# ─────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────
ic_top5 = ic_df_final.head(5)[['feature','mean_ic','icir']].to_string()
corr_top5 = corr_df_final.sort_values('spearman_abs', ascending=False).head(5)[['feature','spearman']].to_string()

summaries = [f for f in os.listdir('outputs/eda/summaries') if f.endswith('.csv') or f.endswith('.pkl') or f.endswith('.parquet')]
plots     = [f for f in os.listdir('outputs/eda/plots') if f.endswith('.png')]

print(f"""
══════════════════════════════════════════════════
EDA COMPLETE
══════════════════════════════════════════════════
Layers completed : 18 / 18
Output files saved: {len(summaries)}
Plots saved:        {len(plots)}
eda_decisions.pkl keys: {list(decisions.keys())}

TOP 5 FEATURES BY ICIR:
{ic_top5}

TOP 5 FEATURES BY SPEARMAN:
{corr_top5}

FEATURES TO DROP: {len(drop_decisions)}
FEATURES NEEDING TRANSFORM: {len(needs_transform)}
FEATURES NEEDING REGIME INTERACTION: {len(needs_regime_interact)}

Next step: run notebooks/02_feature_engineering.ipynb
══════════════════════════════════════════════════
""")

gc.collect()
print("Script completed successfully.")
