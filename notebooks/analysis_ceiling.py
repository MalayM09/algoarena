"""
Ceiling Analysis: What is the maximum achievable R² for compliant predictions?

Multiple independent approaches:
1. Oracle IC → R² conversion (theoretical bound from signal strength)
2. Train-set cross-validated R² ceiling (what's achievable on liquid assets)
3. Noise floor analysis (TARGET kurtosis, SNR)
4. Feature-target mutual information bound
5. Per-day R² distribution (what fraction of days have any signal?)
6. Liquid→illiquid transfer decay (how much signal survives the domain shift?)
7. Overlap vs new-day ceiling split
"""
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, ks_2samp
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')

DATA_DIR = '/Users/malaymishra/Desktop/quant_ml_project/data/raw'
OUT_DIR = '/Users/malaymishra/Desktop/quant_ml_project/outputs'
ORACLE_PATH = os.path.join(OUT_DIR, 'submissions', 'exploit_v2_zero.csv')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

t0 = time.time()

print('Loading data...', flush=True)
train = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
test = pd.read_parquet(os.path.join(DATA_DIR, 'test.parquet'))
oracle_df = pd.read_csv(ORACLE_PATH)

feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET', 'CV_GROUP'}]
all_feat = [c for c in feat_cols if c != 'SO3_T']
train_day = train['SO3_T'].round(5).astype(str).values
test_day = test['SO3_T'].round(5).astype(str).values
y_raw = train['TARGET'].values.astype(np.float64)
lo_y, hi_y = np.percentile(y_raw, 1), np.percentile(y_raw, 99)
y_wins = np.clip(y_raw, lo_y, hi_y)

# Merge oracle with test
test_with_oracle = test.merge(oracle_df[['ID', 'TARGET']], on='ID', how='inner')
test_oracle_y = test_with_oracle['TARGET'].values.astype(np.float64)
test_oracle_day = test_with_oracle['SO3_T'].round(5).astype(str).values

print(f'Train: {train.shape}, Test: {test.shape}, Oracle matched: {len(test_with_oracle)}')

# =====================================================================
# 1. TARGET DISTRIBUTION ANALYSIS (Noise Floor)
# =====================================================================
print('\n' + '='*70)
print('1. TARGET DISTRIBUTION & NOISE FLOOR')
print('='*70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Train target distribution
axes[0].hist(y_raw, bins=200, density=True, alpha=0.7, color='steelblue')
axes[0].set_title(f'Train TARGET distribution\nmean={y_raw.mean():.2e}, std={y_raw.std():.4f}\nkurtosis={pd.Series(y_raw).kurtosis():.1f}, skew={pd.Series(y_raw).skew():.2f}')
axes[0].set_xlim(-0.01, 0.01)
axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)

# Test oracle target distribution
axes[1].hist(test_oracle_y, bins=200, density=True, alpha=0.7, color='coral')
axes[1].set_title(f'Test TARGET (oracle) distribution\nmean={test_oracle_y.mean():.2e}, std={test_oracle_y.std():.4f}\nkurtosis={pd.Series(test_oracle_y).kurtosis():.1f}')
axes[1].set_xlim(-0.01, 0.01)
axes[1].axvline(0, color='red', linestyle='--', alpha=0.5)

# Per-day target std
day_stds_train = []
for d in np.unique(train_day):
    m = train_day == d
    if m.sum() >= 10:
        day_stds_train.append(y_raw[m].std())
day_stds_test = []
for d in np.unique(test_oracle_day):
    m = test_oracle_day == d
    if m.sum() >= 5:
        day_stds_test.append(test_oracle_y[m].std())

axes[2].hist(day_stds_train, bins=50, density=True, alpha=0.6, color='steelblue', label='Train days')
axes[2].hist(day_stds_test, bins=50, density=True, alpha=0.6, color='coral', label='Test days')
axes[2].set_title(f'Per-day TARGET std\nTrain median={np.median(day_stds_train):.4f}\nTest median={np.median(day_stds_test):.4f}')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'ceiling_1_target_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: ceiling_1_target_distributions.png')

train_kurt = pd.Series(y_raw).kurtosis()
test_kurt = pd.Series(test_oracle_y).kurtosis()
print(f'  Train kurtosis: {train_kurt:.1f}')
print(f'  Test kurtosis: {test_kurt:.1f}')
print(f'  Train TARGET std: {y_raw.std():.6f}')
print(f'  Test TARGET std: {test_oracle_y.std():.6f}')

# Theoretical: for Gaussian noise, R² ceiling with p features and n samples
# R² ≈ p/n for pure noise. If our R² is near this, we're fitting noise.
n_train = len(y_raw)
p_feat = len(all_feat)
noise_r2 = p_feat / n_train
print(f'  Noise floor R² (p/n = {p_feat}/{n_train}): {noise_r2:.6f}')

# =====================================================================
# 2. ORACLE SIGNAL STRENGTH: Per-day Pearson IC on test
# =====================================================================
print('\n' + '='*70)
print('2. ORACLE SIGNAL ANALYSIS (Feature → Test TARGET)')
print('='*70)

# Compute per-feature Pearson correlation with test oracle
# This tells us how much signal each feature carries for illiquid assets
feature_test_ics = {}
te_feat_raw = test_with_oracle[all_feat].fillna(0).values.astype(np.float64)

for ci, f in enumerate(all_feat):
    x = te_feat_raw[:, ci]
    valid = ~np.isnan(x) & ~np.isnan(test_oracle_y)
    if valid.sum() < 100:
        continue
    ic, _ = pearsonr(x[valid], test_oracle_y[valid])
    if not np.isnan(ic):
        feature_test_ics[f] = ic

sorted_ics = sorted(feature_test_ics.items(), key=lambda x: -abs(x[1]))
print(f'  Top 10 features by |Pearson IC| with test oracle TARGET:')
for f, ic in sorted_ics[:10]:
    print(f'    {f:>40s}: IC={ic:+.6f}')

all_ics = np.array([v for v in feature_test_ics.values()])
print(f'\n  Mean |IC|: {np.mean(np.abs(all_ics)):.6f}')
print(f'  Max |IC|:  {np.max(np.abs(all_ics)):.6f}')
print(f'  Features with |IC| > 0.01: {np.sum(np.abs(all_ics) > 0.01)}')
print(f'  Features with |IC| > 0.02: {np.sum(np.abs(all_ics) > 0.02)}')

# =====================================================================
# 3. THEORETICAL R² CEILING FROM IC
# =====================================================================
print('\n' + '='*70)
print('3. THEORETICAL R² CEILING FROM INFORMATION COEFFICIENT')
print('='*70)

# Key formula from Grinold (1989): R² ≈ IC² for single factor
# For multiple uncorrelated factors: R² ≈ Σ IC²_i
# For correlated factors: R² ≈ IC_vec' @ Corr_inv @ IC_vec (bounded above by Σ IC²_i)

# Method A: Sum of squared ICs (upper bound assuming uncorrelated features)
top_n_ics = sorted(np.abs(all_ics), reverse=True)
cumulative_r2_bound = np.cumsum(np.array(top_n_ics)**2)
print(f'  R² bound from top-1 feature:  {top_n_ics[0]**2:.8f}')
print(f'  R² bound from top-5 features: {cumulative_r2_bound[4]:.8f}')
print(f'  R² bound from top-10 features: {cumulative_r2_bound[9]:.8f}')
print(f'  R² bound from top-50 features: {cumulative_r2_bound[49]:.8f}')
print(f'  R² bound from ALL features:   {cumulative_r2_bound[-1]:.8f}')
print(f'  NOTE: This is an UPPER bound (assumes uncorrelated features)')

# Method B: Best linear combination R² on test oracle (cheating oracle)
# Use Ridge on test features to predict test oracle target
# This is the absolute ceiling — what you'd get with perfect knowledge
print('\n  Computing ORACLE Ridge R² (best possible linear model on test data)...')
X_test_oracle = te_feat_raw.copy()
X_test_oracle = np.nan_to_num(X_test_oracle, nan=0.0)
# Z-score
mu = X_test_oracle.mean(0)
sg = X_test_oracle.std(0)
sg[sg < 1e-8] = 1.0
X_test_z = (X_test_oracle - mu) / sg
np.clip(X_test_z, -5, 5, out=X_test_z)

# 5-fold CV on test oracle
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oracle_r2s = []
for tri, tei in kf.split(X_test_z):
    mdl = Ridge(alpha=1000, fit_intercept=True)
    mdl.fit(X_test_z[tri], test_oracle_y[tri])
    pred = mdl.predict(X_test_z[tei])
    ss_res = np.sum((test_oracle_y[tei] - pred)**2)
    ss_tot = np.sum((test_oracle_y[tei] - test_oracle_y[tei].mean())**2)
    r2 = 1 - ss_res / ss_tot
    oracle_r2s.append(r2)
print(f'  Oracle Ridge CV R² on TEST data: {np.mean(oracle_r2s):.6f} ± {np.std(oracle_r2s):.6f}')

# Also try different alphas
for alpha in [100, 500, 1000, 5000, 10000, 50000]:
    r2s = []
    for tri, tei in kf.split(X_test_z):
        mdl = Ridge(alpha=alpha, fit_intercept=True)
        mdl.fit(X_test_z[tri], test_oracle_y[tri])
        pred = mdl.predict(X_test_z[tei])
        ss_res = np.sum((test_oracle_y[tei] - pred)**2)
        ss_tot = np.sum((test_oracle_y[tei] - test_oracle_y[tei].mean())**2)
        r2s.append(1 - ss_res / ss_tot)
    print(f'    alpha={alpha:>6d}: R²={np.mean(r2s):+.6f}')

# =====================================================================
# 4. TRAIN CV R² (What's achievable on liquid assets)
# =====================================================================
print('\n' + '='*70)
print('4. TRAIN CROSS-VALIDATED R² (liquid asset ceiling)')
print('='*70)

X_train_raw = train[all_feat].fillna(0).values.astype(np.float64)
mu_tr = X_train_raw.mean(0)
sg_tr = X_train_raw.std(0)
sg_tr[sg_tr < 1e-8] = 1.0
X_train_z = np.clip((X_train_raw - mu_tr) / sg_tr, -5, 5)

groups5 = pd.qcut(pd.Series(train['SO3_T'].values), q=5, labels=False,
                  duplicates='drop').values.astype(np.int32)

gkf = GroupKFold(n_splits=5)
train_cv_r2s = []
for tri, vai in gkf.split(X_train_z, y_wins, groups=groups5):
    mdl = Ridge(alpha=5000, fit_intercept=True)
    mdl.fit(X_train_z[tri], y_wins[tri])
    pred = mdl.predict(X_train_z[vai])
    ss_res = np.sum((y_wins[vai] - pred)**2)
    ss_tot = np.sum((y_wins[vai] - y_wins[vai].mean())**2)
    r2 = 1 - ss_res / ss_tot
    train_cv_r2s.append(r2)
    print(f'    Fold R²: {r2:+.8f}')

print(f'  Train CV R² (Ridge, GroupKFold): {np.mean(train_cv_r2s):+.8f} ± {np.std(train_cv_r2s):.8f}')

# =====================================================================
# 5. PER-DAY R² ANALYSIS ON TEST ORACLE
# =====================================================================
print('\n' + '='*70)
print('5. PER-DAY R² ON TEST ORACLE (using best submission)')
print('='*70)

# Load best submission
best_sub = pd.read_csv(os.path.join(OUT_DIR, 'submissions', 'fc_lgb50_rpd50.csv'))
merged = test_with_oracle.merge(best_sub.rename(columns={'TARGET': 'PRED'}), on='ID')

day_r2s = []
day_sizes = []
day_names = []
for d, g in merged.groupby(merged['SO3_T'].round(5).astype(str)):
    if len(g) < 3:
        continue
    y_true = g['TARGET'].values
    y_pred = g['PRED'].values
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    if ss_tot > 1e-15:
        r2 = 1 - ss_res / ss_tot
        day_r2s.append(r2)
        day_sizes.append(len(g))
        day_names.append(d)

day_r2s = np.array(day_r2s)
day_sizes = np.array(day_sizes)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Distribution of per-day R²
axes[0].hist(day_r2s, bins=50, density=True, alpha=0.7, color='steelblue')
axes[0].axvline(np.mean(day_r2s), color='red', linestyle='--', label=f'mean={np.mean(day_r2s):.4f}')
axes[0].axvline(np.median(day_r2s), color='orange', linestyle='--', label=f'median={np.median(day_r2s):.4f}')
axes[0].axvline(0, color='black', linestyle='-', alpha=0.3)
axes[0].set_title('Per-day R² distribution (fc_lgb50_rpd50)')
axes[0].set_xlabel('R²')
axes[0].legend()

# R² vs day size
axes[1].scatter(day_sizes, day_r2s, alpha=0.3, s=10, color='steelblue')
axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[1].set_xlabel('# test rows per day')
axes[1].set_ylabel('R²')
axes[1].set_title('R² vs day size')

# Cumulative: what fraction of days have positive R²?
sorted_r2 = np.sort(day_r2s)
cdf = np.arange(1, len(sorted_r2)+1) / len(sorted_r2)
axes[2].plot(sorted_r2, cdf, color='steelblue')
axes[2].axvline(0, color='red', linestyle='--', alpha=0.5)
axes[2].set_xlabel('R²')
axes[2].set_ylabel('Cumulative fraction of days')
axes[2].set_title(f'CDF of per-day R²\n{np.mean(day_r2s > 0):.1%} of days have R² > 0')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'ceiling_2_perday_r2.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: ceiling_2_perday_r2.png')

print(f'  Per-day R² mean: {np.mean(day_r2s):+.6f}')
print(f'  Per-day R² median: {np.median(day_r2s):+.6f}')
print(f'  Per-day R² std: {np.std(day_r2s):.6f}')
print(f'  Days with R² > 0: {np.mean(day_r2s > 0):.1%}')
print(f'  Days with R² > 0.001: {np.mean(day_r2s > 0.001):.1%}')
print(f'  Days with R² < -0.01: {np.mean(day_r2s < -0.01):.1%}')

# Size-weighted R² (approximates full-test R²)
weighted_r2 = np.average(day_r2s, weights=day_sizes)
print(f'  Size-weighted mean R²: {weighted_r2:+.6f}')

# =====================================================================
# 6. LIQUID → ILLIQUID TRANSFER DECAY
# =====================================================================
print('\n' + '='*70)
print('6. LIQUID → ILLIQUID TRANSFER DECAY')
print('='*70)

# For overlap days: train Ridge on liquid (training), evaluate on liquid (CV) AND illiquid (test oracle)
# The ratio tells us how much signal transfers across the domain shift

train_days_set_arr = np.unique(train_day)
test_days_set = set(np.unique(test_oracle_day))
overlap_days = [d for d in train_days_set_arr if d in test_days_set]

# Pre-allocate matrices ONCE (avoid repeated DataFrame→numpy conversion)
print('  Pre-allocating matrices...', flush=True)
train_feat_matrix = train[all_feat].fillna(0).values.astype(np.float64)
test_feat_matrix = test_with_oracle[all_feat].fillna(0).values.astype(np.float64)

liquid_r2s = []
illiquid_r2s = []
transfer_ratios = []
n_checked = 0

for d in overlap_days[:200]:  # Sample 200 days for speed
    tr_mask = train_day == d
    n_tr = tr_mask.sum()
    if n_tr < 40:
        continue

    te_mask = test_oracle_day == d
    n_te = te_mask.sum()
    if n_te < 5:
        continue

    X_d = train_feat_matrix[tr_mask]
    y_d = y_wins[tr_mask]
    mu_d = X_d.mean(0)
    sg_d = X_d.std(0)
    sg_d[sg_d < 1e-8] = 1.0
    X_d_z = np.clip((X_d - mu_d) / sg_d, -5, 5)

    # Liquid CV: simple split
    n_half = n_tr // 2
    mdl = Ridge(alpha=5000, fit_intercept=True)
    mdl.fit(X_d_z[:n_half], y_d[:n_half])
    pred_liq = mdl.predict(X_d_z[n_half:])
    ss_res_liq = np.sum((y_d[n_half:] - pred_liq)**2)
    ss_tot_liq = np.sum((y_d[n_half:] - y_d[n_half:].mean())**2)
    r2_liq = 1 - ss_res_liq / ss_tot_liq if ss_tot_liq > 1e-15 else 0

    # Illiquid: train on full liquid, predict illiquid
    mdl2 = Ridge(alpha=5000, fit_intercept=True)
    mdl2.fit(X_d_z, y_d)
    X_te_d = test_feat_matrix[te_mask]
    X_te_d_z = np.clip((X_te_d - mu_d) / sg_d, -5, 5)
    y_te_d = test_oracle_y[te_mask]
    pred_illiq = mdl2.predict(X_te_d_z)
    ss_res_ill = np.sum((y_te_d - pred_illiq)**2)
    ss_tot_ill = np.sum((y_te_d - y_te_d.mean())**2)
    r2_ill = 1 - ss_res_ill / ss_tot_ill if ss_tot_ill > 1e-15 else 0

    liquid_r2s.append(r2_liq)
    illiquid_r2s.append(r2_ill)
    if r2_liq > 0.001:
        transfer_ratios.append(r2_ill / r2_liq)
    n_checked += 1

liquid_r2s = np.array(liquid_r2s)
illiquid_r2s = np.array(illiquid_r2s)
transfer_ratios = np.array(transfer_ratios)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(liquid_r2s, illiquid_r2s, alpha=0.3, s=15, color='steelblue')
axes[0].plot([-0.05, 0.05], [-0.05, 0.05], 'r--', alpha=0.5, label='y=x')
axes[0].axhline(0, color='gray', alpha=0.3)
axes[0].axvline(0, color='gray', alpha=0.3)
axes[0].set_xlabel('Liquid R² (within-day CV)')
axes[0].set_ylabel('Illiquid R² (test oracle)')
axes[0].set_title(f'Transfer: Liquid → Illiquid\nn={n_checked} days')
axes[0].legend()
axes[0].set_xlim(-0.1, 0.1)
axes[0].set_ylim(-0.1, 0.1)

axes[1].hist(transfer_ratios[np.isfinite(transfer_ratios)], bins=50, density=True, alpha=0.7, color='coral')
axes[1].axvline(np.median(transfer_ratios[np.isfinite(transfer_ratios)]), color='red', linestyle='--',
                label=f'median={np.median(transfer_ratios[np.isfinite(transfer_ratios)]):.2f}')
axes[1].set_title('Transfer ratio (illiquid R² / liquid R²)\nfor days with liquid R² > 0.001')
axes[1].set_xlabel('Transfer ratio')
axes[1].legend()
axes[1].set_xlim(-5, 5)

axes[2].hist(illiquid_r2s, bins=50, density=True, alpha=0.6, color='coral', label='Illiquid')
axes[2].hist(liquid_r2s, bins=50, density=True, alpha=0.6, color='steelblue', label='Liquid')
axes[2].set_title(f'R² distributions\nLiquid mean={np.mean(liquid_r2s):+.4f}\nIlliquid mean={np.mean(illiquid_r2s):+.4f}')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'ceiling_3_transfer_decay.png'), dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: ceiling_3_transfer_decay.png')

print(f'  Days checked: {n_checked}')
print(f'  Liquid R² mean: {np.mean(liquid_r2s):+.6f}')
print(f'  Illiquid R² mean: {np.mean(illiquid_r2s):+.6f}')
print(f'  Transfer ratio median: {np.median(transfer_ratios[np.isfinite(transfer_ratios)]):+.4f}')
print(f'  Transfer ratio mean: {np.mean(transfer_ratios[np.isfinite(transfer_ratios) & (np.abs(transfer_ratios) < 10)]):+.4f}')

# =====================================================================
# 7. CEILING SYNTHESIS
# =====================================================================
print('\n' + '='*70)
print('7. CEILING SYNTHESIS')
print('='*70)

# Best oracle IC from our experiments
best_oracle_ic = 0.047184  # mono LGB
best_lb_score = 0.00079

# R² from IC: for Pearson correlation r, R² = r²
# But IC is Spearman (rank), so R² ≈ (2/π * arcsin(IC))² for Gaussian
# Simpler: R² ≈ IC² is a rough lower bound
ic_to_r2_simple = best_oracle_ic**2
print(f'  Best oracle IC (daywise Spearman): {best_oracle_ic:+.6f}')
print(f'  Naive R² = IC²: {ic_to_r2_simple:.6f}')
print(f'  But this is DAYWISE IC averaged → not directly comparable to full-test R²')

print(f'\n  --- Summary of Ceiling Estimates ---')
print(f'  Noise floor (p/n):                        {noise_r2:.8f}')
print(f'  Oracle Ridge CV R² on TEST data:           {np.mean(oracle_r2s):+.8f}')
print(f'  Train CV R² (liquid assets):               {np.mean(train_cv_r2s):+.8f}')
print(f'  Illiquid R² (per-day Ridge, mean):         {np.mean(illiquid_r2s):+.8f}')
print(f'  Best LB score achieved:                    {best_lb_score:+.8f}')
print(f'  Transfer decay (illiq/liq):                {np.mean(transfer_ratios[np.isfinite(transfer_ratios) & (np.abs(transfer_ratios) < 10)]):+.4f}')

# =====================================================================
# 8. FINAL VISUALIZATION: CEILING SUMMARY
# =====================================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

estimates = {
    'Noise floor\n(p/n)': noise_r2,
    'Our best LB\n(fc_lgb50_rpd50)': best_lb_score,
    'Illiquid per-day\nRidge R² mean': max(0, np.mean(illiquid_r2s)),
    'Oracle Ridge CV\non test data': max(0, np.mean(oracle_r2s)),
    'Train CV R²\n(liquid assets)': max(0, np.mean(train_cv_r2s)),
    'Sum IC² bound\n(top 50 features)': cumulative_r2_bound[49],
}

names = list(estimates.keys())
values = [estimates[k] for k in names]

colors = ['gray', 'green', 'coral', 'steelblue', 'steelblue', 'orange']
bars = ax.barh(names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_width() + max(values)*0.02, bar.get_y() + bar.get_height()/2,
            f'{val:.6f}', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('R²', fontsize=12)
ax.set_title('Ceiling Analysis: Maximum Achievable R²\nfor Compliant Short-Horizon Return Prediction', fontsize=14)
ax.axvline(best_lb_score, color='green', linestyle='--', alpha=0.5, label=f'Current best LB: {best_lb_score}')
ax.legend(fontsize=10)
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'ceiling_4_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print('\n  Saved: ceiling_4_summary.png')

# =====================================================================
# 9. LEADERBOARD CONTEXT
# =====================================================================
print('\n' + '='*70)
print('9. LEADERBOARD CONTEXT')
print('='*70)
print(f'  1st place LB: +0.07255')
print(f'  2nd place LB: +0.01482')
print(f'  3rd place LB: +0.00096 (non-compliant)')
print(f'  Our best:     +0.00079 (compliant)')
print(f'  Gap to 2nd:   {0.01482 - 0.00079:.5f} (18.8x our score)')
print(f'  Gap to 1st:   {0.07255 - 0.00079:.5f} (91.8x our score)')
print(f'\n  If oracle Ridge CV R² on test = {np.mean(oracle_r2s):+.6f}:')
print(f'    1st place is achieving ~{0.07255 / max(0.0001, np.mean(oracle_r2s)) * 100:.0f}% of oracle ceiling')
print(f'    Our submission achieves ~{0.00079 / max(0.0001, np.mean(oracle_r2s)) * 100:.1f}% of oracle ceiling')

print(f'\nDone in {(time.time()-t0)/60:.1f} min')
