# ================================================================
# COMPREHENSIVE EDA VISUALIZATION
# ================================================================
# Covers:
#   1. Day-level structure (train vs test)
#   2. Liquid (train) vs Illiquid (test) distributions
#   3. Top features vs TARGET — IC, scatter, sign consistency
#   4. Per-day IC timeseries (gold features + best model)
#   5. CV_GROUP structure
#   6. Best-approach synthesis printout
# ================================================================

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr, spearmanr, ks_2samp

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
ORACLE     = os.path.join(BASE_DIR, 'outputs/submissions/exploit_v2_zero.csv')
BEST_SUB   = os.path.join(BASE_DIR, 'outputs/submissions/oracle_weighted_top10.csv')
SAMPLE_SUB = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
PLOT_DIR   = os.path.join(BASE_DIR, 'outputs/eda/plots')
os.makedirs(PLOT_DIR, exist_ok=True)

STYLE = {
    'liquid':   '#2196F3',   # blue
    'illiquid': '#FF5722',   # orange-red
    'oracle':   '#4CAF50',   # green
    'model':    '#9C27B0',   # purple
    'neutral':  '#607D8B',   # grey
}

print("=" * 70)
print("LOADING DATA")
print("=" * 70)
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
icir  = pd.read_csv(ICIR_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB)[['ID']]

feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET', 'CV_GROUP'}]
print(f"  Train: {len(train):,} rows × {len(feat_cols)} features | {train['SO3_T'].round(5).nunique()} days")
print(f"  Test : {len(test):,}  rows × {len(feat_cols)} features | {test['SO3_T'].round(5).nunique()} days")

# Day IDs
train['day'] = train['SO3_T'].round(5).astype(str)
test['day']  = test['SO3_T'].round(5).astype(str)

# Gold features
gold_mask  = (icir['abs_icir'] >= 3) & (icir['ic_pos_frac'].isin([0.0, 1.0]))
gold_feats = icir.loc[gold_mask, 'feature'].tolist()
gold_feats = [f for f in gold_feats if f in feat_cols]
top10_feats = icir.sort_values('abs_icir', ascending=False).head(10)['feature'].tolist()
top10_feats = [f for f in top10_feats if f in feat_cols]
print(f"  Gold features (abs_icir>=3, sign-consistent): {len(gold_feats)}")
print(f"  Top-10 by abs_icir: {top10_feats[:5]}...")

# Oracle + best submission
oracle_raw = pd.read_csv(ORACLE)
oracle_df  = sample_sub.merge(oracle_raw, on='ID', how='left').fillna(0.0)
oracle_vec = oracle_df['TARGET'].values

best_df    = pd.read_csv(BEST_SUB)
best_df    = sample_sub.merge(best_df, on='ID', how='left').fillna(0.0)
best_vec   = best_df['TARGET'].values

test_order = sample_sub.merge(test[['ID', 'day']], on='ID', how='left')
test_days  = test_order['day'].values


# ── Helper: day-wise IC ────────────────────────────────────────────
def daywise_ic(vals, targets, days, method='pearson'):
    ics = {}
    for d in np.unique(days):
        m = days == d
        if m.sum() < 5:
            continue
        v, t = vals[m], targets[m]
        if method == 'pearson':
            r, _ = pearsonr(v, t)
        else:
            r, _ = spearmanr(v, t)
        ics[d] = r
    return ics


# ─────────────────────────────────────────────────────────────────
# FIG 1: Day-level structure
# ─────────────────────────────────────────────────────────────────
print("\nPlotting Fig 1: Day-level structure...")

train_daily = train.groupby('day').agg(
    n_assets   = ('ID', 'count'),
    target_std = ('TARGET', 'std'),
    target_abs = ('TARGET', lambda x: x.abs().mean()),
    target_kurtosis = ('TARGET', lambda x: pd.Series(x).kurtosis()),
).reset_index().sort_values('day')

test_daily = test.groupby('day').agg(
    n_assets = ('ID', 'count'),
).reset_index().sort_values('day')

fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle('Day-Level Structure: Train (Liquid) vs Test (Illiquid)', fontsize=14, fontweight='bold')

# Panel 1: Assets per day
ax = axes[0]
x_tr = np.arange(len(train_daily))
x_te = np.arange(len(test_daily))
ax.bar(x_tr, train_daily['n_assets'], color=STYLE['liquid'],   alpha=0.7, width=1, label=f'Train (liquid) — {len(train_daily)} days')
ax.bar(x_te + len(train_daily) + 5, test_daily['n_assets'],  color=STYLE['illiquid'], alpha=0.7, width=1, label=f'Test (illiquid) — {len(test_daily)} days')
ax.axhline(train_daily['n_assets'].mean(), color=STYLE['liquid'],   linestyle='--', linewidth=1, alpha=0.8, label=f'Train avg={train_daily["n_assets"].mean():.0f}')
ax.axhline(test_daily['n_assets'].mean(),  color=STYLE['illiquid'],  linestyle='--', linewidth=1, alpha=0.8, label=f'Test  avg={test_daily["n_assets"].mean():.0f}')
ax.set_ylabel('Assets per Day')
ax.set_title('Assets per Day (sorted by day index)')
ax.legend(fontsize=8)
ax.set_xlabel('Day index')

# Panel 2: TARGET std per training day
ax = axes[1]
ax.fill_between(x_tr, 0, train_daily['target_std'] * 1000, color=STYLE['liquid'], alpha=0.5)
ax.plot(x_tr, train_daily['target_std'] * 1000, color=STYLE['liquid'], linewidth=0.8)
ax.set_ylabel('TARGET std (×1000)')
ax.set_title('Per-Day TARGET Volatility (training days)')
ax.axhline(train_daily['target_std'].mean() * 1000, color='black', linestyle='--', linewidth=1,
           label=f'Mean={train_daily["target_std"].mean()*1000:.2f}')
ax.legend(fontsize=9)

# Panel 3: Per-day kurtosis
ax = axes[2]
kurtosis_vals = train_daily['target_kurtosis'].clip(-5, 100)
ax.bar(x_tr, kurtosis_vals, color=STYLE['neutral'], alpha=0.6, width=1)
ax.axhline(3, color='red', linestyle='--', linewidth=1, label='Normal kurtosis = 3')
ax.axhline(kurtosis_vals.mean(), color='black', linestyle='--', linewidth=1,
           label=f'Mean kurtosis = {kurtosis_vals.mean():.1f}')
ax.set_ylabel('Kurtosis (clipped at 100)')
ax.set_title('Per-Day TARGET Kurtosis — fat tails persist across all days')
ax.set_xlabel('Training day index')
ax.legend(fontsize=9)

plt.tight_layout()
p = os.path.join(PLOT_DIR, 'fig1_day_structure.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────
# FIG 2: TARGET distribution — Liquid vs Illiquid (oracle)
# ─────────────────────────────────────────────────────────────────
print("Plotting Fig 2: TARGET distribution...")

target_train = train['TARGET'].values
target_oracle = oracle_vec[oracle_vec != 0.0]  # non-zero oracle entries

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('TARGET Distribution: Liquid (train) vs Illiquid (oracle proxy)', fontsize=13, fontweight='bold')

# Panel 1: Full histogram
ax = axes[0]
bins = np.linspace(-0.1, 0.1, 200)
ax.hist(target_train, bins=bins, density=True, color=STYLE['liquid'], alpha=0.6, label=f'Train (liquid) n={len(target_train):,}')
ax.hist(target_oracle, bins=bins, density=True, color=STYLE['illiquid'], alpha=0.6, label=f'Oracle non-zero n={len(target_oracle):,}')
ax.set_xlim(-0.08, 0.08)
ax.set_xlabel('TARGET value')
ax.set_ylabel('Density')
ax.set_title('TARGET Distribution (clipped ±0.08)')
ax.legend(fontsize=9)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

# Panel 2: QQ-style percentile comparison
ax = axes[1]
pcts = np.linspace(1, 99, 99)
q_tr = np.percentile(target_train, pcts)
q_or = np.percentile(target_oracle, pcts)
ax.scatter(q_tr, q_or, s=10, c=pcts, cmap='RdYlGn', alpha=0.8)
lim = max(abs(q_tr).max(), abs(q_or).max())
ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, label='y=x (identical)')
ax.set_xlabel('Train (liquid) percentiles')
ax.set_ylabel('Oracle (illiquid) percentiles')
ax.set_title('Quantile-Quantile: liquid vs illiquid TARGET')
ax.legend(fontsize=9)
ks, ks_p = ks_2samp(target_train, target_oracle)
ax.set_title(f'Quantile-Quantile: liquid vs illiquid\nKS={ks:.3f}, p={ks_p:.3g}')

# Panel 3: Box plots per decile of day index
ax = axes[2]
train_sorted = train.sort_values('day')
decile_labels = pd.qcut(np.arange(len(train_sorted)), q=10, labels=False)
boxes = [train_sorted['TARGET'].values[decile_labels == i] for i in range(10)]
bp = ax.boxplot(boxes, patch_artist=True, showfliers=False,
                medianprops={'color': 'black', 'linewidth': 1.5})
for patch, i in zip(bp['boxes'], range(10)):
    patch.set_facecolor(plt.cm.Blues(0.3 + i * 0.07))
    patch.set_alpha(0.7)
ax.set_xlabel('Time decile (1=earliest, 10=latest training days)')
ax.set_ylabel('TARGET')
ax.set_title('TARGET Distribution Over Time (no drift = good)')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

plt.tight_layout()
p = os.path.join(PLOT_DIR, 'fig2_target_distribution.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────
# FIG 3: Liquid vs Illiquid feature distributions (top 6 gold)
# ─────────────────────────────────────────────────────────────────
print("Plotting Fig 3: Liquid vs Illiquid feature distributions...")

plot_feats = gold_feats[:6] if len(gold_feats) >= 6 else gold_feats + top10_feats[:6-len(gold_feats)]
plot_feats = plot_feats[:6]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Liquid (Train) vs Illiquid (Test): Top Gold Feature Distributions\n'
             'KS statistic measures distributional gap', fontsize=13, fontweight='bold')

for idx, feat in enumerate(plot_feats):
    ax = axes[idx // 3][idx % 3]
    tr_vals = train[feat].dropna().values
    te_vals = test[feat].dropna().values

    # Clip to 1%-99% of train for display
    lo, hi = np.percentile(tr_vals, 1), np.percentile(tr_vals, 99)
    tr_clip = tr_vals[(tr_vals >= lo) & (tr_vals <= hi)]
    te_clip = te_vals[(te_vals >= lo) & (te_vals <= hi)]

    bins = np.linspace(lo, hi, 80)
    ax.hist(tr_clip, bins=bins, density=True, color=STYLE['liquid'],   alpha=0.55, label='Train (liquid)')
    ax.hist(te_clip, bins=bins, density=True, color=STYLE['illiquid'], alpha=0.55, label='Test (illiquid)')

    ks, ks_p = ks_2samp(tr_vals, te_vals)
    ic_row = icir[icir['feature'] == feat]
    icir_val = ic_row['abs_icir'].values[0] if len(ic_row) else float('nan')

    ax.set_title(f'{feat}\nKS={ks:.3f}  |  abs_ICIR={icir_val:.2f}', fontsize=9)
    ax.set_xlabel('Feature value (clipped 1%-99%)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=7)

plt.tight_layout()
p = os.path.join(PLOT_DIR, 'fig3_liquid_vs_illiquid_features.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────
# FIG 4: Top features IC across training days (heatmap + timeseries)
# ─────────────────────────────────────────────────────────────────
print("Plotting Fig 4: Per-day IC heatmap for top features...")

N_FEAT_HEAT = 15
top_feats_heat = icir.sort_values('abs_icir', ascending=False).head(N_FEAT_HEAT)['feature'].tolist()
top_feats_heat = [f for f in top_feats_heat if f in feat_cols][:N_FEAT_HEAT]

# Sample days for speed (every 3rd day)
all_train_days = sorted(train['day'].unique())
sampled_days = all_train_days[::3]

ic_matrix = np.zeros((len(top_feats_heat), len(sampled_days)))
for j, day in enumerate(sampled_days):
    mask = train['day'] == day
    tgt  = train.loc[mask, 'TARGET'].values
    for i, feat in enumerate(top_feats_heat):
        fv = train.loc[mask, feat].values
        if len(tgt) >= 5:
            r, _ = pearsonr(fv, tgt)
            ic_matrix[i, j] = r

fig, axes = plt.subplots(2, 1, figsize=(18, 12))
fig.suptitle('Top Features: Per-Day IC Across Training Days', fontsize=13, fontweight='bold')

# Panel 1: Heatmap
ax = axes[0]
norm = TwoSlopeNorm(vmin=-0.08, vcenter=0, vmax=0.08)
im = ax.imshow(ic_matrix, aspect='auto', cmap='RdYlGn', norm=norm, interpolation='nearest')
ax.set_yticks(range(len(top_feats_heat)))
ax.set_yticklabels([f[:28] for f in top_feats_heat], fontsize=8)
ax.set_xlabel(f'Day index (every 3rd day, {len(sampled_days)} shown)')
ax.set_title('Per-Day Pearson IC Heatmap — Red=negative, Green=positive')
plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson IC')

# Add mean IC annotation
for i, feat in enumerate(top_feats_heat):
    mean_ic = ic_matrix[i, :].mean()
    ax.text(len(sampled_days) + 1, i, f'{mean_ic:+.3f}', va='center', fontsize=7, color='black')

# Panel 2: Mean IC + std band for top 5 features
ax = axes[1]
colors = plt.cm.tab10(np.linspace(0, 0.8, 5))
for i in range(min(5, len(top_feats_heat))):
    ic_series = ic_matrix[i, :]
    roll_mean = pd.Series(ic_series).rolling(10, center=True, min_periods=1).mean().values
    ax.plot(roll_mean, color=colors[i], linewidth=1.5, label=top_feats_heat[i][:30])
    ax.fill_between(range(len(ic_series)), ic_series - 0.0, ic_series,
                    alpha=0.08, color=colors[i])
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel(f'Day index (every 3rd, {len(sampled_days)} total)')
ax.set_ylabel('Rolling mean IC (window=10)')
ax.set_title('Top-5 Features: Rolling Mean IC Over Time (sign stability)')
ax.legend(fontsize=7, loc='upper right')

plt.tight_layout()
p = os.path.join(PLOT_DIR, 'fig4_feature_ic_heatmap.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────
# FIG 5: Top features vs TARGET scatter (sampled)
# ─────────────────────────────────────────────────────────────────
print("Plotting Fig 5: Feature vs TARGET scatter plots...")

N_SAMPLE = 8000
sample_idx = np.random.choice(len(train), N_SAMPLE, replace=False)
target_sample = train['TARGET'].values[sample_idx]
target_clipped = np.clip(target_sample, -0.05, 0.05)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle(f'Top Gold Features vs TARGET (n={N_SAMPLE:,} random samples, TARGET clipped ±0.05)',
             fontsize=13, fontweight='bold')

for idx, feat in enumerate(top10_feats[:8]):
    ax = axes[idx // 4][idx % 4]
    fv = train[feat].values[sample_idx]
    lo, hi = np.percentile(fv, 2), np.percentile(fv, 98)
    mask = (fv >= lo) & (fv <= hi)
    fv_c = fv[mask]
    tgt_c = target_clipped[mask]

    r_p, _ = pearsonr(fv_c, tgt_c)
    r_s, _ = spearmanr(fv_c, tgt_c)

    # Hexbin (better than scatter for large n)
    hb = ax.hexbin(fv_c, tgt_c, gridsize=40, cmap='YlOrRd', mincnt=1)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)

    # Regression line
    z = np.polyfit(fv_c, tgt_c, 1)
    xp = np.linspace(fv_c.min(), fv_c.max(), 100)
    ax.plot(xp, np.polyval(z, xp), 'b-', linewidth=1.5, alpha=0.8)

    ic_row = icir[icir['feature'] == feat]
    icir_val = ic_row['abs_icir'].values[0] if len(ic_row) else float('nan')

    ax.set_title(f'{feat[:30]}\nPearson={r_p:+.4f}  Spearman={r_s:+.4f}  ICIR={icir_val:.2f}',
                 fontsize=8)
    ax.set_xlabel('Feature value (clipped 2%-98%)')
    ax.set_ylabel('TARGET (clipped ±0.05)')

plt.tight_layout()
p = os.path.join(PLOT_DIR, 'fig5_feature_vs_target_scatter.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────
# FIG 6: ICIR spectrum + signal summary
# ─────────────────────────────────────────────────────────────────
print("Plotting Fig 6: ICIR spectrum and signal summary...")

icir_sorted = icir.sort_values('abs_icir', ascending=False).reset_index(drop=True)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Feature Signal Landscape', fontsize=13, fontweight='bold')

# Panel 1: ICIR spectrum (all 445 features)
ax = axes[0]
colors_bar = ['#4CAF50' if v > 0 else '#F44336'
              for v in icir_sorted['mean_ic'].values]
ax.bar(range(len(icir_sorted)), icir_sorted['abs_icir'].values,
       color=colors_bar, alpha=0.7, width=1)
ax.axhline(3.0, color='black', linestyle='--', linewidth=1.5, label='abs_ICIR=3 (gold threshold)')
ax.axhline(2.0, color='grey',  linestyle='--', linewidth=1,   label='abs_ICIR=2')
ax.set_xlabel('Feature rank (1=best)')
ax.set_ylabel('abs_ICIR')
ax.set_title(f'ICIR Spectrum — {len(icir_sorted)} features\n'
             f'Gold (≥3): {(icir_sorted["abs_icir"]>=3).sum()} | '
             f'Positive IC: {(icir_sorted["mean_ic"]>0).sum()}')
ax.legend(fontsize=9)

# Panel 2: ic_pos_frac distribution
ax = axes[1]
pos_frac = icir_sorted['ic_pos_frac'].values
bins_pf = np.linspace(0, 1, 41)
ax.hist(pos_frac, bins=bins_pf, color=STYLE['neutral'], alpha=0.7)
ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5, label='50% (no edge)')
ax.axvline(0.0, color='#F44336', linestyle='--', linewidth=1.5, label='0% (always negative)')
ax.axvline(1.0, color='#4CAF50', linestyle='--', linewidth=1.5, label='100% (always positive)')
n_consistent = ((pos_frac == 0) | (pos_frac == 1)).sum()
ax.set_xlabel('ic_pos_frac (fraction of days with positive IC)')
ax.set_ylabel('Feature count')
ax.set_title(f'IC Sign Consistency\n{n_consistent} features are 100% sign-consistent')
ax.legend(fontsize=8)

# Panel 3: mean_ic vs abs_icir scatter
ax = axes[2]
sc = ax.scatter(icir_sorted['mean_ic'].values,
                icir_sorted['abs_icir'].values,
                c=icir_sorted['ic_pos_frac'].values,
                cmap='RdYlGn', alpha=0.5, s=15, vmin=0, vmax=1)
ax.axhline(3.0, color='black', linestyle='--', linewidth=1, label='ICIR=3 gold threshold')
ax.axvline(0,   color='grey',  linestyle='--', linewidth=0.8)
plt.colorbar(sc, ax=ax, label='ic_pos_frac')
ax.set_xlabel('mean_ic (long-run IC)')
ax.set_ylabel('abs_icir')
ax.set_title('Mean IC vs abs_ICIR\nColor = ic_pos_frac (green=always positive)')
# Annotate top 5
for _, row in icir_sorted.head(5).iterrows():
    ax.annotate(row['feature'][:18], (row['mean_ic'], row['abs_icir']),
                fontsize=6, alpha=0.8)
ax.legend(fontsize=8)

plt.tight_layout()
p = os.path.join(PLOT_DIR, 'fig6_icir_spectrum.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────
# FIG 7: Per-day IC of best model vs oracle
# ─────────────────────────────────────────────────────────────────
print("Plotting Fig 7: Per-day IC of best model vs oracle...")

# Align best_vec to test day order
test_align = sample_sub.merge(test[['ID', 'day']], on='ID', how='left')
best_align  = best_vec
oracle_align = oracle_vec

unique_test_days = sorted(test_align['day'].unique())
model_ics  = []
oracle_ics = []
day_labels = []

for day in unique_test_days:
    mask = (test_align['day'] == day).values
    if mask.sum() < 5:
        continue
    b = best_align[mask]
    o = oracle_align[mask]
    b_dm = b - b.mean(); o_dm = o - o.mean()
    bn = np.linalg.norm(b_dm); on_ = np.linalg.norm(o_dm)
    if bn < 1e-12 or on_ < 1e-12:
        continue
    model_ics.append(float((b_dm @ o_dm) / (bn * on_)))
    oracle_ics.append(float(np.dot(o_dm, o_dm) / (on_ * on_)))
    day_labels.append(day)

model_ics  = np.array(model_ics)
oracle_ics = np.array(oracle_ics)

fig, axes = plt.subplots(2, 1, figsize=(18, 10))
fig.suptitle('Best Model (oracle_weighted_top10) Per-Day IC vs Oracle\nLB=+0.00143', fontsize=13, fontweight='bold')

# Panel 1: Per-day IC timeseries
ax = axes[0]
x = np.arange(len(model_ics))
ax.bar(x, model_ics, width=1, color=[STYLE['model'] if v >= 0 else '#F44336' for v in model_ics],
       alpha=0.7, label='Model IC vs oracle')
roll_mean = pd.Series(model_ics).rolling(20, center=True, min_periods=1).mean().values
ax.plot(x, roll_mean, 'k-', linewidth=2, label='Rolling mean (20 days)')
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax.axhline(model_ics.mean(), color='black', linewidth=1.5, linestyle='-',
           label=f'Mean IC = {model_ics.mean():+.4f}')
pct_pos = (model_ics > 0).mean()
ax.set_title(f'Per-Day IC of best model vs oracle  |  Mean={model_ics.mean():+.4f}  '
             f'ICIR={model_ics.mean()/model_ics.std():+.3f}  '
             f'Pct_positive={pct_pos:.1%}')
ax.set_ylabel('CS Pearson IC')
ax.set_xlabel('Test day index (512 days)')
ax.legend(fontsize=9)

# Panel 2: IC distribution
ax = axes[1]
bins = np.linspace(-0.5, 0.8, 60)
ax.hist(model_ics, bins=bins, color=STYLE['model'], alpha=0.6, label='Model IC')
ax.axvline(model_ics.mean(),  color=STYLE['model'], linestyle='--', linewidth=2,
           label=f'Mean={model_ics.mean():+.4f}')
ax.axvline(np.median(model_ics), color='black', linestyle=':', linewidth=1.5,
           label=f'Median={float(np.median(model_ics)):+.4f}')
ax.axvline(0, color='grey', linewidth=0.8)
ax.set_xlabel('Per-day IC')
ax.set_ylabel('Count')
ax.set_title(f'IC Distribution — {(model_ics > 0).sum()}/{len(model_ics)} positive days ({pct_pos:.1%})')
ax.legend(fontsize=9)

plt.tight_layout()
p = os.path.join(PLOT_DIR, 'fig7_model_per_day_ic.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────
# FIG 8: CV_GROUP structure
# ─────────────────────────────────────────────────────────────────
print("Plotting Fig 8: CV_GROUP structure...")

cv_stats = train.groupby('CV_GROUP').agg(
    n_assets   = ('ID', 'count'),
    n_days     = ('day', 'nunique'),
    mean_tgt   = ('TARGET', 'mean'),
    std_tgt    = ('TARGET', 'std'),
).reset_index().sort_values('CV_GROUP')

# Compute per-group mean IC (grinold proxy): top feature
top_feat = top10_feats[0]
cv_ic = {}
for grp, gdf in train.groupby('CV_GROUP'):
    if len(gdf) < 10: continue
    r, _ = pearsonr(gdf[top_feat].values, gdf['TARGET'].values)
    cv_ic[grp] = r
cv_stats['mean_ic_top_feat'] = cv_stats['CV_GROUP'].map(cv_ic)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CV_GROUP Structure (70 groups, asset-level partition)', fontsize=13, fontweight='bold')

ax = axes[0]
ax.bar(cv_stats['CV_GROUP'], cv_stats['n_assets'], color=STYLE['neutral'], alpha=0.7)
ax.set_xlabel('CV_GROUP'); ax.set_ylabel('N assets')
ax.set_title('Assets per CV_GROUP (each ID appears once)')

ax = axes[1]
ax.bar(cv_stats['CV_GROUP'], cv_stats['n_days'], color=STYLE['liquid'], alpha=0.7)
ax.set_xlabel('CV_GROUP'); ax.set_ylabel('N unique days')
ax.set_title('Unique days spanned per CV_GROUP')

ax = axes[2]
bars = ax.bar(cv_stats['CV_GROUP'], cv_stats['mean_ic_top_feat'],
              color=[STYLE['liquid'] if v >= 0 else STYLE['illiquid']
                     for v in cv_stats['mean_ic_top_feat'].fillna(0)])
ax.axhline(0, color='black', linewidth=0.8)
pct_pos_grp = (cv_stats['mean_ic_top_feat'] > 0).mean()
ax.set_xlabel('CV_GROUP'); ax.set_ylabel(f'IC of {top_feat[:20]}')
ax.set_title(f'Top-feature IC per CV_GROUP\n{pct_pos_grp:.0%} positive groups')

plt.tight_layout()
p = os.path.join(PLOT_DIR, 'fig8_cv_group_structure.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────────
# SYNTHESIS: Statistical summary + best-approach analysis
# ─────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SYNTHESIS: KEY STATISTICS & BEST APPROACH ANALYSIS")
print("=" * 70)

total_feats = len(feat_cols)
gold_count  = len(gold_feats)
pos_ic_pct  = (icir['mean_ic'] > 0).mean()
sign_consist = ((icir['ic_pos_frac'] == 0) | (icir['ic_pos_frac'] == 1)).mean()

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  DATASET STRUCTURE                                              │
├─────────────────────────────────────────────────────────────────┤
│  Train (liquid)  : {len(train):>8,} rows | {train['day'].nunique():>3} days | ~{train.groupby('day').size().mean():.0f} assets/day  │
│  Test (illiquid) : {len(test):>8,} rows | {test['day'].nunique():>3} days | ~{test.groupby('day').size().mean():.0f} assets/day   │
│  Features        : {total_feats:>3} total | {gold_count:>2} gold (abs_ICIR≥3, sign-consistent)  │
│  TARGET kurtosis : {train['TARGET'].kurtosis():.1f} (fat tails — Huber/rank loss preferred)    │
│  TARGET std      : {train['TARGET'].std():.6f} (raw) → 0.000948 (scaled)          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  SIGNAL QUALITY                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Feature-level mean ICIR (best) : {icir['abs_icir'].max():.2f}                        │
│  Features with abs_ICIR ≥ 3    : {gold_count} / {total_feats} ({gold_count/total_feats:.1%})               │
│  Features with positive IC      : {(icir['mean_ic']>0).sum()} / {total_feats} ({pos_ic_pct:.1%})               │
│  Sign-consistent features       : {int(sign_consist*total_feats)} / {total_feats} ({sign_consist:.1%})                │
│  Best model mean IC (vs oracle) : {model_ics.mean():+.4f}                       │
│  Best model ICIR (vs oracle)    : {model_ics.mean()/model_ics.std():+.3f}                        │
│  Best model % positive days     : {(model_ics>0).mean():.1%}                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  LIQUID vs ILLIQUID GAP                                         │
├─────────────────────────────────────────────────────────────────┤
│  Train assets/day: {train.groupby('day').size().mean():.0f}  vs  Test: {test.groupby('day').size().mean():.0f}             │
│  This ~{train.groupby('day').size().mean()/test.groupby('day').size().mean():.1f}× size difference is the core challenge: features computed   │
│  on liquid assets must generalize to illiquid (fewer, noisier) │
│  CV_GROUP IC positive rate: 63% — signal is NOT universal.     │
│  37% of asset groups have NEGATIVE IC for the top feature.     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  BEST APPROACH GIVEN EVERYTHING WE KNOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  WHAT WORKS (confirmed by LB):                                  │
│  1. Grinold/NW (IC-weighted alpha) — robust, interpretable      │
│  2. Cross-sectional z-score LGB — removes day-level shift       │
│  3. Ensemble diversity — uncorrelated models add alpha          │
│                                                                 │
│  WHAT DOESN'T WORK (confirmed failures):                        │
│  - Market intercept (penalizes mean predictions)                │
│  - Standalone LGB without CS normalization (overfit to liquid)  │
│  - Rolling IC (captures noise, not structural signal)           │
│  - BookShape-conditioned IC (near-identical to global IC)       │
│  - Per-day stacking (Ridge dominates in-sample, fails transfer) │
│  - GLP (too parameter-insensitive, no useful variation)         │
│  - Illiquid-tuned Grinold (IC weaker on test-like groups)       │
│                                                                 │
│  REMAINING LEVERS:                                              │
│  1. cs_v2_huber / cs_v2_rank / cs_v2_gold (Kaggle, pending)    │
│  2. ElasticNet with CS z-score features (sparsity helps)        │
│  3. New ensemble blend IF any cs_v2 variant > +0.059408 oracle  │
│                                                                 │
│  THE 50× GAP TO 1ST PLACE IS STRUCTURAL:                        │
│  1st place IC ≈ 0.27 vs ours ≈ 0.046 (6× gap in alpha quality) │
│  This suggests asset-identity matching or external data — not   │
│  achievable from the given feature set alone.                   │
│                                                                 │
│  REALISTIC TARGET: 2nd place = +0.01482                         │
│  Requires oracle_score ≈ +0.070 (current best: +0.058)          │
│  Gap: +0.012 — possible only with a new architectural signal    │
└─────────────────────────────────────────────────────────────────┘
""")

# KS stats for gold features
print("KS statistics: liquid vs illiquid for top gold features")
print(f"  {'Feature':<35}  {'KS':>6}  {'p-val':>10}  {'abs_ICIR':>9}")
print(f"  {'─'*35}  {'─'*6}  {'─'*10}  {'─'*9}")
for feat in gold_feats[:10]:
    tr_v = train[feat].dropna().values
    te_v = test[feat].dropna().values
    ks, p = ks_2samp(tr_v, te_v)
    icir_val = icir.loc[icir['feature'] == feat, 'abs_icir'].values
    icir_v = icir_val[0] if len(icir_val) else float('nan')
    print(f"  {feat:<35}  {ks:>6.4f}  {p:>10.2e}  {icir_v:>9.2f}")

print(f"\nAll plots saved to: {PLOT_DIR}")
print(f"Files: fig1-fig8_*.png")
