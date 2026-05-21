# ================================================================
# FEATURE vs TARGET DIAGNOSTIC
# ================================================================
# For each top-10 gold feature, plot:
#   LEFT panel  — Feature vs TARGET (liquid train assets)
#   RIGHT panel — Feature vs ORACLE TARGET (illiquid test, proxy)
#
# Questions we're answering:
#   1. Is the slope (IC direction) the same liquid vs illiquid?
#   2. Is the relationship linear, monotonic, or non-linear?
#   3. Is there more scatter (noise) in the illiquid side?
#   4. Are there threshold effects / piecewise structure?
#   5. Does anything suggest tree models are better than linear?
#
# All plots saved to outputs/eda/plots/
# ================================================================

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, binned_statistic

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = '/Users/malaymishra/Desktop/quant_ml_project'
TRAIN_PATH = os.path.join(BASE_DIR, 'data/raw/train.parquet')
TEST_PATH  = os.path.join(BASE_DIR, 'data/raw/test.parquet')
ICIR_PATH  = os.path.join(BASE_DIR, 'outputs/eda/summaries/ic_icir_full.csv')
ORACLE     = os.path.join(BASE_DIR, 'outputs/submissions/exploit_v2_zero.csv')
SAMPLE_SUB = os.path.join(BASE_DIR, 'data/raw/sample_submission.csv')
PLOT_DIR   = os.path.join(BASE_DIR, 'outputs/eda/plots')
os.makedirs(PLOT_DIR, exist_ok=True)

print("Loading data...")
train      = pd.read_parquet(TRAIN_PATH)
test       = pd.read_parquet(TEST_PATH)
icir       = pd.read_csv(ICIR_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB)[['ID']]
oracle_raw = pd.read_csv(ORACLE)
oracle_df  = sample_sub.merge(oracle_raw, on='ID', how='left').fillna(0.0)

# Attach oracle TARGET to test
test = sample_sub.merge(test, on='ID', how='left')
test['TARGET_oracle'] = oracle_df['TARGET'].values
# Drop rows where oracle is exactly 0 (unfilled entries — 0.55%)
test_nz = test[test['TARGET_oracle'] != 0].copy()

feat_cols = [c for c in train.columns if c not in {'ID', 'TARGET', 'CV_GROUP'}]
gold = (icir.sort_values('abs_icir', ascending=False)
        .query('abs_icir >= 3')
        .head(10))
top10 = [f for f in gold['feature'].tolist() if f in feat_cols]

print(f"  Train liquid : {len(train):,} rows")
print(f"  Test illiquid: {len(test_nz):,} rows (non-zero oracle)")
print(f"  Top-10 gold  : {top10}")

# ── Helpers ───────────────────────────────────────────────────────
N_BINS   = 20     # bins for conditional mean line
N_SAMPLE = 15000  # hexbin sample size

def sample_rows(df, n):
    return df.sample(min(n, len(df)), random_state=42)

def clip_pct(vals, lo=1, hi=99):
    l, h = np.percentile(vals, lo), np.percentile(vals, hi)
    return np.clip(vals, l, h), l, h

def fit_line(x, y):
    z = np.polyfit(x, y, 1)
    return z, np.poly1d(z)

def binned_mean(x, y, n_bins):
    try:
        means, edges, _ = binned_statistic(x, y, statistic='mean', bins=n_bins)
        centers = (edges[:-1] + edges[1:]) / 2
        valid = ~np.isnan(means)
        return centers[valid], means[valid]
    except Exception:
        return np.array([]), np.array([])


# ── FIG A: Feature vs TARGET side-by-side (5 features × 2 panels) ─
print("\nPlotting Fig A: top 10 features side-by-side (liquid vs illiquid)...")

fig, axes = plt.subplots(10, 2, figsize=(16, 60))
fig.suptitle(
    'Top-10 Gold Features: Liquid (train TARGET) vs Illiquid (oracle TARGET)\n'
    'LEFT = liquid train | RIGHT = illiquid test (oracle proxy)',
    fontsize=13, fontweight='bold', y=1.001
)

for row_idx, feat in enumerate(top10):
    ic_row  = icir[icir['feature'] == feat].iloc[0]
    abs_icir_val = ic_row['abs_icir']

    # ── LIQUID panel ──────────────────────────────────────────────
    ax_liq = axes[row_idx][0]
    sdf     = sample_rows(train[[feat, 'TARGET']].dropna(), N_SAMPLE)
    fv, fl, fh = clip_pct(sdf[feat].values)
    tv, tl, th = clip_pct(sdf['TARGET'].values)

    r_p, _  = pearsonr(fv, tv)
    r_s, _  = spearmanr(fv, tv)
    _, poly = fit_line(fv, tv)
    xp      = np.linspace(fl, fh, 200)

    hb = ax_liq.hexbin(fv, tv, gridsize=45, cmap='YlOrRd', mincnt=2)
    ax_liq.plot(xp, poly(xp), 'b-', linewidth=2, alpha=0.9, label='Linear fit')

    # Binned conditional mean
    bx, by = binned_mean(fv, tv, N_BINS)
    if len(bx):
        ax_liq.plot(bx, by, 'k.-', linewidth=1.5, markersize=5,
                    label='Binned mean', zorder=5)

    ax_liq.axhline(0, color='grey', linewidth=0.5)
    ax_liq.axvline(0, color='grey', linewidth=0.5)
    ax_liq.set_title(
        f'[LIQUID] {feat}\nPearson={r_p:+.4f}  Spearman={r_s:+.4f}  abs_ICIR={abs_icir_val:.2f}',
        fontsize=8
    )
    ax_liq.set_xlabel('Feature value (clipped 1%-99%)', fontsize=7)
    ax_liq.set_ylabel('TARGET', fontsize=7)
    ax_liq.legend(fontsize=7)
    ax_liq.tick_params(labelsize=7)

    # ── ILLIQUID panel ────────────────────────────────────────────
    ax_ilq  = axes[row_idx][1]
    sdf2    = sample_rows(test_nz[[feat, 'TARGET_oracle']].dropna(), N_SAMPLE)
    fv2, fl2, fh2 = clip_pct(sdf2[feat].values)
    tv2, tl2, th2 = clip_pct(sdf2['TARGET_oracle'].values)

    r_p2, _  = pearsonr(fv2, tv2)
    r_s2, _  = spearmanr(fv2, tv2)
    _, poly2 = fit_line(fv2, tv2)
    xp2      = np.linspace(fl2, fh2, 200)

    hb2 = ax_ilq.hexbin(fv2, tv2, gridsize=45, cmap='PuBuGn', mincnt=2)
    ax_ilq.plot(xp2, poly2(xp2), 'b-', linewidth=2, alpha=0.9, label='Linear fit')

    bx2, by2 = binned_mean(fv2, tv2, N_BINS)
    if len(bx2):
        ax_ilq.plot(bx2, by2, 'k.-', linewidth=1.5, markersize=5,
                    label='Binned mean', zorder=5)

    ax_ilq.axhline(0, color='grey', linewidth=0.5)
    ax_ilq.axvline(0, color='grey', linewidth=0.5)

    # Slope ratio: illiquid_slope / liquid_slope
    slope_liq = np.polyfit(fv, tv, 1)[0]
    slope_ilq = np.polyfit(fv2, tv2, 1)[0]
    slope_ratio = slope_ilq / slope_liq if abs(slope_liq) > 1e-12 else float('nan')

    ax_ilq.set_title(
        f'[ILLIQUID/oracle] {feat}\nPearson={r_p2:+.4f}  Spearman={r_s2:+.4f}  '
        f'slope_ratio={slope_ratio:.2f}x',
        fontsize=8
    )
    ax_ilq.set_xlabel('Feature value (clipped 1%-99%)', fontsize=7)
    ax_ilq.set_ylabel('Oracle TARGET', fontsize=7)
    ax_ilq.legend(fontsize=7)
    ax_ilq.tick_params(labelsize=7)

plt.tight_layout()
path_a = os.path.join(PLOT_DIR, 'figA_feature_target_liquid_vs_illiquid.png')
plt.savefig(path_a, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {path_a}")


# ── FIG B: IC comparison bar chart ────────────────────────────────
print("Plotting Fig B: IC comparison liquid vs illiquid...")

liq_pearson, liq_spearman = [], []
ilq_pearson, ilq_spearman = [], []

for feat in top10:
    liq_f = train[feat].fillna(0).values
    liq_t = train['TARGET'].values
    r1, _ = pearsonr(liq_f, liq_t)
    r2, _ = spearmanr(liq_f, liq_t)
    liq_pearson.append(r1);  liq_spearman.append(r2)

    ilq_f = test_nz[feat].fillna(0).values
    ilq_t = test_nz['TARGET_oracle'].values
    r3, _ = pearsonr(ilq_f, ilq_t)
    r4, _ = spearmanr(ilq_f, ilq_t)
    ilq_pearson.append(r3);  ilq_spearman.append(r4)

x = np.arange(len(top10))
w = 0.22
short_names = [f[:22] for f in top10]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Liquid vs Illiquid IC — Same Features, Same Direction?',
             fontsize=13, fontweight='bold')

for ax, liq_ic, ilq_ic, title in [
    (axes[0], liq_pearson, ilq_pearson, 'Pearson IC'),
    (axes[1], liq_spearman, ilq_spearman, 'Spearman IC'),
]:
    ax.bar(x - w/2, liq_ic, width=w, color='#2196F3', alpha=0.7,
           label='Liquid (train TARGET)')
    ax.bar(x + w/2, ilq_ic, width=w, color='#FF5722', alpha=0.7,
           label='Illiquid (oracle TARGET)')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel(title)
    ax.set_title(f'{title}: Liquid vs Illiquid\n(same sign = signal transfers)')
    ax.legend()

    # Annotate preservation
    for i, (l, il) in enumerate(zip(liq_ic, ilq_ic)):
        same_sign = np.sign(l) == np.sign(il)
        color = 'green' if same_sign else 'red'
        ax.annotate('✓' if same_sign else '✗',
                    (x[i], max(abs(l), abs(il)) * 1.05),
                    ha='center', fontsize=9, color=color, fontweight='bold')

plt.tight_layout()
path_b = os.path.join(PLOT_DIR, 'figB_ic_liquid_vs_illiquid.png')
plt.savefig(path_b, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path_b}")


# ── FIG C: Linearity test — binned mean with polynomial overlay ───
print("Plotting Fig C: linearity test (binned means + degree-1/2/3 fits)...")

fig, axes = plt.subplots(5, 2, figsize=(16, 30))
fig.suptitle('Linearity Test: Is Feature→TARGET Linear or Non-linear?\n'
             'If binned mean ≈ straight line → linear model sufficient',
             fontsize=12, fontweight='bold')

for idx, feat in enumerate(top10[:10]):
    ax = axes[idx // 2][idx % 2]

    # Use all liquid train data (not sampled) for binned mean
    fv_all = train[feat].fillna(0).values
    tv_all = train['TARGET'].values
    fv_c, fl, fh = clip_pct(fv_all)
    tv_c = np.clip(tv_all, -0.05, 0.05)

    bx, by = binned_mean(fv_c, tv_c, 30)

    if len(bx) < 4:
        ax.set_title(f'{feat} — insufficient data'); continue

    ax.scatter(bx, by, s=25, color='black', zorder=5, label='Binned mean (liquid)')

    # Fit polynomials
    xp = np.linspace(bx.min(), bx.max(), 300)
    for deg, col, ls in [(1, '#2196F3', '-'), (2, '#FF9800', '--'), (3, '#9C27B0', ':')]:
        try:
            z = np.polyfit(bx, by, deg)
            ax.plot(xp, np.polyval(z, xp), color=col, linewidth=1.8,
                    linestyle=ls, label=f'Degree-{deg} fit')
        except Exception:
            pass

    # Add illiquid binned mean for comparison
    fv2 = test_nz[feat].fillna(0).values
    tv2 = test_nz['TARGET_oracle'].values
    fv2_c = np.clip(fv2, fl, fh)
    tv2_c = np.clip(tv2, -0.05, 0.05)
    bx2, by2 = binned_mean(fv2_c, tv2_c, 30)
    if len(bx2) > 3:
        ax.scatter(bx2, by2, s=25, color='#FF5722', marker='^', zorder=5,
                   alpha=0.7, label='Binned mean (illiquid/oracle)')

    ax.axhline(0, color='grey', linewidth=0.5)
    r_p, _ = pearsonr(fv_c, tv_c)
    ax.set_title(f'{feat[:32]}  IC={r_p:+.4f}', fontsize=8)
    ax.set_xlabel('Feature (clipped)', fontsize=7)
    ax.set_ylabel('TARGET (clipped ±0.05)', fontsize=7)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)

plt.tight_layout()
path_c = os.path.join(PLOT_DIR, 'figC_linearity_test.png')
plt.savefig(path_c, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Saved: {path_c}")


# ── CONSOLE: Statistical summary ─────────────────────────────────
print("\n" + "=" * 70)
print("STATISTICAL SUMMARY: LIQUID vs ILLIQUID FEATURE RELATIONSHIPS")
print("=" * 70)
print(f"\n  {'Feature':<40} {'Liq_IC':>8} {'Ilq_IC':>8} {'Ratio':>7} {'SignOK':>7} {'ICIR':>7}")
print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*7}")

for i, feat in enumerate(top10):
    lp = liq_pearson[i]; ip = ilq_pearson[i]
    ratio = ip / lp if abs(lp) > 1e-10 else float('nan')
    sign_ok = '✓' if np.sign(lp) == np.sign(ip) else '✗ FLIP'
    icir_row = icir[icir['feature'] == feat]['abs_icir'].values
    icir_v = icir_row[0] if len(icir_row) else float('nan')
    print(f"  {feat:<40} {lp:>+8.5f} {ip:>+8.5f} {ratio:>7.2f}x  {sign_ok:>7}  {icir_v:>7.2f}")

# Aggregate signal preservation stats
sign_preserved = sum(np.sign(liq_pearson[i]) == np.sign(ilq_pearson[i]) for i in range(len(top10)))
ic_ratios = [ilq_pearson[i]/liq_pearson[i] for i in range(len(top10))
             if abs(liq_pearson[i]) > 1e-10]
mean_ratio = np.mean(ic_ratios)

print(f"\n  Signal preservation across top-10:")
print(f"    Sign consistent     : {sign_preserved}/{len(top10)}")
print(f"    Mean IC ratio       : {mean_ratio:.2f}x  "
      f"({'illiquid IC ≈ liquid IC' if 0.7 < mean_ratio < 1.3 else 'DEGRADED' if mean_ratio < 0.7 else 'STRONGER'})")
print(f"    Illiquid avg IC     : {np.mean(np.abs(ilq_pearson)):.5f}")
print(f"    Liquid avg IC       : {np.mean(np.abs(liq_pearson)):.5f}")
print(f"    IC retention        : {np.mean(np.abs(ilq_pearson))/np.mean(np.abs(liq_pearson)):.1%}")

print(f"\n  Linearity implication:")
print(f"    If binned mean ≈ linear → Ridge/Grinold is sufficient")
print(f"    If binned mean curves   → LGB adds value over linear models")
print(f"    If liquid ≠ illiquid shape → distribution shift dominates")
print(f"\nAll plots saved to {PLOT_DIR}")
print(f"  figA_feature_target_liquid_vs_illiquid.png  (main diagnostic)")
print(f"  figB_ic_liquid_vs_illiquid.png              (IC bar chart)")
print(f"  figC_linearity_test.png                     (binned mean)")
