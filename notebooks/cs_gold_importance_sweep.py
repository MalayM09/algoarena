# ================================================================
# GOLD FEATURE IMPORTANCE + SWEEP
# ================================================================
# cs_v2_gold stops at iter 5-23 (very shallow). With 51 features
# and only ~10 trees, most features may never be used.
#
# Goal: find the optimal feature count N where:
#   - signal is retained (top N features capture it)
#   - noise is minimised (fewer features = better illiquid transfer)
#
# Steps:
#   1. Train full model on all 51 gold features → extract importances
#   2. Plot feature importance (gain + split)
#   3. Try SHAP if available, else LGB built-in
#   4. Sweep N = [5, 8, 10, 12, 15, 20, 30, 51] gold features
#   5. For each N: train CS-LGB, compute oracle_score
#   6. Plot oracle_score vs N → find optimal cutoff
#   7. Build best submission
# ================================================================

import os, gc, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
PLOT_DIR   = os.path.join(BASE_DIR, 'outputs/eda/plots')

TARGET_STD = 0.000948
N_EST, LR, ES = 2000, 0.05, 50
t0 = time.time()

def auto_scale(p):
    s = p.std(); return p*(TARGET_STD/s) if s>1e-10 else p

def daywise_oracle_score(pred_vec, oracle_vec, day_ids):
    ics = []
    for day in np.unique(day_ids):
        m = day_ids==day
        if m.sum()<3: continue
        p=pred_vec[m]-pred_vec[m].mean(); o=oracle_vec[m]-oracle_vec[m].mean()
        pn=np.linalg.norm(p); on=np.linalg.norm(o)
        if pn<1e-12 or on<1e-12: ics.append(0.)
        else: ics.append(float((p@o)/(pn*on)))
    return float(np.mean(ics))

# ── Load ──────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
icir  = pd.read_csv(ICIR_PATH)

gold_mask  = (icir['abs_icir']>=3) & (icir['ic_pos_frac'].isin([0.,1.]))
all_feats  = [c for c in train.columns if c not in {'ID','TARGET','CV_GROUP'}]
gold_feats = [f for f in icir[gold_mask].sort_values('abs_icir',ascending=False)['feature']
              if f in all_feats]
print(f"  Gold features: {len(gold_feats)}")

y_raw     = train['TARGET'].values.astype(np.float64)
lo,hi     = np.percentile(y_raw,1), np.percentile(y_raw,99)
y_wins    = np.clip(y_raw, lo, hi)
test_ids  = test['ID'].values
train_day = train['SO3_T'].round(5).astype(str).values
test_day  = test['SO3_T'].round(5).astype(str).values

sample_sub  = pd.read_csv(SAMPLE_SUB)[['ID']]
oracle_vec  = sample_sub.merge(pd.read_csv(ORACLE),on='ID',how='left').fillna(0)['TARGET'].values
oracle_days = sample_sub.merge(
    pd.read_parquet(TEST_PATH,columns=['ID','SO3_T']),on='ID',how='left'
)['SO3_T'].round(5).astype(str).values

# GroupKFold
so3t_vals = train['SO3_T'].values
groups    = pd.qcut(pd.Series(so3t_vals),q=5,labels=False,duplicates='drop').values.astype(np.int32)
n_folds   = len(np.unique(groups))
gkf       = GroupKFold(n_splits=n_folds)

LGB_PARAMS = dict(objective='regression',metric='rmse',num_leaves=63,
                  learning_rate=LR,feature_fraction=0.8,bagging_fraction=0.8,
                  bagging_freq=1,min_child_samples=50,lambda_l1=0.1,lambda_l2=1.0,
                  n_jobs=-1,verbose=-1,seed=42)

# ── Helper: train N-feature CS-LGB ───────────────────────────────
def train_cs_lgb(feats, label):
    tr_raw = train[feats].fillna(0).values.astype(np.float32)
    te_raw = test.reindex(columns=feats,fill_value=0).values.astype(np.float32)
    X_tr   = np.zeros_like(tr_raw)
    X_te   = np.zeros_like(te_raw)
    for d in np.unique(train_day):
        m=train_day==d; x=tr_raw[m]; s=x.std(0); s[s<1e-8]=1.
        X_tr[m]=(x-x.mean(0))/s
    for d in np.unique(test_day):
        m=test_day==d; x=te_raw[m]; s=x.std(0); s[s<1e-8]=1.
        X_te[m]=(x-x.mean(0))/s
    del tr_raw, te_raw; gc.collect()

    folds    = list(gkf.split(X_tr, y_wins, groups=groups))
    oof      = np.zeros(len(y_wins))
    te_preds = np.zeros(len(X_te))
    best_iters=[]; all_models=[]

    for fi,(tri,vai) in enumerate(folds):
        dt = lgb.Dataset(X_tr[tri],label=y_wins[tri],free_raw_data=True)
        dv = lgb.Dataset(X_tr[vai],label=y_wins[vai],reference=dt,free_raw_data=True)
        m  = lgb.train(LGB_PARAMS,dt,num_boost_round=N_EST,valid_sets=[dv],
                       callbacks=[lgb.early_stopping(ES,verbose=False),
                                  lgb.log_evaluation(500)])
        bi=m.best_iteration; best_iters.append(bi)
        oof[vai]  = m.predict(X_tr[vai],num_iteration=bi)
        te_preds += m.predict(X_te,       num_iteration=bi)/n_folds
        all_models.append(m)
        del dt,dv; gc.collect()

    oof_r2 = r2_score(y_wins, oof)
    scaled  = auto_scale(te_preds)
    sub     = sample_sub.merge(
        pd.DataFrame({'ID':test_ids,'TARGET':scaled}),on='ID',how='left').fillna(0)
    sc      = daywise_oracle_score(sub['TARGET'].values, oracle_vec, oracle_days)
    print(f"  [{label}] n_feats={len(feats):2d}  best_iters={best_iters}"
          f"  OOF_R²={oof_r2:+.6f}  oracle={sc:+.6f}")
    return sub, sc, oof_r2, all_models

# ── STEP 1: Train on all 51 gold → extract importance ─────────────
print("\n" + "="*60)
print("STEP 1: Train on 51 gold features → feature importances")
print("="*60)
sub51, score51, oof51, models51 = train_cs_lgb(gold_feats, 'all_51_gold')

# Aggregate importance across folds
imp_gain  = np.zeros(len(gold_feats))
imp_split = np.zeros(len(gold_feats))
for m in models51:
    imp_gain  += m.feature_importance(importance_type='gain')
    imp_split += m.feature_importance(importance_type='split')
imp_gain  /= n_folds
imp_split /= n_folds

imp_df = pd.DataFrame({
    'feature':  gold_feats,
    'gain':     imp_gain,
    'split':    imp_split,
    'abs_icir': [icir.loc[icir['feature']==f,'abs_icir'].values[0]
                 if len(icir.loc[icir['feature']==f])>0 else 0. for f in gold_feats]
}).sort_values('gain', ascending=False).reset_index(drop=True)

print("\n  Top-20 features by LGB gain importance:")
print(f"  {'Rank':<5} {'Feature':<42} {'Gain':>8} {'Split':>7} {'ICIR':>7}")
print(f"  {'─'*5} {'─'*42} {'─'*8} {'─'*7} {'─'*7}")
for i, row in imp_df.head(20).iterrows():
    used = '●' if row['gain'] > 0 else '○'
    print(f"  {i+1:<5} {row['feature']:<42} {row['gain']:>8.1f} "
          f"{row['split']:>7.0f} {row['abs_icir']:>7.2f}  {used}")

n_used = (imp_df['gain'] > 0).sum()
print(f"\n  Features actually used (gain>0): {n_used} / {len(gold_feats)}")
print(f"  Features with zero gain:          {len(gold_feats)-n_used} / {len(gold_feats)}")

# Save importance
imp_df.to_csv(os.path.join(BASE_DIR,'outputs/eda/summaries/cs_gold_importance.csv'), index=False)

# ── STEP 2: Try SHAP if available ────────────────────────────────
shap_available = False
try:
    import shap
    shap_available = True
    print("\n  SHAP available — computing SHAP values on sample...")
    # Use fold-1 model + sample of training data
    feat_names = gold_feats
    tr_sample  = train[feat_names].fillna(0).values.astype(np.float32)[:2000]
    # CS z-score sample
    for d in np.unique(train_day[:2000]):
        m = train_day[:2000]==d
        x=tr_sample[m]; s=x.std(0); s[s<1e-8]=1.
        tr_sample[m]=(x-x.mean(0))/s
    explainer   = shap.TreeExplainer(models51[0])
    shap_values = explainer.shap_values(tr_sample)
    shap_imp    = np.abs(shap_values).mean(0)
    shap_df = pd.DataFrame({'feature':feat_names,'shap_importance':shap_imp})\
                .sort_values('shap_importance',ascending=False)
    print("  Top-10 by SHAP:")
    for _, r in shap_df.head(10).iterrows():
        print(f"    {r['feature']:<42}  shap={r['shap_importance']:.4f}")
    shap_df.to_csv(os.path.join(BASE_DIR,'outputs/eda/summaries/cs_gold_shap.csv'),index=False)
except ImportError:
    print("\n  SHAP not installed — using LGB gain importance only")

# ── STEP 3: Feature count sweep ───────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Feature count sweep")
print("="*60)

# Use features ranked by gain (only features with gain>0)
used_feats = imp_df[imp_df['gain']>0]['feature'].tolist()
N_SWEEP    = sorted(set([3, 5, 8, 10, 12, 15, 20, 30, len(used_feats), len(gold_feats)]))
N_SWEEP    = [n for n in N_SWEEP if n <= len(gold_feats)]

sweep_results = []
best_sub, best_score, best_n = sub51, score51, len(gold_feats)

for n in N_SWEEP:
    feats_n = imp_df['feature'].tolist()[:n]  # top-n by gain
    sub_n, sc_n, oof_n, _ = train_cs_lgb(feats_n, f'top{n}')
    sweep_results.append({'n': n, 'oracle': sc_n, 'oof_r2': oof_n})
    if sc_n > best_score:
        best_score, best_n = sc_n, n
        best_sub = sub_n

# ── STEP 4: Plot oracle_score vs N ───────────────────────────────
ns      = [r['n']     for r in sweep_results]
oracles = [r['oracle'] for r in sweep_results]
oof_r2s = [r['oof_r2'] for r in sweep_results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CS-LGB Gold: Feature Count vs Oracle Score\n'
             'Finding the sweet spot: fewer features = less overfit', fontsize=12)

ax = axes[0]
ax.plot(ns, oracles, 'o-', color='#2196F3', linewidth=2, markersize=8, label='oracle_score')
ax.axhline(0.058349, color='#FF5722', linestyle='--', linewidth=1.5, label='current best (0.058349)')
ax.axhline(0.060349, color='#4CAF50', linestyle='--', linewidth=1.5, label='submit threshold (0.060349)')
ax.axvline(best_n, color='purple', linestyle=':', linewidth=1.5, label=f'best N={best_n}')
ax.scatter([best_n], [best_score], s=150, color='purple', zorder=5)
for n, sc in zip(ns, oracles):
    ax.annotate(f'{sc:.4f}', (n, sc), textcoords='offset points',
                xytext=(0,8), ha='center', fontsize=7)
ax.set_xlabel('Number of features (top-N by LGB gain importance)')
ax.set_ylabel('Oracle score (day-wise CS Pearson vs oracle)')
ax.set_title('Oracle Score vs Feature Count')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.bar(range(min(25, len(imp_df))),
       imp_df['gain'].values[:25], color='#2196F3', alpha=0.7)
ax.set_xticks(range(min(25, len(imp_df))))
ax.set_xticklabels([f[:18] for f in imp_df['feature'].values[:25]],
                   rotation=45, ha='right', fontsize=7)
ax.set_ylabel('LGB Gain Importance (avg across folds)')
ax.set_title(f'Top-25 Feature Importances\n({n_used}/{len(gold_feats)} features used by model)')
ax.grid(axis='y', alpha=0.3)

# Shade unused features
zero_start = n_used if n_used < 25 else 25
if zero_start < 25:
    ax.axvspan(zero_start-0.5, 24.5, alpha=0.1, color='red', label='zero importance')
    ax.legend(fontsize=8)

plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, 'figD_gold_feature_importance_sweep.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {plot_path}")

# ── STEP 5: Summary + save best ──────────────────────────────────
print("\n" + "="*60)
print("SWEEP SUMMARY")
print("="*60)
print(f"\n  {'N':>4} {'oracle':>14} {'OOF_R²':>10}  note")
print(f"  {'─'*4} {'─'*14} {'─'*10}")
for r in sweep_results:
    flag = ' ← BEST' if r['n']==best_n else ''
    flag = ' ← BEATS THRESHOLD' if r['oracle']>0.060349 else flag
    print(f"  {r['n']:>4} {r['oracle']:>+14.6f} {r['oof_r2']:>+10.6f}{flag}")

best_name = f'cs_gold_top{best_n}.csv'
best_sub.to_csv(os.path.join(OUT_DIR, best_name), index=False)
print(f"\n  Best standalone: {best_name}  oracle={best_score:+.6f}")

# Build blend: best_n variant + optimal_blend_v2
print("\n  Blend: cs_gold_top{N} + optimal_blend_v2 (fine grid):")
anchor_v2 = sample_sub.merge(
    pd.read_csv(os.path.join(OUT_DIR,'optimal_blend_v2.csv')),
    on='ID',how='left').fillna(0)['TARGET'].values
best_vec = best_sub['TARGET'].values
blend_best_score, blend_best_w = -np.inf, None
for w in np.arange(0.05, 0.60, 0.01):
    blend = w*best_vec + (1-w)*anchor_v2
    s=blend.std(); blend=blend*(TARGET_STD/s) if s>1e-10 else blend
    sc = daywise_oracle_score(blend, oracle_vec, oracle_days)
    if sc > blend_best_score:
        blend_best_score, blend_best_w = sc, w
    if abs(w - round(w*10)/10) < 0.005:
        flag = '  ← BEATS THRESHOLD' if sc>0.060349 else ''
        print(f"    w_gold={w:.0%}  oracle={sc:+.6f}{flag}")

print(f"\n  Best blend: w_gold={blend_best_w:.2f}  oracle={blend_best_score:+.6f}")
print(f"  Submit threshold:    +0.060349")
print(f"  Gap:                 {blend_best_score-0.060349:+.6f}")

if blend_best_score > 0.060349:
    fname = f'cs_gold_top{best_n}_blend.csv'
    blend_s = blend_best_w*best_vec + (1-blend_best_w)*anchor_v2
    blend_s = blend_s*(TARGET_STD/blend_s.std())
    pd.DataFrame({'ID':sample_sub['ID'].values,'TARGET':blend_s})\
      .to_csv(os.path.join(OUT_DIR, fname), index=False)
    print(f"  SAVED (beats threshold): {fname}")

print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
print(f"\nKey chart: {plot_path}")
