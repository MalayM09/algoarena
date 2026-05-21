"""
Adversarial Validation — notebooks/01b_adversarial_validation.ipynb equivalent
Identifies features driving train-test distinguishability.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
import gc
import warnings
warnings.filterwarnings('ignore')

os.chdir('/Users/malaymishra/Desktop/quant_ml_project')
os.makedirs('outputs/eda/summaries', exist_ok=True)
os.makedirs('outputs/eda/plots', exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("ADVERSARIAL VALIDATION")
print("=" * 60)

# ── Cell 1: Setup ─────────────────────────────────────────────────────
print("\n--- Cell 1: Setup ---")

with open('outputs/eda/summaries/eda_decisions.pkl', 'rb') as f:
    decisions = pickle.load(f)

raw_files  = os.listdir('data/raw/')
train_file = [f for f in raw_files if 'train' in f.lower() and f.endswith('.parquet')][0]
test_file  = [f for f in raw_files if 'test'  in f.lower() and f.endswith('.parquet')][0]
print(f"Train: {train_file}, Test: {test_file}")

train = pd.read_parquet(f'data/raw/{train_file}')
test  = pd.read_parquet(f'data/raw/{test_file}')

feature_cols = [c for c in train.columns if c not in ['ID', 'TARGET']]
for col in feature_cols:
    train[col] = train[col].astype(np.float32)
for col in [c for c in test.columns if c != 'ID']:
    test[col] = test[col].astype(np.float32)

print(f"Train: {train.shape}, Test: {test.shape}")
gc.collect()

# ── Cell 2: Build adversarial dataset ─────────────────────────────────
print("\n--- Cell 2: Build adversarial dataset ---")

train_adv = train[feature_cols].copy()
test_adv  = test[[c for c in feature_cols if c in test.columns]].copy()

train_adv = train_adv.fillna(0)
test_adv  = test_adv.fillna(0)

common_cols = [c for c in feature_cols if c in test_adv.columns]
train_adv = train_adv[common_cols]
test_adv  = test_adv[common_cols]

n_test = len(test_adv)
train_sample = train_adv.sample(n=min(n_test, len(train_adv)), random_state=RANDOM_SEED)

X_adv = pd.concat([train_sample, test_adv], axis=0, ignore_index=True)
y_adv = np.array([0] * len(train_sample) + [1] * len(test_adv))

print(f"Adversarial dataset: {X_adv.shape}")
print(f"Class balance: train={len(train_sample)}, test={len(test_adv)}")

# ── Cell 3: Train adversarial LightGBM classifier ─────────────────────
print("\n--- Cell 3: Train adversarial LightGBM (5-fold CV) ---")

lgb_params = {
    'objective'        : 'binary',
    'metric'           : 'auc',
    'n_estimators'     : 300,
    'learning_rate'    : 0.05,
    'num_leaves'       : 63,
    'feature_fraction' : 0.7,
    'subsample'        : 0.8,
    'min_child_samples': 50,
    'random_state'     : RANDOM_SEED,
    'n_jobs'           : -1,
    'verbose'          : -1,
}

skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
oof    = np.zeros(len(X_adv))
models = []

print("Training adversarial classifier (5-fold CV)...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_adv, y_adv)):
    X_tr, X_val = X_adv.iloc[tr_idx], X_adv.iloc[val_idx]
    y_tr, y_val = y_adv[tr_idx],      y_adv[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    try:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(period=-1)]
        )
    except TypeError:
        # Fallback for older LightGBM API
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=30,
                  verbose=-1)

    oof[val_idx] = model.predict_proba(X_val)[:, 1]
    models.append(model)
    fold_auc = roc_auc_score(y_val, oof[val_idx])
    print(f"  Fold {fold+1}: AUC = {fold_auc:.4f}")

overall_auc = roc_auc_score(y_adv, oof)
print(f"\nOverall OOF AUC-ROC: {overall_auc:.4f}")

if overall_auc > 0.55:
    print(f"WARNING: AUC={overall_auc:.4f} > 0.55")
    print("Train and test are distinguishable. Adversarial drop recommended.")
elif overall_auc > 0.52:
    print(f"MODERATE: AUC={overall_auc:.4f}. Some distinguishability. Review top features.")
else:
    print(f"GOOD: AUC={overall_auc:.4f} near 0.5. Datasets are not easily distinguishable.")

# ── Cell 4: Extract feature importances ───────────────────────────────
print("\n--- Cell 4: Feature importances & drop candidates ---")

importances = np.zeros(len(common_cols))
for model in models:
    importances += model.feature_importances_
importances /= len(models)

imp_df = pd.DataFrame({
    'feature'        : common_cols,
    'adv_importance' : importances
}).sort_values('adv_importance', ascending=False)

print(f"\nTop 30 adversarially-important features:")
print(imp_df.head(30).to_string())

# Cross-reference with KS stats
ks_df = pd.read_csv('outputs/eda/summaries/ks_train_test.csv')
imp_df = imp_df.merge(
    ks_df[['feature', 'ks_stat', 'ks_p']],
    on='feature', how='left'
)

imp_df['adv_rank'] = imp_df['adv_importance'].rank(ascending=False)
adversarial_drop_candidates = imp_df[
    (imp_df['adv_rank'] <= 20) &
    (imp_df['ks_stat'] >= 0.3)
]['feature'].tolist()

print(f"\nAdversarial drop candidates (top-20 adv importance AND ks_stat>=0.3):")
print(f"  Count: {len(adversarial_drop_candidates)}")
print(f"  Features: {adversarial_drop_candidates}")

with open('outputs/eda/summaries/adversarial_drop_list.pkl', 'wb') as f:
    pickle.dump(adversarial_drop_candidates, f)
print("SAVED: outputs/eda/summaries/adversarial_drop_list.pkl")

# Save full importance table too
imp_df.to_csv('outputs/eda/summaries/adversarial_importances.csv', index=False)
print("SAVED: outputs/eda/summaries/adversarial_importances.csv")

# ── Cell 5: Visualise ─────────────────────────────────────────────────
print("\n--- Cell 5: Visualise ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

top30 = imp_df.head(30)
axes[0].barh(range(30), top30['adv_importance'].values[::-1], color='coral')
axes[0].set_yticks(range(30))
axes[0].set_yticklabels(top30['feature'].values[::-1], fontsize=7)
axes[0].set_title(f'Top 30 Adversarial Feature Importances\n(AUC={overall_auc:.4f})')
axes[0].set_xlabel('Mean importance across 5 folds')

axes[1].hist(imp_df['adv_importance'], bins=50, color='steelblue', alpha=0.7)
top20_thresh = imp_df[imp_df['adv_rank'] <= 20]['adv_importance'].min()
axes[1].axvline(top20_thresh, color='red', lw=2, linestyle='--', label='Top-20 threshold')
axes[1].set_title('Distribution of Adversarial Importances')
axes[1].set_xlabel('Importance')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/eda/plots/14_adversarial_validation.png', dpi=150)
plt.close()
print("SAVED: outputs/eda/plots/14_adversarial_validation.png")

# ── Cell 6: Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ADVERSARIAL VALIDATION COMPLETE")
print("=" * 60)
print(f"  Overall AUC-ROC      : {overall_auc:.4f}")
print(f"  Features evaluated   : {len(common_cols)}")
print(f"  Drop candidates      : {len(adversarial_drop_candidates)}")
print(f"  Drop criteria used   : adv_rank<=20 AND ks_stat>=0.3")
print()
print("INTERPRETATION:")
if overall_auc > 0.70:
    print("  SEVERE drift. The model will struggle to generalise.")
    print("  Consider increasing adversarial drop threshold.")
elif overall_auc > 0.55:
    print("  SIGNIFICANT drift. Adversarial drops are important.")
    print("  Quantile normalisation in feature engineering is mandatory.")
elif overall_auc > 0.52:
    print("  MODERATE drift. Drops are a precaution, not an emergency.")
else:
    print("  LOW drift. Datasets are reasonably similar after sampling.")

print()
print("NEXT STEPS:")
print("  1. adversarial_drop_list.pkl saved to outputs/eda/summaries/")
print("  2. FEATURE_ENGINEERING_INSTRUCTIONS.md loads this in STEP 0")
print("  3. Proceeding to: notebooks/02_feature_engineering.ipynb")
print("=" * 60)

gc.collect()
