import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from datetime import datetime
import pickle
from time import time

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score
)

warnings.filterwarnings('ignore')

# ============================================================
# STEP C: MODEL TRAINING & EVALUATION (BASELINE MODELS)
# ============================================================
# Temporal split + baseline models (Logistic Regression & Random Forest)
# OPTIMIZED VERSION with progress indicators
# ============================================================

print("=" * 80)
print("FOREST FIRE OCCURRENCE PREDICTION MODEL (FFOPM)")
print("Step C: Model Training & Evaluation - Baseline Models (OPTIMIZED)")
print("=" * 80)

# ------------------------------
# Configuration
# ------------------------------
DATA_DIR = Path("/Users/prabhatrawal/Minor_project_code/data/integrated_data")
EDA_DIR = DATA_DIR / "eda_results"
OUTPUT_DIR = DATA_DIR / "model_results"
OUTPUT_DIR.mkdir(exist_ok=True)

MASTER_FILE = DATA_DIR / "Master_FFOPM_Table.parquet"

# CRITICAL: Use the NO LEAKAGE feature list
FEATURES_FILE = EDA_DIR / "final_features_for_ml_NO_LEAKAGE.json"

# Check if clean features exist, otherwise warn
if not FEATURES_FILE.exists():
    print("\n" + "=" * 80)
    print("⚠️  WARNING: Clean feature list not found!")
    print("=" * 80)
    print("Run the target leakage fix script first:")
    print("  python fix_target_leakage.py")
    print("=" * 80)
    FEATURES_FILE = EDA_DIR / "final_features_for_ml.json"
    print(f"Using original features (MAY CONTAIN LEAKAGE): {FEATURES_FILE.name}\n")

# Temporal split configuration
TRAIN_END_YEAR = 2018
VAL_END_YEAR = 2020
TEST_START_YEAR = 2021

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ------------------------------
# Step C1: Load Data & Temporal Split
# ------------------------------
print("\n" + "=" * 80)
print("C1: TEMPORAL TRAIN/VALIDATION/TEST SPLIT")
print("=" * 80)

# Load master table
print("\n[C1.1] Loading Master Table")
print("-" * 80)
df = pd.read_parquet(MASTER_FILE)
print(f"✓ Loaded data: {len(df):,} rows × {len(df.columns)} columns")

# Add year for splitting
df['year'] = df['date'].dt.year

# Load final features from EDA
print("\n[C1.2] Loading Final Feature List from EDA")
print("-" * 80)
with open(FEATURES_FILE, 'r') as f:
    feature_metadata = json.load(f)

final_features = feature_metadata['final_feature_list']
print(f"✓ Loaded {len(final_features)} features for modeling")

# Ensure all features exist in dataframe
available_features = [f for f in final_features if f in df.columns]
missing_features = [f for f in final_features if f not in df.columns]

if missing_features:
    print(f"⚠ WARNING: {len(missing_features)} features not found in data:")
    print(f"  {missing_features[:5]}...")
    final_features = available_features

print(f"✓ Using {len(final_features)} features for modeling")

# Temporal split
print("\n[C1.3] Performing Temporal Split")
print("-" * 80)
print(f"  Train: 2000 - {TRAIN_END_YEAR}")
print(f"  Validation: {TRAIN_END_YEAR + 1} - {VAL_END_YEAR}")
print(f"  Test: {TEST_START_YEAR} - 2025")

train_mask = df['year'] <= TRAIN_END_YEAR
val_mask = (df['year'] > TRAIN_END_YEAR) & (df['year'] <= VAL_END_YEAR)
test_mask = df['year'] >= TEST_START_YEAR

train_df = df[train_mask].copy()
val_df = df[val_mask].copy()
test_df = df[test_mask].copy()

print(f"\n📊 Split Summary:")
print(f"  Train: {len(train_df):,} rows ({train_df['year'].min()}-{train_df['year'].max()})")
print(f"    Fire events: {train_df['fire_label'].sum():,} ({train_df['fire_label'].mean()*100:.2f}%)")
print(f"  Validation: {len(val_df):,} rows ({val_df['year'].min()}-{val_df['year'].max()})")
print(f"    Fire events: {val_df['fire_label'].sum():,} ({val_df['fire_label'].mean()*100:.2f}%)")
print(f"  Test: {len(test_df):,} rows ({test_df['year'].min()}-{test_df['year'].max()})")
print(f"    Fire events: {test_df['fire_label'].sum():,} ({test_df['fire_label'].mean()*100:.2f}%)")

# Extract features and target
X_train = train_df[final_features].copy()
y_train = train_df['fire_label'].copy()

X_val = val_df[final_features].copy()
y_val = val_df['fire_label'].copy()

X_test = test_df[final_features].copy()
y_test = test_df['fire_label'].copy()

# Handle any remaining missing values
print("\n[C1.4] Handling Missing Values")
print("-" * 80)

# Check for missing values
train_missing = X_train.isnull().sum().sum()
val_missing = X_val.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()

print(f"Missing values - Train: {train_missing}, Val: {val_missing}, Test: {test_missing}")

if train_missing > 0 or val_missing > 0 or test_missing > 0:
    print("⚠ Filling remaining missing values with median (fit on train only)")
    print("This may take 1-2 minutes...")
    
    # Fit on train, transform all
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    
    start = time()
    print("  [1/3] Fitting imputer on training data...")
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    print(f"  ✓ Done in {time()-start:.1f}s")
    
    start = time()
    print("  [2/3] Transforming validation data...")
    X_val = pd.DataFrame(
        imputer.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    print(f"  ✓ Done in {time()-start:.1f}s")
    
    start = time()
    print("  [3/3] Transforming test data...")
    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    print(f"  ✓ Done in {time()-start:.1f}s")
    
    print("✓ Missing values filled")

# Feature Scaling
print("\n[C1.5] Feature Scaling (RobustScaler)")
print("-" * 80)

scaler = RobustScaler()

start = time()
print("  [1/3] Fitting scaler on training data...")
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
print(f"  ✓ Done in {time()-start:.1f}s")

start = time()
print("  [2/3] Transforming validation data...")
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=X_val.columns,
    index=X_val.index
)
print(f"  ✓ Done in {time()-start:.1f}s")

start = time()
print("  [3/3] Transforming test data...")
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)
print(f"  ✓ Done in {time()-start:.1f}s")

print("✓ Features scaled using RobustScaler (fit on train only)")

# Save preprocessed data and scaler
with open(OUTPUT_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved scaler for future use")

# ------------------------------
# Step C2: Baseline Models
# ------------------------------
print("\n" + "=" * 80)
print("C2: BASELINE MODELS")
print("=" * 80)

# Metrics calculation function
def calculate_metrics(y_true, y_pred, y_pred_proba, model_name="Model"):
    """Calculate comprehensive metrics for model evaluation"""
    
    metrics = {}
    
    # ROC-AUC
    metrics['ROC_AUC'] = roc_auc_score(y_true, y_pred_proba)
    
    # PR-AUC (Precision-Recall AUC)
    metrics['PR_AUC'] = average_precision_score(y_true, y_pred_proba)
    
    # Recall @ Fire (True Positive Rate)
    metrics['Recall_Fire'] = recall_score(y_true, y_pred, zero_division=0)
    
    # Precision @ Fire
    metrics['Precision_Fire'] = precision_score(y_true, y_pred, zero_division=0)
    
    # F1 Score
    metrics['F1_Score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics['True_Positives'] = tp
    metrics['False_Positives'] = fp
    metrics['True_Negatives'] = tn
    metrics['False_Negatives'] = fn
    
    # False Alarm Rate (FPR)
    metrics['False_Alarm_Rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Specificity (True Negative Rate)
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics

# Storage for results
baseline_results = {}

# C2.1: Logistic Regression
print("\n[C2.1] Training Logistic Regression (class_weight='balanced')")
print("-" * 80)

# Use smaller sample for faster training (optional)
USE_SAMPLE = True
SAMPLE_SIZE = 100000  # Use 100K samples for faster training

if USE_SAMPLE and len(X_train_scaled) > SAMPLE_SIZE:
    print(f"⚡ Using {SAMPLE_SIZE:,} samples for faster training...")
    sample_idx = np.random.choice(len(X_train_scaled), SAMPLE_SIZE, replace=False)
    X_train_lr = X_train_scaled.iloc[sample_idx]
    y_train_lr = y_train.iloc[sample_idx]
else:
    X_train_lr = X_train_scaled
    y_train_lr = y_train

lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=100,  # Reduced from 1000
    random_state=RANDOM_SEED,
    n_jobs=-1,
    solver='lbfgs',  # Faster than saga for this size
    verbose=1,  # Show progress
    warm_start=False
)

print(f"Training Logistic Regression on {len(X_train_lr):,} samples...")
print("This may take 2-5 minutes...")
lr_model.fit(X_train_lr, y_train_lr)
print("✓ Training complete")

# Predictions
print("Making predictions...")
lr_val_pred_proba = lr_model.predict_proba(X_val_scaled)[:, 1]
lr_val_pred = lr_model.predict(X_val_scaled)

lr_test_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_test_pred = lr_model.predict(X_test_scaled)

# Calculate metrics
print("\nValidation Set Performance:")
lr_val_metrics = calculate_metrics(y_val, lr_val_pred, lr_val_pred_proba, "Logistic Regression")
for metric, value in lr_val_metrics.items():
    if 'AUC' in metric or 'Rate' in metric or 'Recall' in metric or 'Precision' in metric or 'F1' in metric or 'Specificity' in metric:
        print(f"  {metric}: {value:.4f}")

print("\nTest Set Performance:")
lr_test_metrics = calculate_metrics(y_test, lr_test_pred, lr_test_pred_proba, "Logistic Regression")
for metric, value in lr_test_metrics.items():
    if 'AUC' in metric or 'Rate' in metric or 'Recall' in metric or 'Precision' in metric or 'F1' in metric or 'Specificity' in metric:
        print(f"  {metric}: {value:.4f}")

baseline_results['Logistic_Regression'] = {
    'model': lr_model,
    'validation': lr_val_metrics,
    'test': lr_test_metrics,
    'val_pred_proba': lr_val_pred_proba,
    'test_pred_proba': lr_test_pred_proba
}

# Save model
with open(OUTPUT_DIR / 'logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("\n✓ Saved Logistic Regression model")

# C2.2: Random Forest
print("\n[C2.2] Training Random Forest (class_weight='balanced_subsample')")
print("-" * 80)

# Use smaller sample for faster training
if USE_SAMPLE and len(X_train_scaled) > SAMPLE_SIZE:
    print(f"⚡ Using {SAMPLE_SIZE:,} samples for faster training...")
    X_train_rf = X_train_scaled.iloc[sample_idx]
    y_train_rf = y_train.iloc[sample_idx]
else:
    X_train_rf = X_train_scaled
    y_train_rf = y_train

rf_model = RandomForestClassifier(
    n_estimators=50,  # Reduced from 100 for speed
    max_depth=10,
    min_samples_split=50,  # Increased for faster training
    min_samples_leaf=20,   # Increased for faster training
    max_features='sqrt',   # Faster than 'auto'
    class_weight='balanced_subsample',
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=2  # Show detailed progress
)

print(f"Training Random Forest on {len(X_train_rf):,} samples...")
print("Building 50 trees with progress updates...")
from time import time
start_time = time()
rf_model.fit(X_train_rf, y_train_rf)
elapsed = time() - start_time
print(f"✓ Training complete in {elapsed:.1f} seconds")

# Predictions
print("Making predictions...")
rf_val_pred_proba = rf_model.predict_proba(X_val_scaled)[:, 1]
rf_val_pred = rf_model.predict(X_val_scaled)

rf_test_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_test_pred = rf_model.predict(X_test_scaled)

# Calculate metrics
print("\nValidation Set Performance:")
rf_val_metrics = calculate_metrics(y_val, rf_val_pred, rf_val_pred_proba, "Random Forest")
for metric, value in rf_val_metrics.items():
    if 'AUC' in metric or 'Rate' in metric or 'Recall' in metric or 'Precision' in metric or 'F1' in metric or 'Specificity' in metric:
        print(f"  {metric}: {value:.4f}")

print("\nTest Set Performance:")
rf_test_metrics = calculate_metrics(y_test, rf_test_pred, rf_test_pred_proba, "Random Forest")
for metric, value in rf_test_metrics.items():
    if 'AUC' in metric or 'Rate' in metric or 'Recall' in metric or 'Precision' in metric or 'F1' in metric or 'Specificity' in metric:
        print(f"  {metric}: {value:.4f}")

baseline_results['Random_Forest'] = {
    'model': rf_model,
    'validation': rf_val_metrics,
    'test': rf_test_metrics,
    'val_pred_proba': rf_val_pred_proba,
    'test_pred_proba': rf_test_pred_proba
}

# Save model
with open(OUTPUT_DIR / 'random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("\n✓ Saved Random Forest model")

# ------------------------------
# Step C3: Baseline Comparison Table
# ------------------------------
print("\n" + "=" * 80)
print("C3: BASELINE COMPARISON TABLE")
print("=" * 80)

# Create comparison dataframe
comparison_data = []

for model_name, results in baseline_results.items():
    # Validation metrics
    comparison_data.append({
        'Model': model_name,
        'Split': 'Validation',
        'ROC_AUC': results['validation']['ROC_AUC'],
        'PR_AUC': results['validation']['PR_AUC'],
        'Recall_Fire': results['validation']['Recall_Fire'],
        'Precision_Fire': results['validation']['Precision_Fire'],
        'F1_Score': results['validation']['F1_Score'],
        'False_Alarm_Rate': results['validation']['False_Alarm_Rate'],
        'Specificity': results['validation']['Specificity']
    })
    
    # Test metrics
    comparison_data.append({
        'Model': model_name,
        'Split': 'Test',
        'ROC_AUC': results['test']['ROC_AUC'],
        'PR_AUC': results['test']['PR_AUC'],
        'Recall_Fire': results['test']['Recall_Fire'],
        'Precision_Fire': results['test']['Precision_Fire'],
        'F1_Score': results['test']['F1_Score'],
        'False_Alarm_Rate': results['test']['False_Alarm_Rate'],
        'Specificity': results['test']['Specificity']
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Save comparison table
comparison_df.to_csv(OUTPUT_DIR / 'baseline_model_comparison.csv', index=False)
print(f"\n✓ Saved: baseline_model_comparison.csv")

# ------------------------------
# Step C4: Visualization - ROC & PR Curves
# ------------------------------
print("\n" + "=" * 80)
print("C4: GENERATING PERFORMANCE VISUALIZATIONS")
print("=" * 80)

# C4.1: ROC Curves
print("\n[C4.1] Plotting ROC Curves")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Validation ROC
for model_name, results in baseline_results.items():
    fpr, tpr, _ = roc_curve(y_val, results['val_pred_proba'])
    auc = results['validation']['ROC_AUC']
    axes[0].plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", linewidth=2)

axes[0].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curve - Validation Set', fontsize=14, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

# Test ROC
for model_name, results in baseline_results.items():
    fpr, tpr, _ = roc_curve(y_test, results['test_pred_proba'])
    auc = results['test']['ROC_AUC']
    axes[1].plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", linewidth=2)

axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roc_curves_baseline.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curves_baseline.png")
plt.close()

# C4.2: Precision-Recall Curves
print("\n[C4.2] Plotting Precision-Recall Curves")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Validation PR
for model_name, results in baseline_results.items():
    precision, recall, _ = precision_recall_curve(y_val, results['val_pred_proba'])
    pr_auc = results['validation']['PR_AUC']
    axes[0].plot(recall, precision, label=f"{model_name} (AP={pr_auc:.3f})", linewidth=2)

baseline_fire_rate = y_val.mean()
axes[0].axhline(y=baseline_fire_rate, color='k', linestyle='--', 
                label=f'Baseline ({baseline_fire_rate:.3f})', linewidth=1)
axes[0].set_xlabel('Recall', fontsize=12)
axes[0].set_ylabel('Precision', fontsize=12)
axes[0].set_title('Precision-Recall Curve - Validation Set', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(alpha=0.3)

# Test PR
for model_name, results in baseline_results.items():
    precision, recall, _ = precision_recall_curve(y_test, results['test_pred_proba'])
    pr_auc = results['test']['PR_AUC']
    axes[1].plot(recall, precision, label=f"{model_name} (AP={pr_auc:.3f})", linewidth=2)

baseline_fire_rate = y_test.mean()
axes[1].axhline(y=baseline_fire_rate, color='k', linestyle='--', 
                label=f'Baseline ({baseline_fire_rate:.3f})', linewidth=1)
axes[1].set_xlabel('Recall', fontsize=12)
axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('Precision-Recall Curve - Test Set', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pr_curves_baseline.png', dpi=300, bbox_inches='tight')
print("✓ Saved: pr_curves_baseline.png")
plt.close()

# C4.3: Confusion Matrices
print("\n[C4.3] Plotting Confusion Matrices")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_idx = 0
for model_name, results in baseline_results.items():
    # Validation
    cm_val = confusion_matrix(y_val, (results['val_pred_proba'] > 0.5).astype(int))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=axes[plot_idx, 0])
    axes[plot_idx, 0].set_title(f'{model_name} - Validation', fontsize=12, fontweight='bold')
    axes[plot_idx, 0].set_ylabel('True Label')
    axes[plot_idx, 0].set_xlabel('Predicted Label')
    
    # Test
    cm_test = confusion_matrix(y_test, (results['test_pred_proba'] > 0.5).astype(int))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', ax=axes[plot_idx, 1])
    axes[plot_idx, 1].set_title(f'{model_name} - Test', fontsize=12, fontweight='bold')
    axes[plot_idx, 1].set_ylabel('True Label')
    axes[plot_idx, 1].set_xlabel('Predicted Label')
    
    plot_idx += 1

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrices_baseline.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrices_baseline.png")
plt.close()

# ------------------------------
# Step C5: Feature Importance (Random Forest)
# ------------------------------
print("\n" + "=" * 80)
print("C5: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

print("\n[C5.1] Random Forest Feature Importance")
print("-" * 80)

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': final_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

# Save full importance list
feature_importance.to_csv(OUTPUT_DIR / 'feature_importance_random_forest.csv', index=False)
print(f"\n✓ Saved: feature_importance_random_forest.csv")

# Plot top 20
fig, ax = plt.subplots(figsize=(10, 8))
top_20 = feature_importance.head(20)
ax.barh(range(len(top_20)), top_20['Importance'], color='steelblue', alpha=0.7)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['Feature'])
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Top 20 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance_top20.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance_top20.png")
plt.close()

# ------------------------------
# Final Summary
# ------------------------------
print("\n" + "=" * 80)
print("STEP C COMPLETE - BASELINE MODELS TRAINED")
print("=" * 80)

summary = f"""
BASELINE MODEL TRAINING SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

1. TEMPORAL SPLIT
   - Train: 2000-{TRAIN_END_YEAR} ({len(train_df):,} samples, {train_df['fire_label'].sum():,} fires)
   - Validation: {TRAIN_END_YEAR+1}-{VAL_END_YEAR} ({len(val_df):,} samples, {val_df['fire_label'].sum():,} fires)
   - Test: {TEST_START_YEAR}-2025 ({len(test_df):,} samples, {test_df['fire_label'].sum():,} fires)

2. FEATURES
   - Total features used: {len(final_features)}
   - Scaling: RobustScaler (fit on train only)

3. BASELINE MODELS TRAINED
   - Logistic Regression (class_weight='balanced')
   - Random Forest (class_weight='balanced_subsample', n_estimators=100)

4. TEST SET PERFORMANCE
"""

for model_name, results in baseline_results.items():
    summary += f"\n   {model_name}:\n"
    summary += f"     ROC-AUC: {results['test']['ROC_AUC']:.4f}\n"
    summary += f"     PR-AUC: {results['test']['PR_AUC']:.4f}\n"
    summary += f"     Recall@Fire: {results['test']['Recall_Fire']:.4f}\n"
    summary += f"     Precision@Fire: {results['test']['Precision_Fire']:.4f}\n"
    summary += f"     F1-Score: {results['test']['F1_Score']:.4f}\n"
    summary += f"     False Alarm Rate: {results['test']['False_Alarm_Rate']:.4f}\n"

summary += f"""
5. OUTPUT FILES
   - baseline_model_comparison.csv (full metrics table)
   - roc_curves_baseline.png
   - pr_curves_baseline.png
   - confusion_matrices_baseline.png
   - feature_importance_random_forest.csv
   - feature_importance_top20.png
   - logistic_regression_model.pkl
   - random_forest_model.pkl
   - scaler.pkl

6. NEXT STEPS
   - Compare with advanced models (XGBoost, LightGBM)
   - Tune hyperparameters on validation set
   - Consider SMOTE or focal loss for better recall
   
================================================================================
"""

print(summary)

# Save summary
with open(OUTPUT_DIR / 'baseline_training_summary.txt', 'w') as f:
    f.write(summary)

print(f"\n📁 All outputs saved to: {OUTPUT_DIR}")
print("\n✓ Ready for advanced model training!")
print("=" * 80)