"""
Weighted Ensemble for Wildfire Prediction - S-TIER FINAL VERSION
Handles models trained on different feature sets with robust error handling

CRITICAL FIXES APPLIED (from review):
1. ✅ Robust CatBoost feature detection (get_feature_names)
2. ✅ Feature JSON validation after loading
3. ✅ Linear stacking collapse protection
4. ✅ Calibration metrics on validation (not test)
5. ✅ Feature provenance tracking (hash + count)
6. ✅ Monotonicity check after calibration
7. ✅ Ensemble diversity metrics
8. ✅ Per-model feature count logging

Author: Research-Grade Ensemble System
Version: 4.0 (S-Tier Final)
"""

import numpy as np
import pandas as pd
import joblib
import pickle
import json
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, f1_score, precision_score, recall_score, confusion_matrix,
    brier_score_loss
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for ensemble model"""
    
    # Model paths
    BASE_PATH = Path("/Users/prabhatrawal/Minor_project_code/data/integrated_data")
    
    CATBOOST_MODEL = BASE_PATH / "catboost_tuning/catboost_s_tier_model.pkl"
    LIGHTGBM_MODEL = BASE_PATH / "lightgbm_tuning/lightgbm_best_model.pkl"
    XGBOOST_MODEL = BASE_PATH / "xgboost_tuning/xgboost_enhanced_model.joblib"
    
    # Feature list paths (optional - will be auto-detected)
    CATBOOST_FEATURES = BASE_PATH / "catboost_tuning/feature_names.json"
    LIGHTGBM_FEATURES = BASE_PATH / "lightgbm_tuning/feature_names.json"
    XGBOOST_FEATURES = BASE_PATH / "xgboost_tuning/feature_names.json"
    
    # Data paths
    DATA_DIR = BASE_PATH / "ensemble_ready"
    X_TRAIN_PATH = DATA_DIR / "X_train.csv"
    Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
    X_VAL_PATH = DATA_DIR / "X_val.csv"
    Y_VAL_PATH = DATA_DIR / "y_val.csv"
    X_TEST_PATH = DATA_DIR / "X_test.csv"
    Y_TEST_PATH = DATA_DIR / "y_test.csv"
    
    # Output directory
    OUTPUT_DIR = BASE_PATH / "ensemble_results_s_tier"
    
    # Ensemble method
    ENSEMBLE_METHOD = 'weighted_voting'  # 'weighted_voting' or 'linear_stacking'
    
    # Fixed weights (for weighted voting)
    FIXED_WEIGHTS = {
        'catboost': 0.5,   # High recall sensor
        'lightgbm': 0.3,   # Low false alarm filter
        'xgboost': 0.2     # Tie-breaker/diversity
    }
    
    # Calibration settings
    APPLY_CALIBRATION = True
    CALIBRATION_METHOD = 'isotonic'  # 'isotonic' or 'platt'
    CALIBRATE_ENSEMBLE = True
    
    # Cost-sensitive thresholding
    COST_SENSITIVE = True
    COST_FN = 10.0
    COST_FP = 1.0
    
    # Threshold optimization
    THRESHOLD_MIN = 0.0
    THRESHOLD_MAX = 1.0
    THRESHOLD_STEPS = 300
    
    # Validation checks
    CHECK_MONOTONICITY = True
    CHECK_DIVERSITY = True
    
    # Visualization
    FIGURE_SIZE = (12, 8)
    DPI = 300


# ============================================================================
# FEATURE MANAGEMENT (UPGRADED)
# ============================================================================

class FeatureManager:
    """
    Manages feature alignment for models trained on different feature sets
    
    UPGRADES:
    - Robust CatBoost detection
    - Feature JSON validation
    - Provenance tracking (hash + count)
    """
    
    def __init__(self):
        self.model_features = {}
        self.feature_hashes = {}
        self.feature_counts = {}
    
    def _compute_feature_hash(self, features):
        """Compute hash of feature list for provenance tracking"""
        if features is None:
            return None
        feature_str = ','.join(sorted(features))
        return hashlib.md5(feature_str.encode()).hexdigest()[:8]
    
    def detect_features(self, model, model_name, feature_path=None):
        """
        Detect or load the features a model expects
        
        UPGRADED:
        - CatBoost: get_feature_names() before feature_names_
        - JSON validation after loading
        - Hash tracking
        """
        features = None
        
        # Try loading from file
        if feature_path and Path(feature_path).exists():
            try:
                with open(feature_path, 'r') as f:
                    features = json.load(f)
                
                # UPGRADE: Validate loaded features
                if not isinstance(features, list):
                    raise ValueError(f"Feature list must be a list, got {type(features)}")
                
                if not all(isinstance(f, str) for f in features):
                    raise ValueError("All features must be strings")
                
                if len(features) == 0:
                    raise ValueError("Feature list is empty")
                
                print(f"  ✅ Loaded {len(features)} features from {feature_path}")
                self.model_features[model_name] = features
                self.feature_counts[model_name] = len(features)
                self.feature_hashes[model_name] = self._compute_feature_hash(features)
                return features
                
            except Exception as e:
                print(f"  ⚠️  Could not load features from file: {e}")
        
        # Try extracting from model object
        try:
            # UPGRADE: CatBoost - try get_feature_names() FIRST
            if hasattr(model, 'get_feature_names'):
                features = model.get_feature_names()
                print(f"  ✅ Extracted {len(features)} features from model.get_feature_names()")
            
            # CatBoost fallback
            elif hasattr(model, 'feature_names_'):
                features = list(model.feature_names_)
                print(f"  ✅ Extracted {len(features)} features from model.feature_names_")
            
            # LightGBM
            elif hasattr(model, 'feature_name_'):
                features = model.feature_name_
                print(f"  ✅ Extracted {len(features)} features from model.feature_name_")
            
            # XGBoost
            elif hasattr(model, 'get_booster'):
                features = model.get_booster().feature_names
                print(f"  ✅ Extracted {len(features)} features from booster.feature_names")
            
            # Sklearn-like
            elif hasattr(model, 'feature_names_in_'):
                features = list(model.feature_names_in_)
                print(f"  ✅ Extracted {len(features)} features from model.feature_names_in_")
            
            else:
                print(f"  ⚠️  Could not auto-detect features for {model_name}")
                print(f"     Model will use all available features (may cause errors)")
        
        except Exception as e:
            print(f"  ⚠️  Error detecting features: {e}")
        
        self.model_features[model_name] = features
        if features:
            self.feature_counts[model_name] = len(features)
            self.feature_hashes[model_name] = self._compute_feature_hash(features)
        
        return features
    
    def align_features(self, X_data, model_name):
        """
        Align dataset features to match what the model expects
        
        Parameters:
        -----------
        X_data : DataFrame
            Full feature set
        model_name : str
            Name of the model
        
        Returns:
        --------
        X_aligned : DataFrame
            Features aligned for this model
        """
        expected_features = self.model_features.get(model_name)
        
        if expected_features is None:
            # No feature list available - use all features
            return X_data
        
        # Check for missing features
        missing_features = set(expected_features) - set(X_data.columns)
        if missing_features:
            raise ValueError(
                f"Model '{model_name}' expects features not in data: {missing_features}"
            )
        
        # Select and reorder features
        X_aligned = X_data[expected_features]
        
        return X_aligned
    
    def save_feature_mapping(self, output_dir):
        """Save feature mapping with provenance info"""
        mapping = {}
        
        for model_name in self.model_features.keys():
            features = self.model_features[model_name]
            mapping[model_name] = {
                'features': features if features is not None else "all_features",
                'count': self.feature_counts.get(model_name, 'unknown'),
                'hash': self.feature_hashes.get(model_name, 'unknown')
            }
        
        output_path = output_dir / "feature_mapping.json"
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=4)
        
        print(f"\n✅ Saved feature mapping with provenance to {output_path}")
    
    def get_summary(self):
        """Get summary of feature mappings for logging"""
        summary = {}
        for model_name in self.model_features.keys():
            summary[model_name] = {
                'num_features': self.feature_counts.get(model_name, 'unknown'),
                'hash': self.feature_hashes.get(model_name, 'unknown')
            }
        return summary


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_model(model_path, model_name):
    """Load a trained model from file"""
    model_path = Path(model_path)
    print(f"\n📥 Loading {model_name} from: {model_path}")
    
    try:
        if model_path.suffix == '.pkl':
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_path.suffix == '.joblib':
            model = joblib.load(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
        
        print(f"✅ {model_name} loaded successfully")
        return model
    
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        raise


def load_data(X_path, y_path, name="Data"):
    """Load features and labels from CSV"""
    print(f"\n📂 Loading {name}...")
    
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()
    
    print(f"✅ {name}: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y


# ============================================================================
# PREDICTION GENERATION (WITH FEATURE ALIGNMENT)
# ============================================================================

def generate_predictions(models, X_data, model_names, feature_manager):
    """
    Generate predictions with automatic feature alignment
    
    Parameters:
    -----------
    models : dict
        Dictionary of models
    X_data : DataFrame
        Full feature set
    model_names : list
        List of model names
    feature_manager : FeatureManager
        Feature alignment manager
    
    Returns:
    --------
    predictions : dict
        Dictionary of predictions
    """
    predictions = {}
    
    print("\n🔮 Generating predictions with feature alignment...")
    
    for name in model_names:
        model = models[name]
        
        try:
            # Align features for this model
            X_aligned = feature_manager.align_features(X_data, name)
            
            print(f"\n  {name}:")
            print(f"    Using {X_aligned.shape[1]} features (hash: {feature_manager.feature_hashes.get(name, 'N/A')})")
            
            # Generate predictions
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_aligned)[:, 1]
            elif hasattr(model, 'predict'):
                proba = model.predict(X_aligned)
            else:
                raise ValueError(f"Model {name} has no predict method")
            
            predictions[name] = proba
            print(f"    ✅ Predictions generated (range: {proba.min():.4f} - {proba.max():.4f})")
        
        except Exception as e:
            print(f"    ❌ Error generating predictions for {name}: {e}")
            raise
    
    return predictions


# ============================================================================
# WEIGHT LEARNING (UPGRADED)
# ============================================================================

def learn_weights_linear_stacking(predictions_dict, y_true):
    """
    Learn ensemble weights using logistic regression
    
    UPGRADE: Protection against weight collapse
    """
    print("\n🎓 Learning weights via Linear Stacking...")
    
    model_names = list(predictions_dict.keys())
    X_meta = np.column_stack([predictions_dict[name] for name in model_names])
    
    lr = LogisticRegression(fit_intercept=False, max_iter=1000, random_state=42)
    lr.fit(X_meta, y_true)
    
    raw_weights = lr.coef_[0]
    raw_weights = np.maximum(raw_weights, 0)
    
    # UPGRADE: Protect against collapse
    if raw_weights.sum() == 0 or np.isnan(raw_weights.sum()):
        print("  ⚠️  All weights ≈ 0, using uniform weights")
        normalized_weights = np.ones_like(raw_weights) / len(raw_weights)
    else:
        normalized_weights = raw_weights / raw_weights.sum()
    
    weights = {name: w for name, w in zip(model_names, normalized_weights)}
    
    print("✅ Learned weights:")
    for name, weight in weights.items():
        print(f"   • {name}: {weight:.4f} ({weight*100:.1f}%)")
    
    # Warn if one model dominates
    max_weight = max(weights.values())
    if max_weight > 0.8:
        print(f"  ⚠️  Warning: {max(weights, key=weights.get)} dominates with {max_weight*100:.1f}%")
        print(f"     Ensemble may not add much value over single model")
    
    return weights


# ============================================================================
# WEIGHTED ENSEMBLE
# ============================================================================

def create_weighted_ensemble(predictions, weights):
    """Create weighted ensemble predictions"""
    print("\n⚖️  Creating weighted ensemble...")
    
    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0):
        print(f"⚠️  Warning: Weights sum to {weight_sum:.4f}, normalizing...")
        weights = {k: v/weight_sum for k, v in weights.items()}
    
    print("\nEnsemble weights:")
    for model_name, weight in weights.items():
        print(f"  • {model_name}: {weight:.4f} ({weight*100:.1f}%)")
    
    ensemble_proba = np.zeros_like(predictions[list(predictions.keys())[0]])
    
    for model_name, proba in predictions.items():
        weight = weights[model_name]
        ensemble_proba += weight * proba
    
    print(f"\n✅ Ensemble created (range: {ensemble_proba.min():.4f} - {ensemble_proba.max():.4f})")
    
    return ensemble_proba


# ============================================================================
# CALIBRATION (UPGRADED)
# ============================================================================

def calibrate_predictions(y_true, y_proba, method='isotonic'):
    """Calibrate probability predictions"""
    print(f"\n🔧 Calibrating predictions (method: {method})...")
    
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(y_proba, y_true)
    elif method == 'platt' or method == 'sigmoid':
        calibrator = LogisticRegression()
        calibrator.fit(y_proba.reshape(-1, 1), y_true)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    print(f"✅ Calibration complete")
    return calibrator


def apply_calibration(calibrator, y_proba, method='isotonic'):
    """Apply fitted calibrator to predictions"""
    if method == 'isotonic':
        return calibrator.transform(y_proba)
    elif method == 'platt' or method == 'sigmoid':
        return calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def check_monotonicity(y_proba_uncal, y_proba_cal):
    """
    UPGRADE: Check calibration didn't break monotonicity
    
    Returns:
    --------
    is_monotonic : bool
    violations : int
    """
    print("\n🔍 Checking calibration monotonicity...")
    
    # Sort indices by uncalibrated probabilities
    sort_idx = np.argsort(y_proba_uncal)
    
    # Check if calibrated probabilities are also sorted
    calibrated_sorted = y_proba_cal[sort_idx]
    
    # Count violations (where diff < 0)
    diffs = np.diff(calibrated_sorted)
    violations = np.sum(diffs < -1e-10)  # Allow tiny numerical errors
    
    is_monotonic = violations == 0
    
    if is_monotonic:
        print(f"  ✅ Calibration preserved monotonicity")
    else:
        violation_pct = 100 * violations / len(diffs)
        print(f"  ⚠️  {violations} monotonicity violations ({violation_pct:.2f}%)")
        if violation_pct > 1.0:
            print(f"     WARNING: Calibration may be unreliable")
    
    return is_monotonic, violations


def compute_calibration_metrics(y_true, y_proba_uncal, y_proba_cal):
    """Compute calibration quality metrics"""
    brier_uncal = brier_score_loss(y_true, y_proba_uncal)
    brier_cal = brier_score_loss(y_true, y_proba_cal)
    
    def compute_ece(y_true, y_proba, n_bins=10):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bin_edges[1:-1])
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                avg_pred = y_proba[mask].mean()
                avg_true = y_true[mask].mean()
                ece += mask.sum() / len(y_true) * abs(avg_pred - avg_true)
        
        return ece
    
    ece_uncal = compute_ece(y_true, y_proba_uncal)
    ece_cal = compute_ece(y_true, y_proba_cal)
    
    metrics = {
        'brier_uncalibrated': brier_uncal,
        'brier_calibrated': brier_cal,
        'brier_improvement': brier_uncal - brier_cal,
        'ece_uncalibrated': ece_uncal,
        'ece_calibrated': ece_cal,
        'ece_improvement': ece_uncal - ece_cal
    }
    
    print("\n📊 Calibration Quality Metrics:")
    print(f"   Brier Score: {brier_uncal:.4f} → {brier_cal:.4f} (Δ {metrics['brier_improvement']:.4f})")
    print(f"   ECE: {ece_uncal:.4f} → {ece_cal:.4f} (Δ {metrics['ece_improvement']:.4f})")
    
    return metrics


# ============================================================================
# DIVERSITY ANALYSIS (UPGRADE)
# ============================================================================

def compute_diversity_metrics(predictions_dict):
    """
    UPGRADE: Compute ensemble diversity metrics
    
    Returns correlation matrix and diversity score
    """
    print("\n🎨 Computing ensemble diversity...")
    
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    # Stack predictions
    pred_matrix = np.column_stack([predictions_dict[name] for name in model_names])
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(pred_matrix.T)
    
    # Average pairwise correlation (lower = more diverse)
    upper_tri = corr_matrix[np.triu_indices(n_models, k=1)]
    avg_correlation = upper_tri.mean()
    
    # Diversity score (1 - avg correlation)
    diversity_score = 1 - avg_correlation
    
    print(f"\n📊 Diversity Metrics:")
    print(f"   Average pairwise correlation: {avg_correlation:.4f}")
    print(f"   Diversity score: {diversity_score:.4f}")
    
    print(f"\n   Correlation matrix:")
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            if i < j:
                print(f"     {name_i} ↔ {name_j}: {corr_matrix[i, j]:.4f}")
    
    if avg_correlation > 0.95:
        print(f"\n  ⚠️  WARNING: Models are highly correlated (>{0.95:.2f})")
        print(f"     Ensemble may not add much value")
    elif diversity_score > 0.3:
        print(f"\n  ✅ Good diversity (>{0.3:.2f}) - ensemble should add value")
    
    return {
        'correlation_matrix': corr_matrix,
        'model_names': model_names,
        'avg_correlation': avg_correlation,
        'diversity_score': diversity_score
    }


def plot_diversity_matrix(diversity_metrics, output_dir):
    """Plot correlation matrix heatmap"""
    plt.figure(figsize=(8, 7))
    
    corr_matrix = diversity_metrics['correlation_matrix']
    model_names = diversity_metrics['model_names']
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',
        xticklabels=model_names,
        yticklabels=model_names,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation'}
    )
    
    plt.title('Model Prediction Correlation Matrix\n(Lower = More Diverse)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'diversity_matrix.png', dpi=Config.DPI, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved diversity matrix")


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def optimize_threshold_vectorized(y_true, y_proba, metric='f1', 
                                   cost_sensitive=False, cost_fn=10.0, cost_fp=1.0):
    """Vectorized threshold optimization"""
    if cost_sensitive:
        print(f"\n🎯 Optimizing threshold (cost-sensitive: FN={cost_fn}, FP={cost_fp})...")
    else:
        print(f"\n🎯 Optimizing threshold (metric: {metric})...")
    
    thresholds = np.linspace(Config.THRESHOLD_MIN, Config.THRESHOLD_MAX, Config.THRESHOLD_STEPS)
    
    y_true_expanded = y_true.reshape(-1, 1)
    y_proba_expanded = y_proba.reshape(-1, 1)
    
    y_pred_all = (y_proba_expanded >= thresholds).astype(int)
    
    tp = ((y_pred_all == 1) & (y_true_expanded == 1)).sum(axis=0)
    fp = ((y_pred_all == 1) & (y_true_expanded == 0)).sum(axis=0)
    tn = ((y_pred_all == 0) & (y_true_expanded == 0)).sum(axis=0)
    fn = ((y_pred_all == 0) & (y_true_expanded == 1)).sum(axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        specificity = tn / (tn + fp)
        fpr = fp / (fp + tn)
        
        precision = np.nan_to_num(precision, 0)
        recall = np.nan_to_num(recall, 0)
        f1 = np.nan_to_num(f1, 0)
        specificity = np.nan_to_num(specificity, 0)
        fpr = np.nan_to_num(fpr, 0)
    
    expected_cost = (fn * cost_fn + fp * cost_fp) / len(y_true)
    
    threshold_metrics = pd.DataFrame({
        'threshold': thresholds,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'specificity': specificity,
        'expected_cost': expected_cost,
        'fn': fn,
        'fp': fp
    })
    
    if cost_sensitive or metric == 'cost':
        optimal_idx = threshold_metrics['expected_cost'].idxmin()
    elif metric == 'f1':
        optimal_idx = threshold_metrics['f1'].idxmax()
    elif metric == 'recall':
        optimal_idx = threshold_metrics['recall'].idxmax()
    elif metric == 'precision':
        optimal_idx = threshold_metrics['precision'].idxmax()
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    optimal_threshold = threshold_metrics.loc[optimal_idx, 'threshold']
    optimal_metrics = threshold_metrics.loc[optimal_idx].to_dict()
    
    print(f"\n✅ Optimal threshold: {optimal_threshold:.4f}")
    print(f"   F1 Score: {optimal_metrics['f1']:.4f}")
    print(f"   Precision: {optimal_metrics['precision']:.4f}")
    print(f"   Recall: {optimal_metrics['recall']:.4f}")
    if cost_sensitive:
        print(f"   Expected Cost: {optimal_metrics['expected_cost']:.4f}")
    
    return optimal_threshold, threshold_metrics


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(y_true, y_proba, threshold, model_name="Model"):
    """Comprehensive model evaluation"""
    print(f"\n📊 Evaluating {model_name}...")
    
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'model_name': model_name,
        'threshold': threshold,
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'brier_score': brier_score_loss(y_true, y_proba)
    }
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'accuracy': (tp + tn) / (tp + tn + fp + fn)
    })
    
    print(f"\n{model_name} Performance:")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    return metrics


def compare_models(all_metrics):
    """Create comparison table"""
    comparison_df = pd.DataFrame(all_metrics)
    
    column_order = [
        'model_name', 'threshold', 'roc_auc', 'pr_auc', 'f1_score',
        'precision', 'recall', 'specificity', 'false_alarm_rate',
        'brier_score', 'accuracy', 'true_positives', 'true_negatives',
        'false_positives', 'false_negatives'
    ]
    
    comparison_df = comparison_df[column_order]
    
    return comparison_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_threshold_analysis(threshold_metrics, optimal_threshold, output_dir, cost_sensitive=False):
    """Plot threshold vs metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # F1 Score
    axes[0].plot(threshold_metrics['threshold'], threshold_metrics['f1'], 'b-', linewidth=2)
    axes[0].axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal: {optimal_threshold:.3f}')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score vs Threshold', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall
    axes[1].plot(threshold_metrics['threshold'], threshold_metrics['precision'], 'g-', label='Precision', linewidth=2)
    axes[1].plot(threshold_metrics['threshold'], threshold_metrics['recall'], 'b-', label='Recall', linewidth=2)
    axes[1].axvline(optimal_threshold, color='r', linestyle='--')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Precision and Recall vs Threshold', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Specificity
    axes[2].plot(threshold_metrics['threshold'], threshold_metrics['specificity'], 'purple', linewidth=2)
    axes[2].axvline(optimal_threshold, color='r', linestyle='--')
    axes[2].set_xlabel('Threshold')
    axes[2].set_ylabel('Specificity')
    axes[2].set_title('Specificity vs Threshold', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Expected Cost
    axes[3].plot(threshold_metrics['threshold'], threshold_metrics['expected_cost'], 'orange', linewidth=2)
    axes[3].axvline(optimal_threshold, color='r', linestyle='--')
    axes[3].set_xlabel('Threshold')
    axes[3].set_ylabel('Expected Cost')
    axes[3].set_title('Expected Cost vs Threshold', fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_analysis.png', dpi=Config.DPI, bbox_inches='tight')
    plt.close()


def plot_roc_curves(models_data, output_dir):
    """Plot ROC curves"""
    plt.figure(figsize=Config.FIGURE_SIZE)
    
    for model_name, data in models_data.items():
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_proba'])
        auc = roc_auc_score(data['y_true'], data['y_proba'])
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'roc_curves.png', dpi=Config.DPI, bbox_inches='tight')
    plt.close()


def plot_pr_curves(models_data, output_dir):
    """Plot PR curves"""
    plt.figure(figsize=Config.FIGURE_SIZE)
    
    for model_name, data in models_data.items():
        precision, recall, _ = precision_recall_curve(data['y_true'], data['y_proba'])
        pr_auc = average_precision_score(data['y_true'], data['y_proba'])
        plt.plot(recall, precision, linewidth=2, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Model Comparison', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'pr_curves.png', dpi=Config.DPI, bbox_inches='tight')
    plt.close()


# ============================================================================
# SAVE OUTPUTS (UPGRADED)
# ============================================================================

def save_ensemble_model(models, weights, threshold, ensemble_calibrator, 
                       feature_manager, diversity_metrics, output_dir, config_info):
    """Save ensemble model with full provenance"""
    print("\n💾 Saving ensemble model...")
    
    ensemble_config = {
        'ensemble_method': config_info['ensemble_method'],
        'weights': weights,
        'threshold': threshold,
        'model_names': list(models.keys()),
        'feature_mapping': feature_manager.model_features,
        'feature_counts': feature_manager.feature_counts,  # UPGRADE
        'feature_hashes': feature_manager.feature_hashes,  # UPGRADE
        'diversity_score': diversity_metrics['diversity_score'],  # UPGRADE
        'avg_correlation': diversity_metrics['avg_correlation'],  # UPGRADE
        'calibration_applied': ensemble_calibrator is not None,
        'calibration_method': Config.CALIBRATION_METHOD,
        'cost_sensitive': config_info.get('cost_sensitive', False),
        'cost_fn': config_info.get('cost_fn', None),
        'cost_fp': config_info.get('cost_fp', None),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(output_dir / 'ensemble_config.json', 'w') as f:
        json.dump(ensemble_config, f, indent=4)
    
    ensemble_package = {
        'models': models,
        'config': ensemble_config,
        'ensemble_calibrator': ensemble_calibrator,
        'feature_manager': feature_manager,
        'diversity_metrics': diversity_metrics
    }
    
    joblib.dump(ensemble_package, output_dir / 'ensemble_model_complete.joblib')
    
    print(f"✅ Ensemble model saved with full provenance")


# ============================================================================
# MAIN EXECUTION (S-TIER FINAL)
# ============================================================================

def main():
    """Main execution with all S-tier upgrades"""
    
    print("\n" + "="*80)
    print("🔥 WILDFIRE ENSEMBLE - S-TIER FINAL VERSION")
    print("="*80)
    print("\nUPGRADES APPLIED:")
    print("  ✅ Robust CatBoost feature detection")
    print("  ✅ Feature JSON validation")
    print("  ✅ Linear stacking collapse protection")
    print("  ✅ Calibration metrics on validation (not test)")
    print("  ✅ Feature provenance tracking (hash + count)")
    print("  ✅ Monotonicity check after calibration")
    print("  ✅ Ensemble diversity analysis")
    print("  ✅ Per-model feature count logging")
    print("="*80)
    
    output_dir = ensure_output_dir(Config.OUTPUT_DIR)
    
    # Initialize feature manager
    feature_manager = FeatureManager()
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    X_train, y_train = load_data(Config.X_TRAIN_PATH, Config.Y_TRAIN_PATH, "Training Set")
    X_val, y_val = load_data(Config.X_VAL_PATH, Config.Y_VAL_PATH, "Validation Set")
    X_test, y_test = load_data(Config.X_TEST_PATH, Config.Y_TEST_PATH, "Test Set")
    
    print("\n📊 Splitting validation set...")
    X_val_weight, X_val_thresh, y_val_weight, y_val_thresh = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42, stratify=y_val
    )
    print(f"   Weight learning set: {len(y_val_weight)} samples")
    print(f"   Threshold optimization set: {len(y_val_thresh)} samples")
    
    # ========================================================================
    # STEP 2: Load Models and Detect Features (UPGRADED)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: LOADING MODELS & DETECTING FEATURES")
    print("="*80)
    
    model_configs = {
        'catboost': (Config.CATBOOST_MODEL, Config.CATBOOST_FEATURES),
        'lightgbm': (Config.LIGHTGBM_MODEL, Config.LIGHTGBM_FEATURES),
        'xgboost': (Config.XGBOOST_MODEL, Config.XGBOOST_FEATURES)
    }
    
    models = {}
    
    for model_name, (model_path, feature_path) in model_configs.items():
        model = load_model(model_path, model_name.upper())
        models[model_name] = model
        
        print(f"\n🔍 Detecting features for {model_name}:")
        feature_manager.detect_features(model, model_name, feature_path)
    
    # Save feature mapping with provenance
    feature_manager.save_feature_mapping(output_dir)
    
    # UPGRADE: Display feature summary
    print("\n" + "="*80)
    print("FEATURE SUMMARY")
    print("="*80)
    summary = feature_manager.get_summary()
    for model_name, info in summary.items():
        print(f"{model_name}: {info['num_features']} features (hash: {info['hash']})")
    
    # ========================================================================
    # STEP 3: Generate Predictions
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: GENERATING PREDICTIONS")
    print("="*80)
    
    print("\n📍 Weight learning set:")
    weight_predictions = generate_predictions(
        models, X_val_weight, list(models.keys()), feature_manager
    )
    
    print("\n📍 Threshold optimization set:")
    thresh_predictions = generate_predictions(
        models, X_val_thresh, list(models.keys()), feature_manager
    )
    
    print("\n📍 Test set:")
    test_predictions = generate_predictions(
        models, X_test, list(models.keys()), feature_manager
    )
    
    # ========================================================================
    # STEP 4: Diversity Analysis (UPGRADE)
    # ========================================================================
    if Config.CHECK_DIVERSITY:
        print("\n" + "="*80)
        print("STEP 4: DIVERSITY ANALYSIS")
        print("="*80)
        
        diversity_metrics = compute_diversity_metrics(test_predictions)
        plot_diversity_matrix(diversity_metrics, output_dir)
    else:
        diversity_metrics = {'diversity_score': 'not_computed', 'avg_correlation': 'not_computed'}
    
    # ========================================================================
    # STEP 5: Learn Weights (UPGRADED)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: DETERMINING ENSEMBLE WEIGHTS")
    print("="*80)
    
    if Config.ENSEMBLE_METHOD == 'linear_stacking':
        weights = learn_weights_linear_stacking(weight_predictions, y_val_weight)
    else:
        weights = Config.FIXED_WEIGHTS
        print("\n📌 Using Weighted Voting (fixed weights):")
        for name, weight in weights.items():
            print(f"   • {name}: {weight:.4f} ({weight*100:.1f}%)")
    
    # ========================================================================
    # STEP 6: Create Ensemble
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: CREATING ENSEMBLE")
    print("="*80)
    
    ensemble_weight_proba = create_weighted_ensemble(weight_predictions, weights)
    ensemble_thresh_proba = create_weighted_ensemble(thresh_predictions, weights)
    ensemble_test_proba = create_weighted_ensemble(test_predictions, weights)
    
    # ========================================================================
    # STEP 7: Calibrate (UPGRADED)
    # ========================================================================
    ensemble_calibrator = None
    calibration_metrics = None
    
    if Config.APPLY_CALIBRATION and Config.CALIBRATE_ENSEMBLE:
        print("\n" + "="*80)
        print("STEP 7: CALIBRATING ENSEMBLE")
        print("="*80)
        
        ensemble_calibrator = calibrate_predictions(
            y_val_weight,
            ensemble_weight_proba,
            method=Config.CALIBRATION_METHOD
        )
        
        # Apply calibration
        ensemble_thresh_proba_cal = apply_calibration(
            ensemble_calibrator,
            ensemble_thresh_proba,
            method=Config.CALIBRATION_METHOD
        )
        
        ensemble_test_proba_cal = apply_calibration(
            ensemble_calibrator,
            ensemble_test_proba,
            method=Config.CALIBRATION_METHOD
        )
        
        # UPGRADE: Check monotonicity
        if Config.CHECK_MONOTONICITY:
            check_monotonicity(ensemble_test_proba, ensemble_test_proba_cal)
        
        # UPGRADE: Compute calibration metrics on VALIDATION (not test)
        calibration_metrics = compute_calibration_metrics(
            y_val_thresh,  # FIXED: Use validation, not test
            ensemble_thresh_proba,
            ensemble_thresh_proba_cal
        )
        
        # Use calibrated versions
        ensemble_thresh_proba = ensemble_thresh_proba_cal
        ensemble_test_proba = ensemble_test_proba_cal
    
    # ========================================================================
    # STEP 8: Optimize Threshold
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 8: THRESHOLD OPTIMIZATION")
    print("="*80)
    
    optimal_threshold, threshold_metrics = optimize_threshold_vectorized(
        y_val_thresh,
        ensemble_thresh_proba,
        metric='f1',
        cost_sensitive=Config.COST_SENSITIVE,
        cost_fn=Config.COST_FN,
        cost_fp=Config.COST_FP
    )
    
    plot_threshold_analysis(
        threshold_metrics,
        optimal_threshold,
        output_dir,
        cost_sensitive=Config.COST_SENSITIVE
    )
    
    threshold_metrics.to_csv(output_dir / 'threshold_metrics.csv', index=False)
    
    # ========================================================================
    # STEP 9: Evaluate All Models
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 9: MODEL EVALUATION (Test Set)")
    print("="*80)
    
    all_metrics = []
    all_thresholds = {}
    
    for model_name in models.keys():
        thresh, _ = optimize_threshold_vectorized(
            y_val_thresh,
            thresh_predictions[model_name],
            metric='f1',
            cost_sensitive=Config.COST_SENSITIVE,
            cost_fn=Config.COST_FN,
            cost_fp=Config.COST_FP
        )
        all_thresholds[model_name] = thresh
        
        metrics = evaluate_model(
            y_test,
            test_predictions[model_name],
            thresh,
            model_name.upper()
        )
        all_metrics.append(metrics)
    
    ensemble_metrics = evaluate_model(
        y_test,
        ensemble_test_proba,
        optimal_threshold,
        "ENSEMBLE"
    )
    all_metrics.append(ensemble_metrics)
    all_thresholds['Ensemble'] = optimal_threshold
    
    comparison_df = compare_models(all_metrics)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON (Test Set)")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    # ========================================================================
    # STEP 10: Visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 10: GENERATING VISUALIZATIONS")
    print("="*80)
    
    models_data = {
        'CatBoost': {'y_true': y_test, 'y_proba': test_predictions['catboost']},
        'LightGBM': {'y_true': y_test, 'y_proba': test_predictions['lightgbm']},
        'XGBoost': {'y_true': y_test, 'y_proba': test_predictions['xgboost']},
        'Ensemble': {'y_true': y_test, 'y_proba': ensemble_test_proba}
    }
    
    plot_roc_curves(models_data, output_dir)
    plot_pr_curves(models_data, output_dir)
    
    print("✅ Visualizations complete")
    
    # ========================================================================
    # STEP 11: Save Model (UPGRADED)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 11: SAVING ENSEMBLE MODEL")
    print("="*80)
    
    config_info = {
        'ensemble_method': Config.ENSEMBLE_METHOD,
        'cost_sensitive': Config.COST_SENSITIVE,
        'cost_fn': Config.COST_FN if Config.COST_SENSITIVE else None,
        'cost_fp': Config.COST_FP if Config.COST_SENSITIVE else None
    }
    
    save_ensemble_model(
        models,
        weights,
        optimal_threshold,
        ensemble_calibrator,
        feature_manager,
        diversity_metrics,
        output_dir,
        config_info
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✅ S-TIER ENSEMBLE COMPLETE")
    print("="*80)
    
    print(f"\n📁 Results saved to: {output_dir}")
    
    best_model = comparison_df.loc[comparison_df['f1_score'].idxmax()]
    print(f"\n🏆 Best Model: {best_model['model_name']}")
    print(f"   F1 Score: {best_model['f1_score']:.4f}")
    print(f"   ROC-AUC: {best_model['roc_auc']:.4f}")
    print(f"   PR-AUC: {best_model['pr_auc']:.4f}")
    
    if calibration_metrics:
        print(f"\n📊 Calibration Improvement (on validation):")
        print(f"   Brier: {calibration_metrics['brier_improvement']:.4f}")
        print(f"   ECE: {calibration_metrics['ece_improvement']:.4f}")
    
    if Config.CHECK_DIVERSITY:
        print(f"\n🎨 Ensemble Diversity:")
        print(f"   Diversity Score: {diversity_metrics['diversity_score']:.4f}")
        print(f"   Avg Correlation: {diversity_metrics['avg_correlation']:.4f}")
    
    print("\n" + "="*80)
    
    return {
        'comparison_df': comparison_df,
        'ensemble_proba': ensemble_test_proba,
        'optimal_threshold': optimal_threshold,
        'weights': weights,
        'feature_manager': feature_manager,
        'diversity_metrics': diversity_metrics,
        'calibration_metrics': calibration_metrics
    }


if __name__ == "__main__":
    results = main()
    print("\n🎉 S-tier ensemble complete!")
    print(f"Check: {Config.OUTPUT_DIR}")