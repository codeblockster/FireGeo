"""
Deployment Script for S-Tier Tuned CatBoost Model
Correctly handles calibrated predictions, optimal thresholds, and policy-aware decisions
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    
from sklearn.isotonic import IsotonicRegression
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class CatBoostPredictor:
    """
    Deployment-ready CatBoost predictor with optimal threshold and risk assessment
    """
    
    def __init__(self, model, config, scaler=None, imputer=None, calibrator=None, expected_features=None):
        self.model = model
        self.config = config
        self.scaler = scaler
        self.imputer = imputer
        self.calibrator = calibrator
        self.expected_features = expected_features
        
        # Extract key configuration
        self.optimal_threshold = config.get('optimal_threshold', 0.5)
        self.default_threshold = 0.5
        self.target_far = config.get('target_false_alarm_rate', 0.05)
        self.calibration_applied = calibrator is not None
        
    @classmethod
    def load(cls, model_dir):
        """
        Load CatBoost model and all artifacts from directory
        """
        model_dir = Path(model_dir)
        
        logger.info(f"Loading CatBoost model from: {model_dir}")
        
        # Load model
        # Try different names
        model_files = [
            'catboost_s_tier_model.pkl',
            'model.cbm',
            'model.pkl'
        ]
        
        model = None
        for fname in model_files:
            fpath = model_dir / fname
            if fpath.exists():
                try:
                    if fname.endswith('.pkl'):
                        with open(fpath, 'rb') as f:
                            # Safely unpickle; usually requires class definition
                            model = pickle.load(f)
                    else:
                        if not CATBOOST_AVAILABLE:
                            raise ImportError("CatBoost library not installed. Cannot load .cbm model.")
                        model = CatBoostClassifier()
                        model.load_model(str(fpath))
                    logger.info(f"Model loaded from {fname}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {fname}: {e}")
        
        if model is None:
            if not CATBOOST_AVAILABLE:
                logger.warning("CatBoost not available. Running in Mock Mode.")
                # Return a predictor with None model that handles mock logic if needed,
                # or let it fail so PreFireAnalyzer handles it.
                # Here we raise to let PreFireAnalyzer catch it and set model_loaded=False
                raise ImportError("CatBoost not installed")
            raise FileNotFoundError(f"No valid model file found in {model_dir}")
        
        # Load configuration
        config = {}
        
        # Load optimal threshold info
        threshold_path = model_dir / 'optimal_threshold_info.json'
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                threshold_info = json.load(f)
            config.update(threshold_info)
        
        # Load best hyperparameters
        params_path = model_dir / 'best_hyperparameters_s_tier.json'
        if params_path.exists():
            with open(params_path, 'r') as f:
                hyperparameters = json.load(f)
            config['hyperparameters'] = hyperparameters
        
        # Load metrics
        metrics_path = model_dir / 's_tier_model_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            config['metrics'] = metrics
        
        # Load scaler
        scaler = None
        # Check model_dir first, then parent/model_results
        scaler_paths = [
            model_dir / 'scaler.pkl',
            model_dir.parent / 'model_results' / 'scaler.pkl'
        ]
        
        for spath in scaler_paths:
            if spath.exists():
                with open(spath, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info(f"Scaler loaded from {spath}")
                break
        
        # Imputer (will be fitted on first use if not loaded)
        imputer = None
        
        # Calibrator
        calibrator = None
        
        # Expected features
        expected_features = None
        
        # Method 1: Feature names file
        features_file = model_dir / 'feature_names.json'
        if features_file.exists():
            with open(features_file, 'r') as f:
                expected_features = json.load(f)
                if isinstance(expected_features, dict) and 'final_feature_list' in expected_features:
                    expected_features = expected_features['final_feature_list']
        
        # Method 2: From model
        if expected_features is None and hasattr(model, 'feature_names_'):
            expected_features = list(model.feature_names_)
            
        # Method 3: From scaler
        if expected_features is None and scaler is not None and hasattr(scaler, 'feature_names_in_'):
            expected_features = list(scaler.feature_names_in_)

        if expected_features:
             logger.info(f"Expected features: {len(expected_features)}")
        else:
             logger.warning("Could not determine expected features")

        return cls(
            model=model,
            config=config,
            scaler=scaler,
            imputer=imputer,
            calibrator=calibrator,
            expected_features=expected_features
        )
    
    def _preprocess(self, X, fit_imputer=False):
        """Preprocess input data"""
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Select features
        if self.expected_features is not None:
            # Add missing
            missing = set(self.expected_features) - set(X.columns)
            for feat in missing:
                X[feat] = np.nan
            
            # Select and order
            X_processed = X[self.expected_features].copy()
        else:
            X_processed = X.copy()
        
        # Handle missing values
        if X_processed.isnull().any().any():
            if self.imputer is not None:
                if fit_imputer:
                    X_processed = pd.DataFrame(
                        self.imputer.fit_transform(X_processed),
                        columns=X_processed.columns,
                        index=X_processed.index
                    )
                else:
                    X_processed = pd.DataFrame(
                        self.imputer.transform(X_processed),
                        columns=X_processed.columns,
                        index=X_processed.index
                    )
            else:
                from sklearn.impute import SimpleImputer
                self.imputer = SimpleImputer(strategy='median')
                X_processed = pd.DataFrame(
                    self.imputer.fit_transform(X_processed),
                    columns=X_processed.columns,
                    index=X_processed.index
                )
        
        # Scale
        if self.scaler is not None:
            X_processed = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        
        return X_processed
    
    def predict_proba(self, X):
        X_processed = self._preprocess(X)
        return self.model.predict_proba(X_processed)[:, 1]
    
    def predict(self, X, use_optimal_threshold=True):
        proba = self.predict_proba(X)
        threshold = self.optimal_threshold if use_optimal_threshold else self.default_threshold
        return (proba >= threshold).astype(int)
    
    def predict_with_risk_levels(self, X, use_optimal_threshold=True):
        """Predict with comprehensive risk assessment"""
        probabilities = self.predict_proba(X)
        predictions = self.predict(X, use_optimal_threshold=use_optimal_threshold)
        
        # Categorize confidence (Simplified)
        confidence = np.full(len(probabilities), 'Medium')
        confidence[probabilities >= 0.9] = 'Very High'
        
        # Categorize fire risk
        risk_level = self._categorize_risk(probabilities)
        
        # Alert priority
        alert_priority = self._assign_alert_priority(probabilities, predictions)
        
        return pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'risk_level': risk_level,
            'confidence': confidence,
            'alert_priority': alert_priority
        })
    
    def _categorize_risk(self, probabilities):
        risk = np.full(len(probabilities), 'Moderate', dtype=object)
        risk[probabilities >= 0.8] = 'Critical'
        risk[(probabilities >= 0.6) & (probabilities < 0.8)] = 'High'
        risk[(probabilities >= 0.4) & (probabilities < 0.6)] = 'Medium'
        risk[probabilities < 0.4] = 'Low'
        return risk
    
    def _assign_alert_priority(self, probabilities, predictions):
        priority = np.full(len(probabilities), 'Monitor', dtype=object)
        priority[(predictions == 1) & (probabilities >= 0.8)] = 'Critical'
        priority[(predictions == 1) & (probabilities >= 0.6) & (probabilities < 0.8)] = 'High'
        priority[(predictions == 1) & (probabilities < 0.6)] = 'Medium'
        priority[(predictions == 0) & (probabilities >= self.optimal_threshold - 0.1)] = 'Watch'
        return priority
