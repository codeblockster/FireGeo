
import numpy as np
import pandas as pd
import joblib
import pickle
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import sys

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import config

# Configure logging
logger = logging.getLogger(__name__)

class FeatureManager:
    """
    Manages feature alignment for models trained on different feature sets.
    Adapted from 'Weighted Ensemble for Wildfire Prediction - S-TIER FINAL VERSION'
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
        Detect or load the features a model expects.
        """
        features = None
        
        # Try loading from file
        if feature_path and Path(feature_path).exists():
            try:
                with open(feature_path, 'r') as f:
                    features = json.load(f)
                
                if isinstance(features, list) and all(isinstance(f, str) for f in features):
                    self.model_features[model_name] = features
                    self.feature_counts[model_name] = len(features)
                    self.feature_hashes[model_name] = self._compute_feature_hash(features)
                    logger.info(f"Loaded {len(features)} features for {model_name} from {feature_path}")
                    return features
            except Exception as e:
                logger.warning(f"Could not load features from file for {model_name}: {e}")
        
        # Try extracting from model object
        try:
            if hasattr(model, 'get_feature_names'): # CatBoost
                features = model.get_feature_names()
            elif hasattr(model, 'feature_names_'): # Sklearn/CatBoost
                features = list(model.feature_names_)
            elif hasattr(model, 'feature_name_'): # LightGBM
                features = model.feature_name_
            elif hasattr(model, 'get_booster'): # XGBoost
                features = model.get_booster().feature_names
            elif hasattr(model, 'feature_names_in_'): # Sklearn
                features = list(model.feature_names_in_)
            
            if features:
                logger.info(f"Auto-detected {len(features)} features for {model_name}")
        except Exception as e:
            logger.warning(f"Error detecting features for {model_name}: {e}")
        
        self.model_features[model_name] = features
        if features:
            self.feature_counts[model_name] = len(features)
            self.feature_hashes[model_name] = self._compute_feature_hash(features)
        
        return features
    
    def align_features(self, input_data: Dict[str, float], model_name: str) -> pd.DataFrame:
        """
        Align single input dictionary to match what the model expects (DataFrame).
        """
        expected_features = self.model_features.get(model_name)
        
        # Convert single dict to DataFrame
        df = pd.DataFrame([input_data])
        
        if expected_features is None:
            # If no features known, return as is (might fail if model is picky)
            return df
        
        # If expected features are known, ensure they exist
        # We handle missing features by filling with 0 or NaN, but ideally they should be present
        missing = set(expected_features) - set(df.columns)
        if missing:
            logger.warning(f"Model {model_name} expects missing features: {missing}. Filling with 0.")
            for col in missing:
                df[col] = 0.0
                
        # Reorder columns
        return df[expected_features]

class WildfireEnsemble:
    """
    Weighted Ensemble Model for Wildfire Risk Prediction.
    Integrates CatBoost, LightGBM, and XGBoost.
    """
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = str(config.PRE_FIRE_MODELS_DIR)
        self.models_dir = Path(models_dir)
        self.feature_manager = FeatureManager()
        self.models = {}
        self.weights = {
            'catboost': 0.5,
            'lightgbm': 0.3, 
            'xgboost': 0.2
        }
        self.loaded = False
        self._load_models()
        
    def _load_models(self):
        """Load available models from the directory."""
        model_files = {
            'catboost': self.models_dir / "catboost_s_tier_model.pkl",
            'lightgbm': self.models_dir / "lightgbm_best_model.pkl",
            'xgboost': self.models_dir / "xgboost_enhanced_model.joblib"
        }
        
        loaded_count = 0
        for name, path in model_files.items():
            if path.exists():
                try:
                    if name == 'catboost':
                        try:
                            import catboost
                        except ImportError:
                            logger.warning("CatBoost library not found, skipping CatBoost model.")
                            continue
                    elif name == 'lightgbm':
                        try:
                            import lightgbm
                        except ImportError:
                            logger.warning("LightGBM library not found, skipping LightGBM model.")
                            continue

                    model = None
                    if path.suffix == '.pkl':
                        with open(path, 'rb') as f:
                            # Use joblib for robust pickling regarding different module versions
                            model = pickle.load(f)
                    elif path.suffix == '.joblib':
                        model = joblib.load(path)
                    else:
                        continue
                        
                    if model is not None:
                        self.models[name] = model
                        self.feature_manager.detect_features(model, name)
                        loaded_count += 1
                        logger.info(f"Loaded {name} from {path}")
                        
                except Exception as e:
                    logger.error(f"Failed to load {name}: {e}")
            else:
                logger.warning(f"Model file not found: {path}")

        if loaded_count > 0:
            self.loaded = True
            # Renormalize weights based on loaded models
            total_weight = sum(self.weights.get(name, 0) for name in self.models.keys())
            if total_weight > 0:
                for name in self.models:
                    self.weights[name] /= total_weight
            else:
                # Fallback to uniform
                w = 1.0 / len(self.models)
                for name in self.models:
                    self.weights[name] = w
        else:
            logger.warning("No ensemble models loaded. Running in MOCK mode.")

    def predict_risk(self, input_data: Dict[str, float]) -> float:
        """
        Predict fire risk probability (0-1).
        
        Args:
            input_data: Dictionary of environmental features.
        """
        if not self.loaded:
            return self._mock_predict(input_data)
        
        ensemble_proba = 0.0
        
        for name, model in self.models.items():
            try:
                X = self.feature_manager.align_features(input_data, name)
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[:, 1][0]
                elif hasattr(model, 'predict'):
                    # Assume simple predict returns class or regression score
                    # If regression, we might need sigmoid if not probability
                    # For now assume probability-like output
                    proba = float(model.predict(X)[0])
                    proba = max(0.0, min(1.0, proba)) # Clip
                else:
                    proba = 0.0
                
                ensemble_proba += proba * self.weights[name]
                
            except Exception as e:
                logger.error(f"Prediction failed for {name}: {e}")
                
        return float(ensemble_proba)

    def _mock_predict(self, data: Dict[str, float]) -> float:
        """Fallback logic if no models are available."""
        # Simple heuristic based on common features
        # Assuming keys match GEE output
        temp = data.get('temp_max', 30)
        humidity = data.get('humidity', 50)
        wind = data.get('wind_speed', 10)
        # Normalize to 0-1 range roughly
        # T > 40 is bad (1), T < 10 is good (0)
        t_score = (temp - 10) / 30.0
        # H < 20 is bad (1), H > 80 is good (0)
        h_score = (80 - humidity) / 60.0
        # W > 50 is bad (1), W < 5 is good (0)
        w_score = (wind - 5) / 45.0
        
        score = (t_score + h_score + w_score) / 3.0
        return max(0.0, min(1.0, score))
