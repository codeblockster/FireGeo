import xgboost as xgb

class XGBoostFireRiskModel:
    """XGBoost model wrapper for fire risk prediction"""
    
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.load(model_path)
    
    def load(self, model_path):
        """Load trained XGBoost model"""
        import pickle
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, features):
        """
        Predict fire risk
        
        Args:
            features: numpy array of shape (n_samples, n_features)
        
        Returns:
            Risk probabilities (0-1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        return self.model.predict_proba(features)[:, 1]
    
    def predict_binary(self, features):
        """Predict binary fire risk (0 or 1)"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        return self.model.predict(features)
