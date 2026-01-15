import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle

class XGBoostFireRiskTrainer:
    def __init__(self, params=None):
        """
        Initialize XGBoost trainer for fire risk prediction
        
        Args:
            params: Dictionary of XGBoost parameters
        """
        self.params = params or {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'random_state': 42
        }
        self.model = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features (N, features)
            y_train: Training labels (N,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        self.model = xgb.XGBClassifier(**self.params)
        
        if X_val is not None and y_val is not None:
            # Train with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance scores"""
        importance = self.model.feature_importances_
        
        if feature_names:
            importance_dict = dict(zip(feature_names, importance))
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True))
            return importance_dict
        
        return importance
    
    def save_model(self, filepath):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Placeholder for actual data loading
    # In practice, load your preprocessed wildfire data
    
    # Example: Generate dummy data
    np.random.seed(42)
    X = np.random.randn(1000, 12)  # 1000 samples, 12 features
    y = np.random.randint(0, 2, 1000)  # Binary labels
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Initialize trainer
    trainer = XGBoostFireRiskTrainer()
    
    # Train model
    print("Training XGBoost model...")
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    print("\nEvaluating model...")
    results = trainer.evaluate(X_test, y_test)
    
    # Feature importance
    feature_names = [f"feature_{i}" for i in range(12)]
    importance = trainer.get_feature_importance(feature_names)
    print("\nTop 5 Important Features:")
    for feat, imp in list(importance.items())[:5]:
        print(f"{feat}: {imp:.4f}")
    
    # Save model
    trainer.save_model("data/models/xgboost_fire_risk.pkl")
