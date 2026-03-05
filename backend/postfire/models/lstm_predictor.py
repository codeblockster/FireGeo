"""
LSTM Model Predictor for Fire Risk Assessment
Load and use trained LSTM model for predictions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')

class FireRiskLSTM(nn.Module):
    """LSTM model architecture"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout=0.3, bidirectional=False):
        super(FireRiskLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()


class LSTMPredictor:
    """
    LSTM-based Fire Risk Predictor
    """
    
    def __init__(self, model, config, scaler=None, imputer=None, 
                 expected_features=None, sequence_length=14):
        self.model = model
        self.config = config
        self.scaler = scaler
        self.imputer = imputer
        self.expected_features = expected_features
        self.sequence_length = sequence_length
        
        self.optimal_threshold = config.get('optimal_threshold', 0.5)
        self.default_threshold = 0.5
    
    @classmethod
    def load(cls, model_path, device=None):
        """Load LSTM model from checkpoint"""
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model
        model = FireRiskLSTM(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            bidirectional=checkpoint['bidirectional']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load config
        config = {
            'optimal_threshold': checkpoint.get('optimal_threshold', 0.5),
            'n_features': checkpoint.get('n_features'),
            'sequence_length': checkpoint.get('sequence_length', 14)
        }
        
        # Load scaler
        scaler = None
        model_dir = Path(model_path).parent
        scaler_path = model_dir / 'lstm_scaler.pkl'
        
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        # Load expected features
        expected_features = checkpoint.get('feature_names', None)
        
        return cls(
            model=model,
            config=config,
            scaler=scaler,
            expected_features=expected_features,
            sequence_length=config['sequence_length']
        )
    
    def _preprocess(self, X):
        """Preprocess input data for LSTM"""
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Select expected features
        if self.expected_features is not None:
            missing = set(self.expected_features) - set(X.columns)
            for feat in missing:
                X[feat] = np.nan
            
            X_processed = X[self.expected_features].copy()
        else:
            X_processed = X.copy()
        
        # Handle missing values
        if X_processed.isnull().any().any():
            from sklearn.impute import SimpleImputer
            if self.imputer is None:
                self.imputer = SimpleImputer(strategy='median')
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
        
        # Scale
        if self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)
        
        # Create sequences - repeat features for sequence length
        # For single prediction, we create a dummy sequence
        n_samples = len(X_processed)
        X_seq = np.tile(X_processed, (1, self.sequence_length))
        X_seq = X_seq.reshape(n_samples, self.sequence_length, -1)
        
        return X_seq
    
    def predict_proba(self, X):
        """Get probability predictions"""
        X_seq = self._preprocess(X)
        
        X_tensor = torch.FloatTensor(X_seq).to(next(self.model.parameters()).device)
        
        with torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy()
        
        return proba
    
    def predict(self, X, use_optimal_threshold=True):
        """Get binary predictions"""
        proba = self.predict_proba(X)
        threshold = self.optimal_threshold if use_optimal_threshold else self.default_threshold
        return (proba >= threshold).astype(int)
    
    def predict_with_risk_levels(self, X, use_optimal_threshold=True):
        """Predict with comprehensive risk assessment"""
        probabilities = self.predict_proba(X)
        predictions = self.predict(X, use_optimal_threshold=use_optimal_threshold)
        
        # Categorize confidence
        confidence = np.full(len(probabilities), 'Medium', dtype=object)
        confidence[probabilities >= 0.9] = 'Very High'
        
        # Categorize fire risk
        risk_level = np.full(len(probabilities), 'Moderate', dtype=object)
        risk_level[probabilities >= 0.8] = 'Critical'
        risk_level[(probabilities >= 0.6) & (probabilities < 0.8)] = 'High'
        risk_level[(probabilities >= 0.4) & (probabilities < 0.6)] = 'Medium'
        risk_level[probabilities < 0.4] = 'Low'
        
        # Alert priority
        alert_priority = np.full(len(probabilities), 'Monitor', dtype=object)
        alert_priority[(predictions == 1) & (probabilities >= 0.8)] = 'Critical'
        alert_priority[(predictions == 1) & (probabilities >= 0.6) & (probabilities < 0.8)] = 'High'
        alert_priority[(predictions == 1) & (probabilities < 0.6)] = 'Medium'
        alert_priority[(predictions == 0) & (probabilities >= self.optimal_threshold - 0.1)] = 'Watch'
        
        return pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'risk_level': risk_level,
            'confidence': confidence,
            'alert_priority': alert_priority
        })


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Example: Load and use model
    MODEL_PATH = Path(__file__).parent / "models" / "lstm_fire_model.pth"
    
    if MODEL_PATH.exists():
        print("Loading LSTM model...")
        predictor = LSTMPredictor.load(MODEL_PATH)
        
        # Example: Create dummy features
        import random
        n_features = len(predictor.expected_features) if predictor.expected_features else 81
        
        # Dummy input
        X = pd.DataFrame(np.random.rand(5, n_features), columns=predictor.expected_features)
        
        # Predict
        result = predictor.predict_with_risk_levels(X)
        print("\nPredictions:")
        print(result)
    else:
        print(f"Model not found at: {MODEL_PATH}")
        print("Please run lstm_trainer.py first to train the model.")
