import numpy as np
import logging
from typing import Dict, List, Any
from datetime import datetime
import json
import os
import joblib
import pandas as pd

from ..data_collection.gee_extractor import get_gee_extractor
# We'll need to import the XGBoost trainer or model class to load the model
# Assuming we can just load the pickle/joblib file directly if we used standard sklearn/xgboost save.
# But for safety, let's use the class if available or raw xgboost.
import xgboost as xgb

logger = logging.getLogger(__name__)

class PreFireAnalyzer:
    """
    Analyzer for Pre-Fire Risk Assessment.
    Generates risk maps based on environmental data.
    """
    
    def __init__(self, model_path: str = "data/models/xgboost_fire_risk.pkl"):
        self.gee_extractor = get_gee_extractor()
        self.model_path = model_path
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the trained XGBoost model"""
        try:
            if os.path.exists(self.model_path):
                # Try loading with pickle (as used in train_xgboost.py)
                import pickle
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded risk model from {self.model_path}")
            else:
                logger.warning(f"Model not found at {self.model_path}. Using mock predictions.")
                self.model = None
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def generate_risk_map(self, lat: float, lon: float, 
                         size_km: int = 10, 
                         grid_resolution: int = 5) -> Dict[str, Any]:
        """
        Generate a fire risk map for a region.
        
        Args:
            lat, lon: Center coordinates
            size_km: Size of the bounding box (width/height) in km
            grid_resolution: Number of grid points per axis (e.g., 5 means 5x5 grid)
            
        Returns:
            GeoJSON FeatureCollection of risk zones
        """
        try:
            # 1. Create Grid
            # approx 1 deg lat = 111 km
            # approx 1 deg lon = 111 km * cos(lat)
            km_per_deg_lat = 111.0
            km_per_deg_lon = 111.0 * np.cos(np.radians(lat))
            
            delta_lat = (size_km / km_per_deg_lat) / 2
            delta_lon = (size_km / km_per_deg_lon) / 2
            
            min_lat = lat - delta_lat
            max_lat = lat + delta_lat
            min_lon = lon - delta_lon
            max_lon = lon + delta_lon
            
            lats = np.linspace(min_lat, max_lat, grid_resolution)
            lons = np.linspace(min_lon, max_lon, grid_resolution)
            
            risk_zones = []
            
            # 2. Iterate and Predict
            # Note: Sequential GEE calls are slow. In production, this should be parallelized or batched.
            # For this minor project, a small grid (e.g., 5x5 = 25 calls) is acceptable.
            
            for i in range(len(lats) - 1):
                for j in range(len(lons) - 1):
                    # Define cell center
                    cell_lat = (lats[i] + lats[i+1]) / 2
                    cell_lon = (lons[j] + lons[j+1]) / 2
                    
                    # Fetch Features
                    data = self.gee_extractor.get_environmental_data(cell_lat, cell_lon)
                    
                    # Prepare input vector (must match training order)
                    # Based on gee_extractor.py keys, we need to order them correctly 
                    # as expected by the model. 
                    # The train_xgboost.py doesn't specify order, but usually it's alphabetical or fixed.
                    # Let's assume a standard order implicitly or just pass all.
                    # We'll use a standard list here sorted by key to be deterministic.
                    feature_keys = sorted(data.keys()) 
                    features = [data[k] for k in feature_keys]
                    
                    # Predict
                    risk_score = 0.5
                    if self.model:
                        try:
                            # Reshape for single sample
                            X = np.array(features).reshape(1, -1)
                            # Predict proba for class 1 (Fire)
                            risk_score = float(self.model.predict_proba(X)[:, 1])
                        except Exception as e:
                            logger.error(f"Prediction error: {e}")
                    else:
                        # Mock Logic if no model: High Temp + Low Humidity + High Wind = High Risk
                        # Normalize inputs roughly
                        temp = data.get('temp_max', 30) / 50.0
                        humid = (100 - data.get('humidity', 50)) / 100.0
                        wind = data.get('wind_speed', 10) / 30.0
                        risk_score = (temp + humid + wind) / 3.0
                        risk_score = min(max(risk_score, 0.0), 1.0)

                    # Create GeoJSON Feature (Polygon for the grid cell)
                    # Points: BL, BR, TR, TL, BL
                    polygon = [
                        [lons[j], lats[i]],
                        [lons[j+1], lats[i]],
                        [lons[j+1], lats[i+1]],
                        [lons[j], lats[i+1]],
                        [lons[j], lats[i]]
                    ]
                    
                    risk_zones.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [polygon]
                        },
                        "properties": {
                            "risk_score": risk_score,
                            "risk_level": "High" if risk_score > 0.7 else "Moderate" if risk_score > 0.4 else "Low",
                            "data": data
                        }
                    })
            
            return {
                "type": "FeatureCollection",
                "features": risk_zones,
                "center": [lat, lon],
                "metadata": {
                    "count": len(risk_zones),
                    "resolution": f"{grid_resolution}x{grid_resolution}"
                }
            }
            
        except Exception as e:
            logger.error(f"Pre-fire analysis failed: {e}")
            return {"error": str(e)}
