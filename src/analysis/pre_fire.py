import numpy as np
import logging
from typing import Dict, List, Any
from datetime import datetime
import json
import os
import joblib
import pandas as pd
import sys
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

from ..data_collection.gee_extractor import get_gee_extractor
# We'll need to import the XGBoost trainer or model class to load the model
# Assuming we can just load the pickle/joblib file directly if we used standard sklearn/xgboost save.
# But for safety, let's use the class if available or raw xgboost.
import xgboost as xgb

logger = logging.getLogger(__name__)

class PreFireAnalyzer:
    """
    Analyzer for Pre-Fire Risk Assessment.
    Generates risk maps based on environmental data using an Ensemble Model.
    """
    
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = str(config.PRE_FIRE_MODELS_DIR)
        self.gee_extractor = get_gee_extractor()
        # Import here to avoid circular dependencies if any
        from src.models.pre_fire.ensemble_model import WildfireEnsemble
        self.model = WildfireEnsemble(models_dir=models_dir)
        
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
            
            for i in range(len(lats) - 1):
                for j in range(len(lons) - 1):
                    # Define cell center
                    cell_lat = (lats[i] + lats[i+1]) / 2
                    cell_lon = (lons[j] + lons[j+1]) / 2
                    
                    # Fetch Features
                    data = self.gee_extractor.get_environmental_data(cell_lat, cell_lon)
                    
                    # Predict using Ensemble
                    risk_score = self.model.predict_risk(data)

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
                            "risk_score": float(risk_score),
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
                    "resolution": f"{grid_resolution}x{grid_resolution}",
                    "model_loaded": self.model.loaded
                }
            }
            
        except Exception as e:
            logger.error(f"Pre-fire analysis failed: {e}")
            return {"error": str(e)}
