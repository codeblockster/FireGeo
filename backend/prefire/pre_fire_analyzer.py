"""
Pre-Fire Risk Analyzer
Integrates CatBoost model with feature engineering for wildfire risk assessment
Includes parallel processing for fast risk map generation
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend import config

from backend.prefire.catboost_predictor import CatBoostPredictor
from backend.prefire.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)

class PreFireAnalyzer:
    """
    Analyzer for Pre-Fire Risk Assessment.
    
    Features:
    - Integrates CatBoost model with 20-feature engineering pipeline
    - Single location risk analysis
    - Parallel risk map generation for regions
    - Mock predictions when model unavailable
    
    Example:
        >>> analyzer = PreFireAnalyzer()
        >>> result = analyzer.analyze_location(28.3949, 84.1240)
        >>> print(f"Risk: {result['risk_level']}, Probability: {result['probability']:.2%}")
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize Pre-Fire Analyzer
        
        Args:
            models_dir: Path to model directory (default: from config)
        """
        if models_dir is None:
            models_dir = str(config.PRE_FIRE_MODELS_DIR)
        
        logger.info("Initializing PreFireAnalyzer...")
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Load CatBoost model
        try:
            self.model = CatBoostPredictor.load(models_dir)
            self.model_loaded = True
            logger.info(" CatBoost model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CatBoost model: {e}")
            logger.warning("Using mock predictions (model not available)")
            self.model = None
            self.model_loaded = False
        
    def analyze_location(self, lat: float, lon: float, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete risk analysis pipeline for a single location.
        
        Workflow:
        1. Fetch 20 environmental features
        2. Make risk prediction using CatBoost model
        3. Return comprehensive risk assessment
        
        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
            date: Date for analysis (YYYY-MM-DD), defaults to today
            
        Returns:
            Dictionary containing:
                - probability: Fire risk probability (0-1)
                - risk_level: 'Low', 'Medium', 'High', or 'Critical'
                - alert_priority: 'Monitor', 'Watch', 'Medium', 'High', or 'Critical'
                - confidence: Confidence level of prediction
                - features: All 20 features used in prediction
                
        Example:
            >>> result = analyzer.analyze_location(28.3949, 84.1240)
            >>> print(result['risk_level'])
            'High'
        """
        try:
            # 1. Fetch Features
            logger.debug(f"Fetching features for ({lat}, {lon})")
            features = self.feature_engineer.get_all_features(lat, lon, date)
            
            # 2. Prepare Input
            X = pd.DataFrame([features])
            
            # 3. Predict
            if self.model_loaded:
                result = self.model.predict_with_risk_levels(X)
                
                # Extract scalar values
                risk_data = {
                    "probability": float(result['probability'].iloc[0]),
                    "risk_level": result['risk_level'].iloc[0],
                    "alert_priority": result['alert_priority'].iloc[0],
                    "confidence": result['confidence'].iloc[0],
                    "features": features,
                    "model_used": True
                }
                
                logger.debug(f"Prediction: {risk_data['risk_level']} ({risk_data['probability']:.2%})")
                return risk_data
            else:
                logger.warning("Using mock risk prediction (model not loaded)")
                return self._get_mock_prediction(features)
                
        except Exception as e:
            logger.error(f"Analysis failed for ({lat}, {lon}): {e}", exc_info=True)
            return {"error": str(e), "location": [lat, lon]}

    def predict_risk(self, lat: float, lon: float, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Alias for analyze_location() for backward compatibility.
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Optional date string
            
        Returns:
            Risk analysis result with risk_score, risk_level, probability
        """
        result = self.analyze_location(lat, lon, date)
        
        # Add risk_score as alias for probability
        if 'probability' in result and 'risk_score' not in result:
            result['risk_score'] = result['probability']
        
        return result

    def predict_from_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make risk prediction from pre-fetched features.
        
        Useful for two-step workflow:
        1. Fetch features once
        2. Make multiple predictions with variations
        
        Args:
            features: Dictionary of 20 features from FeatureEngineer
            
        Returns:
            Dictionary with prediction results
            
        Example:
            >>> features = engineer.get_all_features(28.3949, 84.1240)
            >>> result = analyzer.predict_from_features(features)
        """
        try:
            if not self.model_loaded:
                logger.warning("Using mock risk prediction (model not loaded)")
                return self._get_mock_prediction(features)
            
            # Check feature count mismatch
            if self.model.expected_features and len(features) != len(self.model.expected_features):
                logger.warning(f"Feature count mismatch: model expects {len(self.model.expected_features)} features, got {len(features)}. Using rule-based prediction.")
                return self._get_mock_prediction(features)
            
            X = pd.DataFrame([features])
            result = self.model.predict_with_risk_levels(X)
            
            return {
                "probability": float(result['probability'].iloc[0]),
                "risk_level": result['risk_level'].iloc[0],
                "alert_priority": result['alert_priority'].iloc[0],
                "confidence": result['confidence'].iloc[0],
                "model_used": True
            }
        except ValueError as e:
            # Catch shape mismatch errors
            if "Shape of passed values" in str(e):
                logger.warning(f"Feature shape mismatch: {e}. Using rule-based prediction.")
                return self._get_mock_prediction(features)
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return self._get_mock_prediction(features)

    def _get_mock_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate plausible mock prediction based on key features.
        
        Uses simple heuristics:
        - High temperature + Low humidity + High VPD = Higher risk
        
        Args:
            features: Dictionary of features
            
        Returns:
            Mock prediction matching real prediction format
        """
        import random
        
        # Extract key features (using correct names from FeatureEngineer)
        temp = features.get('lst_day_c', 30)  # Land Surface Temperature
        humidity = features.get('relative_humidity_pct', 50)
        vpd = features.get('vapor_pressure_deficit_kpa', 1.0)
        precipitation = features.get('precipitation_mm_lag1', 0)
        vegetation = features.get('landsat_savi', 0.4)
        
        # Calculate baseline probability
        baseline_prob = 0.3
        
        # Adjust based on conditions
        if temp > 35: baseline_prob += 0.2
        if temp > 40: baseline_prob += 0.1
        
        if humidity < 30: baseline_prob += 0.2
        if humidity < 20: baseline_prob += 0.1
        
        if vpd > 2.5: baseline_prob += 0.15
        if vpd > 3.5: baseline_prob += 0.1
        
        if precipitation < 1: baseline_prob += 0.05
        
        if vegetation < 0.2: baseline_prob += 0.1  # Dry vegetation
        
        # Add random variation
        prob = baseline_prob + random.uniform(-0.1, 0.1)
        prob = min(0.95, max(0.05, prob))
        
        # Derive risk levels
        if prob > 0.8: 
            risk_level = "Critical"
            alert = "Critical"
        elif prob > 0.6: 
            risk_level = "High"
            alert = "High"
        elif prob > 0.4: 
            risk_level = "Medium"
            alert = "Medium"
        else: 
            risk_level = "Low"
            alert = "Monitor"
        
        return {
            "probability": round(prob, 3),
            "risk_level": risk_level,
            "alert_priority": alert,
            "confidence": "Low (Mock Mode)",
            "features": features,
            "model_used": False
        }

    def _analyze_grid_cell(self, cell_data: Dict) -> Optional[Dict[str, Any]]:
        """
        Analyze a single grid cell for risk map generation.
        
        This is called by parallel workers.
        
        Args:
            cell_data: Dictionary containing:
                - lat, lon: Cell center coordinates
                - i, j: Grid indices
                - lats, lons: Grid arrays
                - date: Analysis date
                
        Returns:
            GeoJSON Feature for this cell or None on error
        """
        try:
            # Analyze this cell
            result = self.analyze_location(
                cell_data['lat'], 
                cell_data['lon'], 
                cell_data.get('date')
            )
            
            # Extract risk data
            if "error" not in result:
                risk_score = result['probability']
                risk_level = result['risk_level']
                alert_priority = result.get('alert_priority', 'Monitor')
            else:
                logger.warning(f"Cell analysis failed: {result.get('error')}")
                risk_score = 0.0
                risk_level = "Unknown"
                alert_priority = "Unknown"
            
            # Get grid coordinates
            i, j = cell_data['i'], cell_data['j']
            lats, lons = cell_data['lats'], cell_data['lons']
            
            # Create GeoJSON polygon (note: GeoJSON uses [lon, lat] order)
            polygon = [
                [lons[j], lats[i]],
                [lons[j+1], lats[i]],
                [lons[j+1], lats[i+1]],
                [lons[j], lats[i+1]],
                [lons[j], lats[i]]  # Close the polygon
            ]
            
            return {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon]
                },
                "properties": {
                    "risk_score": round(risk_score, 3),
                    "risk_level": risk_level,
                    "alert_priority": alert_priority,
                    "center_lat": cell_data['lat'],
                    "center_lon": cell_data['lon'],
                    "grid_i": i,
                    "grid_j": j
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cell ({cell_data['lat']}, {cell_data['lon']}): {e}")
            return None

    def generate_risk_map(self, 
                         lat: float, 
                         lon: float, 
                         size_km: int = 10, 
                         grid_resolution: int = 5,
                         max_workers: int = 5,
                         date: Optional[str] = None,
                         show_progress: bool = True) -> Dict[str, Any]:
        """
        Generate a fire risk map for a region using parallel processing.
        
        Creates a grid of cells around the center point and analyzes each cell
        in parallel for faster processing.
        
        Args:
            lat: Center latitude
            lon: Center longitude
            size_km: Size of area in kilometers (creates size_km x size_km square)
            grid_resolution: Number of grid divisions per side (e.g., 5 = 5x5 = 25 cells)
            max_workers: Number of parallel workers (default: 5, recommended: 3-10)
            date: Date for analysis (YYYY-MM-DD), defaults to today
            show_progress: Show progress information (default: True)
        
        Returns:
            GeoJSON FeatureCollection containing:
                - features: List of polygons with risk scores
                - center: Map center coordinates
                - metadata: Generation info
        
        Performance:
            - 5x5 grid (25 cells) with 5 workers: ~25-30 seconds
            - 10x10 grid (100 cells) with 10 workers: ~2-3 minutes
            - Sequential would take 5x longer
        
        Example:
            >>> risk_map = analyzer.generate_risk_map(
            ...     lat=28.3949, 
            ...     lon=84.1240, 
            ...     size_km=20,
            ...     grid_resolution=8,
            ...     max_workers=8
            ... )
            >>> print(f"Generated {risk_map['metadata']['count']} risk zones")
        """
        try:
            logger.info(f"Generating risk map centered at ({lat}, {lon})")
            logger.info(f"Area: {size_km}x{size_km} km, Resolution: {grid_resolution}x{grid_resolution}")
            
            # 1. Calculate grid boundaries
            km_per_deg_lat = 111.0  # Approximate km per degree latitude
            km_per_deg_lon = 111.0 * np.cos(np.radians(lat))  # Adjusted for latitude
            
            delta_lat = (size_km / km_per_deg_lat) / 2
            delta_lon = (size_km / km_per_deg_lon) / 2
            
            min_lat = lat - delta_lat
            max_lat = lat + delta_lat
            min_lon = lon - delta_lon
            max_lon = lon + delta_lon
            
            # 2. Create grid arrays
            lats = np.linspace(min_lat, max_lat, grid_resolution)
            lons = np.linspace(min_lon, max_lon, grid_resolution)
            
            # 3. Prepare all grid cells for processing
            grid_cells = []
            for i in range(len(lats) - 1):
                for j in range(len(lons) - 1):
                    cell_lat = (lats[i] + lats[i+1]) / 2
                    cell_lon = (lons[j] + lons[j+1]) / 2
                    
                    grid_cells.append({
                        'lat': cell_lat,
                        'lon': cell_lon,
                        'i': i,
                        'j': j,
                        'lats': lats,
                        'lons': lons,
                        'date': date
                    })
            
            total_cells = len(grid_cells)
            logger.info(f"Processing {total_cells} cells with {max_workers} parallel workers")
            
            # 4. Process cells in parallel
            risk_zones = []
            errors = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_cell = {
                    executor.submit(self._analyze_grid_cell, cell): cell 
                    for cell in grid_cells
                }
                
                # Setup progress tracking
                if show_progress:
                    try:
                        from tqdm import tqdm
                        progress_bar = tqdm(
                            total=total_cells, 
                            desc="Generating risk map",
                            unit="cell"
                        )
                        has_tqdm = True
                    except ImportError:
                        has_tqdm = False
                        logger.info("Install tqdm for progress bar: pip install tqdm")
                else:
                    has_tqdm = False
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_cell):
                    completed += 1
                    
                    try:
                        result = future.result(timeout=60)  # 60 second timeout per cell
                        if result is not None:
                            risk_zones.append(result)
                        else:
                            errors += 1
                    except Exception as e:
                        errors += 1
                        logger.error(f"Cell analysis failed: {e}")
                    
                    # Update progress
                    if has_tqdm:
                        progress_bar.update(1)
                    elif show_progress and completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{total_cells} cells ({completed/total_cells*100:.1f}%)")
                
                if has_tqdm:
                    progress_bar.close()
            
            # 5. Log results
            success_rate = (len(risk_zones) / total_cells * 100) if total_cells > 0 else 0
            logger.info(f"Risk map complete: {len(risk_zones)}/{total_cells} cells successful ({success_rate:.1f}%)")
            
            if errors > 0:
                logger.warning(f"{errors} cells failed to analyze")
            
            # 6. Return GeoJSON FeatureCollection
            return {
                "type": "FeatureCollection",
                "features": risk_zones,
                "center": [lat, lon],
                "bounds": {
                    "min_lat": float(min_lat),
                    "max_lat": float(max_lat),
                    "min_lon": float(min_lon),
                    "max_lon": float(max_lon)
                },
                "metadata": {
                    "count": len(risk_zones),
                    "total_cells": total_cells,
                    "resolution": f"{grid_resolution}x{grid_resolution}",
                    "area_km": size_km,
                    "errors": errors,
                    "success_rate": round(success_rate, 1),
                    "workers": max_workers,
                    "generated_at": pd.Timestamp.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Risk map generation failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "location": [lat, lon]
            }

    def get_status(self) -> Dict[str, Any]:
        """
        Get analyzer status and configuration.
        
        Returns:
            Dictionary with status information
        """
        feature_status = self.feature_engineer.get_status()
        
        return {
            "model_loaded": self.model_loaded,
            "feature_engineer": feature_status,
            "ready": self.model_loaded and feature_status.get('ready', False),
            "mode": "production" if self.model_loaded else "mock"
        }

# Convenience function
def create_analyzer(models_dir: Optional[str] = None) -> PreFireAnalyzer:
    """
    Factory function to create PreFireAnalyzer instance.
    
    Args:
        models_dir: Optional path to models directory
        
    Returns:
        Configured PreFireAnalyzer instance
    """
    return PreFireAnalyzer(models_dir=models_dir)

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("Pre-Fire Risk Analyzer Test")
    print("=" * 70)
    
    # 1. Initialize
    print("\n1. Initializing analyzer...")
    analyzer = PreFireAnalyzer()
    
    # Check status
    status = analyzer.get_status()
    print(f"   Status: {status}")
    
    # 2. Test single location
    print("\n2. Testing single location analysis (Kathmandu, Nepal)...")
    result = analyzer.analyze_location(lat=28.3949, lon=84.1240)
    
    if "error" not in result:
        print(f"    Risk Level: {result['risk_level']}")
        print(f"    Probability: {result['probability']:.1%}")
        print(f"    Alert Priority: {result['alert_priority']}")
        print(f"    Model Used: {result.get('model_used', False)}")
    else:
        print(f"    Error: {result['error']}")
    
    # 3. Test risk map generation
    print("\n3. Testing risk map generation (small 3x3 grid)...")
    risk_map = analyzer.generate_risk_map(
        lat=28.3949,
        lon=84.1240,
        size_km=5,
        grid_resolution=3,  # 3x3 = 9 cells (fast for testing)
        max_workers=3,
        show_progress=True
    )
    
    if "error" not in risk_map:
        print(f"    Generated {risk_map['metadata']['count']} risk zones")
        print(f"    Success rate: {risk_map['metadata']['success_rate']}%")
        print(f"    Area covered: {risk_map['metadata']['area_km']} km²")
    else:
        print(f"    Error: {risk_map['error']}")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
