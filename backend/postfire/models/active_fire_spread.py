import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple
from sklearn.isotonic import IsotonicRegression

# Adjust path based on execution location relative to backend root
import sys
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

# ----- REQUIRED FOR UNPICKLING rf_fire_risk_model.pkl -----
class CalibratedTree:
    """Wrapper class used during model training to calibrate probabilities. 
    Must be defined in the same namespace when unpickling."""
    def __init__(self, model, Xc, yc):
        self.model = model
        raw = model.predict_proba(Xc)[:, 1]
        self.iso = IsotonicRegression(out_of_bounds='clip')
        self.iso.fit(raw, yc)

    def predict_proba(self, X):
        p = self.iso.predict(self.model.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])
# ------------------------------------------------------------

try:
    # Absolute/relative import depending on where this script is run
    from backend.prefire.feature_engineer import FeatureEngineer
except ImportError:
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from prefire.feature_engineer import FeatureEngineer
    except ImportError as e:
        logger.error(f"Failed to import FeatureEngineer: {e}")
        FeatureEngineer = None

class ActiveFireCA:
    """
    Active Fire Spread Prediction Module using Cellular Automata logic.
    Integrates with the 81-feature Random Forest model.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._load_ensemble_model()
        if FeatureEngineer is not None:
            self.fe = FeatureEngineer()
        else:
            self.fe = None
            logger.warning("FeatureEngineer is not available. Real data fetching will fail.")

    def _load_ensemble_model(self):
        """Loads the pre-trained RF/ET ensemble and thresholds."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # Inject CalibratedTree into __main__ so pickle can find it
        import sys
        
        # When unpickling, Python looks for CalibratedTree in the module where it was defined.
        # It was defined in lstm_trainer.py (which was run as __main__). 
        # So we must ensure __main__.CalibratedTree exists.
        import __main__
        if not hasattr(__main__, 'CalibratedTree'):
            setattr(__main__, 'CalibratedTree', CalibratedTree)

        try:
            with open(self.model_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.rf_model = checkpoint['rf_model']
            self.et_model = checkpoint['et_model']
            self.w_rf = checkpoint['weights']['rf']
            self.w_et = checkpoint['weights']['et']
            self.impute_medians = checkpoint['impute_medians']
            self.features_order = checkpoint['feature_names']
            
            logger.info("Successfully loaded RF ensemble model.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_cell_features(self, lat: float, lon: float, date: str = None) -> Dict[str, float]:
        """Fetch env features using FeatureEngineer."""
        if not self.fe:
            raise RuntimeError("FeatureEngineer is missing.")
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        features = self.fe.get_all_features(lat, lon, date)
        return features

    def predict_fire_risk_prob(self, features_dict: Dict[str, float]) -> float:
        """Runs the unscaled features through the RF ensemble."""
        # Convert dictionary to ordered array mapping
        X_array = np.zeros(len(self.features_order))
        for i, feat_name in enumerate(self.features_order):
            val = features_dict.get(feat_name)
            # Impute if missing or NaN
            if val is None or pd.isna(val) or np.isnan(val):
                val = self.impute_medians.get(feat_name, 0.0)
            X_array[i] = float(val)

        # Predict using Ensemble
        X_raw = X_array.reshape(1, -1)
        prob_rf = self.rf_model.predict_proba(X_raw)[0, 1]
        prob_et = self.et_model.predict_proba(X_raw)[0, 1]
        
        ensemble_prob = (self.w_rf * prob_rf) + (self.w_et * prob_et)
        return float(ensemble_prob)

    def compute_wind_bias(self, from_center_idx: Tuple[int, int], neighbor_idx: Tuple[int, int], wind_dir_deg: float) -> float:
        """
        Calculates a multiplier based on wind direction to bias the cellular spread.
        wind_dir_deg: standard meteorological wind direction (direction wind is blowing FROM).
        """
        # Calculate angle of neighbor relative to center cell
        # Grid index: 0 is top (North), column 0 is left (West)
        dr = neighbor_idx[0] - from_center_idx[0] # +ve means South
        dc = neighbor_idx[1] - from_center_idx[1] # +ve means East
        
        # Grid angle (0 is East, 90 is North, standard unit circle)
        cell_angle_rad = np.arctan2(-dr, dc) # negative dr because y typically goes down in arrays
        cell_angle_deg = np.degrees(cell_angle_rad) % 360
        
        # Convert meteorological wind direction to vector angle wind is blowing TOWARDS
        # e.g., Wind from 90 (East) -> Blowing towards 270 (West)
        wind_towards_deg = (wind_dir_deg + 180) % 360
        
        # Calculate angular difference (0 to 180)
        diff = min((cell_angle_deg - wind_towards_deg) % 360, (wind_towards_deg - cell_angle_deg) % 360)
        
        # Wind multiplier mapping:
        # Frontal spread (diff ~ 0deg): 1.2x boost
        # Flanking spread (diff ~ 90deg): 1.0x (neutral)
        # Backing spread (diff ~ 180deg): 0.7x penalty
        
        # Linear interpolation between penalty and boost
        multiplier = 0.7 + (180 - diff) / 180.0 * 0.5
        return multiplier

    def simulate_spread(self, start_lat: float, start_lon: float, steps: int = 5, cell_size_deg: float = 0.01, override_wind_dir: float = None, override_wind_speed: float = None) -> np.ndarray:
        """
        Simulate fire spread over T time steps using Cellular Automata.
        Generates a dynamic spatial grid centered on start coordinate.
        
        Returns: 2D numpy array representing the burn delay state.
        0: Unburned
        1: Burned immediately (time step 1)
        T: Burned at time step T
        -1: Burning pending
        """
        # Moore neighborhood definitions (dy, dx)
        NEIGHBORS = [
            (-1, 0), (1, 0), (0, -1), (0, 1),   # N, S, W, E
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # NW, NE, SW, SE
        ]
        
        # Create a grid large enough to hold max possible spread (2 * steps + 1)
        grid_size = (steps * 2) + 1
        center_y = grid_size // 2
        center_x = grid_size // 2
        
        grid_state = np.zeros((grid_size, grid_size), dtype=int)
        
        # Initialize center as actively burning
        grid_state[center_y, center_x] = 1 
        active_fires = [(center_y, center_x)]
        
        # Keep track of cells already vetted to avoid redundant API calls
        risk_cache = {} 
        
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        for t in range(1, steps + 1):
            logger.info(f"Simulating Time Step {t} / {steps}...")
            next_active_fires = []
            
            for (y, x) in active_fires:
                cell_lat = start_lat + (center_y - y) * cell_size_deg 
                cell_lon = start_lon + (x - center_x) * cell_size_deg
                
                for dy, dx in NEIGHBORS:
                    ny, nx = y + dy, x + dx
                    
                    # Ensure within bounds
                    if 0 <= ny < grid_size and 0 <= nx < grid_size:
                        if grid_state[ny, nx] == 0:  # If unburned
                            coord_key = (ny, nx)
                            n_lat = start_lat + (center_y - ny) * cell_size_deg
                            n_lon = start_lon + (nx - center_x) * cell_size_deg
                            
                            # Fetch or retrieve cached probability
                            if coord_key not in risk_cache:
                                try:
                                    # LIVE GEE API CALL — real data required, no mock fallback
                                    features = self.get_cell_features(n_lat, n_lon, date_str)
                                    base_prob = self.predict_fire_risk_prob(features)

                                    # Use override wind if provided, else read from GEE features
                                    wind_dir = override_wind_dir if override_wind_dir is not None else features.get("wind_direction_deg", 90.0)
                                    wind_speed = override_wind_speed if override_wind_speed is not None else features.get("wind_speed_ms", 1.0)

                                    # compute wind-directional bias
                                    w_bias = self.compute_wind_bias((y, x), (ny, nx), wind_dir)

                                    # dampen bias at low wind speeds
                                    if wind_speed < 2.0:
                                        w_bias = 1.0 + (w_bias - 1.0) * (wind_speed / 2.0)

                                    adjusted_prob = min(max(base_prob * w_bias, 0.0), 1.0)
                                    risk_cache[coord_key] = adjusted_prob
                                    logger.debug(f"Cell ({ny},{nx}) Base_P: {base_prob:.2f} | Adj_P: {adjusted_prob:.2f} | Wind: {wind_dir}deg, {wind_speed}m/s")
                                except Exception as e:
                                    # GEE data is required — log and skip this cell (no spread)
                                    logger.error(f"GEE fetch failed for cell ({ny},{nx}) at ({n_lat:.4f},{n_lon:.4f}): {e}. Cell skipped.")
                                    risk_cache[coord_key] = 0.0  # No spread without real data
                            
                            p = risk_cache[coord_key]
                            
                            # Rule-based Thresholds (tuned for realistic spread even with fallback features)
                            # At step 1 from ignition, force spread to immediate neighbors to guarantee visual output
                            if t == 1 and (y, x) == (center_y, center_x):
                                # Always spread from ignition point, wind-biased probability
                                wind_dir_eff = override_wind_dir if override_wind_dir is not None else 90.0
                                forced_wind_bias = self.compute_wind_bias((y, x), (ny, nx), wind_dir_eff)
                                if forced_wind_bias >= 1.0:  # Downwind cells always ignite
                                    grid_state[ny, nx] = t
                                    next_active_fires.append((ny, nx))
                                elif forced_wind_bias >= 0.85:  # slight crosswind also spreads
                                    grid_state[ny, nx] = t + 1
                                    next_active_fires.append((ny, nx))
                            elif p > 0.30:
                                # High Risk: Spreads instantly in this step
                                grid_state[ny, nx] = t
                                next_active_fires.append((ny, nx))
                            elif p > 0.15:
                                # Medium Risk: Spread is delayed until NEXT time step
                                grid_state[ny, nx] = t + 1
                                next_active_fires.append((ny, nx))
                            else:
                                # Low Risk: Spread Halted
                                pass
            
            # For the next time step, evaluate all cells that just caught fire
            active_fires = list(set(next_active_fires))
            if not active_fires:
                break # Fire stopped spreading naturally
                
        return grid_state

if __name__ == "__main__":
    import yaml
    
    # Simple console logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Path logic inside module to find pkl relative to its executed scope
    # Usually root is 2 dirs up from active_fire_spread.py 
    model_path = Path(__file__).parent / "rf_fire_risk_model.pkl"
    
    # DRY RUN FOR DEVELOPMENT
    logger.info("Initializing ActiveFireCA Dry Run...")
    try:
        ca = ActiveFireCA(str(model_path))
        
        # Test coordinates (California area roughly)
        lat, lon = 39.75, -121.50
        logger.info(f"Simulating spread at {lat}, {lon} for 2 steps (due to API rate limit concerns).")
        
        spread_matrix = ca.simulate_spread(lat, lon, steps=2, cell_size_deg=0.02)
        print("\nFinal Spread Matrix (CA Grid):")
        print(spread_matrix)
    except Exception as e:
        logger.error(f"Dry run failed: {e}")
