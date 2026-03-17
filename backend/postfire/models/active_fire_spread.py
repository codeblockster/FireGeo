import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.isotonic import IsotonicRegression
import random

import sys
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

# ----- REQUIRED FOR UNPICKLING rf_fire_risk_model.pkl -----
class CalibratedTree:
    """Wrapper class used during model training to calibrate probabilities."""
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
    from backend.prefire.feature_engineer import FeatureEngineer
except ImportError:
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from prefire.feature_engineer import FeatureEngineer
    except ImportError as e:
        logger.error(f"Failed to import FeatureEngineer: {e}")
        FeatureEngineer = None


# ─── Grid Resolution: ~0.1 km × 0.1 km ─────────────────────────────────────
# 1 degree latitude ≈ 111 km → 0.1 km ≈ 0.0009 degrees
# 1 degree longitude ≈ 111 km * cos(lat); at lat=39° cos≈0.777 → 0.1km ≈ 0.00116 deg
# We use a uniform 0.0009° for lat and adjust lon per latitude below.
CELL_SIZE_LAT_DEG = 0.0045   # ≈ 500 m north-south
# Lon cell size is computed dynamically in simulate_spread to stay ~100 m east-west


class ActiveFireCA:
    """
    Active Fire Spread Prediction Module using Cellular Automata logic.
    Integrates with the 81-feature Random Forest model.

    Key improvements over v1:
    - Each grid cell is visited only ONCE (deduplication via a `visited` set).
    - Burn time is set atomically; later steps cannot overwrite an earlier assignment.
    - Wind bias is applied consistently; no special-case forced-spread branch that
      could double-assign cells.
    - `build_map_layers()` returns GeoJSON-ready feature collections with RGBA
      colours (transparent red / orange / yellow) for direct Leaflet / Mapbox use.
    - Grid resolution reduced to ≈ 0.1 km × 0.1 km.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._load_ensemble_model()
        if FeatureEngineer is not None:
            self.fe = FeatureEngineer()
        else:
            self.fe = None
            logger.warning("FeatureEngineer unavailable – real GEE fetching will fail.")

    # ── Model Loading ────────────────────────────────────────────────────────

    def _load_ensemble_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        import __main__
        if not hasattr(__main__, 'CalibratedTree'):
            setattr(__main__, 'CalibratedTree', CalibratedTree)

        try:
            with open(self.model_path, 'rb') as f:
                checkpoint = pickle.load(f)

            self.rf_model       = checkpoint['rf_model']
            self.et_model       = checkpoint['et_model']
            self.w_rf           = checkpoint['weights']['rf']
            self.w_et           = checkpoint['weights']['et']
            self.impute_medians = checkpoint['impute_medians']
            self.features_order = checkpoint['feature_names']

            logger.info("Successfully loaded RF ensemble model.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    # ── Feature Fetching ─────────────────────────────────────────────────────

    def get_cell_features(self, lat: float, lon: float, date: str = None) -> Dict[str, float]:
        if not self.fe:
            raise RuntimeError("FeatureEngineer is missing.")
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        features = self.fe.get_all_features(lat, lon, date)

        if features is None or not features:
            raise RuntimeError(
                f"Google Earth Engine returned no data for ({lat}, {lon}). "
                "Verify GEE authentication."
            )
        if len(features) < 50:
            raise RuntimeError(
                f"Incomplete feature data from GEE: got {len(features)}, expected ~81."
            )
        return features

    # ── Risk Prediction ──────────────────────────────────────────────────────

    def predict_fire_risk_prob(self, features_dict: Dict[str, float]) -> float:
        X_array = np.zeros(len(self.features_order))
        for i, feat_name in enumerate(self.features_order):
            val = features_dict.get(feat_name)
            if val is None or pd.isna(val) or np.isnan(float(val)):
                val = self.impute_medians.get(feat_name, 0.0)
            X_array[i] = float(val)

        X_raw = X_array.reshape(1, -1)
        try:
            prob_rf = self.rf_model.predict_proba(X_raw)[0, 1]
            prob_et = self.et_model.predict_proba(X_raw)[0, 1]
            return float(self.w_rf * prob_rf + self.w_et * prob_et)
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}. Using default 0.35.")
            return 0.35

    # ── Wind Bias ────────────────────────────────────────────────────────────

    def compute_wind_bias(
        self,
        from_center_idx: Tuple[int, int],
        neighbor_idx: Tuple[int, int],
        wind_dir_deg: float,
    ) -> float:
        """
        Returns a spread multiplier [0.7 – 1.2] based on whether the neighbour
        is downwind (1.2×) or upwind (0.7×) of the burning cell.

        wind_dir_deg: meteorological convention – direction wind is blowing FROM.
        """
        dr = neighbor_idx[0] - from_center_idx[0]   # +ve → South in array coords
        dc = neighbor_idx[1] - from_center_idx[1]   # +ve → East

        cell_angle_deg = np.degrees(np.arctan2(-dr, dc)) % 360

        # Direction the wind is blowing TOWARDS
        wind_towards_deg = (wind_dir_deg + 180) % 360

        diff = min(
            (cell_angle_deg - wind_towards_deg) % 360,
            (wind_towards_deg - cell_angle_deg) % 360,
        )

        # diff=0  → full downwind → 1.2×
        # diff=90 → crosswind    → 0.95×
        # diff=180→ upwind       → 0.7×
        multiplier = 0.7 + (180 - diff) / 180.0 * 0.5
        return multiplier

    # ── Cellular Automata Simulation ─────────────────────────────────────────

    def simulate_spread(
        self,
        start_lat: float,
        start_lon: float,
        steps: int = 5,
        override_wind_dir: Optional[float] = None,
        override_wind_speed: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """
        Simulate fire spread over *steps* time steps.

        Resolution: ~0.1 km × 0.1 km per cell.

        Returns
        -------
        grid_state : 2-D int array
            0  = unburned
            t  = burned at time step t  (1 … steps)
        cell_size_lat, cell_size_lon : cell dimensions in degrees
        origin_lat, origin_lon       : top-left corner of the grid
        """

        # ── Dynamic lon cell size to keep cells square at this latitude ──────
        cell_size_lat = CELL_SIZE_LAT_DEG
        cell_size_lon = cell_size_lat / np.cos(np.radians(abs(start_lat)))

        # Moore neighbourhood offsets (dy, dx)
        NEIGHBORS = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

        grid_size = (steps * 2) + 1
        center_y  = grid_size // 2
        center_x  = grid_size // 2

        grid_state = np.zeros((grid_size, grid_size), dtype=int)

        # ── Single source of truth: `visited` prevents any double-assignment ──
        visited: set = set()

        # Mark ignition cell
        grid_state[center_y, center_x] = 1
        visited.add((center_y, center_x))
        active_fires = [(center_y, center_x)]

        # Pre-compute grid's top-left lat/lon origin for mapping
        origin_lat = start_lat + center_y * cell_size_lat   # top row lat
        origin_lon = start_lon - center_x * cell_size_lon   # left col lon

        risk_cache: Dict[Tuple[int, int], float] = {}
        date_str = datetime.now().strftime('%Y-%m-%d')

        for t in range(1, steps + 1):
            logger.info(f"Time step {t}/{steps} | Active fire fronts: {len(active_fires)}")

            # Collect candidates for this time step (deduplicated within the step too)
            candidates_this_step: Dict[Tuple[int, int], int] = {}  # (ny,nx) → burn_time

            # 1. Gather all unburned neighbors and their burning sources
            target_candidates: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
            for (y, x) in active_fires:
                for dy, dx in NEIGHBORS:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < grid_size and 0 <= nx < grid_size:
                        if (ny, nx) not in visited:
                            coord_key = (ny, nx)
                            if coord_key not in target_candidates:
                                target_candidates[coord_key] = []
                            target_candidates[coord_key].append((y, x))

            # 2. Evaluate each target cell ONCE using the best wind bias source
            for coord_key, sources in target_candidates.items():
                ny, nx = coord_key
                n_lat = start_lat + (center_y - ny) * cell_size_lat
                n_lon = start_lon + (nx - center_x) * cell_size_lon

                # ── Risk probability (cached per grid cell) ──────────────
                if coord_key not in risk_cache:
                    try:
                        features = self.get_cell_features(n_lat, n_lon, date_str)
                        base_prob = self.predict_fire_risk_prob(features)

                        wind_dir   = override_wind_dir   if override_wind_dir   is not None else features.get("wind_direction_deg", 90.0)
                        wind_speed = override_wind_speed if override_wind_speed is not None else features.get("wind_speed_ms", 1.0)

                        risk_cache[coord_key] = {"base": base_prob, "wind_dir": wind_dir, "wind_speed": wind_speed}
                    except Exception as e:
                        logger.error(f"GEE fetch failed for ({ny},{nx}): {e}")
                        raise RuntimeError(f"GEE feature fetch failed: {e}")

                cached = risk_cache[coord_key]
                
                # Find maximum probability across all firing sources mapping to this cell
                max_p = 0.0
                for (y, x) in sources:
                    w_bias = self.compute_wind_bias((y, x), (ny, nx), cached["wind_dir"])
                    if cached["wind_speed"] < 2.0:
                        w_bias = 1.0 + (w_bias - 1.0) * (cached["wind_speed"] / 2.0)
                    
                    p = float(np.clip(cached["base"] * w_bias, 0.0, 1.0))
                    if p > max_p:
                        max_p = p

                # ── Spread rules (evaluate maximum potential) ─
                is_center_source = (center_y, center_x) in sources
                
                if t == 1 and is_center_source:
                    # Always spread from ignition point, wind-biased probability
                    wind_dir_eff = override_wind_dir if override_wind_dir is not None else cached["wind_dir"]
                    forced_wind_bias = self.compute_wind_bias((center_y, center_x), coord_key, wind_dir_eff)
                    if forced_wind_bias >= 1.0:  # Downwind cells always ignite
                        candidates_this_step[coord_key] = t
                    elif forced_wind_bias >= 0.85:  # slight crosswind also spreads
                        candidates_this_step[coord_key] = t + 1
                elif max_p > 0.30:
                    # High Risk: Spreads instantly in this step
                    candidates_this_step[coord_key] = t
                elif max_p > 0.15:
                    # Medium Risk: Spread is delayed until NEXT time step
                    candidates_this_step[coord_key] = t + 1
                elif max_p > 0.05:
                    # Lower risk -> delayed by 2 steps
                    candidates_this_step[coord_key] = t + 2

            # ── Commit candidates ────────────────────────────────────────────
            next_active = []
            for (ny, nx), burn_time in candidates_this_step.items():
                if (ny, nx) not in visited:          # final guard
                    grid_state[ny, nx] = burn_time
                    visited.add((ny, nx))
                    next_active.append((ny, nx))

            active_fires = next_active
            if not active_fires:
                logger.info("Fire stopped spreading naturally.")
                break

        return grid_state, cell_size_lat, cell_size_lon, origin_lat, origin_lon

    # ── Map Layer Builder ────────────────────────────────────────────────────

    def build_map_layers(
        self,
        grid_state: np.ndarray,
        cell_size_lat: float,
        cell_size_lon: float,
        origin_lat: float,
        origin_lon: float,
        steps: int,
    ) -> Dict:
        """
        Convert a burn-time grid into a GeoJSON FeatureCollection suitable for
        Leaflet / Mapbox rendering.

        Colour scheme (all semi-transparent):
        - Step 1 (fastest / most intense) : rgba(220, 38,  38,  0.75)  deep red
        - Steps 2–3                        : rgba(249, 115,  22, 0.65)  orange
        - Steps 4+                         : rgba(250, 204,  21, 0.50)  yellow
        - Ignition cell                    : rgba(127,  29,  29, 0.90)  dark red

        No cell is ever duplicated in the output – each grid position appears
        in at most one feature.
        """

        def _rgba_for_step(burn_time: int, max_step: int) -> str:
            if burn_time == 1 and burn_time == max_step:
                # Sole ignition point
                return "rgba(127, 29, 29, 0.90)"
            if burn_time <= 1:
                return "rgba(220, 38, 38, 0.75)"
            ratio = (burn_time - 1) / max(max_step - 1, 1)
            if ratio < 0.4:
                return "rgba(249, 115, 22, 0.65)"
            return "rgba(250, 204, 21, 0.50)"

        rows, cols = grid_state.shape
        max_step   = int(grid_state.max()) or 1

        features = []
        seen: set = set()  # Deduplicate output cells

        for r in range(rows):
            for c in range(cols):
                burn_time = int(grid_state[r, c])
                if burn_time == 0:
                    continue
                if (r, c) in seen:
                    continue
                seen.add((r, c))

                # Cell centre coordinates
                cell_lat = origin_lat - r * cell_size_lat
                cell_lon = origin_lon + c * cell_size_lon

                # Cell bounding box (GeoJSON Polygon, CCW)
                half_lat = cell_size_lat / 2
                half_lon = cell_size_lon / 2
                coords = [[
                    [cell_lon - half_lon, cell_lat + half_lat],  # NW
                    [cell_lon + half_lon, cell_lat + half_lat],  # NE
                    [cell_lon + half_lon, cell_lat - half_lat],  # SE
                    [cell_lon - half_lon, cell_lat - half_lat],  # SW
                    [cell_lon - half_lon, cell_lat + half_lat],  # close
                ]]

                colour = _rgba_for_step(burn_time, max_step)

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coords,
                    },
                    "properties": {
                        "burn_step":   burn_time,
                        "fill_color":  colour,
                        "fill_opacity": 0.75 if burn_time <= 1 else (0.65 if burn_time <= 3 else 0.50),
                        "stroke":      False,
                    },
                })

        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_burned_cells":  len(features),
                "max_spread_step":     max_step,
                "cell_size_lat_deg":   cell_size_lat,
                "cell_size_lon_deg":   cell_size_lon,
                "approx_cell_size_m":  round(cell_size_lat * 111000),
            },
        }

    # ── Convenience: run simulation + build map in one call ─────────────────

    def run(
        self,
        start_lat: float,
        start_lon: float,
        steps: int = 5,
        override_wind_dir: Optional[float] = None,
        override_wind_speed: Optional[float] = None,
    ) -> Dict:
        """
        Full pipeline: simulate spread → build GeoJSON map layers.

        Returns a dict with:
          'grid'    : raw burn-time numpy array
          'geojson' : GeoJSON FeatureCollection for map rendering
        """
        grid, csl, cslon, orig_lat, orig_lon = self.simulate_spread(
            start_lat, start_lon, steps,
            override_wind_dir=override_wind_dir,
            override_wind_speed=override_wind_speed,
        )

        geojson = self.build_map_layers(grid, csl, cslon, orig_lat, orig_lon, steps)

        logger.info(
            f"Simulation complete. Burned cells: {geojson['metadata']['total_burned_cells']} | "
            f"Cell size: ~{geojson['metadata']['approx_cell_size_m']} m"
        )

        return {"grid": grid, "geojson": geojson}


# ── CLI / Dry-run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Use config path for model
    try:
        from backend.config import ACTIVE_FIRE_CA_MODEL
        model_path = ACTIVE_FIRE_CA_MODEL
    except ImportError:
        model_path = Path(__file__).parent / "models" / "rf_fire_risk_model.pkl"

    logger.info("Initializing ActiveFireCA …")
    try:
        ca = ActiveFireCA(str(model_path))

        lat, lon = 39.75, -121.50
        logger.info(f"Simulating spread at ({lat}, {lon}), 3 steps, ~0.1 km cells.")

        result = ca.run(lat, lon, steps=3, override_wind_dir=270, override_wind_speed=4.0)

        print("\n── Burn-time grid ──")
        print(result["grid"])

        print("\n── GeoJSON metadata ──")
        print(json.dumps(result["geojson"]["metadata"], indent=2))

        print(f"\nTotal GeoJSON features: {len(result['geojson']['features'])}")

        # Optional: dump full GeoJSON for QGIS / Leaflet inspection
        out_path = Path(__file__).parent / "fire_spread_output.geojson"
        with open(out_path, "w") as f:
            json.dump(result["geojson"], f)
        logger.info(f"GeoJSON saved to {out_path}")

    except Exception as e:
        logger.error(f"Dry run failed: {e}")