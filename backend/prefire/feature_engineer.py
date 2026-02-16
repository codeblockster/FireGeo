import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import streamlit as st
import os
try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    logging.warning("Earth Engine API not available. Using mock data.")
from datetime import datetime

from backend.src.data_collection.weather_api import WeatherDataFetcher
from backend.src.data_collection.gee_extractor import get_gee_extractor
from backend.prefire.calculations import calculate_vpd, calculate_rolling_mean

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        # Initialize API clients
        weather_api_key = os.getenv('WEATHER_API_KEY')
        self.weather_api = WeatherDataFetcher(api_key=weather_api_key)
        self.gee_client = get_gee_extractor()
        
    def get_all_features(self, lat: float, lon: float, date: str = None) -> Dict[str, Any]:
        """
        Fetch all 81 features required for CatBoost model
        
        Returns:
            Dictionary with 81 feature names matching the model's expected inputs
        """
        features = {}
        
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')

            current = self.weather_api.fetch_current_weather(lat, lon)
            hist_data = self.weather_api.get_historical_weather(lat, lon, days_back=30)
            
            # Helper to get lag values
            def get_lag(data_list, days_ago, default):
                try:
                    if not data_list or len(data_list) == 0:
                        return default
                    idx = -1 * days_ago
                    if abs(idx) <= len(data_list):
                        return float(data_list[idx])
                    return float(data_list[0])
                except Exception:
                    return default
            
            # Helper for rolling sums/means
            def get_rolling_mean(data_list, window, default):
                try:
                    if not data_list or len(data_list) < window:
                        return default
                    return float(np.mean(data_list[-window:]))
                except Exception:
                    return default
            
            def get_rolling_sum(data_list, window, default):
                try:
                    if not data_list or len(data_list) < window:
                        return default
                    return float(np.sum(data_list[-window:]))
                except Exception:
                    return default

            features['dewpoint_2m_celsius'] = current.get('dewpoint', 15.0)
            features['relative_humidity_pct'] = current.get('humidity', 50.0)
            features['clear_day_coverage'] = max(0, 100 - current.get('cloud_cover', 20.0))
            features['clear_night_coverage'] = max(0, 100 - current.get('cloud_cover', 20.0))  # Approximation
            features['vapor_pressure_deficit_kpa'] = calculate_vpd(
                current.get('temp', 25.0), current.get('humidity', 50.0)
            )
            
            # Wind features
            features['wind_speed_ms'] = current.get('wind_speed', 10.0)
            features['wind_direction_deg'] = current.get('wind_direction', 180.0)
            features['u_wind_component_ms'] = current.get('wind_u', 0.0)
            features['v_wind_component_ms'] = current.get('wind_v', 0.0)
            
            # Skin temperature and soil
            features['skin_temperature_celsius'] = current.get('skin_temp', 25.0)
            features['soil_moisture_m3m3'] = current.get('soil_moisture', 0.3)
            
            # Precipitation
            features['precipitation_mm'] = current.get('precip', 0.0)

            temps = hist_data.get('temp_mean', [])
            skin_temps = hist_data.get('skin_temp', [])
            soil_temps = hist_data.get('soil_temp', [])
            
            features['temperature_2m_celsius_lag1'] = get_lag(temps, 1, 25.0)
            features['temperature_2m_celsius_lag3'] = get_lag(temps, 3, 25.0)
            features['temperature_2m_celsius_lag7'] = get_lag(temps, 7, 25.0)
            features['temperature_2m_celsius_roll7_mean'] = get_rolling_mean(temps, 7, 25.0)
            features['temperature_2m_celsius_roll14_mean'] = get_rolling_mean(temps, 14, 25.0)
            
            features['skin_temperature_celsius_lag1'] = get_lag(skin_temps, 1, 25.0)
            features['skin_temperature_celsius_lag3'] = get_lag(skin_temps, 3, 25.0)
            features['skin_temperature_celsius_lag7'] = get_lag(skin_temps, 7, 25.0)
            
            features['soil_temperature_celsius_lag1'] = get_lag(soil_temps, 1, 20.0)
            features['soil_temperature_celsius_lag3'] = get_lag(soil_temps, 3, 20.0)
            features['soil_temperature_celsius_lag7'] = get_lag(soil_temps, 7, 20.0)

            soil_moisture = hist_data.get('soil_moisture', [])
            features['soil_moisture_m3m3_lag1'] = get_lag(soil_moisture, 1, 0.3)
            features['soil_moisture_m3m3_lag3'] = get_lag(soil_moisture, 3, 0.3)
            features['soil_moisture_m3m3_lag7'] = get_lag(soil_moisture, 7, 0.3)
            features['soil_moisture_m3m3_lag14'] = get_lag(soil_moisture, 14, 0.3)

            hums = hist_data.get('humidity', [])
            features['relative_humidity_pct_lag1'] = get_lag(hums, 1, 50.0)
            features['relative_humidity_pct_lag3'] = get_lag(hums, 3, 50.0)
            features['relative_humidity_pct_lag7'] = get_lag(hums, 7, 50.0)

            precips = hist_data.get('precip', [])
            features['precipitation_mm_lag1'] = get_lag(precips, 1, 0.0)
            features['precipitation_mm_lag5'] = get_lag(precips, 5, 0.0)
            features['precipitation_mm_lag10'] = get_lag(precips, 10, 0.0)
            features['precipitation_mm_lag30'] = get_lag(precips, 30, 0.0)
            features['precipitation_mm_roll7_sum'] = get_rolling_sum(precips, 7, 0.0)
            features['precipitation_mm_roll14_sum'] = get_rolling_sum(precips, 14, 0.0)
            features['precipitation_mm_roll30_sum'] = get_rolling_sum(precips, 30, 0.0)

            vpds = hist_data.get('vpd', [])
            features['vapor_pressure_deficit_kpa_lag1'] = get_lag(vpds, 1, 1.0)
            features['vapor_pressure_deficit_kpa_lag3'] = get_lag(vpds, 3, 1.0)
            features['vapor_pressure_deficit_kpa_lag7'] = get_lag(vpds, 7, 1.0)
            features['vapor_pressure_deficit_kpa_lag14'] = get_lag(vpds, 14, 1.0)
            features['vapor_pressure_deficit_kpa_roll7_mean'] = get_rolling_mean(vpds, 7, 1.0)
            features['vapor_pressure_deficit_kpa_roll14_mean'] = get_rolling_mean(vpds, 14, 1.0)

            if EE_AVAILABLE and not self.gee_client.is_mock_mode:
                point = ee.Geometry.Point([lon, lat])
                
                # TERRAIN METRICS
                terrain = self.gee_client.get_terrain_metrics(point)
                features['mtpi_min'] = terrain.get('mtpi_min', -10.0)
                features['mtpi_mean'] = terrain.get('mtpi_mean', 0.0)
                features['mtpi_max'] = terrain.get('mtpi_max', 10.0)
                features['mtpi_stddev'] = terrain.get('mtpi_stddev', 5.0)
                features['elevation_min_m'] = terrain.get('elevation_min', 0.0)
                features['elevation_range_m'] = terrain.get('elevation_range', 500.0)
                features['elevation_stddev_m'] = terrain.get('elevation_stddev', 100.0)
                features['aspect_mean_deg'] = terrain.get('aspect_mean', 180.0)
                features['aspect_stddev_deg'] = terrain.get('aspect_stddev', 45.0)
                features['slope_min_deg'] = terrain.get('slope_min', 0.0)
                features['slope_stddev_deg'] = terrain.get('slope_stddev', 5.0)
                features['slope_max_deg'] = terrain.get('slope_max', 15.0)
                
                # LST
                features['lst_day_c'] = self.gee_client.get_modis_lst(point, date)
                features['lst_missing_flag'] = 0.0  # 0 = data available, 1 = missing
                
                # LANDSAT INDICES
                landsat = self.gee_client.get_landsat_indices(point, date)
                features['landsat_ndvi'] = landsat.get('landsat_ndvi', 0.5)
                features['landsat_gndvi'] = landsat.get('landsat_gndvi', 0.5)
                features['landsat_nbr'] = landsat.get('landsat_nbr', 0.3)
                features['landsat_savi'] = self.gee_client.get_landsat_savi(point, date)
                
                # For lags and rolling means, we'd need historical satellite data
                # For now, use current values as approximation
                features['landsat_ndvi_lag1'] = landsat.get('landsat_ndvi', 0.5)
                features['landsat_ndvi_lag3'] = landsat.get('landsat_ndvi', 0.5)
                features['landsat_ndvi_lag7'] = landsat.get('landsat_ndvi', 0.5)
                features['landsat_ndvi_lag14'] = landsat.get('landsat_ndvi', 0.5)
                features['landsat_ndvi_roll7_mean'] = landsat.get('landsat_ndvi', 0.5)
                features['landsat_ndvi_roll14_mean'] = landsat.get('landsat_ndvi', 0.5)
                
                # SENTINEL-2 INDICES
                s2 = self.gee_client.get_sentinel2_indices(point, date)
                features['s2_ndvi'] = s2.get('s2_ndvi', 0.5)
                features['s2_gndvi'] = s2.get('s2_gndvi', 0.5)
                features['s2_ndsi'] = s2.get('s2_ndsi', 0.0)
                features['s2_ndwi'] = s2.get('s2_ndwi', 0.0)
                features['s2_savi'] = s2.get('s2_savi', 0.3)
                features['s2_evi'] = s2.get('s2_evi', 0.3)
                features['s2_cloud_cover_percent'] = s2.get('s2_cloud_cover_percent', 20.0)
                
                # S2 lags (approximation with current values)
                features['s2_ndvi_lag1'] = s2.get('s2_ndvi', 0.5)
                features['s2_ndvi_lag3'] = s2.get('s2_ndvi', 0.5)
                features['s2_ndvi_lag7'] = s2.get('s2_ndvi', 0.5)
                features['s2_ndvi_lag14'] = s2.get('s2_ndvi', 0.5)
                features['s2_ndvi_roll7_mean'] = s2.get('s2_ndvi', 0.5)
                features['s2_ndvi_roll14_mean'] = s2.get('s2_ndvi', 0.5)
                
                # QUALITY FLAGS
                features['veg_data_quality'] = 1.0  # 1 = good quality, 0 = poor
                
            else:
                # Fallback values if GEE is not available
                features.update(self._get_mock_gee_features())
            
            logger.info(f"Generated {len(features)} features for ({lat}, {lon})")
            return features
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}", exc_info=True)
            logger.warning("Returning mock features due to error")
            return self._get_mock_features()

    def _get_mock_gee_features(self):
        """Mock GEE features when GEE is unavailable"""
        return {
            'mtpi_min': -10.0,
            'mtpi_mean': 0.0,
            'mtpi_max': 10.0,
            'mtpi_stddev': 5.0,
            'elevation_min_m': 0.0,
            'elevation_range_m': 500.0,
            'elevation_stddev_m': 100.0,
            'aspect_mean_deg': 180.0,
            'aspect_stddev_deg': 45.0,
            'slope_min_deg': 0.0,
            'slope_stddev_deg': 5.0,
            'slope_max_deg': 15.0,
            'lst_day_c': 25.0,
            'lst_missing_flag': 0.0,
            'landsat_ndvi': 0.5,
            'landsat_gndvi': 0.5,
            'landsat_nbr': 0.3,
            'landsat_savi': 0.3,
            'landsat_ndvi_lag1': 0.5,
            'landsat_ndvi_lag3': 0.5,
            'landsat_ndvi_lag7': 0.5,
            'landsat_ndvi_lag14': 0.5,
            'landsat_ndvi_roll7_mean': 0.5,
            'landsat_ndvi_roll14_mean': 0.5,
            's2_ndvi': 0.5,
            's2_gndvi': 0.5,
            's2_ndsi': 0.0,
            's2_ndwi': 0.0,
            's2_savi': 0.3,
            's2_evi': 0.3,
            's2_cloud_cover_percent': 20.0,
            's2_ndvi_lag1': 0.5,
            's2_ndvi_lag3': 0.5,
            's2_ndvi_lag7': 0.5,
            's2_ndvi_lag14': 0.5,
            's2_ndvi_roll7_mean': 0.5,
            's2_ndvi_roll14_mean': 0.5,
            'veg_data_quality': 1.0
        }

    def _get_mock_features(self):
        """Fallback mock features for all 81 features"""
        return {
            # Current weather
            'dewpoint_2m_celsius': 15.0,
            'relative_humidity_pct': 50.0,
            'clear_day_coverage': 80.0,
            'clear_night_coverage': 80.0,
            'vapor_pressure_deficit_kpa': 1.2,
            'wind_speed_ms': 10.0,
            'wind_direction_deg': 180.0,
            'u_wind_component_ms': 0.0,
            'v_wind_component_ms': 0.0,
            'skin_temperature_celsius': 25.0,
            'soil_moisture_m3m3': 0.3,
            'precipitation_mm': 0.0,
            
            # Temperature lags
            'temperature_2m_celsius_lag1': 25.0,
            'temperature_2m_celsius_lag3': 25.0,
            'temperature_2m_celsius_lag7': 25.0,
            'temperature_2m_celsius_roll7_mean': 25.0,
            'temperature_2m_celsius_roll14_mean': 25.0,
            'skin_temperature_celsius_lag1': 25.0,
            'skin_temperature_celsius_lag3': 25.0,
            'skin_temperature_celsius_lag7': 25.0,
            'soil_temperature_celsius_lag1': 20.0,
            'soil_temperature_celsius_lag3': 20.0,
            'soil_temperature_celsius_lag7': 20.0,
            
            # Soil moisture lags
            'soil_moisture_m3m3_lag1': 0.3,
            'soil_moisture_m3m3_lag3': 0.3,
            'soil_moisture_m3m3_lag7': 0.3,
            'soil_moisture_m3m3_lag14': 0.3,
            
            # Humidity lags
            'relative_humidity_pct_lag1': 50.0,
            'relative_humidity_pct_lag3': 50.0,
            'relative_humidity_pct_lag7': 50.0,
            
            # Precipitation lags and sums
            'precipitation_mm_lag1': 0.0,
            'precipitation_mm_lag5': 0.0,
            'precipitation_mm_lag10': 0.0,
            'precipitation_mm_lag30': 0.0,
            'precipitation_mm_roll7_sum': 0.0,
            'precipitation_mm_roll14_sum': 0.0,
            'precipitation_mm_roll30_sum': 0.0,
            
            # VPD lags and means
            'vapor_pressure_deficit_kpa_lag1': 1.2,
            'vapor_pressure_deficit_kpa_lag3': 1.2,
            'vapor_pressure_deficit_kpa_lag7': 1.2,
            'vapor_pressure_deficit_kpa_lag14': 1.2,
            'vapor_pressure_deficit_kpa_roll7_mean': 1.2,
            'vapor_pressure_deficit_kpa_roll14_mean': 1.2,
            
            # Terrain
            'mtpi_min': -10.0,
            'mtpi_mean': 0.0,
            'mtpi_max': 10.0,
            'mtpi_stddev': 5.0,
            'elevation_min_m': 0.0,
            'elevation_range_m': 500.0,
            'elevation_stddev_m': 100.0,
            'aspect_mean_deg': 180.0,
            'aspect_stddev_deg': 45.0,
            'slope_min_deg': 0.0,
            'slope_stddev_deg': 5.0,
            'slope_max_deg': 15.0,
            
            # LST
            'lst_day_c': 25.0,
            'lst_missing_flag': 0.0,
            
            # Landsat
            'landsat_ndvi': 0.5,
            'landsat_gndvi': 0.5,
            'landsat_nbr': 0.3,
            'landsat_savi': 0.3,
            'landsat_ndvi_lag1': 0.5,
            'landsat_ndvi_lag3': 0.5,
            'landsat_ndvi_lag7': 0.5,
            'landsat_ndvi_lag14': 0.5,
            'landsat_ndvi_roll7_mean': 0.5,
            'landsat_ndvi_roll14_mean': 0.5,
            
            # Sentinel-2
            's2_ndvi': 0.5,
            's2_gndvi': 0.5,
            's2_ndsi': 0.0,
            's2_ndwi': 0.0,
            's2_savi': 0.3,
            's2_evi': 0.3,
            's2_cloud_cover_percent': 20.0,
            's2_ndvi_lag1': 0.5,
            's2_ndvi_lag3': 0.5,
            's2_ndvi_lag7': 0.5,
            's2_ndvi_lag14': 0.5,
            's2_ndvi_roll7_mean': 0.5,
            's2_ndvi_roll14_mean': 0.5,
            
            # Quality
            'veg_data_quality': 1.0
        }
