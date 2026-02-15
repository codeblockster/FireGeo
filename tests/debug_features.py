"""
Debug script to identify the extra feature
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.prefire.feature_engineer import FeatureEngineer

# Expected 81 features based on implementation plan
EXPECTED_FEATURES = [
    # Current weather (13)
    'dewpoint_2m_celsius', 'relative_humidity_pct', 'clear_day_coverage', 'clear_night_coverage',
    'vapor_pressure_deficit_kpa', 'wind_speed_ms', 'wind_direction_deg', 
    'u_wind_component_ms', 'v_wind_component_ms',
    'skin_temperature_celsius', 'soil_temperature_celsius', 'soil_moisture_m3m3', 'precipitation_mm',
    
    # Temperature lags (8)
    'temperature_2m_celsius_lag1', 'temperature_2m_celsius_lag3', 'temperature_2m_celsius_lag7',
    'temperature_2m_celsius_roll7_mean', 'temperature_2m_celsius_roll14_mean',
    'skin_temperature_celsius_lag1', 'skin_temperature_celsius_lag3', 'skin_temperature_celsius_lag7',
    'soil_temperature_celsius_lag1', 'soil_temperature_celsius_lag3', 'soil_temperature_celsius_lag7',
    
    # Soil moisture lags (4)
    'soil_moisture_m3m3_lag1', 'soil_moisture_m3m3_lag3', 'soil_moisture_m3m3_lag7', 'soil_moisture_m3m3_lag14',
    
    # Humidity lags (3)
    'relative_humidity_pct_lag1', 'relative_humidity_pct_lag3', 'relative_humidity_pct_lag7',
    
    # Precipitation (7)
    'precipitation_mm_lag1', 'precipitation_mm_lag5', 'precipitation_mm_lag10', 'precipitation_mm_lag30',
    'precipitation_mm_roll7_sum', 'precipitation_mm_roll14_sum', 'precipitation_mm_roll30_sum',
    
    # VPD (6)
    'vapor_pressure_deficit_kpa_lag1', 'vapor_pressure_deficit_kpa_lag3', 
    'vapor_pressure_deficit_kpa_lag7', 'vapor_pressure_deficit_kpa_lag14',
    'vapor_pressure_deficit_kpa_roll7_mean', 'vapor_pressure_deficit_kpa_roll14_mean',
    
    # Terrain (12)
    'mtpi_min', 'mtpi_mean', 'mtpi_max', 'mtpi_stddev',
    'elevation_min_m', 'elevation_range_m', 'elevation_stddev_m',
    'aspect_mean_deg', 'aspect_stddev_deg',
    'slope_min_deg', 'slope_stddev_deg', 'slope_max_deg',
    
    # LST (2)
    'lst_day_c', 'lst_missing_flag',
    
    # Landsat (10)
    'landsat_ndvi', 'landsat_gndvi', 'landsat_nbr', 'landsat_savi',
    'landsat_ndvi_lag1', 'landsat_ndvi_lag3', 'landsat_ndvi_lag7', 'landsat_ndvi_lag14',
    'landsat_ndvi_roll7_mean', 'landsat_ndvi_roll14_mean',
    
    # Sentinel-2 (14)
    's2_ndvi', 's2_gndvi', 's2_ndsi', 's2_ndwi', 's2_savi', 's2_evi', 's2_cloud_cover_percent',
    's2_ndvi_lag1', 's2_ndvi_lag3', 's2_ndvi_lag7', 's2_ndvi_lag14',
    's2_ndvi_roll7_mean', 's2_ndvi_roll14_mean',
    
    # Quality (1)
    'veg_data_quality'
]

engineer = FeatureEngineer()
features = engineer._get_mock_features()

print(f"Expected: {len(EXPECTED_FEATURES)} features")
print(f"Generated: {len(features)} features")
print(f"Difference: {len(features) - len(EXPECTED_FEATURES)}\n")

# Find extra features
generated_set = set(features.keys())
expected_set = set(EXPECTED_FEATURES)

extra = generated_set - expected_set
missing = expected_set - generated_set

if extra:
    print(f"EXTRA features ({len(extra)}):")
    for f in sorted(extra):
        print(f"  - {f}")

if missing:
    print(f"\nMISSING features ({len(missing)}):")
    for f in sorted(missing):
        print(f"  - {f}")

if not extra and not missing:
    print("✅ All features match!")
