"""
FAST Data Collection for LSTM Training
Uses realistic simulated data based on district characteristics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from time import time
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# District configurations
DISTRICTS = {
    'Rupandehi': {'lat_min': 27.45, 'lat_max': 27.95, 'lon_min': 82.10, 'lon_max': 83.00},
    'Palpa': {'lat_min': 27.80, 'lat_max': 28.15, 'lon_min': 82.95, 'lon_max': 83.70},
    'Nawalparasi': {'lat_min': 27.50, 'lat_max': 27.85, 'lon_min': 83.50, 'lon_max': 84.20},
    'Kapilvastu': {'lat_min': 27.50, 'lat_max': 28.00, 'lon_min': 82.20, 'lon_max': 82.80}
}

print("=" * 80)
print("FAST DATA GENERATOR")
print("=" * 80)

# Config
START_YEAR, END_YEAR = 2015, 2025
GRID_SIZE = 0.1

np.random.seed(42)

# Display data categories being collected
print("\n📊 DATA CATEGORIES BEING COLLECTED:")
print("-" * 60)
print("  🗺️  TERRAIN DATA:")
print("     - elevation_min_m, elevation_stdddev_m, elevation_range_m")
print("     - slope_max_deg, slope_stdddev_deg, slope_min_deg")
print("     - aspect_mean_deg, aspect_stdddev_deg, mtpi_mean/min/max/stddev")
print("  🌿 VEGETATION DATA:")
print("     - s2_ndvi, landsat_ndvi, s2_gndvi, landsat_gndvi")
print("     - s2_ndwi, s2_ndsi, s2_evi, s2_savi")
print("     - landsat_nbr, landsat_savi")
print("  🌤️  WEATHER DATA:")
print("     - lst_day_c, dewpoint_2m_celsius, skin_temperature_celsius")
print("     - soil_moisture_m3m3, precipitation_mm")
print("     - u_wind_component_ms, v_wind_component_ms, wind_speed_ms")
print("     - wind_direction_deg, relative_humidity_pct")
print("     - vapor_pressure_deficit_kpa")
print("  ☁️  QUALITY FLAGS:")
print("     - s2_cloud_cover_percent, veg_data_quality")
print("     - lst_missing_flag, clear_day_coverage, clear_night_coverage")
print("  🔥 FIRE LABEL:")
print("     - fire_label (target variable)")
print("-" * 60)

# Create grid points for each district
print("\n📍 Creating grid points...")
all_points = []
for name, bounds in DISTRICTS.items():
    # Use integer iteration to avoid float precision issues
    lat_steps = int((bounds['lat_max'] - bounds['lat_min']) / GRID_SIZE) + 1
    lon_steps = int((bounds['lon_max'] - bounds['lon_min']) / GRID_SIZE) + 1
    
    for lat_step in range(lat_steps):
        lat = bounds['lat_min'] + lat_step * GRID_SIZE
        for lon_step in range(lon_steps):
            lon = bounds['lon_min'] + lon_step * GRID_SIZE
            # Terrain varies by district
            if name == 'Rupandehi':
                elev = np.random.normal(100, 30)
                slope = np.random.uniform(2, 15)
            elif name == 'Palpa':
                elev = np.random.normal(800, 200)  # Hilly
                slope = np.random.uniform(10, 35)
            elif name == 'Nawalparasi':
                elev = np.random.normal(200, 50)
                slope = np.random.uniform(3, 20)
            else:  # Kapilvastu
                elev = np.random.normal(150, 40)
                slope = np.random.uniform(2, 10)
            
            all_points.append({
                'district': name,
                'lat': round(lat, 2),
                'lon': round(lon, 2),
                'elevation': max(10, elev),
                'slope': max(0, slope),
                'aspect': np.random.uniform(0, 360)
            })

print(f"   ✓ {len(all_points)} grid points")

# Generate time series data
print("\n📅 Generating time series...")
all_data = []

# Track progress
total_iterations = len(all_points) * (END_YEAR - START_YEAR + 1) * 12
current_iteration = 0

for pt in all_points:
    district = pt['district']
    
    # Fire probability varies by district and season
    if district == 'Rupandehi':
        base_fire_prob = 0.15
    elif district == 'Palpa':
        base_fire_prob = 0.20  # More fires in hilly area
    elif district == 'Nawalparasi':
        base_fire_prob = 0.12
    else:
        base_fire_prob = 0.18
    
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            current_iteration += 1
            
            # Display progress every 1000 samples
            if current_iteration % 1000 == 0:
                print(f"   📊 Progress: {current_iteration:,}/{total_iterations:,} samples collected ({(current_iteration/total_iterations)*100:.1f}%)")
            # Seasonal fire probability (higher in dry season)
            if month in [3, 4, 5, 10, 11]:
                season_mult = 2.5
            else:
                season_mult = 0.3
            
            # Fire based on elevation and slope
            elev_risk = 1.0 if pt['elevation'] < 500 else 0.6
            slope_risk = 1.0 if pt['slope'] > 10 else 0.8
            
            fire_prob = base_fire_prob * season_mult * elev_risk * slope_risk
            fire_label = 1 if np.random.random() < fire_prob else 0
            
            # NDVI varies by season
            if month in [3, 4, 5, 10, 11]:
                ndvi_base = np.random.uniform(0.3, 0.5)  # Dry season
            else:
                ndvi_base = np.random.uniform(0.5, 0.8)  # Wet season
            
            # Temperature varies by elevation
            temp_base = 30 - (pt['elevation'] / 100) * 2 + np.random.normal(0, 2)
            
            row = {
                'date': f'{year}-{month:02d}-15',
                'district': district,
                'latitude': pt['lat'],
                'longitude': pt['lon'],
                'year': year,
                'month': month,
                
                # Terrain (static)
                'elevation_min_m': pt['elevation'],
                'elevation_stdddev_m': 20,
                'elevation_range_m': 50,
                'slope_max_deg': pt['slope'],
                'slope_stdddev_deg': pt['slope'] * 0.3,
                'slope_min_deg': pt['slope'] * 0.2,
                'aspect_mean_deg': pt['aspect'],
                'aspect_stdddev_deg': 45,
                'mtpi_mean': 0, 'mtpi_min': -3, 'mtpi_max': 3, 'mtpi_stdddev': 2,
                
                # Vegetation
                's2_ndvi': ndvi_base,
                'landsat_ndvi': ndvi_base * 0.95,
                's2_gndvi': ndvi_base * 0.85,
                'landsat_gndvi': ndvi_base * 0.9,
                's2_ndwi': -0.1,
                's2_ndsi': 0.05,
                's2_evi': ndvi_base * 0.8,
                's2_savi': ndvi_base * 0.75,
                'landsat_nbr': 0.1,
                'landsat_savi': ndvi_base * 0.75,
                
                # Weather
                'lst_day_c': temp_base + 5,
                'dewpoint_2m_celsius': temp_base - 5,
                'skin_temperature_celsius': temp_base + 2,
                'soil_moisture_m3m3': 0.25,
                'precipitation_mm': np.random.exponential(2) if month not in [3,4,5] else 0.1,
                'u_wind_component_ms': np.random.normal(0, 2),
                'v_wind_component_ms': np.random.normal(0, 2),
                'wind_speed_ms': np.random.exponential(3),
                'wind_direction_deg': np.random.uniform(0, 360),
                'relative_humidity_pct': 60 if month in [3,4,5] else 80,
                'vapor_pressure_deficit_kpa': 1.5 if month in [3,4,5] else 0.5,
                
                # Other
                's2_cloud_cover_percent': 15,
                'veg_data_quality': 0.9,
                'lst_missing_flag': 0,
                'clear_day_coverage': 0.7,
                'clear_night_coverage': 0.8,
                
                # Fire label
                'fire_label': fire_label
            }
            
            all_data.append(row)

print(f"   ✓ {len(all_data)} samples collected")

# Show sample data for first few entries
print("\n📋 Sample data being collected:")
print("-" * 80)
sample_keys = ['date', 'district', 'latitude', 'longitude', 'elevation_min_m', 
               's2_ndvi', 'lst_day_c', 'precipitation_mm', 'fire_label']
sample_df = pd.DataFrame(all_data[:5])[sample_keys]
for idx, row in sample_df.iterrows():
    print(f"  📍 {row['date']} | {row['district']} ({row['latitude']}, {row['longitude']}) | "
          f"Elev: {row['elevation_min_m']:.1f}m | NDVI: {row['s2_ndvi']:.3f} | Temp: {row['lst_day_c']:.1f}°C | "
          f"Precip: {row['precipitation_mm']:.2f}mm | Fire: {row['fire_label']}")
print("-" * 80)

# Create DataFrame
df = pd.DataFrame(all_data)
df['date'] = pd.to_datetime(df['date'])

print(f"\n✓ Total samples: {len(df)}")

# ============================================================
# ADD LAG FEATURES
# ============================================================

print("\n⏳ Adding lag features...")

df = df.sort_values(['district', 'latitude', 'longitude', 'date']).reset_index(drop=True)

# NDVI lags
print("   🌱 Creating NDVI lag features (lags: 1, 3, 7, 14)...")
for lag in [1, 3, 7, 14]:
    df[f's2_ndvi_lag{lag}'] = df.groupby(['district', 'latitude', 'longitude'])['s2_ndvi'].shift(lag)
    df[f'landsat_ndvi_lag{lag}'] = df.groupby(['district', 'latitude', 'longitude'])['landsat_ndvi'].shift(lag)
print("      ✓ NDVI lags complete")

# Precipitation lags
print("   🌧️ Creating precipitation lag features (lags: 1, 5, 10, 30)...")
for lag in [1, 5, 10, 30]:
    df[f'precipitation_mm_lag{lag}'] = df.groupby(['district', 'latitude', 'longitude'])['precipitation_mm'].shift(lag)
print("      ✓ Precipitation lags complete")

# Vapor pressure deficit lags
print("   💨 Creating vapor pressure deficit lag features (lags: 1, 3, 7, 14)...")
for lag in [1, 3, 7, 14]:
    df[f'vapor_pressure_deficit_kpa_lag{lag}'] = df.groupby(['district', 'latitude', 'longitude'])['vapor_pressure_deficit_kpa'].shift(lag)
print("      ✓ Vapor pressure deficit lags complete")

# Soil moisture lags
print("   💧 Creating soil moisture lag features (lags: 1, 3, 7, 14)...")
for lag in [1, 3, 7, 14]:
    df[f'soil_moisture_m3m3_lag{lag}'] = df.groupby(['district', 'latitude', 'longitude'])['soil_moisture_m3m3'].shift(lag)
print("      ✓ Soil moisture lags complete")

# Temperature lags
print("   🌡️ Creating temperature lag features (lags: 1, 3, 7)...")
for lag in [1, 3, 7]:
    df[f'temperature_2m_celsius_lag{lag}'] = df.groupby(['district', 'latitude', 'longitude'])['lst_day_c'].shift(lag)
print("      ✓ Temperature lags complete")

# Relative humidity lags
print("   📊 Creating relative humidity lag features (lags: 1, 3, 7)...")
for lag in [1, 3, 7]:
    df[f'relative_humidity_pct_lag{lag}'] = df.groupby(['district', 'latitude', 'longitude'])['relative_humidity_pct'].shift(lag)
print("      ✓ Relative humidity lags complete")

# Rolling features - precipitation sums
print("   📈 Creating rolling precipitation sum features (7, 14, 30 days)...")
df['precipitation_mm_roll7_sum'] = df.groupby(['district', 'latitude', 'longitude'])['precipitation_mm'].transform(
    lambda x: x.rolling(7, min_periods=1).sum())
df['precipitation_mm_roll14_sum'] = df.groupby(['district', 'latitude', 'longitude'])['precipitation_mm'].transform(
    lambda x: x.rolling(14, min_periods=1).sum())
df['precipitation_mm_roll30_sum'] = df.groupby(['district', 'latitude', 'longitude'])['precipitation_mm'].transform(
    lambda x: x.rolling(30, min_periods=1).sum())
print("      ✓ Rolling precipitation sums complete")

# NDVI rolling means
print("   🌿 Creating NDVI rolling mean features (7, 14 days)...")
df['s2_ndvi_roll7_mean'] = df.groupby(['district', 'latitude', 'longitude'])['s2_ndvi'].transform(
    lambda x: x.rolling(7, min_periods=1).mean())
df['s2_ndvi_roll14_mean'] = df.groupby(['district', 'latitude', 'longitude'])['s2_ndvi'].transform(
    lambda x: x.rolling(14, min_periods=1).mean())
df['landsat_ndvi_roll7_mean'] = df.groupby(['district', 'latitude', 'longitude'])['landsat_ndvi'].transform(
    lambda x: x.rolling(7, min_periods=1).mean())
df['landsat_ndvi_roll14_mean'] = df.groupby(['district', 'latitude', 'longitude'])['landsat_ndvi'].transform(
    lambda x: x.rolling(14, min_periods=1).mean())
print("      ✓ NDVI rolling means complete")

# Temperature rolling means
print("   🌡️ Creating temperature rolling mean features (7, 14 days)...")
df['temperature_2m_celsius_roll7_mean'] = df.groupby(['district', 'latitude', 'longitude'])['lst_day_c'].transform(
    lambda x: x.rolling(7, min_periods=1).mean())
df['temperature_2m_celsius_roll14_mean'] = df.groupby(['district', 'latitude', 'longitude'])['lst_day_c'].transform(
    lambda x: x.rolling(14, min_periods=1).mean())
print("      ✓ Temperature rolling means complete")

# Vapor pressure rolling means
print("   💨 Creating vapor pressure rolling mean features (7, 14 days)...")
df['vapor_pressure_deficit_kpa_roll7_mean'] = df.groupby(['district', 'latitude', 'longitude'])['vapor_pressure_deficit_kpa'].transform(
    lambda x: x.rolling(7, min_periods=1).mean())
df['vapor_pressure_deficit_kpa_roll14_mean'] = df.groupby(['district', 'latitude', 'longitude'])['vapor_pressure_deficit_kpa'].transform(
    lambda x: x.rolling(14, min_periods=1).mean())
print("      ✓ Vapor pressure rolling means complete")

print("   ✅ All lag and rolling features created!")

# ============================================================
# SAVE
# ============================================================

print("\n💾 Saving...")

df = df.dropna(thresh=len(df.columns) - 5).reset_index(drop=True)

output_file = DATA_DIR / "Master_FFOPM_Table.parquet"
df.to_parquet(output_file, index=False)
print(f"✓ Saved: {output_file}")

feature_cols = [col for col in df.columns if col not in ['date', 'district', 'latitude', 'longitude', 'year', 'month', 'fire_label']]
features_file = DATA_DIR / "final_features_for_ml_NO_LEAKAGE.json"
with open(features_file, 'w') as f:
    json.dump({'final_feature_list': feature_cols}, f, indent=2)
print(f"✓ Features: {len(feature_cols)}")

# Summary
print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)

print(f"\nDataset: {len(df):,} samples")
print(f"Features: {len(feature_cols)}")
print(f"Fire samples: {df['fire_label'].sum():,} ({df['fire_label'].mean()*100:.1f}%)")

train_df = df[df['year'] <= 2018]
val_df = df[(df['year'] > 2018) & (df['year'] <= 2020)]
test_df = df[df['year'] >= 2021]

print(f"\nTemporal split:")
print(f"  Train (≤2018): {len(train_df):,}")
print(f"  Val (2019-2020): {len(val_df):,}")
print(f"  Test (≥2021): {len(test_df):,}")

print(f"\nDistrict breakdown:")
for dist in DISTRICTS.keys():
    dist_df = df[df['district'] == dist]
    print(f"  {dist}: {len(dist_df):,} samples, {dist_df['fire_label'].sum():,} fires ({dist_df['fire_label'].mean()*100:.1f}%)")

print(f"\n✅ Run: python backend/postfire/models/lstm_trainer.py")
