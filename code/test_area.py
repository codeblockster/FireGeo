import json
from pathlib import Path

# Your paths
DATA_DIR = Path("/Users/prabhatrawal/Minor_project_code/data/integrated_data")
FEATURES_FILE = DATA_DIR / "eda_results" / "final_features_for_ml_NO_LEAKAGE.json"

print("=" * 80)
print("FEATURE ANALYSIS: Finding Missing 21 Features")
print("=" * 80)

# Master CSV columns (from your provided list)
master_columns = """date,district,zone,fire_label,low_confidence_fire_pixels,nominal_confidence_fire_pixels,high_confidence_fire_pixels,total_fire_pixels,fire_percentage,lst_day_c,s2_ndvi,s2_gndvi,s2_nbr,s2_ndwi,s2_ndsi,s2_evi,s2_savi,s2_cloud_cover_percent,landsat_ndvi,landsat_gndvi,landsat_nbr,landsat_ndwi,landsat_evi,landsat_savi,ndvi_composite,veg_data_quality,temperature_2m_celsius,dewpoint_2m_celsius,skin_temperature_celsius,soil_temperature_celsius,soil_moisture_m3m3,precipitation_mm,u_wind_component_ms,v_wind_component_ms,wind_speed_ms,wind_direction_deg,surface_pressure_hpa,relative_humidity_pct,vapor_pressure_deficit_kpa,elevation_mean_m,elevation_min_m,elevation_max_m,elevation_median_m,elevation_stddev_m,elevation_range_m,elevation_p10_m,elevation_p25_m,elevation_p75_m,elevation_p90_m,mtpi_mean,mtpi_min,mtpi_max,mtpi_stddev,slope_mean_deg,slope_min_deg,slope_max_deg,slope_stddev_deg,aspect_mean_deg,aspect_stddev_deg,lst_missing_flag,s2_ndvi_lag1,s2_ndvi_lag3,s2_ndvi_lag7,s2_ndvi_lag14,landsat_ndvi_lag1,landsat_ndvi_lag3,landsat_ndvi_lag7,landsat_ndvi_lag14,precipitation_mm_lag1,precipitation_mm_lag5,precipitation_mm_lag10,precipitation_mm_lag30,vapor_pressure_deficit_kpa_lag1,vapor_pressure_deficit_kpa_lag3,vapor_pressure_deficit_kpa_lag7,vapor_pressure_deficit_kpa_lag14,soil_temperature_celsius_lag1,soil_temperature_celsius_lag3,soil_temperature_celsius_lag7,soil_moisture_m3m3_lag1,soil_moisture_m3m3_lag3,soil_moisture_m3m3_lag7,soil_moisture_m3m3_lag14,temperature_2m_celsius_lag1,temperature_2m_celsius_lag3,temperature_2m_celsius_lag7,skin_temperature_celsius_lag1,skin_temperature_celsius_lag3,skin_temperature_celsius_lag7,relative_humidity_pct_lag1,relative_humidity_pct_lag3,relative_humidity_pct_lag7,precipitation_mm_roll7_sum,precipitation_mm_roll14_sum,precipitation_mm_roll30_sum,s2_ndvi_roll7_mean,s2_ndvi_roll14_mean,landsat_ndvi_roll7_mean,landsat_ndvi_roll14_mean,temperature_2m_celsius_roll7_mean,temperature_2m_celsius_roll14_mean,vapor_pressure_deficit_kpa_roll7_mean,vapor_pressure_deficit_kpa_roll14_mean,total_pixels,clear_day_coverage,clear_night_coverage"""

all_columns = [col.strip() for col in master_columns.split(',')]

# Exclude non-feature columns
exclude_cols = ['date', 'district', 'zone', 'fire_label']
potential_features = [col for col in all_columns if col not in exclude_cols]

print(f"\n1. Total columns in master file: {len(all_columns)}")
print(f"   Excluding: {exclude_cols}")
print(f"   Potential features: {len(potential_features)}")

# Load current features from JSON
with open(FEATURES_FILE, 'r') as f:
    feature_data = json.load(f)
    current_features = feature_data['final_feature_list']

print(f"\n2. Current features in JSON: {len(current_features)}")

# Features that were dropped (from JSON)
dropped_high_corr = feature_data['features_dropped']['high_correlation']
dropped_leakage = feature_data['features_dropped']['target_leakage']

print(f"\n3. Dropped features:")
print(f"   High correlation: {len(dropped_high_corr)}")
print(f"   Target leakage: {len(dropped_leakage)}")
print(f"   Total dropped: {len(dropped_high_corr) + len(dropped_leakage)}")

# Calculate what we should have
print(f"\n4. Math check:")
print(f"   Potential features: {len(potential_features)}")
print(f"   Minus dropped features: {len(dropped_high_corr) + len(dropped_leakage)}")
print(f"   Should equal: {len(potential_features) - len(dropped_high_corr) - len(dropped_leakage)}")
print(f"   Actual in JSON: {len(current_features)}")

# Find what's in master but not in current features (and not dropped)
all_dropped = set(dropped_high_corr + dropped_leakage)
current_set = set(current_features)
potential_set = set(potential_features)

missing_features = potential_set - current_set - all_dropped

print(f"\n5. MISSING FEATURES (not in JSON, not dropped): {len(missing_features)}")
if missing_features:
    for feat in sorted(missing_features):
        print(f"   - {feat}")

# Find what's in JSON but not in master (this would be an error)
extra_features = current_set - potential_set
print(f"\n6. Features in JSON but NOT in master file: {len(extra_features)}")
if extra_features:
    print("   ⚠️  WARNING: These should not exist!")
    for feat in sorted(extra_features):
        print(f"   - {feat}")

# Solution options
print("\n" + "=" * 80)
print("SOLUTIONS TO GET 102 FEATURES:")
print("=" * 80)

print("\nOption 1: ADD BACK some low-risk dropped features")
print("   Current: 81 features")
print("   Need: 21 more features")
print("   Available from high_correlation list: 15 features")
print("   Available from other sources: 6+ features")
print("\n   Suggested features to add back (lower correlation risk):")

# Show the dropped features that might be safer to include
safe_to_add = [
    "landsat_ndwi",  # Water index, different from NDVI
    "elevation_mean_m",  # Central tendency
    "slope_mean_deg",  # Central tendency
    "temperature_2m_celsius",  # Important weather variable
    "soil_temperature_celsius",  # Important soil variable
    "surface_pressure_hpa",  # Weather variable
]

for feat in safe_to_add:
    if feat in dropped_high_corr:
        print(f"   - {feat} (from high_correlation)")

print("\nOption 2: Use EXACTLY the features that give 102")
print("   This would be: 81 (current) + 21 (selected from dropped)")

print("\nOption 3: Recalculate from scratch")
print("   Review correlation threshold and keep 102 features")

# Calculate exact count needed
print(f"\n7. EXACT COUNT:")
print(f"   Total potential features in master: {len(potential_features)}")
print(f"   Target leakage (must drop): {len(dropped_leakage)}")
print(f"   Remaining after leakage removal: {len(potential_features) - len(dropped_leakage)}")
print(f"   To get 102 features, drop only: {len(potential_features) - len(dropped_leakage) - 102} high-correlation features")
print(f"   Currently dropped: {len(dropped_high_corr)} high-correlation features")

print("\n" + "=" * 80)