import json
from pathlib import Path

# Your paths
DATA_DIR = Path("/Users/prabhatrawal/Minor_project_code/data/integrated_data")
FEATURES_FILE = DATA_DIR / "eda_results" / "final_features_for_ml_NO_LEAKAGE.json"
OUTPUT_FILE = DATA_DIR / "eda_results" / "final_features_102_NO_LEAKAGE.json"

print("=" * 80)
print("GENERATING 102-FEATURE LIST")
print("=" * 80)

# Load current 81 features
with open(FEATURES_FILE, 'r') as f:
    feature_data = json.load(f)
    current_features = feature_data['final_feature_list']
    dropped_high_corr = feature_data['features_dropped']['high_correlation']
    dropped_leakage = feature_data['features_dropped']['target_leakage']

print(f"\nCurrent state:")
print(f"  ✓ Current features: {len(current_features)}")
print(f"  ✓ Dropped (high correlation): {len(dropped_high_corr)}")
print(f"  ✓ Dropped (target leakage): {len(dropped_leakage)} [CANNOT ADD BACK]")
print(f"  ✓ Need to add: {102 - len(current_features)} features")

# Strategy: Add back 21 features from high_correlation list
# Priority: Features with lower correlation risk and high importance

# CAREFULLY SELECTED 21 features to add back
# These are ordered by importance and correlation risk
features_to_add_back = [
    # Core weather features (very important despite correlation)
    "temperature_2m_celsius",      # Important baseline temperature
    "soil_temperature_celsius",    # Soil conditions
    "surface_pressure_hpa",        # Atmospheric pressure
    
    # Elevation features (spatial context, moderate correlation)
    "elevation_mean_m",            # Central tendency
    "elevation_median_m",          # Robust central measure
    "elevation_p25_m",             # Lower quartile
    "elevation_p75_m",             # Upper quartile
    "elevation_p10_m",             # 10th percentile
    "elevation_p90_m",             # 90th percentile
    "elevation_max_m",             # Maximum elevation
    
    # Terrain features (topographic context)
    "slope_mean_deg",              # Average slope
    
    # Vegetation indices (different perspectives)
    "ndvi_composite",              # Composite NDVI
    "landsat_evi",                 # Enhanced vegetation index (Landsat)
    "landsat_ndwi",                # Water content (Landsat)
    
    # Sentinel-2 indices
    "s2_nbr",                      # Normalized Burn Ratio
    
    # These total 15 features (all from dropped_high_corr)
    # We have 15 in the dropped list, so we'll add all
]

# Verify all are in the dropped list
available_to_add = [f for f in features_to_add_back if f in dropped_high_corr]
print(f"\n✓ Features available to add back: {len(available_to_add)}")

# If we need more, we'll need to reconsider
needed = 102 - len(current_features)
if len(available_to_add) < needed:
    print(f"\n⚠️  WARNING: Only {len(available_to_add)} features available from dropped list")
    print(f"   Need {needed} features to reach 102")
    print(f"   Will add all {len(available_to_add)} available features")
    print(f"   Final count will be: {len(current_features) + len(available_to_add)}")
    features_to_add = available_to_add
elif len(available_to_add) > needed:
    # Take only what we need
    features_to_add = available_to_add[:needed]
    print(f"\n✓ Will add {len(features_to_add)} features (exactly what's needed)")
else:
    features_to_add = available_to_add
    print(f"\n✓ Will add all {len(features_to_add)} features")

# Create new feature list (102 features)
final_102_features = current_features + features_to_add

print(f"\nFinal feature count: {len(final_102_features)}")

if len(final_102_features) != 102:
    print(f"⚠️  WARNING: Final count is {len(final_102_features)}, not 102!")
    print(f"   This is because only {len(dropped_high_corr)} features were dropped for high correlation")
    print(f"   And we started with {len(current_features)} features")
    print(f"\n   SOLUTION: We need to reconsider which features to include")
else:
    print(f"✅ SUCCESS: Exactly 102 features!")

# Show what was added
print(f"\nFeatures added back ({len(features_to_add)}):")
for i, feat in enumerate(features_to_add, 1):
    print(f"  {i:2d}. {feat}")

# Create updated feature categories
feature_categories = {
    "LST": 2,
    "Vegetation": 20 + sum(1 for f in features_to_add if any(x in f for x in ['ndvi', 'evi', 'nbr', 'ndwi'])),
    "Weather": 36 + sum(1 for f in features_to_add if any(x in f for x in ['temperature', 'pressure'])),
    "Terrain": 12 + sum(1 for f in features_to_add if any(x in f for x in ['elevation', 'slope'])),
    "Lag_Features": 33,
    "Rolling_Features": 11,
    "Other": 7,
}

# Save the 102-feature list
output_data = {
    "total_features": len(final_102_features),
    "note": "Generated from 81-feature list by adding back 21 strategically selected features",
    "generation_method": "Added back features from high_correlation dropped list",
    "features_added_back": features_to_add,
    "features_kept_dropped": {
        "target_leakage": dropped_leakage,
        "high_correlation": [f for f in dropped_high_corr if f not in features_to_add]
    },
    "final_feature_list": final_102_features,
    "feature_categories": feature_categories,
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✅ Saved 102-feature list to: {OUTPUT_FILE.name}")

# Verification
print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)
print(f"Original features: {len(current_features)}")
print(f"Added back: {len(features_to_add)}")
print(f"Total: {len(final_102_features)}")
print(f"Target: 102")
print(f"Match: {'✅ YES' if len(final_102_features) == 102 else '❌ NO'}")

# Show first and last 10 features
print(f"\nFirst 10 features:")
for i, feat in enumerate(final_102_features[:10], 1):
    print(f"  {i:2d}. {feat}")

print(f"\nLast 10 features:")
for i, feat in enumerate(final_102_features[-10:], len(final_102_features)-9):
    print(f"  {i:3d}. {feat}")

print("\n" + "=" * 80)