import json
from pathlib import Path

# ============================================================
# FIX TARGET LEAKAGE - REMOVE FIRE METRICS FROM FEATURES
# ============================================================
# These variables are derived from MOD14A1 on the SAME day
# They essentially contain the answer -> MUST BE REMOVED
# ============================================================

print("=" * 80)
print("FIXING TARGET LEAKAGE - REMOVING FIRE DETECTION VARIABLES")
print("=" * 80)

# Paths
EDA_DIR = Path("/Users/prabhatrawal/Minor_project_code/data/integrated_data/eda_results")
FEATURES_FILE = EDA_DIR / "final_features_for_ml.json"
FIXED_FEATURES_FILE = EDA_DIR / "final_features_for_ml_NO_LEAKAGE.json"

# Load current feature list
with open(FEATURES_FILE, 'r') as f:
    feature_metadata = json.load(f)

current_features = feature_metadata['final_feature_list']
print(f"\nCurrent feature count: {len(current_features)}")

# CRITICAL: Remove ALL fire-related metrics from MOD14A1
# These are computed on the same day and contain the answer
LEAKAGE_KEYWORDS = [
    'fire_pixels',           # total_fire_pixels
    'fire_percentage',       # fire_percentage
    'confidence_fire',       # low/nominal/high_confidence_fire_pixels
    'total_pixels',          # Could indirectly leak info
]

# Identify leakage features
leakage_features = []
for feature in current_features:
    feature_lower = feature.lower()
    if any(keyword in feature_lower for keyword in LEAKAGE_KEYWORDS):
        leakage_features.append(feature)

print(f"\n🚨 LEAKAGE FEATURES IDENTIFIED ({len(leakage_features)}):")
for feat in leakage_features:
    print(f"  ❌ {feat}")

# Remove leakage features
clean_features = [f for f in current_features if f not in leakage_features]

print(f"\n✓ Clean feature count: {len(clean_features)}")
print(f"  Removed: {len(leakage_features)} leakage features")

# Update metadata
feature_metadata_clean = {
    'total_features': len(clean_features),
    'features_dropped': {
        'high_correlation': feature_metadata['features_dropped']['high_correlation'],
        'low_variance': feature_metadata['features_dropped']['low_variance'],
        'target_leakage': leakage_features,  # NEW: Track removed leakage
        'total_dropped': (feature_metadata['features_dropped']['total_dropped'] + 
                         len(leakage_features))
    },
    'feature_categories': {},  # Will be recalculated
    'final_feature_list': clean_features,
    'feature_details_by_category': {}
}

# Recategorize features
feature_categories = {
    'LST': [f for f in clean_features if 'lst' in f.lower()],
    'Vegetation': [f for f in clean_features if any(x in f.lower() for x in ['ndvi', 'evi', 'savi', 'nbr', 'gndvi'])],
    'Weather': [f for f in clean_features if any(x in f.lower() for x in ['temperature', 'humidity', 'wind', 'precipitation', 'pressure', 'dewpoint'])],
    'Terrain': [f for f in clean_features if any(x in f.lower() for x in ['elevation', 'slope', 'aspect', 'mtpi'])],
    'Lag_Features': [f for f in clean_features if 'lag' in f.lower()],
    'Rolling_Features': [f for f in clean_features if 'roll' in f.lower()],
    'Other': []
}

# Categorize remaining features
categorized = set()
for cat_features in feature_categories.values():
    categorized.update(cat_features)

feature_categories['Other'] = [f for f in clean_features if f not in categorized]

feature_metadata_clean['feature_categories'] = {k: len(v) for k, v in feature_categories.items()}
feature_metadata_clean['feature_details_by_category'] = feature_categories

print(f"\n📈 Clean Feature Breakdown:")
for category, count in feature_metadata_clean['feature_categories'].items():
    print(f"  {category}: {count}")

# Save cleaned feature list
with open(FIXED_FEATURES_FILE, 'w') as f:
    json.dump(feature_metadata_clean, f, indent=2)

print(f"\n✓ Saved clean feature list: {FIXED_FEATURES_FILE.name}")

# Also update the original file with warning
feature_metadata['WARNING'] = "TARGET LEAKAGE DETECTED - USE final_features_for_ml_NO_LEAKAGE.json instead"
with open(FEATURES_FILE, 'w') as f:
    json.dump(feature_metadata, f, indent=2)

print("\n" + "=" * 80)
print("EXPECTED RESULTS AFTER FIX:")
print("=" * 80)
print("""
After retraining with clean features, expect REALISTIC performance:

Logistic Regression:
  ROC-AUC: 0.75 - 0.85
  PR-AUC: 0.25 - 0.45
  Recall@Fire: 0.60 - 0.75
  Precision@Fire: 0.30 - 0.50
  False Alarm Rate: 0.02 - 0.05

Random Forest:
  ROC-AUC: 0.80 - 0.92
  PR-AUC: 0.30 - 0.55
  Recall@Fire: 0.65 - 0.80
  Precision@Fire: 0.40 - 0.65
  False Alarm Rate: 0.01 - 0.04

These are REALISTIC numbers for highly imbalanced fire prediction!
Perfect scores (1.000) were impossible and indicated cheating.
""")

print("\n🚀 NEXT STEP:")
print("=" * 80)
print(f"1. Rerun Step C with: {FIXED_FEATURES_FILE.name}")
print("2. Update the training script to use this file")
print("3. Expect lower but HONEST performance metrics")
print("=" * 80)