"""
Test script to verify feature engineering generates exactly 81 features
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.prefire.feature_engineer import FeatureEngineer

def test_feature_count():
    """Test that feature engineer generates exactly 81 features"""
    print("=" * 70)
    print("Testing Feature Engineer - 81 Feature Generation")
    print("=" * 70)
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Test location (Kathmandu, Nepal)
    lat, lon = 28.3949, 84.1240
    
    print(f"\nGenerating features for location: ({lat}, {lon})")
    
    # Get all features
    features = engineer.get_all_features(lat, lon)
    
    # Count features
    feature_count = len(features)
    
    print(f"\n✓ Generated {feature_count} features")
    
    # Verify count
    if feature_count == 81:
        print("\n✅ SUCCESS: Feature count matches expected 81 features!")
    else:
        print(f"\n❌ FAILURE: Expected 81 features, got {feature_count}")
        print(f"\nDifference: {81 - feature_count} features")
        
        # List all features
        print("\nGenerated features:")
        for i, feature_name in enumerate(sorted(features.keys()), 1):
            print(f"  {i:2d}. {feature_name}: {features[feature_name]}")
        
        return False
    
    # Print feature categories
    print("\nFeature breakdown:")
    print(f"  - Total features: {feature_count}")
    
    # Sample some features
    print("\nSample features:")
    sample_features = [
        'temperature_2m_celsius_lag1',
        'precipitation_mm_roll7_sum',
        'vapor_pressure_deficit_kpa_lag14',
        'landsat_ndvi',
        's2_evi',
        'mtpi_max',
        'soil_moisture_m3m3_lag7'
    ]
    
    for feat in sample_features:
        if feat in features:
            print(f"  ✓ {feat}: {features[feat]}")
        else:
            print(f"  ✗ {feat}: MISSING")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = test_feature_count()
    sys.exit(0 if success else 1)
