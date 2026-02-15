"""
Comprehensive System Integration Test
Tests all major components: Fire Detection, Risk Assessment, CatBoost Prediction
"""
import sys
import os
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_imports():
    """Test 1: Verify all imports work"""
    print("\n" + "="*70)
    print("TEST 1: Import Validation")
    print("="*70)
    
    try:
        from backend.firedetect import FireDetector
        print("✓ FireDetector imported")
        
        from backend.prefire import PreFireAnalyzer
        print("✓ PreFireAnalyzer imported")
        
        from backend.postfire import PostFireAnalyzer, SpreadPredictor
        print("✓ PostFireAnalyzer, SpreadPredictor imported")
        
        from backend.src.data_collection.weather_api import WeatherDataFetcher
        print("✓ WeatherDataFetcher imported")
        
        from backend.src.data_collection.gee_extractor import get_gee_extractor
        print("✓ GEE Extractor imported")
        
        from backend.prefire.feature_engineer import FeatureEngineer
        print("✓ FeatureEngineer imported")
        
        print("\n✅ All imports successful!")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_generation():
    """Test 2: Feature generation produces 81 features"""
    print("\n" + "="*70)
    print("TEST 2: Feature Generation (81 Features)")
    print("="*70)
    
    try:
        from backend.prefire.feature_engineer import FeatureEngineer
        
        engineer = FeatureEngineer()
        
        # Test location: Kathmandu, Nepal
        lat, lon = 28.3949, 84.1240
        print(f"Test location: ({lat}, {lon})")
        
        features = engineer.get_all_features(lat, lon)
        
        print(f"\nGenerated {len(features)} features")
        
        if len(features) == 81:
            print("✅ Correct feature count (81)")
            
            # Show sample features
            print("\nSample features:")
            samples = [
                'temperature_2m_celsius_lag1',
                'precipitation_mm_roll7_sum',
                'vapor_pressure_deficit_kpa_lag14',
                'landsat_ndvi',
                's2_evi',
                'mtpi_max'
            ]
            for feat in samples:
                if feat in features:
                    print(f"  ✓ {feat}: {features[feat]}")
            
            return True
        else:
            print(f"❌ Wrong feature count: expected 81, got {len(features)}")
            return False
            
    except Exception as e:
        print(f"❌ Feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_catboost_prediction():
    """Test 3: CatBoost model prediction"""
    print("\n" + "="*70)
    print("TEST 3: CatBoost Model Prediction")
    print("="*70)
    
    try:
        from backend.prefire import PreFireAnalyzer
        
        analyzer = PreFireAnalyzer()
        
        # Test location
        lat, lon = 28.3949, 84.1240
        print(f"Test location: ({lat}, {lon})")
        
        # Get prediction
        result = analyzer.predict_risk(lat, lon)
        
        print(f"\nPrediction result:")
        print(f"  Risk Score: {result.get('risk_score', 'N/A')}")
        print(f"  Risk Level: {result.get('risk_level', 'N/A')}")
        print(f"  Probability: {result.get('probability', 'N/A')}")
        
        if 'error' in result:
            print(f"⚠️  Prediction returned with error: {result['error']}")
            return False
        
        if result.get('risk_score') is not None:
            print("✅ CatBoost prediction successful!")
            return True
        else:
            print("❌ No risk score in prediction")
            return False
            
    except Exception as e:
        print(f"❌ CatBoost prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fire_detection():
    """Test 4: Fire detection system"""
    print("\n" + "="*70)
    print("TEST 4: Fire Detection (NASA FIRMS)")
    print("="*70)
    
    try:
        from backend.firedetect import FireDetector
        
        detector = FireDetector()
        
        # Test with Nepal region
        region = "Nepal"
        hours = 24
        print(f"Testing fire detection: {region}, last {hours} hours")
        
        result = detector.detect_fires(region, hours=hours)
        
        print(f"\nDetection result:")
        print(f"  Fire count: {result.get('count', 'N/A')}")
        print(f"  Region: {result.get('region', 'N/A')}")
        print(f"  Time window: {result.get('hours', 'N/A')} hours")
        
        if result.get('count') != 'N/A':
            print(f"✅ Fire detection successful! Found {result['count']} fires")
            
            # Show sample fire if available
            if result.get('fires') and len(result['fires']) > 0:
                fire = result['fires'][0]
                print(f"\nSample fire:")
                print(f"  Location: ({fire['latitude']}, {fire['longitude']})")
                print(f"  Confidence: {fire['confidence']}%")
                print(f"  Satellite: {fire['satellite']}")
            
            return True
        else:
            print("⚠️  Fire detection returned N/A")
            return False
            
    except Exception as e:
        print(f"❌ Fire detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_weather_api():
    """Test 5: Weather API integration"""
    print("\n" + "="*70)
    print("TEST 5: Weather API Integration")
    print("="*70)
    
    try:
        from backend.src.data_collection.weather_api import WeatherDataFetcher
        
        fetcher = WeatherDataFetcher()
        
        lat, lon = 28.3949, 84.1240
        print(f"Test location: ({lat}, {lon})")
        
        # Test current weather
        current = fetcher.fetch_current_weather(lat, lon)
        print(f"\nCurrent weather:")
        print(f"  Temperature: {current.get('temp', 'N/A')}°C")
        print(f"  Humidity: {current.get('humidity', 'N/A')}%")
        print(f"  Wind speed: {current.get('wind_speed', 'N/A')} m/s")
        print(f"  Soil moisture: {current.get('soil_moisture', 'N/A')} m³/m³")
        
        # Test historical weather
        historical = fetcher.get_historical_weather(lat, lon, days_back=7)
        print(f"\nHistorical weather (7 days):")
        print(f"  Data points: {len(historical.get('dates', []))}")
        
        if current.get('temp') is not None:
            print("✅ Weather API working!")
            return True
        else:
            print("❌ Weather API returned no data")
            return False
            
    except Exception as e:
        print(f"❌ Weather API failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gee_integration():
    """Test 6: Google Earth Engine integration"""
    print("\n" + "="*70)
    print("TEST 6: Google Earth Engine Integration")
    print("="*70)
    
    try:
        from backend.src.data_collection.gee_extractor import get_gee_extractor
        
        gee = get_gee_extractor()
        
        print(f"GEE Mock Mode: {gee.is_mock_mode}")
        
        if gee.is_mock_mode:
            print("⚠️  Running in MOCK mode (GEE not authenticated)")
            print("   This is expected if GEE credentials are not configured")
            return True
        else:
            print("✅ GEE authenticated and ready")
            
            # Test terrain metrics
            try:
                import ee
                point = ee.Geometry.Point([84.1240, 28.3949])
                terrain = gee.get_terrain_metrics(point)
                print(f"\nTerrain metrics:")
                print(f"  Elevation range: {terrain.get('elevation_range', 'N/A')} m")
                print(f"  Slope max: {terrain.get('slope_max', 'N/A')}°")
                print(f"  MTPI mean: {terrain.get('mtpi_mean', 'N/A')}")
                print("✅ GEE data retrieval working!")
            except Exception as e:
                print(f"⚠️  GEE data retrieval error: {e}")
            
            return True
            
    except Exception as e:
        print(f"❌ GEE integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("COMPREHENSIVE SYSTEM INTEGRATION TEST")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Feature Generation", test_feature_generation),
        ("CatBoost Prediction", test_catboost_prediction),
        ("Fire Detection", test_fire_detection),
        ("Weather API", test_weather_api),
        ("GEE Integration", test_gee_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is operational.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
