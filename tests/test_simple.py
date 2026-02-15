"""
Simple ASCII-only system test to avoid encoding issues
"""
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

print("="*70)
print("SYSTEM INTEGRATION TEST")
print("="*70)

# Test 1: Feature Generation
print("\n[TEST 1] Feature Generation")
try:
    from backend.prefire.feature_engineer import FeatureEngineer
    engineer = FeatureEngineer()
    features = engineer.get_all_features(28.3949, 84.1240)
    print(f"Generated {len(features)} features")
    if len(features) == 81:
        print("[PASS] Correct feature count")
    else:
        print(f"[FAIL] Expected 81, got {len(features)}")
except Exception as e:
    print(f"[FAIL] {e}")

# Test 2: CatBoost Prediction
print("\n[TEST 2] CatBoost Prediction")
try:
    from backend.prefire import PreFireAnalyzer
    analyzer = PreFireAnalyzer()
    result = analyzer.analyze_location(28.3949, 84.1240)  # Fixed: use analyze_location
    print(f"Risk Score: {result.get('probability', 'N/A')}")
    print(f"Risk Level: {result.get('risk_level', 'N/A')}")
    if 'error' in result:
        print(f"[FAIL] Error: {result['error']}")
    elif result.get('probability') is not None:
        print("[PASS] Prediction successful")
    else:
        print("[FAIL] No risk score returned")
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()

# Test 3: Fire Detection
print("\n[TEST 3] Fire Detection")
try:
    from backend.firedetect import FireDetector
    detector = FireDetector()
    result = detector.detect_fires("Nepal", hours=24)
    print(f"Fire count: {result.get('count', 'N/A')}")
    if result.get('count') != 'N/A':
        print(f"[PASS] Found {result['count']} fires")
    else:
        print("[FAIL] No fire data")
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()

# Test 4: Weather API
print("\n[TEST 4] Weather API")
try:
    from backend.src.data_collection.weather_api import WeatherDataFetcher
    fetcher = WeatherDataFetcher()
    current = fetcher.fetch_current_weather(28.3949, 84.1240)
    print(f"Temperature: {current.get('temp', 'N/A')} C")
    print(f"Humidity: {current.get('humidity', 'N/A')} %")
    if current.get('temp') is not None:
        print("[PASS] Weather data retrieved")
    else:
        print("[FAIL] No weather data")
except Exception as e:
    print(f"[FAIL] {e}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
