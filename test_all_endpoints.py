import requests
import json

URL = "http://127.0.0.1:8000"

print("="*60)
print("WILDFIRE MANAGEMENT SYSTEM - COMPREHENSIVE TEST")
print("="*60)

# Test 1: Health Check
print("\n1. Testing Health Endpoint...")
try:
    response = requests.get(f"{URL}/api/health", timeout=5)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=6)}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Environmental Data
print("\n2. Testing Environmental Data Endpoint...")
try:
    response = requests.get(
        f"{URL}/predictions/environmental",
        params={"lat": 27.7172, "lon": 85.3240, "date": "2023-01-01"},
        timeout=30
    )
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Features available: {list(data.get('features', {}).keys())}")
    print(f"   Sample values:")
    for key, value in list(data.get('features', {}).items())[:5]:
        print(f"      {key}: {value}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Pre-Fire Risk Map
print("\n3. Testing Pre-Fire Risk Map Endpoint...")
try:
    response = requests.post(
        f"{URL}/predictions/pre-fire/risk-map",
        json={"latitude": 27.7172, "longitude": 85.3240, "date": "2023-01-01"},
        timeout=60
    )
    print(f"   Status: {response.status_code}")
    data = response.json()
    metadata = data.get("metadata", {})
    print(f"   Metadata: {json.dumps(metadata, indent=6)}")
    features = data.get("features", [])
    print(f"   Generated {len(features)} risk zones")
    if features:
        sample = features[0]['properties']
        print(f"   Sample zone:")
        print(f"      Risk Score: {sample['risk_score']}")
        print(f"      Risk Level: {sample['risk_level']}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Active Fires
print("\n4. Testing Active Fires Endpoint...")
try:
    response = requests.get(
        f"{URL}/predictions/fires/active",
        params={"region": "NPL", "hours": 24},
        timeout=30
    )
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Fire count: {data.get('count', 0)}")
    print(f"   Source: {data.get('source', 'N/A')}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 5: Simple Risk Prediction
print("\n5. Testing Simple Risk Prediction Endpoint...")
try:
    response = requests.post(
        f"{URL}/predictions/predict",
        json={"latitude": 27.7172, "longitude": 85.3240, "date": "2023-01-01", "fire_detected": False},
        timeout=30
    )
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Risk Level: {data.get('risk', 'N/A')}")
    print(f"   Risk Score: {data.get('risk_score', 'N/A')}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("ALL TESTS COMPLETED")
print("="*60)
