
import requests
import json
import time

URL = "http://127.0.0.1:8000"

def test_health():
    print("Testing server health...")
    try:
        response = requests.get(f"{URL}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_risk_map():
    print("\nTesting Pre-Fire Risk Map (Integration with Ensemble Model)...")
    endpoint = f"{URL}/predictions/pre-fire/risk-map"
    payload = {
        "latitude": 27.7172,
        "longitude": 85.3240,
        "date": "2023-01-01"
    }
    
    try:
        response = requests.post(endpoint, json=payload, timeout=60)
        if response.status_code == 200:
            print("✅ Success!")
            data = response.json()
            metadata = data.get("metadata", {})
            print("Metadata:", json.dumps(metadata, indent=2))
            features = data.get("features", [])
            print(f"Generated {len(features)} risk zones.")
            if features:
                print("Sample Feature Properties:", json.dumps(features[0]['properties'], indent=2))
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_environmental_data():
    print("\nTesting Environmental Data Endpoint...")
    endpoint = f"{URL}/predictions/environmental"
    params = {
        "lat": 27.7172,
        "lon": 85.3240,
        "date": "2023-01-01"
    }
    
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        if response.status_code == 200:
            print("✅ Success!")
            data = response.json()
            print("Features:", list(data.get("features", {}).keys()))
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Waiting 3 seconds for server to be ready...")
    time.sleep(3)
    
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Environmental Data", test_environmental_data()))
    results.append(("Risk Map", test_risk_map()))
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
