from fastapi.testclient import TestClient
from api.backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "models_loaded": True}

def test_pre_fire_prediction():
    payload = {
        "latitude": 27.7,
        "longitude": 85.3,
        "date": "2023-10-01",
        "fire_detected": False
    }
    response = client.post("/predictions/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk" in data

def test_post_fire_prediction():
    payload = {
        "latitude": 27.7,
        "longitude": 85.3,
        "date": "2023-10-01",
        "fire_detected": True
    }
    response = client.post("/predictions/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
