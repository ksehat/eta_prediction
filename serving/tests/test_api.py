from fastapi.testclient import TestClient
from serving.api import app

client = TestClient(app)


def test_predict():
    response = client.post("/predict", json={
        "features": {
            "city_id": 'C',
            "accept_event_timestamp": "2024-09-10T12:34:56Z",
            "origin_lat": 35.6892,
            "origin_lon": 51.3890,
            "destination_lat": 36.292,
            "destination_lon": 52.3890,
            "edd": 12000,
            "provider_A": 3600,
            "provider_B": 3700,
            "provider_C": 3400,
            "provider_D": 3300
        }
    })
    assert response.status_code == 200
    assert "prediction" in response.json()


test_predict()
