from datetime import datetime, timezone
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_websocket_ping_pong():
    with client.websocket_connect("/ws/readings") as websocket:
        websocket.send_text('{"type": "ping"}')
        reply = websocket.receive_json()
        assert reply["type"] == "pong"


def test_websocket_events_endpoint_and_status():
    with client.websocket_connect("/ws/events") as websocket:
        websocket.send_text('{"type": "status"}')
        reply = websocket.receive_json()
        assert reply["type"] == "status_response"
        assert "active_connections" in reply
        assert reply["active_connections"] >= 1


def test_websocket_broadcast_delivery():
    reading_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "simulator",
        "scenario": "heatwave",
        "heart_rate": 110,
        "spo2": 95,
        "body_temperature": 38.2,
        "activity_level": 0.8,
        "ambient_temperature": 39,
        "humidity": 75,
        "aqi": 80,
        "pm25": 25,
    }

    with client.websocket_connect("/ws/events") as websocket:
        post_res = client.post("/api/readings", json=reading_payload)
        assert post_res.status_code == 201

        event = websocket.receive_json()
        assert event["type"] == "reading_processed"
        assert event["reading"]["heart_rate"] == 110
        assert event["risk"]["severity"] >= 2
        # Verify backward compatibility fields
        assert event["heart_rate"] == 110
