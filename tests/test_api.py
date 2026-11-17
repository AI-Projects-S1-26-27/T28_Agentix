from datetime import datetime, timezone

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def payload(**overrides):
    value = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "simulator",
        "scenario": "normal",
        "heart_rate": 72,
        "spo2": 98,
        "body_temperature": 36.8,
        "activity_level": 0.4,
        "ambient_temperature": 28,
        "humidity": 55,
        "aqi": 50,
        "pm25": 12,
    }
    value.update(overrides)
    return value


def test_health_and_reading_endpoints():
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/api/readings", json=payload(spo2=None))

    assert response.status_code == 201
    assert response.json()["spo2"] is None
    assert client.get("/api/readings?limit=1").status_code == 200


def test_invalid_reading_is_rejected():
    response = client.post("/api/readings", json=payload(spo2=101))

    assert response.status_code == 422


def test_websocket_receives_persisted_reading():
    with client.websocket_connect("/ws/readings") as websocket:
        response = client.post("/api/readings", json=payload())
        assert response.status_code == 201
        message = websocket.receive_json()
        assert message["source"] == "simulator"
        assert "risk" in message


def test_latest_reading_summary():
    # Ingest a reading first
    client.post("/api/readings", json=payload())
    res = client.get("/api/readings/latest")
    assert res.status_code == 200
    data = res.json()
    assert "reading" in data
    assert "risk" in data


def test_risks_routes():
    # Ingest high risk reading
    client.post(
        "/api/readings",
        json=payload(ambient_temperature=38, humidity=80, heart_rate=115),
    )
    recent = client.get("/api/risks/recent?limit=5")
    assert recent.status_code == 200
    assert len(recent.json()) > 0

    latest = client.get("/api/risks/latest")
    assert latest.status_code == 200
    assert latest.json()["risk_level"] in ("very_high", "severe", "high")

    # On-demand evaluation
    eval_res = client.post(
        "/api/risks/evaluate",
        json={"ambient_temperature": 40, "humidity": 75, "heart_rate": 110},
    )
    assert eval_res.status_code == 200
    assert eval_res.json()["severity"] >= 3


def test_alerts_routes():
    # Trigger high heat/pollution alert
    client.post(
        "/api/readings",
        json=payload(aqi=350, spo2=91),
    )
    alerts = client.get("/api/alerts")
    assert alerts.status_code == 200
    alert_list = alerts.json()
    assert len(alert_list) > 0

    first_id = alert_list[0]["id"]
    ack_res = client.post(f"/api/alerts/{first_id}/acknowledge")
    assert ack_res.status_code == 200
    assert ack_res.json()["acknowledged"] is True


def test_history_and_stats_routes():
    stats = client.get("/api/history/stats")
    assert stats.status_code == 200
    assert "total_count" in stats.json()

    history = client.get("/api/history/readings?limit=10")
    assert history.status_code == 200
    assert isinstance(history.json(), list)


def test_baseline_routes():
    save_res = client.post(
        "/api/baseline/calibrate",
        json={
            "participant_id": "test_user",
            "signal": "heart_rate_bpm",
            "mean": 71.5,
            "std": 3.8,
            "observation_count": 120,
            "status": "stable",
        },
    )
    assert save_res.status_code == 200

    get_res = client.get("/api/baseline?participant_id=test_user")
    assert get_res.status_code == 200
    assert len(get_res.json()) >= 1


def test_scenarios_routes():
    list_res = client.get("/api/scenarios")
    assert list_res.status_code == 200
    scenarios = list_res.json()
    assert len(scenarios) >= 4

    run_res = client.post(
        "/api/scenarios/run",
        json={"scenario_id": "normal", "count": 3},
    )
    assert run_res.status_code == 200
    assert run_res.json()["readings_generated"] == 3


def test_settings_routes():
    res = client.get("/api/settings")
    assert res.status_code == 200
    assert "settings" in res.json()

    patch_res = client.patch(
        "/api/settings",
        json={"offline_mode": True, "alert_sound": False},
    )
    assert patch_res.status_code == 200
    assert patch_res.json()["settings"]["alert_sound"] is False
