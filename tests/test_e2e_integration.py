from datetime import datetime, timezone
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_end_to_end_scenario_to_alert_lifecycle():
    # 1. Run heatwave scenario simulation via API
    run_res = client.post(
        "/api/scenarios/run",
        json={"scenario_id": "heatwave", "count": 5},
    )
    assert run_res.status_code == 200
    assert run_res.json()["readings_generated"] == 5

    # 2. Verify readings were persisted and ordered recent-first
    readings_res = client.get("/api/history/readings?limit=10")
    assert readings_res.status_code == 200
    readings = readings_res.json()
    assert len(readings) >= 5
    heatwave_readings = [r for r in readings if r.get("scenario") == "heatwave"]
    assert len(heatwave_readings) >= 1
    assert heatwave_readings[0]["source"] == "simulator"

    # 3. Verify risk assessments were automatically generated
    risks_res = client.get("/api/risks/recent?limit=5")
    assert risks_res.status_code == 200
    risks = risks_res.json()
    assert len(risks) >= 5
    # In heatwave, ambient temp is 36-45C so severity should be >= 2
    assert any(risk["severity"] >= 2 for risk in risks)

    # 4. Verify alerts were triggered by high heat severity
    alerts_res = client.get("/api/alerts")
    assert alerts_res.status_code == 200
    alerts = alerts_res.json()
    assert len(alerts) > 0
    heat_alert = next((a for a in alerts if a["category"] in ("heat_stress", "multimodal")), None)
    assert heat_alert is not None
    assert heat_alert["level"] in ("warning", "critical")

    # 5. Acknowledge the alert
    ack_res = client.post(f"/api/alerts/{heat_alert['id']}/acknowledge")
    assert ack_res.status_code == 200
    assert ack_res.json()["acknowledged"] is True

    # 6. Verify summary stats
    stats_res = client.get("/api/history/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total_count"] >= 5
    assert stats["avg_ambient_temperature"] > 30.0

    # 7. Check latest reading summary
    summary_res = client.get("/api/readings/latest")
    assert summary_res.status_code == 200
    summary = summary_res.json()
    assert "reading" in summary
    assert "risk" in summary
