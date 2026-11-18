from datetime import datetime, timezone
import pytest

from backend.db.database import MeasurementRepository
from backend.models.measurement import Measurement
from backend.services.alert_service import AlertService
from backend.services.data_service import DataService
from backend.services.risk_service import RiskService


def make_measurement(**overrides):
    values = {
        "timestamp": datetime(2026, 11, 1, 15, 0, tzinfo=timezone.utc),
        "source": "simulator",
        "scenario": "normal",
        "heart_rate": 72,
        "spo2": 98,
        "body_temperature": 36.8,
        "activity_level": 0.4,
        "ambient_temperature": 22,
        "humidity": 45,
        "aqi": 45,
        "pm25": 10,
    }
    values.update(overrides)
    return Measurement(**values)


def test_risk_service_evaluates_normal_conditions(tmp_path):
    repo = MeasurementRepository(tmp_path / "test.db")
    repo.initialize()
    service = RiskService(repository=repo)

    m = make_measurement()
    risk = service.evaluate_reading(m)

    assert risk.risk_level == "low"
    assert risk.severity == 0
    assert "low" in risk.explanation.lower()


def test_risk_service_escalates_on_heat_stress(tmp_path):
    repo = MeasurementRepository(tmp_path / "test.db")
    repo.initialize()
    service = RiskService(repository=repo)

    # Ambient 36 C, 80% humidity (dangerous heat index) + heart rate 115 bpm
    m = make_measurement(
        ambient_temperature=36,
        humidity=80,
        heart_rate=115,
        body_temperature=38.0,
    )
    risk = service.evaluate_reading(m)

    assert risk.severity >= 3
    assert risk.risk_level in ("very_high", "severe")
    assert "heart_rate" in risk.contributing_signals
    assert "ambient_temperature" in risk.contributing_signals
    assert "Elevated physiological risk" in risk.explanation


def test_alert_service_triggers_on_high_risk(tmp_path):
    repo = MeasurementRepository(tmp_path / "test.db")
    repo.initialize()
    risk_svc = RiskService(repository=repo)
    alert_svc = AlertService(repository=repo)

    # Trigger air quality hazard: aqi 350, pm25 250
    m = make_measurement(aqi=350, pm25=250, spo2=92)
    risk = risk_svc.evaluate_reading(m)

    alert = alert_svc.evaluate_and_create_alert(m, risk, reading_id=1)
    assert alert is not None
    assert alert.level == "critical"
    assert alert.category in ("air_quality", "multimodal")

    saved_alert = alert_svc.persist_alert_if_eligible(alert)
    assert saved_alert is not None
    assert saved_alert.id is not None
    assert saved_alert.acknowledged is False


@pytest.mark.anyio
async def test_data_service_orchestrates_flow(tmp_path):
    repo = MeasurementRepository(tmp_path / "test.db")
    repo.initialize()
    data_svc = DataService(repository=repo)

    m = make_measurement(ambient_temperature=38, humidity=75, heart_rate=110)
    result = await data_svc.ingest_reading(m)

    assert result["reading"].id is not None
    assert result["risk"].id is not None
    assert result["risk"].severity >= 2
    assert result["alert"] is not None
    assert result["alert"].id is not None

    # Check persistence
    assert len(repo.recent()) == 1
    assert len(repo.recent_risks()) == 1
    assert len(repo.recent_alerts()) == 1


@pytest.mark.anyio
async def test_data_service_persists_personalized_anomaly_without_health_alert(tmp_path):
    repo = MeasurementRepository(tmp_path / "test.db")
    repo.initialize()
    repo.save_baseline("S2", "heart_rate_bpm", 70.0, 5.0, 600, "stable")
    repo.save_baseline("S2", "eda_microsiemens", 1.0, 0.2, 600, "stable")
    repo.save_baseline("S2", "body_temperature_c", 36.8, 1.0, 600, "stable")
    data_service = DataService(repository=repo)

    measurement = make_measurement(
        source="wesad",
        participant_id="S2",
        protocol_label=None,
        heart_rate=90.0,
        eda=1.2,
    )
    result = await data_service.ingest_reading(measurement)

    anomaly = result["risk"].components["personalized_anomaly"]
    assert anomaly["overall_status"] == "anomaly_detected"
    assert anomaly["signals"]["heart_rate_bpm"]["anomaly"] == 1
    assert result["risk"].severity == 0
    assert result["alert"] is None

    stored_risk = repo.get_risk_by_reading_id(result["reading"].id)
    assert stored_risk is not None
    assert stored_risk.components["personalized_anomaly"]["overall_anomaly"] == 1
