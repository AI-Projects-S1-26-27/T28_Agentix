from datetime import datetime, timezone
import sqlite3

import pytest

from backend.db.database import MeasurementRepository
from backend.models.alert import Alert
from backend.models.measurement import Measurement
from backend.models.risk import RiskAssessment


def make_measurement(**overrides):
    values = {
        "timestamp": datetime(2026, 11, 1, 15, 0, tzinfo=timezone.utc),
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
        "eda": 1.2,
    }
    values.update(overrides)
    return Measurement(**values)


def test_measurement_requires_timezone_and_valid_ranges():
    with pytest.raises(ValueError, match="timezone"):
        make_measurement(timestamp=datetime(2026, 11, 1, 15, 0))

    with pytest.raises(ValueError):
        make_measurement(spo2=101)

    assert make_measurement(protocol_label=7).protocol_label == 7
    with pytest.raises(ValueError):
        make_measurement(protocol_label=2.5)


def test_repository_persists_optional_fields_and_orders_recent_first(tmp_path):
    repository = MeasurementRepository(tmp_path / "health.sqlite3")
    repository.initialize()
    repository.add(
        make_measurement(
            timestamp=datetime(2026, 11, 1, 14, 0, tzinfo=timezone.utc)
        )
    )
    latest = repository.add(
        make_measurement(
            timestamp=datetime(2026, 11, 1, 15, 0, tzinfo=timezone.utc),
            spo2=None,
        )
    )

    readings = repository.recent()

    assert readings[0].id == latest.id
    assert readings[0].spo2 is None
    assert readings[0].eda == 1.2
    assert readings[1].source == "simulator"


def test_repository_migrates_existing_database_for_wesad_metadata(tmp_path):
    database_path = tmp_path / "existing.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                scenario TEXT,
                heart_rate REAL,
                spo2 REAL,
                body_temperature REAL,
                activity_level REAL,
                ambient_temperature REAL,
                humidity REAL,
                aqi REAL,
                pm25 REAL
            )
            """
        )

    repository = MeasurementRepository(database_path)
    repository.initialize()
    saved = repository.add(
        make_measurement(participant_id="S2", protocol_label=2)
    )

    stored = repository.recent(limit=1)[0]
    assert saved.participant_id == "S2"
    assert stored.participant_id == "S2"
    assert stored.protocol_label == 2


def test_repository_persists_risk_assessment(tmp_path):
    repository = MeasurementRepository(tmp_path / "health.sqlite3")
    repository.initialize()
    saved_m = repository.add(make_measurement())

    assessment = RiskAssessment(
        timestamp=datetime(2026, 11, 1, 15, 0, tzinfo=timezone.utc),
        reading_id=saved_m.id,
        risk_level="moderate",
        severity=1,
        components={"air_quality": {"category": "moderate", "severity": 1, "risk_level": "moderate"}},
        unavailable_inputs=["heat_index"],
        contributing_signals=["aqi"],
        explanation="Environmental screening is moderate.",
    )

    saved_risk = repository.add_risk_assessment(assessment)
    assert saved_risk.id is not None
    assert saved_risk.risk_level == "moderate"
    assert saved_risk.components["air_quality"]["category"] == "moderate"

    recent = repository.recent_risks(10)
    assert len(recent) == 1
    assert recent[0].id == saved_risk.id

    by_reading = repository.get_risk_by_reading_id(saved_m.id)
    assert by_reading is not None
    assert by_reading.id == saved_risk.id


def test_repository_persists_and_acknowledges_alerts(tmp_path):
    repository = MeasurementRepository(tmp_path / "health.sqlite3")
    repository.initialize()
    saved_m = repository.add(make_measurement())

    alert = Alert(
        timestamp=datetime(2026, 11, 1, 15, 0, tzinfo=timezone.utc),
        reading_id=saved_m.id,
        level="warning",
        category="heat_stress",
        message="High heat index detected with elevated heart rate",
        explanation="Heat index exceeds 35C while heart rate is 105 bpm.",
        contributing_signals=["heart_rate", "ambient_temperature", "humidity"],
        acknowledged=False,
    )

    saved_alert = repository.add_alert(alert)
    assert saved_alert.id is not None
    assert saved_alert.acknowledged is False

    active = repository.recent_alerts(unacknowledged_only=True)
    assert len(active) == 1

    ack = repository.acknowledge_alert(saved_alert.id)
    assert ack is not None
    assert ack.acknowledged is True

    active_after = repository.recent_alerts(unacknowledged_only=True)
    assert len(active_after) == 0


def test_repository_manages_personal_baselines(tmp_path):
    repository = MeasurementRepository(tmp_path / "health.sqlite3")
    repository.initialize()

    res = repository.save_baseline("user1", "heart_rate_bpm", 72.5, 3.2, 120, "stable")
    assert res["signal"] == "heart_rate_bpm"

    baselines = repository.get_baselines("user1")
    assert len(baselines) == 1
    assert baselines[0]["mean"] == 72.5
