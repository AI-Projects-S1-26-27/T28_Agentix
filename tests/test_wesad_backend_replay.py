from urllib.parse import urlsplit

import pandas as pd
from fastapi.testclient import TestClient

from backend.api.routes.current import get_repository
from backend.db.database import MeasurementRepository
from backend.main import app
from data.src.replay_wesad_to_backend import (
    build_reading_payload,
    replay_wesad_files,
)


def test_build_reading_payload_preserves_wesad_metadata_and_missing_values():
    payload = build_reading_payload({
        "participant_id": "S2",
        "timestamp": "2017-05-22T07:15:25Z",
        "heart_rate_bpm": None,
        "eda_microsiemens": 1.25,
        "body_temperature_c": 35.4,
        "stress_label": 2,
    })

    assert payload["source"] == "wesad"
    assert payload["participant_id"] == "S2"
    assert payload["protocol_label"] == 2
    assert payload["heart_rate"] is None
    assert payload["eda"] == 1.25
    assert payload["body_temperature"] == 35.4
    assert payload["timestamp"] == "2017-05-22T07:15:25Z"


def test_build_reading_payload_preserves_unmapped_protocol_codes():
    payload = build_reading_payload({
        "participant_id": "S2",
        "timestamp": "2017-05-22T07:15:25Z",
        "heart_rate_bpm": 70.0,
        "eda_microsiemens": 1.0,
        "body_temperature_c": 35.0,
        "stress_label": 7,
    })

    assert payload["protocol_label"] == 7


def test_replay_calibrates_and_posts_rows_through_backend(tmp_path):
    csv_path = tmp_path / "wesad_s2.csv"
    frame = pd.DataFrame({
        "participant_id": ["S2"] * 10,
        "timestamp": pd.date_range(
            "2017-05-22T07:15:25Z", periods=10, freq="s"
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "heart_rate_bpm": [70, 71, 69, 70, 72, 70, 71, 69, 100, 95],
        "eda_microsiemens": [1.0, 1.1, 0.9, 1.0, 1.05, 0.98, 1.02, 0.99, 2.0, 1.8],
        "body_temperature_c": [34.5, 34.6, 34.4, 34.5, 34.6, 34.5, 34.4, 34.5, 35.0, 35.1],
        "stress_label": [1] * 8 + [2] * 2,
    })
    frame.to_csv(csv_path, index=False)

    repository = MeasurementRepository(tmp_path / "backend.sqlite3")
    repository.initialize()
    previous_override = app.dependency_overrides.get(get_repository)

    try:
        app.dependency_overrides[get_repository] = lambda: repository
        with TestClient(app) as client:
            def post_json(endpoint, payload, timeout):
                response = client.post(urlsplit(endpoint).path, json=dict(payload))
                response.raise_for_status()
                return response.json()

            summary = replay_wesad_files(
                [csv_path],
                base_url="http://testserver",
                max_readings=2,
                request_sender=post_json,
            )
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(get_repository, None)
        else:
            app.dependency_overrides[get_repository] = previous_override

    assert summary["participants"] == ["S2"]
    assert summary["baselines_saved"] == 3
    assert len(summary["baseline_statuses"]) == 3
    assert summary["readings_sent"] == 2
    assert summary["eligible_readings"] == 5
    assert summary["input_readings"] == 10

    saved_readings = repository.recent(limit=2)
    assert len(saved_readings) == 2
    assert all(reading.source == "wesad" for reading in saved_readings)
    assert all(reading.participant_id == "S2" for reading in saved_readings)
    assert all(reading.protocol_label == 1 for reading in saved_readings)
    assert all(reading.timestamp.second >= 30 for reading in saved_readings)

    latest_risk = repository.get_risk_by_reading_id(saved_readings[0].id)
    assert latest_risk is not None
    assert latest_risk.components["personalized_anomaly"]["overall_status"] == "unevaluable"
