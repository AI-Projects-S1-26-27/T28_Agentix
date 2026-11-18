import json
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from backend.core.config import database_path
from backend.core.metrics import metrics_collector
from backend.db.database import MeasurementRepository
from backend.db.retention import StorageManager
from backend.main import app
from backend.models.measurement import Measurement
from backend.services.analytics_service import AnalyticsService
from backend.services.export_service import ExportService


@pytest.fixture
def repo(tmp_path):
    db_file = tmp_path / "test_analytics.sqlite3"
    repository = MeasurementRepository(str(db_file))
    repository.initialize()
    # Add dummy readings
    for i in range(20):
        repository.add(
            Measurement(
                timestamp=datetime.now(timezone.utc),
                source="test",
                heart_rate=70.0 + (i % 5) * 3.0,
                eda=1.0 + (i % 4) * 0.2,
                body_temperature=36.5 + (i % 3) * 0.1,
                ambient_temperature=24.0,
                humidity=50.0,
                pm25=12.0,
            )
        )
    return repository


def test_analytics_service_quantiles(repo):
    service = AnalyticsService(repository=repo)
    summary = service.compute_summary(window_minutes=30)

    assert summary.total_samples == 20
    assert "heart_rate" in summary.signals
    hr_q = summary.signals["heart_rate"]
    assert hr_q.min_value >= 70.0
    assert hr_q.max_value <= 85.0
    assert hr_q.mean > 0.0
    assert hr_q.sample_count == 20
    assert summary.stability.stability_index >= 0.0
    assert summary.stability.stability_index <= 100.0


def test_export_service_csv(repo):
    service = ExportService(repository=repo)
    csv_str, manifest = service.generate_csv_export(limit=10)

    assert "heart_rate" in csv_str
    assert "eda" in csv_str
    assert manifest.record_count == 10
    assert manifest.format == "csv"
    assert len(manifest.sha256_checksum) == 64


def test_export_service_json(repo):
    service = ExportService(repository=repo)
    json_str, manifest = service.generate_json_export(limit=5)

    data = json.loads(json_str)
    assert len(data) == 5
    assert manifest.format == "json"
    assert manifest.record_count == 5


def test_storage_manager_diagnostics(tmp_path):
    db_file = tmp_path / "test_storage.sqlite3"
    repo = MeasurementRepository(str(db_file))
    repo.initialize()
    for _ in range(50):
        repo.add(Measurement(timestamp=datetime.now(timezone.utc), source="test", heart_rate=75.0))

    sm = StorageManager(str(db_file))
    diag = sm.get_diagnostics()

    assert diag.total_pages > 0
    assert diag.page_size > 0
    assert diag.table_counts["measurements"] == 50

    prune = sm.prune_old_readings(keep_latest=20, vacuum=True)
    assert prune.deleted_readings == 30
    assert sm.get_diagnostics().table_counts["measurements"] == 20


def test_metrics_collector():
    metrics_collector.record_ingest(12.5)
    metrics_collector.record_ingest(18.2)
    metrics_collector.record_broadcast()

    telemetry = metrics_collector.get_metrics(active_ws_count=2)
    assert telemetry.packets_ingested_total >= 2
    assert telemetry.active_websocket_connections == 2
    assert telemetry.system_status == "healthy"


def test_api_analytics_and_export_routes():
    client = TestClient(app)

    # Test analytics route
    res_analytics = client.get("/api/analytics/summary?window_minutes=15")
    assert res_analytics.status_code == 200
    assert "stability" in res_analytics.json()

    # Test export routes
    res_csv = client.get("/api/export/csv?limit=5")
    assert res_csv.status_code == 200
    assert res_csv.headers["content-type"] == "text/csv; charset=utf-8"

    res_json = client.get("/api/export/json?limit=5")
    assert res_json.status_code == 200
    assert isinstance(res_json.json(), list)

    # Test storage stats route
    res_storage = client.get("/api/storage/stats")
    assert res_storage.status_code == 200
    assert "database_size_bytes" in res_storage.json()

    # Test metrics route
    res_metrics = client.get("/api/metrics/telemetry")
    assert res_metrics.status_code == 200
    assert "packets_ingested_total" in res_metrics.json()
