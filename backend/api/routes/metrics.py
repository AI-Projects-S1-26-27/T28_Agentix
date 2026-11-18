from fastapi import APIRouter

from backend.core.metrics import TelemetryMetrics, metrics_collector
from backend.websocket.live_stream import manager

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/telemetry", response_model=TelemetryMetrics)
def get_telemetry_metrics() -> TelemetryMetrics:
    active_clients = len(getattr(manager, "_connections", []))
    return metrics_collector.get_metrics(active_ws_count=active_clients)
