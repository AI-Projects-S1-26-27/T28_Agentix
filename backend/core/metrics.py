import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class LatencyBreakdown(BaseModel):
    last_eval_ms: float
    avg_eval_ms: float
    p95_eval_ms: float


class TelemetryMetrics(BaseModel):
    server_uptime_seconds: float
    packets_ingested_total: int
    packets_per_second: float
    evaluation_latency: LatencyBreakdown
    active_websocket_connections: int
    broadcast_messages_total: int
    system_status: str
    recorded_at: str


class MetricsCollector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsCollector, cls).__new__(cls)
            cls._instance._start_time = time.time()
            cls._instance._packets_count = 0
            cls._instance._broadcasts_count = 0
            cls._instance._latencies = []
            cls._instance._max_latency_history = 100
        return cls._instance

    def record_ingest(self, latency_ms: float):
        self._packets_count += 1
        self._latencies.append(latency_ms)
        if len(self._latencies) > self._max_latency_history:
            self._latencies.pop(0)

    def record_broadcast(self):
        self._broadcasts_count += 1

    def get_metrics(self, active_ws_count: int = 0) -> TelemetryMetrics:
        uptime = max(0.1, time.time() - self._start_time)
        rate = round(self._packets_count / uptime, 2)

        if self._latencies:
            last_lat = round(self._latencies[-1], 2)
            avg_lat = round(sum(self._latencies) / len(self._latencies), 2)
            sorted_lat = sorted(self._latencies)
            p95_idx = int(len(sorted_lat) * 0.95)
            p95_lat = round(sorted_lat[min(p95_idx, len(sorted_lat) - 1)], 2)
        else:
            last_lat, avg_lat, p95_lat = 0.0, 0.0, 0.0

        return TelemetryMetrics(
            server_uptime_seconds=round(uptime, 1),
            packets_ingested_total=self._packets_count,
            packets_per_second=rate,
            evaluation_latency=LatencyBreakdown(
                last_eval_ms=last_lat,
                avg_eval_ms=avg_lat,
                p95_eval_ms=p95_lat,
            ),
            active_websocket_connections=active_ws_count,
            broadcast_messages_total=self._broadcasts_count,
            system_status="healthy" if avg_lat < 100.0 else "degraded",
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )


metrics_collector = MetricsCollector()
