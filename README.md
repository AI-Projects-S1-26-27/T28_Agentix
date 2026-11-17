# HealthSensing

HealthSensing is a local-first personal health analytics prototype for
personalized metrics and early warnings about environmental health risks. It
is a risk-screening project, not a medical diagnostic system.

## Overview & Architecture

HealthSensing integrates physiological and environmental data locally to
provide explainable risk screening without relying on external cloud dependencies:

- **Ingestion & Persistence:** Validated health and environmental readings with source provenance stored in SQLite.
- **Multimodal Risk Screening:** Rule-based environmental risk calculation (heat index, AQI) fused with physiological stress indicators (heart rate, SpO₂, body temperature).
- **Proactive Alerts:** Automatic generation, deduplication, and acknowledgment tracking for actionable early warnings.
- **Live Streaming:** Low-latency WebSocket streaming (`/ws/readings` and `/ws/events`) for dashboard telemetry.
- **Scenario Simulation:** Controlled simulation of heatwaves, pollution spikes, fatigue, and nominal conditions.

## Run the backend locally

Install dependencies from `requirements.txt`, then run:

```text
uvicorn backend.main:app --reload
```

The default database is `data/healthsensing.sqlite3`. Set
`HEALTHSENSING_DATABASE_PATH` to use another local path. Do not commit raw
wearable exports, secrets, or other sensitive health data.

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### Simulator Integration

The simulator can run a finite integration stream and submit records to the
local API from Python:

```python
from simulator.simulator import run_simulator

run_simulator(
    "heatwave",
    count=10,
    endpoint="http://127.0.0.1:8000/api/readings",
)
```

Alternatively, scenario bursts can be triggered directly via HTTP:

```bash
curl -X POST "http://127.0.0.1:8000/api/scenarios/run" \
     -H "Content-Type: application/json" \
     -d '{"scenario_id": "heatwave", "count": 10}'
```

### Real-time Streaming

Dashboard clients can subscribe to `ws://127.0.0.1:8000/ws/readings` or
`ws://127.0.0.1:8000/ws/events`. Each persisted reading and derived risk
state is broadcast in real time.

See [BACKEND_PROGRESS.md](BACKEND_PROGRESS.md) for the detailed implementation
status, architecture, API table, and cross-module integration details.
