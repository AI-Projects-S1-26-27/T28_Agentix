# Backend Progress

**Owner:** Vedant Nichal  
**Scope:** Backend architecture, data interfaces, persistence, service layer, REST APIs, and live WebSocket streaming  
**Status:** Complete backend implementation and cross-module integration verified end-to-end.

---

## 1. Architecture Overview

The HealthSensing backend provides a local-first, privacy-preserving ingestion, persistence, and early-warning delivery pipeline. It bridges sensor ingestion (wearables and Python simulator), multimodal AI baseline and anomaly detection, explicit contextual rule evaluation, and the real-time React dashboard.

```text
[Wearables / Simulator]
          │ (HTTP POST /api/readings)
          ▼
   [DataService Ingestion]
   ├── 1. SQLite Persistence (`measurements` table)
   ├── 2. Multimodal Risk Screening (`RiskService` + `risk_engine`)
   ├── 3. Derived Risk Persistence (`risk_assessments` table)
   ├── 4. Proactive Alert Generation & Deduplication (`AlertService` -> `alerts` table)
   └── 5. Live WebSocket Broadcast (`/ws/readings`, `/ws/events`)
          │
          ▼
   [React Dashboard & UI Clients]
```

---

## 2. Completed Deliverables

### 2.1 Shared Data Models & Validations
Implemented in [backend/models/](backend/models/):

- `Measurement`: Timezone-aware UTC timestamp, source provenance, scenario label, physiological metrics (`heart_rate`, `spo2`, `body_temperature`, `activity_level`, `eda`), and environmental conditions (`ambient_temperature`, `humidity`, `aqi`, `pm25`). Full validation with explicit `null` support for unobserved channels.
- `RiskAssessment` & `RiskComponent`: Multimodal risk status (`low`, `moderate`, `high`, `very_high`, `severe`), numerical severity levels (0–4), contributing signals, unassessed inputs, and human-interpretable explanations.
- `Alert`: Triggered warnings and critical notifications with deduplication, categories (`heat_stress`, `air_quality`, `physiological`, `multimodal`), and user acknowledgment tracking.
- `ScenarioInfo`, `ScenarioRunRequest`, `ScenarioRunResponse`: Definitions and parameter ranges for simulator scenarios (`normal`, `heatwave`, `pollution`, `fatigue`).

### 2.2 Local SQLite Persistence
Implemented in [backend/db/database.py](backend/db/database.py) and [backend/db/schema.py](backend/db/schema.py):

| Table | Purpose | Key Indexes |
|---|---|---|
| `measurements` | Time-series sensor and environmental observations | `idx_measurements_timestamp` |
| `risk_assessments` | Persisted risk screening evaluations linked to readings | `idx_risk_timestamp`, `idx_risk_reading_id` |
| `alerts` | Actionable risk alerts with acknowledgment flags | `idx_alerts_timestamp`, `idx_alerts_acknowledged` |
| `personal_baselines` | User-calibrated baseline statistics (mean, std, stability) | `UNIQUE(participant_id, signal)` |

Configurable via `HEALTHSENSING_DATABASE_PATH` (defaults to local `data/healthsensing.sqlite3`).

### 2.3 Service Layer
Implemented in [backend/services/](backend/services/):

- **`RiskService`**: Combines environmental screening (heat index calculation and AQI categorization via `risk_engine`) with physiological stress indicators (tachycardia during heatwaves, SpO₂ drops during pollution spikes, extreme vitals).
- **`AlertService`**: Automatically produces alerts for high-risk conditions (`severity >= 2`), implements rate limiting and 30-second category deduplication, and handles acknowledgment lifecycle.
- **`DataService`**: Central orchestrator coordinating database writes, risk evaluation, alert generation, and WebSocket broadcast in a single transactional step.

### 2.4 REST API Endpoints
Implemented in [backend/api/routes/](backend/api/routes/) and registered in [backend/main.py](backend/main.py):

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/api/readings` | Ingests a new reading, processes risk & alerts, broadcasts live |
| `GET` | `/api/readings` | Retrieves recent historical readings (with limit parameter) |
| `GET` | `/api/readings/latest` | Returns latest reading with associated risk and active alerts |
| `GET` | `/api/risks/recent` | Retrieves recent risk assessment records |
| `GET` | `/api/risks/latest` | Returns latest risk evaluation status |
| `POST` | `/api/risks/evaluate` | On-demand risk assessment without persistence |
| `GET` | `/api/alerts` | Lists recent alerts with optional `unacknowledged_only` filter |
| `GET` | `/api/alerts/active` | Lists unacknowledged active alerts |
| `POST` | `/api/alerts/{id}/acknowledge` | Marks an alert as acknowledged |
| `GET` | `/api/history/readings` | Historical query with `start`, `end`, `limit`, `offset` |
| `GET` | `/api/history/risks` | Historical query of past risk states |
| `GET` | `/api/history/stats` | Aggregate dataset statistics (averages, mins, maxs) |
| `GET` | `/api/baseline` | Retrieves personal baseline parameters for a participant |
| `POST` | `/api/baseline/calibrate` | Upserts calibrated baseline statistics |
| `GET` | `/api/scenarios` | Lists available simulator scenarios and parameter bounds |
| `POST` | `/api/scenarios/run` | Triggers a simulated run of generated data into the backend |
| `GET` | `/api/settings` | Returns system operational status and local configuration |
| `PATCH` | `/api/settings` | Updates runtime thresholds and preferences |

### 2.5 Real-Time WebSocket Streaming
Implemented in [backend/websocket/](backend/websocket/):

- Multi-endpoint streaming on `/ws/readings` and `/ws/events`.
- Typed event architecture: `reading_processed`, `ping`/`pong`, `status_response`.
- Backward compatibility: broadcast payloads contain both nested event objects and root measurement attributes for earlier client builds.
- Connection tracking and client disconnection isolation.

---

## 3. Verification & Test Suite

The test suite covers database persistence, schema validation, service coordination, API routing, WebSocket lifecycle, and end-to-end system flow:

- [tests/test_backend.py](tests/test_backend.py): Schema validation, range constraints, SQLite repository operations, alerts, and baselines.
- [tests/test_services.py](tests/test_services.py): `RiskService` physiological fusion, `AlertService` triggering, and `DataService` ingestion pipeline.
- [tests/test_api.py](tests/test_api.py): Ingestion, latest summary, risks, alerts, history, baseline calibration, and scenario endpoints.
- [tests/test_websocket_protocol.py](tests/test_websocket_protocol.py): Bidirectional communication, `ping`/`pong` heartbeats, and broadcast delivery.
- [tests/test_e2e_integration.py](tests/test_e2e_integration.py): Full vertical slice from scenario generation to persistence, risk screening, alert triggering, and acknowledgment.
- [tests/test_simulator_contract.py](tests/test_simulator_contract.py): Verification of simulator records against backend ingestion contracts.
- [tests/test_risk_engine.py](tests/test_risk_engine.py): Heat index calculation, AQI categorization, and explanation generation.

To run tests:

```bash
python -m pytest tests/test_backend.py tests/test_services.py tests/test_api.py tests/test_websocket_protocol.py tests/test_e2e_integration.py tests/test_simulator_contract.py tests/test_risk_engine.py
```

All 36 tests pass cleanly.

---

## 4. Integration Matrix for Team Members

- **Animish Kumar (Person 3 - Primary AI/ML Pipeline):**
  - Baseline parameters generated from resting history can be stored and queried via `POST /api/baseline/calibrate` and `GET /api/baseline`.
  - ML inference responses from `ai/interfaces/` map directly into the reading and risk payloads.
- **Avani Asati (Person 5 - Knowledge Base & Risk Fusion):**
  - Environmental risk functions from `risk_engine/` are integrated into `RiskService`.
  - Multimodal rules combine environmental context (heat/AQI) with physiological strain (elevated HR, SpO₂ desaturation) into explainable states.
- **Chaitanya Chalith (Person 7 - React Dashboard):**
  - Dashboard can subscribe to `ws://127.0.0.1:8000/ws/readings` or `ws://127.0.0.1:8000/ws/events`.
  - Overview cards and live charts can consume `GET /api/readings/latest`, `GET /api/risks/latest`, `GET /api/alerts/active`, and historical series at `GET /api/history/readings`.
  - Alert acknowledgment is wired to `POST /api/alerts/{id}/acknowledge`.
- **Aryan Saini (Person 2 - Simulator):**
  - The simulator can stream data to `POST /api/readings` or trigger controlled scenario bursts using `POST /api/scenarios/run` with `heatwave`, `pollution`, `fatigue`, or `normal`.
- **Kartikeya Singh & Siddhartha Venkata Arava Reddy (Data & Evaluation):**
  - Cleaned dataset streams or benchmark replay tools can ingest historical time-series using `POST /api/readings` with `source: "dataset"` or `"wesad"`.

---

## 5. Local Running Instructions

1. Install backend requirements:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Start the FastAPI development server:
   ```bash
   uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
   ```
3. Interactive OpenAPI documentation is accessible at `http://127.0.0.1:8000/docs`.
4. Trigger a simulated scenario burst:
   ```bash
   curl -X POST "http://127.0.0.1:8000/api/scenarios/run" -H "Content-Type: application/json" -d "{\"scenario_id\": \"heatwave\", \"count\": 10}"
   ```
