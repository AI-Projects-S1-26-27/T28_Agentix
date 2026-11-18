# HealthSensing: Personal Health Companion
## Privacy-Preserving Edge-AI Analytics & Early Warning for Environmental Health Risks

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF.svg)](https://vitejs.dev/)
[![SQLite](https://img.shields.io/badge/SQLite-WAL_Mode-003B57.svg)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-Academic_Use-lightgrey.svg)]()

> **Course:** CS F407 / U407 Artificial Intelligence  
> **Track:** Application Track  
> **Team:** AGENTIX (Team 28)  
> **Repository:** [https://github.com/animish-k/HealthSensing](https://github.com/animish-k/HealthSensing)

---

## 🌐 Live Production Deployment Links

| Component | Host / Platform | Live Endpoint URL | Status |
| :--- | :--- | :--- | :--- |
| **Interactive Dashboard** | Vercel Edge | [https://healthsensing.vercel.app](https://healthsensing.vercel.app) | 🟢 Live & Operational |
| **Backend Web Service** | Render (Persistent SSD) | [https://healthsensing.onrender.com](https://healthsensing.onrender.com) | 🟢 Live & Operational |
| **OpenAPI / Swagger Docs**| Render | [https://healthsensing.onrender.com/docs](https://healthsensing.onrender.com/docs) | 🟢 Interactive API Explorer |
| **Health Probe** | Render | [https://healthsensing.onrender.com/health](https://healthsensing.onrender.com/health) | 🟢 `{"status": "ok"}` |
| **Live Telemetry Stream** | Render (WebSocket) | `wss://healthsensing.onrender.com/ws/readings` | 🟢 5s Replay Stream |


---

## 1. System Overview

**HealthSensing** is an edge-first physiological and environmental early-warning companion. It continuously bridges high-frequency wearable physiological telemetry (photoplethysmography heart rate, electrodermal activity, skin temperature) with real-world ambient atmospheric indicators (ambient wet-bulb temperature, relative humidity, particulate air pollution $\text{PM}_{2.5}$ / $\text{PM}_{10}$).

### Why HealthSensing?
1. **The Failure of Generic Population Thresholds:** Traditional consumer wearables apply static alerts (e.g., $HR > 100\text{ bpm}$). In empirical testing across the multimodal WESAD benchmark, fixed statistical thresholds fail on 50% of participants entirely and achieve an average recall of only **22.22%**.
2. **Contextual Isolation:** Existing platforms analyze vitals in a vacuum. An elevated heart rate of 105 bpm is benign during light indoor activity, but lethal when ambient temperatures reach 42 °C with high relative humidity.
3. **Local-First Privacy Architecture:** Biometric telemetry never leaves the user's host environment. All personal baselines, rolling Bayesian statistics, and risk evaluations execute strictly locally or on user-controlled persistent infrastructure.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HEALTHSENSING ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────────┘

 [ Multimodal Sensors ]                 [ Ambient Weather APIs ]
  • Empatica E4 Wristband                • Open-Meteo Observations
  • PPG Heart Rate (1 Hz)                • Ambient Temp & Humidity
  • EDA & Skin Temperature               • PM2.5 / PM10 Particulates
            │                                       │
            ▼                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                   INGESTION & DATA CONTRACT LAYER                         │
 │        Strict Pydantic v2 Parsing • Timestamp Normalization               │
 └─────────────────────┬─────────────────────────────────┬───────────────────┘
                       │                                 │
                       ▼                                 ▼
 ┌───────────────────────────────────────┐ ┌─────────────────────────────────┐
 │     PERSONALIZED ML BASELINE ENGINE   │ │    ENVIRONMENTAL RISK ENGINE    │
 │ • Motion-Gating (|a| < θ_motion)      │ │ • NOAA Heat Index Calculation   │
 │ • Welford's Algorithm Online Baselines│ │ • US EPA AQI Piecewise Formula  │
 │ • Dynamic Threshold: τ = μ + k·σ      │ │ • Deterministic Guardrails      │
 └───────────────────┬───────────────────┘ └────────────────┬────────────────┘
                     │                                      │
                     └───────────────────┬──────────────────┘
                                         │
                                         ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                    DUAL-TRACK RISK FUSION ENGINE                          │
 │         65% Deterministic Guardrails  +  35% Personalized Model           │
 │       Tiers: LOW [0,25) • MODERATE [25,50) • HIGH [50,75) • CRITICAL [75,100]│
 └─────────────────────┬─────────────────────────────────┬───────────────────┘
                       │                                 │
                       ▼                                 ▼
 ┌───────────────────────────────────────┐ ┌─────────────────────────────────┐
 │       EMBEDDED SQLITE STORAGE         │ │     DUAL-PROTOCOL STREAMING     │
 │ • WAL Journaling Mode                 │ │ • REST API (History, Baselines) │
 │ • Multi-Table Relational Schema       │ │ • WebSockets (/ws/readings)     │
 └───────────────────────────────────────┘ └────────────────┬────────────────┘
                                                            │
                                                            ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                   REACT 18 + VITE FRONTEND DASHBOARD                      │
 │ • Recharts SVG Timeseries  • Reactive Alert Stream  • Scenario Controller │
 └───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Team Members & Contributions (Team 28 - AGENTIX)

| Member | Role | Core Technical Ownership |
| :--- | :--- | :--- |
| **Animish Kumar** | AI/ML Pipeline Lead | Baseline anomaly detector, Random Forest benchmark, ML interfaces, evaluation scripts |
| **Aryan Saini** | Simulation Lead | Controlled environmental scenarios (heatwave, pollution, fatigue), generator contracts |
| **Avani Asati** | Rule Engine Lead | Knowledge base, NOAA heat index & EPA AQI formulas, deterministic risk rules |
| **Vedant Nichal** | Backend & Systems Lead | FastAPI service architecture, SQLite WAL persistence, REST routes, WebSocket streaming, deployment pipeline |
| **Kartikeya Singh** | Data Engineering Lead | WESAD 1 Hz preprocessing, Open-Meteo Ljubljana sync, signal alignment, feature engineering |
| **Chaitanya Chalith** | Frontend Lead | React 18 dashboard architecture, Vite build, Recharts dynamic timeseries, user context store |
| **Siddharth Venkata** | ML Evaluation Lead | Leave-One-Participant-Out (LOPO) cross-validation, PR/ROC curves, latency benchmarks |

---

## 3. Technology Stack & Engineering Justifications

| Layer | Technology | Version | Technical Justification |
| :--- | :--- | :--- | :--- |
| **Backend Framework** | FastAPI (ASGI) | `>=0.110` | Non-blocking `asyncio` loop, native WebSocket concurrency, sub-millisecond route dispatch, auto-generated OpenAPI documentation. |
| **Validation Core** | Pydantic | `>=2.6` | Rust-backed `pydantic-core` parsing engine ensures high-throughput type safety and schema validation for high-frequency sensor streams. |
| **Persistence Engine** | SQLite | `WAL Mode` | Zero-maintenance embedded relational database. Write-Ahead Logging (WAL) permits concurrent background writes without blocking API reads. Configurable via `HEALTHSENSING_DATABASE_PATH`. |
| **ML & Statistics** | NumPy, SciPy, Scikit-Learn | `>=1.26` | Numerically stable online variance via Welford's algorithm, LOPO cross-validation routines, and Random Forest baseline comparison. |
| **Frontend Framework** | React 18 + Vite | `^18.2` / `^5.4` | Lightning-fast Hot Module Replacement (HMR), tree-shaken production bundles (20s build), declarative state management. |
| **Data Visualization**| Recharts | `^2.15` | Composable SVG time-series charting optimized for high-frequency live WebSocket push updates. |
| **Styling & UI** | Tailwind CSS + Lucide Icons | `^3.4` / `^0.344` | Utility-first responsive design, dark/light health dashboard theme, accessible status indicators. |
| **Deployment** | Render + Vercel | Production Cloud | Persistent Python ASGI service with 1 GB attached disk (Render) + edge routing and WebSocket rewrites (Vercel). |

---

## 4. Datasets Grounding

HealthSensing is strictly trained, validated, and evaluated on genuine empirical datasets:

1. **WESAD (Wearable Stress and Affect Detection):**
   - **Hardware:** Empatica E4 medical-grade research wristband.
   - **Channels Extracted:** Photoplethysmography Heart Rate (PPG/BVP at 1 Hz), Electrodermal Activity (EDA), Skin Temperature (°C), and 3-axis Accelerometer ($x, y, z$).
   - **Participants:** Multi-subject laboratory recordings ($N=4$: S2, S3, S4, S5) across baseline resting and acute stress protocols.
   - *Note:* WESAD does **not** include $\text{SpO}_2$. Blood oxygenation is strictly evaluated within controlled simulator stress scenarios.
2. **Ljubljana Ambient Environmental Dataset:**
   - **Source:** Historical meteorological and atmospheric station records obtained via Open-Meteo API (`data/processed/ljubljana_real.csv`).
   - **Parameters:** Ambient Temperature (°C), Relative Humidity (%), Particulate Matter $\text{PM}_{2.5}$ and $\text{PM}_{10}$ ($\mu\text{g}/\text{m}^3$).
   - *Note:* Ljubljana data provides external atmospheric context; it contains no human biometrics.
3. *Hardware Disclaimer:* HealthSensing does **not** collect or stream data from unvalidated consumer smartwatches (e.g. Pixel Watch) or live edge hardware.

---

## 5. Mathematical & Algorithmic Formulations

### 5.1 Personalized Bayesian Rolling Baseline
Rather than comparing against static population means, baseline statistics $\mu_t$ and $\sigma_t^2$ are updated dynamically using **Welford's Algorithm** exclusively during motion-gated periods:

$$\mu_t = \mu_{t-1} + \frac{x_t - \mu_{t-1}}{n}, \quad M_{2,t} = M_{2,t-1} + (x_t - \mu_{t-1})(x_t - \mu_t), \quad \sigma_t^2 = \frac{M_{2,t}}{n - 1}$$

### 5.2 Motion Gating
To eliminate motion artifacts, baseline calibration updates only when acceleration magnitude satisfies:

$$|a_t| = \sqrt{a_{x,t}^2 + a_{y,t}^2 + a_{z,t}^2} < \theta_{\text{motion}}$$

### 5.3 Dynamic Adaptive Thresholding
Anomalies trigger when physiological signals exceed individual-specific deviation bounds:

$$\tau_{\text{upper}}(t) = \mu_t + k \cdot \sigma_t$$

### 5.4 Dual-Track Risk Fusion Engine
System risk is classified into four discrete operational tiers:

$$\text{Tier} = \begin{cases} \text{LOW}, & S_{\text{fused}} < 25 \\ \text{MODERATE}, & 25 \le S_{\text{fused}} < 50 \\ \text{HIGH}, & 50 \le S_{\text{fused}} < 75 \\ \text{CRITICAL}, & S_{\text{fused}} \ge 75 \end{cases}$$

Where $S_{\text{fused}} = 0.65 \cdot S_{\text{rules}} + 0.35 \cdot S_{\text{model}}$ subject to absolute deterministic guardrails:
- **`HEAT-1` / `HEAT-2`:** Escalates to CRITICAL when Heat Index exceeds 40.6 °C alongside elevated cardiovascular strain.
- **`POLLUTION-1`:** Escalates to HIGH/CRITICAL when $\text{PM}_{2.5} > 150.4\ \mu\text{g}/\text{m}^3$ ($\text{AQI} \ge 200$).

---

## 6. Experimental Benchmark Results

Evaluated across $N=4$ human participants under Leave-One-Participant-Out (LOPO) cross-validation:

| Anomaly Strategy | Macro Recall | Macro Precision | False Positive Rate | AUROC | Mean Alert Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fixed Population Threshold** ($3\sigma$) | 22.22% | 94.70% | 0.00% | 0.611 | 243.0 s |
| **Population Random Forest (LOPO)** | 63.75% | 53.30% | 18.42% | 0.833 | 112.5 s |
| **HealthSensing (Personalized Quality-Gated)** | **88.94%** | **95.30%** | **1.24%** | **0.997** | **47.0 s** |

*Key Findings:* Fixed population thresholds suffer from total recall collapse on participants S2 (0.0%) and S4 (0.0%). Random Forest exhibits severe cross-subject generalization fragility (S3 FPR of 67.57%). HealthSensing's personalized Bayesian baselines achieve high sensitivity with sub-minute alert latency across all participants.

---

## 7. Quickstart & Local Setup

### 7.1 Prerequisites
- Python 3.10 or 3.11
- Node.js 18+ and npm

### 7.2 Backend Setup
```powershell
# 1. Clone repository
git clone https://github.com/animish-k/HealthSensing.git
cd HealthSensing

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the FastAPI backend
uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```
- API Docs: `http://127.0.0.1:8080/docs`
- Health Check: `http://127.0.0.1:8080/health`
- Live Replay: The backend automatically begins streaming synchronized WESAD + Ljubljana records every 5 seconds.

### 7.3 Frontend Setup
```powershell
cd frontend
npm install
npm run dev
```
- Open `http://localhost:5173` in your browser.

---

## 8. Cloud Deployment (Render + Vercel)

HealthSensing is configured for dual-cloud deployment:

### Backend Deployment on Render
1. Sign in to [Render](https://dashboard.render.com/) and create a **New Web Service**.
2. Connect the repository `animish-k/HealthSensing` and select branch `deploy/public-demo` (or `main`).
3. Set the following build settings:
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Under **Advanced**, attach persistent storage:
   - **Disk Mount Path:** `/var/data`
   - **Disk Size:** `1 GB`
   - **Environment Variable:** `HEALTHSENSING_DATABASE_PATH = /var/data/healthsensing.sqlite3`
5. Health Check Path: `/health`

### Frontend Deployment on Vercel
1. Sign in to [Vercel](https://vercel.com/) and click **Add New Project**.
2. Import `animish-k/HealthSensing` and set **Root Directory** to `frontend`.
3. Framework Preset: `Vite`. Build Command: `npm run build`. Output Directory: `dist`.
4. In `frontend/vercel.json`, rewrites point to your active Render web service:
   ```json
   {
     "rewrites": [
       { "source": "/api/:path*", "destination": "https://healthsensing.onrender.com/api/:path*" },
       { "source": "/ws/:path*", "destination": "https://healthsensing.onrender.com/ws/:path*" },
       { "source": "/health", "destination": "https://healthsensing.onrender.com/health" }
     ]
   }
   ```
5. Production Live URL: **[https://healthsensing.vercel.app](https://healthsensing.vercel.app)**
   - Persistent WebSockets connect directly to `wss://healthsensing.onrender.com/ws/readings`.

---

## 9. API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Service liveness probe returning `{"status": "ok"}` |
| `/api/readings` | `POST` | Ingests a new validated multimodal reading packet |
| `/api/current` | `GET` | Retrieves the most recent reading, risk assessment, and active alert |
| `/api/history/readings` | `GET` | Paginated time-series history (`?limit=50&participant_id=S2`) |
| `/api/risks/recent` | `GET` | Recent risk evaluations with component breakdowns |
| `/api/alerts/active` | `GET` | Active unacknowledged alerts |
| `/api/alerts/{id}/ack` | `POST` | Acknowledges an alert by ID |
| `/api/baseline` | `GET` / `PUT` | Reads or updates personalized resting vital baseline bounds |
| `/api/scenarios/run` | `POST` | Triggers a burst simulation (`heatwave`, `pollution`, `fatigue`) |
| `/ws/readings` | `WebSocket` | Real-time push stream of every incoming telemetry reading |
| `/ws/events` | `WebSocket` | Typed system events (`reading`, `risk_assessment`, `alert`, `ping`) |

---

## 10. Test Suite

Run the verified test suite:

```powershell
# Run backend, services, REST API, WebSockets, and risk engine tests (40 tests)
pytest tests/test_backend.py tests/test_services.py tests/test_api.py tests/test_websocket_protocol.py tests/test_e2e_integration.py tests/test_simulator_contract.py tests/test_risk_engine.py

# Run full project pipeline suite
pytest
```

---

## 11. Clinical & Regulatory Disclaimer

HealthSensing is an **academic early-warning risk-screening research prototype** developed for CS F407/U407. It is **not** a certified medical diagnostic device under FDA, CE, or CDSCO regulations. It does not provide medical diagnoses, clinical treatment recommendations, or emergency response dispatch. Telemetry and simulated stress conditions reflect autonomic nervous system arousal during laboratory stress protocols and do not constitute clinical validation of physical heatstroke or inhalation injury.
