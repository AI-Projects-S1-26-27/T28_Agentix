import csv
import itertools
import os
import time
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import (
    alerts,
    baseline,
    current,
    health,
    history,
    risks,
    scenarios,
    settings,
)
from backend.api.routes.current import get_repository
from backend.core.config import database_path
from backend.db.database import MeasurementRepository
from backend.websocket.live_stream import router as live_stream_router

repository = MeasurementRepository(database_path())
repository.initialize()

_BASE_URL = "http://127.0.0.1:8080"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data" / "processed"
_DELAY_SECONDS = 5  # one tick every 5 seconds, both datasets update together


def _optional_float(val):
    try:
        return float(val) if val not in (None, "", "nan", "NaN") else None
    except (ValueError, TypeError):
        return None


def _load_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _post(payload: dict) -> bool:
    try:
        r = requests.post(f"{_BASE_URL}/api/readings", json=payload, timeout=5)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[replay] POST failed: {e}")
        return False


def _combined_replay_loop():
    """Tick every 5 s: post one WESAD row + one Ljubljana row together."""
    time.sleep(3)  # let uvicorn finish binding

    wesad_csvs = sorted(_DATA_DIR.glob("wesad_*_real.csv"))
    ljub_csv = _DATA_DIR / "ljubljana_real.csv"

    if not wesad_csvs:
        print("[replay] No WESAD CSVs — health data will not stream.")
    if not ljub_csv.exists():
        print("[replay] Ljubljana CSV not found — env data will not stream.")
    if not wesad_csvs and not ljub_csv.exists():
        return

    # Load all rows upfront
    wesad_rows = []
    for p in wesad_csvs:
        wesad_rows.extend(_load_csv(p))

    ljub_rows = _load_csv(ljub_csv) if ljub_csv.exists() else []

    print(
        f"[replay] Loaded {len(wesad_rows)} WESAD rows, "
        f"{len(ljub_rows)} Ljubljana rows. Ticking every {_DELAY_SECONDS}s."
    )

    wesad_cycle = itertools.cycle(wesad_rows) if wesad_rows else None
    ljub_cycle = itertools.cycle(ljub_rows) if ljub_rows else None

    while True:
        ts = _now_iso()

        # --- WESAD (health) ---
        if wesad_cycle:
            row = next(wesad_cycle)
            protocol = None
            if row.get("stress_label"):
                try:
                    protocol = int(float(row["stress_label"]))
                except (ValueError, TypeError):
                    pass
            payload = {
                "timestamp": ts,
                "source": "wesad",
                "heart_rate":       _optional_float(row.get("heart_rate_bpm")),
                "eda":              _optional_float(row.get("eda_microsiemens")),
                "body_temperature": _optional_float(row.get("body_temperature_c")),
                "protocol_label":   protocol,
            }
            _post(payload)

        # --- Ljubljana (environment) ---
        if ljub_cycle:
            row = next(ljub_cycle)
            payload = {
                "timestamp": ts,
                "source": "ljubljana",
                "pm25":                _optional_float(row.get("pm25_ug_m3")),
                "ambient_temperature": _optional_float(row.get("ambient_temperature_c")),
                "humidity":            _optional_float(row.get("humidity_percent")),
            }
            _post(payload)

        time.sleep(_DELAY_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_combined_replay_loop, daemon=True, name="combined-replay").start()
    print("[replay] Combined replay thread started (5 s tick, both datasets in sync).")
    yield


app = FastAPI(
    title="HealthSensing API",
    description="Local-first health and environmental reading service with multimodal early-risk screening.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.dependency_overrides[get_repository] = lambda: repository

app.include_router(health.router)
app.include_router(current.router, prefix="/api")
app.include_router(risks.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(baseline.router, prefix="/api")
app.include_router(scenarios.router, prefix="/api")
app.include_router(settings.router, prefix="/api")
app.include_router(live_stream_router)
