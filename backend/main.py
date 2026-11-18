import subprocess
import sys
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path

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
_DATA_DIR = Path("data/processed")


def _wesad_loop():
    """Replay WESAD health data, looping forever."""
    time.sleep(3)  # let uvicorn finish binding
    csvs = sorted(_DATA_DIR.glob("wesad_*_real.csv"))
    if not csvs:
        print("[replay] No WESAD CSVs found — health data will not stream.")
        return
    csv_args = [str(p) for p in csvs]
    print(f"[replay] Starting WESAD replay ({len(csvs)} file(s))...")
    while True:
        try:
            subprocess.run(
                [
                    sys.executable, "data/src/replay_wesad_to_backend.py",
                    "--csv", *csv_args,
                    "--base-url", _BASE_URL,
                    "--delay-seconds", "0.5",
                ],
                check=True,
            )
            print("[replay] WESAD pass done — looping.")
        except subprocess.CalledProcessError as e:
            print(f"[replay] WESAD error (retrying in 5s): {e}")
            time.sleep(5)
        except Exception as e:
            print(f"[replay] WESAD unexpected error: {e}")
            time.sleep(5)


def _ljubljana_loop():
    """Replay Ljubljana env data, looping forever."""
    time.sleep(4)  # offset slightly so both don't hammer at t=0
    csv = _DATA_DIR / "ljubljana_real.csv"
    if not csv.exists():
        print("[replay] Ljubljana CSV not found — env data will not stream.")
        return
    print("[replay] Starting Ljubljana env replay...")
    while True:
        try:
            subprocess.run(
                [
                    sys.executable, "data/src/replay_ljubljana_to_backend.py",
                    "--csv", str(csv),
                    "--base-url", _BASE_URL,
                    "--delay-seconds", "2",
                ],
                check=True,
            )
            print("[replay] Ljubljana pass done — looping.")
        except subprocess.CalledProcessError as e:
            print(f"[replay] Ljubljana error (retrying in 5s): {e}")
            time.sleep(5)
        except Exception as e:
            print(f"[replay] Ljubljana unexpected error: {e}")
            time.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_wesad_loop,     daemon=True, name="wesad-replay").start()
    threading.Thread(target=_ljubljana_loop, daemon=True, name="ljubljana-replay").start()
    print("[replay] Background data replay threads started.")
    yield
    # Daemon threads die automatically with the process — nothing to clean up.


app = FastAPI(
    title="HealthSensing API",
    description="Local-first health and environmental reading service with multimodal early-risk screening.",
    version="0.2.0",
    lifespan=lifespan,
)

# Enable CORS for local React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.dependency_overrides[get_repository] = lambda: repository

# Register API routes
app.include_router(health.router)
app.include_router(current.router, prefix="/api")
app.include_router(risks.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(baseline.router, prefix="/api")
app.include_router(scenarios.router, prefix="/api")
app.include_router(settings.router, prefix="/api")
app.include_router(live_stream_router)
