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

app = FastAPI(
    title="HealthSensing API",
    description="Local-first health and environmental reading service with multimodal early-risk screening.",
    version="0.2.0",
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
