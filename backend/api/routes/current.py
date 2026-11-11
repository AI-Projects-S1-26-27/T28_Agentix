from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder

from backend.db.database import MeasurementRepository
from backend.models.measurement import Measurement, MeasurementResponse
from backend.services.data_service import DataService
from backend.websocket.live_stream import manager


router = APIRouter(prefix="/readings", tags=["readings"])


def get_repository() -> MeasurementRepository:
    raise RuntimeError("The repository dependency is not configured")


def get_data_service(
    repository: MeasurementRepository = Depends(get_repository),
) -> DataService:
    return DataService(repository=repository, websocket_manager=manager)


@router.post("", response_model=MeasurementResponse, status_code=201)
async def ingest_reading(
    measurement: Measurement,
    data_service: DataService = Depends(get_data_service),
) -> MeasurementResponse:
    result = await data_service.ingest_reading(measurement)
    return result["reading"]


@router.get("", response_model=list[MeasurementResponse])
def list_readings(
    limit: int = Query(default=50, ge=1, le=500),
    repository: MeasurementRepository = Depends(get_repository),
) -> list[MeasurementResponse]:
    return repository.recent(limit)


@router.get("/latest")
def get_latest_reading_summary(
    repository: MeasurementRepository = Depends(get_repository),
) -> Dict[str, Any]:
    readings = repository.recent(limit=1)
    if not readings:
        raise HTTPException(status_code=404, detail="No readings available")

    latest_reading = readings[0]
    latest_risk = repository.get_risk_by_reading_id(latest_reading.id)
    active_alerts = repository.recent_alerts(limit=5, unacknowledged_only=True)

    return {
        "reading": latest_reading.to_dict(),
        "risk": latest_risk.to_dict() if latest_risk else None,
        "active_alerts": [a.to_dict() for a in active_alerts],
    }
