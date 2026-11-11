from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from backend.api.routes.current import get_repository
from backend.db.database import MeasurementRepository
from backend.models.measurement import MeasurementResponse
from backend.models.risk import RiskAssessmentResponse


router = APIRouter(prefix="/history", tags=["history"])


@router.get("/readings", response_model=List[MeasurementResponse])
def get_historical_readings(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    repository: MeasurementRepository = Depends(get_repository),
) -> List[MeasurementResponse]:
    return repository.query_measurements(
        start_time=start, end_time=end, limit=limit, offset=offset
    )


@router.get("/risks", response_model=List[RiskAssessmentResponse])
def get_historical_risks(
    limit: int = Query(default=50, ge=1, le=500),
    repository: MeasurementRepository = Depends(get_repository),
) -> List[RiskAssessmentResponse]:
    return repository.recent_risks(limit=limit)


@router.get("/stats")
def get_reading_statistics(
    repository: MeasurementRepository = Depends(get_repository),
) -> Dict[str, Any]:
    return repository.get_measurement_stats()
