from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.routes.current import get_repository
from backend.db.database import MeasurementRepository
from backend.models.measurement import Measurement
from backend.models.risk import (
    RiskAssessment,
    RiskAssessmentResponse,
    RiskEvaluationRequest,
)
from backend.services.risk_service import RiskService


router = APIRouter(prefix="/risks", tags=["risks"])


def get_risk_service(
    repository: MeasurementRepository = Depends(get_repository),
) -> RiskService:
    return RiskService(repository=repository)


@router.get("/recent", response_model=List[RiskAssessmentResponse])
def get_recent_risks(
    limit: int = Query(default=50, ge=1, le=500),
    repository: MeasurementRepository = Depends(get_repository),
) -> List[RiskAssessmentResponse]:
    return repository.recent_risks(limit=limit)


@router.get("/latest", response_model=RiskAssessmentResponse)
def get_latest_risk(
    repository: MeasurementRepository = Depends(get_repository),
) -> RiskAssessmentResponse:
    latest = repository.get_latest_risk()
    if latest is None:
        raise HTTPException(status_code=404, detail="No risk assessments recorded")
    return latest


@router.post("/evaluate", response_model=RiskAssessment)
def evaluate_risk_on_demand(
    request: RiskEvaluationRequest,
    risk_service: RiskService = Depends(get_risk_service),
) -> RiskAssessment:
    ts = request.timestamp or datetime.now(timezone.utc)
    # Construct a synthetic measurement for evaluation
    m = Measurement(
        timestamp=ts,
        source="on_demand_api",
        heart_rate=request.heart_rate,
        spo2=request.spo2,
        body_temperature=request.body_temperature,
        activity_level=request.activity_level,
        ambient_temperature=request.ambient_temperature,
        humidity=request.humidity,
        aqi=request.aqi,
        pm25=request.pm25,
        eda=request.eda,
    )
    return risk_service.evaluate_reading(m)
