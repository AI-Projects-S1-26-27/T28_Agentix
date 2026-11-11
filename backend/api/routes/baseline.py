from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.routes.current import get_repository
from backend.db.database import MeasurementRepository


router = APIRouter(prefix="/baseline", tags=["baseline"])


class CalibrateRequest(BaseModel):
    participant_id: str = Field(default="default")
    signal: str = Field(..., description="Signal name e.g. heart_rate_bpm, eda, etc.")
    mean: float
    std: float
    observation_count: int = Field(default=60, ge=1)
    status: str = Field(default="stable")


@router.get("", response_model=List[Dict[str, Any]])
def get_personal_baselines(
    participant_id: str = Query(default="default"),
    repository: MeasurementRepository = Depends(get_repository),
) -> List[Dict[str, Any]]:
    return repository.get_baselines(participant_id=participant_id)


@router.post("/calibrate", response_model=Dict[str, Any])
def save_calibration(
    request: CalibrateRequest,
    repository: MeasurementRepository = Depends(get_repository),
) -> Dict[str, Any]:
    return repository.save_baseline(
        participant_id=request.participant_id,
        signal=request.signal,
        mean=request.mean,
        std=request.std,
        observation_count=request.observation_count,
        status=request.status,
    )
