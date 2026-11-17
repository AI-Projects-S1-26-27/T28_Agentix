from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.routes.current import get_repository
from backend.db.database import MeasurementRepository
from backend.models.alert import (
    AlertAcknowledgeRequest,
    AlertResponse,
)


router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=List[AlertResponse])
def get_alerts(
    limit: int = Query(default=50, ge=1, le=500),
    unacknowledged_only: bool = Query(default=False),
    repository: MeasurementRepository = Depends(get_repository),
) -> List[AlertResponse]:
    return repository.recent_alerts(
        limit=limit, unacknowledged_only=unacknowledged_only
    )


@router.get("/active", response_model=List[AlertResponse])
def get_active_alerts(
    limit: int = Query(default=50, ge=1, le=500),
    repository: MeasurementRepository = Depends(get_repository),
) -> List[AlertResponse]:
    return repository.recent_alerts(limit=limit, unacknowledged_only=True)


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
def acknowledge_alert(
    alert_id: int,
    repository: MeasurementRepository = Depends(get_repository),
) -> AlertResponse:
    updated = repository.acknowledge_alert(alert_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return updated
