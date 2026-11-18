from typing import Optional
from fastapi import APIRouter, Depends, Query

from backend.api.routes.current import get_repository
from backend.db.database import MeasurementRepository
from backend.models.analytics import AutonomicStabilityMetrics, TelemetryAnalyticsSummary
from backend.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])


def get_analytics_service(
    repository: MeasurementRepository = Depends(get_repository),
) -> AnalyticsService:
    return AnalyticsService(repository=repository)


@router.get("/summary", response_model=TelemetryAnalyticsSummary)
def get_analytics_summary(
    window_minutes: int = Query(default=60, ge=5, le=1440),
    participant_id: Optional[str] = Query(default=None),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> TelemetryAnalyticsSummary:
    return analytics_service.compute_summary(
        window_minutes=window_minutes, participant_id=participant_id
    )


@router.get("/stability", response_model=AutonomicStabilityMetrics)
def get_stability_metrics(
    window_minutes: int = Query(default=30, ge=5, le=360),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> AutonomicStabilityMetrics:
    summary = analytics_service.compute_summary(window_minutes=window_minutes)
    return summary.stability
