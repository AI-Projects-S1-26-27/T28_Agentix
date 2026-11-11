from typing import Any, Dict
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends

from backend.api.routes.current import get_repository
from backend.core.config import database_path
from backend.db.database import MeasurementRepository


router = APIRouter(prefix="/settings", tags=["settings"])


_SYSTEM_SETTINGS: Dict[str, Any] = {
    "offline_mode": True,
    "local_ai_enabled": True,
    "alert_sound": True,
    "temperature_unit": "celsius",
    "heat_alert_threshold_c": 35.0,
    "aqi_alert_threshold": 150.0,
}


class SettingsUpdateRequest(BaseModel):
    offline_mode: bool = Field(default=True)
    local_ai_enabled: bool = Field(default=True)
    alert_sound: bool = Field(default=True)
    temperature_unit: str = Field(default="celsius")
    heat_alert_threshold_c: float = Field(default=35.0, ge=20.0, le=60.0)
    aqi_alert_threshold: float = Field(default=150.0, ge=50.0, le=500.0)


@router.get("")
def get_settings(
    repository: MeasurementRepository = Depends(get_repository),
) -> Dict[str, Any]:
    stats = repository.get_measurement_stats()
    return {
        "status": "operational",
        "database_path": str(database_path()),
        "total_readings_stored": stats.get("total_count", 0),
        "settings": _SYSTEM_SETTINGS,
    }


@router.patch("")
def update_settings(
    request: SettingsUpdateRequest,
) -> Dict[str, Any]:
    if hasattr(request, "model_dump"):
        data = request.model_dump()
    else:
        data = request.dict()
    _SYSTEM_SETTINGS.update(data)
    return {
        "status": "updated",
        "settings": _SYSTEM_SETTINGS,
    }
