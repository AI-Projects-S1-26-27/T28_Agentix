from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

try:
    from pydantic import field_validator

    def _validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone offset")
        return value

    def _validate_non_blank_text(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("text fields must not be blank")
        return value

    _TIMESTAMP_VALIDATOR = field_validator("timestamp")(_validate_timestamp)
    _TEXT_VALIDATOR = field_validator("source", "scenario")(_validate_non_blank_text)
except ImportError:
    from pydantic import validator

    def _validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone offset")
        return value

    def _validate_non_blank_text(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("text fields must not be blank")
        return value

    _TIMESTAMP_VALIDATOR = validator("timestamp", allow_reuse=True)(_validate_timestamp)
    _TEXT_VALIDATOR = validator("source", "scenario", allow_reuse=True)(_validate_non_blank_text)


class Measurement(BaseModel):
    """A health/environment reading shared by the simulator and backend."""

    timestamp: datetime
    source: str = Field(..., min_length=1, max_length=64)
    scenario: Optional[str] = Field(default=None, max_length=64)
    heart_rate: Optional[float] = Field(default=None, ge=0, le=300)
    spo2: Optional[float] = Field(default=None, ge=0, le=100)
    body_temperature: Optional[float] = Field(default=None, ge=25, le=45)
    activity_level: Optional[float] = Field(default=None, ge=0, le=1)
    ambient_temperature: Optional[float] = Field(default=None, ge=-80, le=80)
    humidity: Optional[float] = Field(default=None, ge=0, le=100)
    aqi: Optional[float] = Field(default=None, ge=0, le=1000)
    pm25: Optional[float] = Field(default=None, ge=0, le=1000)
    eda: Optional[float] = Field(default=None, ge=0, le=100)

    # Apply validators
    _val_ts = _TIMESTAMP_VALIDATOR
    _val_txt = _TEXT_VALIDATOR

    def to_dict(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()

    def as_storage_tuple(self) -> tuple:
        values = self.to_dict()
        timestamp = values["timestamp"].astimezone(timezone.utc).isoformat()
        return (
            timestamp,
            values["source"],
            values["scenario"],
            values["heart_rate"],
            values["spo2"],
            values["body_temperature"],
            values["activity_level"],
            values["ambient_temperature"],
            values["humidity"],
            values["aqi"],
            values["pm25"],
            values.get("eda"),
        )


class MeasurementResponse(Measurement):
    id: int
