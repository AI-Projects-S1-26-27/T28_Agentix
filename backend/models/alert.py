from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Alert(BaseModel):
    timestamp: datetime
    reading_id: Optional[int] = None
    level: str = Field(..., description="warning, critical, or info")
    category: str = Field(..., description="heat_stress, air_quality, physiological, fatigue, multimodal")
    message: str
    explanation: str
    contributing_signals: List[str] = Field(default_factory=list)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


class AlertResponse(Alert):
    id: int


class AlertAcknowledgeRequest(BaseModel):
    acknowledged: bool = True
