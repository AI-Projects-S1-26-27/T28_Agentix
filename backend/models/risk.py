from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RiskComponent(BaseModel):
    category: Optional[str] = None
    risk_level: str
    severity: int
    value: Optional[float] = None
    value_c: Optional[float] = None
    value_f: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class RiskAssessment(BaseModel):
    timestamp: datetime
    reading_id: Optional[int] = None
    risk_level: str = Field(..., description="Risk tier: low, moderate, high, very_high, severe, or unknown")
    severity: Optional[int] = Field(default=None, description="0=low, 1=moderate, 2=high, 3=very_high, 4=severe")
    components: Dict[str, Any] = Field(default_factory=dict)
    unavailable_inputs: List[str] = Field(default_factory=list)
    contributing_signals: List[str] = Field(default_factory=list)
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


class RiskAssessmentResponse(RiskAssessment):
    id: int


class RiskEvaluationRequest(BaseModel):
    timestamp: Optional[datetime] = None
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    body_temperature: Optional[float] = None
    activity_level: Optional[float] = None
    ambient_temperature: Optional[float] = None
    humidity: Optional[float] = None
    aqi: Optional[float] = None
    pm25: Optional[float] = None
    eda: Optional[float] = None
