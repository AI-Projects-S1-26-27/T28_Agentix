from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScenarioInfo(BaseModel):
    id: str
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ScenarioRunRequest(BaseModel):
    scenario_id: str = Field(..., description="Scenario identifier e.g. heatwave, pollution_spike, normal, fatigue")
    count: int = Field(default=10, ge=1, le=1000)
    speed: float = Field(default=1.0, gt=0, le=50.0)


class ScenarioRunResponse(BaseModel):
    scenario_id: str
    status: str
    readings_generated: int
    message: str
