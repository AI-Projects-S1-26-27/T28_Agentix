from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from backend.api.routes.current import get_data_service
from backend.models.measurement import Measurement
from backend.models.scenario import (
    ScenarioInfo,
    ScenarioRunRequest,
    ScenarioRunResponse,
)
from backend.services.data_service import DataService
from simulator.data_generator import generate_data
from simulator.scenarios import SCENARIOS


router = APIRouter(prefix="/scenarios", tags=["scenarios"])

# Tracks the last scenario that was activated (for GET /scenarios/active)
_active_scenario: str = "normal"


@router.get("", response_model=List[ScenarioInfo])
def list_scenarios() -> List[ScenarioInfo]:
    result = []
    descriptions = {
        "normal": "Baseline environmental and physiological indicators within standard nominal ranges.",
        "heatwave": "Extreme ambient heat and humidity accompanied by cardiac and thermal response.",
        "pollution": "Elevated particulate matter and hazardous AQI with respiratory impact.",
        "fatigue": "Low activity patterns and physiological fluctuations associated with fatigue.",
    }
    for scenario_id, ranges in SCENARIOS.items():
        result.append(
            ScenarioInfo(
                id=scenario_id,
                name=scenario_id.capitalize(),
                description=descriptions.get(scenario_id, "Simulated stress scenario."),
                parameters=ranges,
            )
        )
    return result


@router.get("/active")
def get_active_scenario() -> Dict[str, str]:
    """Return the most recently activated scenario."""
    return {"scenario": _active_scenario}


@router.post("/run", response_model=ScenarioRunResponse)
async def run_scenario_stream(
    request: ScenarioRunRequest,
    data_service: DataService = Depends(get_data_service),
) -> ScenarioRunResponse:
    global _active_scenario
    if request.scenario_id not in SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario '{request.scenario_id}'. Available: {list(SCENARIOS.keys())}",
        )

    _active_scenario = request.scenario_id
    count = request.count
    # Stagger timestamps so readings spread across a window on the chart
    # (all 10 readings otherwise get the same datetime.now() millisecond)
    now = datetime.now(timezone.utc)
    interval = timedelta(seconds=2)  # 2 s between readings
    for i in range(count):
        raw = generate_data(request.scenario_id)
        # Override timestamp: oldest reading first, newest last
        raw["timestamp"] = (now - interval * (count - 1 - i)).isoformat()
        m = Measurement(**raw)
        await data_service.ingest_reading(m)

    return ScenarioRunResponse(
        scenario_id=request.scenario_id,
        status="completed",
        readings_generated=count,
        message=f"Successfully generated and processed {count} readings for '{request.scenario_id}'.",
    )
