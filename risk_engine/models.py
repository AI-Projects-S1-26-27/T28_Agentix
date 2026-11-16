"""Output types for the LOW / MODERATE / HIGH / CRITICAL risk assessment."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from reasoning.models import Category, RuleResult

DISCLAIMER = (
    "Informational early-warning only. This is a risk-screening prototype, "
    "not a medical diagnosis."
)


class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


LEVEL_RANK = {RiskLevel.LOW: 0, RiskLevel.MODERATE: 1, RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}


@dataclass
class CategoryAssessment:
    category: Category
    level: RiskLevel
    score: float  # fused 0..1 (rules + model)
    evidence: float  # rule evidence only, 0..1
    triggered_rules: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    status: str  # "ok" or "insufficient_data"
    level: RiskLevel
    primary_category: Optional[Category]
    anomaly_score: Optional[float]
    model_available: bool
    readings_used: int
    categories: list[CategoryAssessment]
    rule_results: list[RuleResult]
    triggered_rules: list[str]
    contributing_signals: dict[str, Any]
    unavailable_inputs: list[str]
    warnings: list[str]
    explanation: str
    environment: Optional[dict] = None  # context from risk_engine.risk_fusion (AQI / heat index bands)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict (enums become plain strings)."""
        return _plain(self)


def _plain(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {name: _plain(getattr(obj, name)) for name in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj