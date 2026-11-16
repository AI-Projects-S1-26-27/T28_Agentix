"""Shared types for the reasoning layer (rule results)."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Category(str, Enum):
    HEAT = "heat_environmental_stress"
    POLLUTION = "pollution_respiratory"
    FATIGUE = "fatigue"
    GENERAL = "general_anomaly"


class RuleStatus(str, Enum):
    TRIGGERED = "TRIGGERED"
    NOT_TRIGGERED = "NOT_TRIGGERED"
    NOT_EVALUABLE = "NOT_EVALUABLE"  # required inputs missing or too little history


@dataclass
class RuleResult:
    rule_id: str
    name: str
    category: Category
    status: RuleStatus
    weight: float
    hits: int = 0  # readings in the window where the condition held
    evaluable: int = 0  # readings in the window that had every required input
    window: int = 0  # readings actually examined
    min_hits: int = 0  # persistence requirement
    signals: dict[str, Any] = field(default_factory=dict)
    missing_inputs: list[str] = field(default_factory=list)
    explanation: str = ""