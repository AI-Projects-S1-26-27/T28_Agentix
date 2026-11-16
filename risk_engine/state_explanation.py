"""Human-readable explanation for a risk assessment."""
from typing import Optional

from reasoning.models import Category, RuleResult, RuleStatus
from risk_engine.models import DISCLAIMER, CategoryAssessment, RiskLevel

_LABEL = {
    Category.HEAT: "heat / environmental stress",
    Category.POLLUTION: "pollution-associated respiratory risk pattern",
    Category.FATIGUE: "fatigue pattern",
    Category.GENERAL: "general physiological anomaly",
}


def build_state_explanation(
    level: RiskLevel,
    primary: Optional[Category],
    categories: list[CategoryAssessment],
    rule_results: list[RuleResult],
    anomaly_score: Optional[float],
    unavailable: list[str],
    warnings: list[str],
) -> str:
    if primary is None:
        lines = ["Risk level: LOW. No contextual rule was triggered and no notable anomaly was reported."]
    else:
        lines = [f"Risk level: {level.value} ({_LABEL[primary]})."]

    triggered = [r for r in rule_results if r.status == RuleStatus.TRIGGERED]
    if triggered:
        lines.append("Why: " + "; ".join(r.explanation for r in triggered) + ".")

    if anomaly_score is not None:
        lines.append(f"Model anomaly score: {anomaly_score:.2f}.")
    else:
        lines.append("Model anomaly score unavailable; assessment is based on rules only.")

    for category in categories:
        lines.extend(category.notes)
    if unavailable:
        lines.append("Inputs unavailable, so some rules could not be checked: " + ", ".join(unavailable) + ".")
    lines.extend(warnings)
    lines.append(DISCLAIMER)
    return " ".join(lines)