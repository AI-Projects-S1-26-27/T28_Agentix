"""Public entry point for the backend: ``assess_risk``."""
import math
from typing import Any, Iterable, Optional

from reasoning.knowledge_base.knowledge_base import Baseline, Reading
from reasoning.models import RuleStatus
from reasoning.rule_engine import evaluate_rules
from risk_engine.models import RiskAssessment, RiskLevel
from risk_engine.risk_fusion import evaluate_environmental_risk
from risk_engine.state_explanation import build_state_explanation
from risk_engine.state_fusion import fuse, overall


def _as_dict(reading: Any) -> Reading:
    """Accept dicts or pydantic-style objects (backend Measurement / MeasurementResponse)."""
    if isinstance(reading, dict):
        return reading
    for method in ("model_dump", "dict"):
        fn = getattr(reading, method, None)
        if callable(fn):
            return fn()
    return dict(getattr(reading, "__dict__", {}))


def _clean_score(score: Any) -> tuple[Optional[float], Optional[str]]:
    if score is None:
        return None, None
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        return None, "Model anomaly score was invalid and was ignored."
    if not 0.0 <= score <= 1.0:
        return None, f"Model anomaly score {score} is outside 0..1 and was ignored."
    return float(score), None


def _environment(latest: Reading) -> Optional[dict]:
    try:
        return evaluate_environmental_risk(latest)
    except (ValueError, TypeError):
        return None


def assess_risk(
    readings: Iterable[Any],
    anomaly_score: Optional[float] = None,
    baseline: Optional[Baseline] = None,
    newest_first: bool = False,
) -> RiskAssessment:
    """Assess risk from recent readings.

    readings:      recent readings (dicts or objects). Oldest -> newest unless
                   newest_first=True (``GET /api/readings`` returns newest-first).
    anomaly_score: detector score for the latest reading, 0..1, or None.
    baseline:      optional personal baseline, e.g.
                   {"heart_rate": {"mean": 70, "std": 6}, "spo2": {"mean": 97, "std": 0.8}}.
                   Without it, fixed illustrative thresholds are used.
    """
    data = [_as_dict(r) for r in readings]
    if newest_first:
        data.reverse()
    score, score_warning = _clean_score(anomaly_score)
    warnings = [score_warning] if score_warning else []

    if not data:
        warnings.append("No readings available; no assessment possible.")
        return RiskAssessment(
            status="insufficient_data", level=RiskLevel.LOW, primary_category=None,
            anomaly_score=score, model_available=score is not None, readings_used=0,
            categories=fuse([], score), rule_results=[], triggered_rules=[],
            contributing_signals={}, unavailable_inputs=["readings"], warnings=warnings,
            explanation="No readings available, so no risk assessment is possible. "
            "Informational early-warning only; not a medical diagnosis.",
        )

    if baseline is None:
        warnings.append("No personal baseline supplied; fixed illustrative thresholds were used.")

    results = evaluate_rules(data, baseline)
    categories = fuse(results, score)
    level, primary = overall(categories)

    unavailable = sorted(
        {m for r in results if r.status == RuleStatus.NOT_EVALUABLE for m in r.missing_inputs}
    )
    short_history = [
        r.rule_id for r in results if r.status == RuleStatus.NOT_EVALUABLE and not r.missing_inputs
    ]
    if short_history:
        warnings.append("Not enough history to evaluate: " + ", ".join(short_history) + ".")

    triggered = [r for r in results if r.status == RuleStatus.TRIGGERED]
    signals: dict[str, Any] = {r.rule_id: r.signals for r in triggered}
    if score is not None:
        signals["anomaly_score"] = score

    return RiskAssessment(
        status="ok", level=level, primary_category=primary, anomaly_score=score,
        model_available=score is not None, readings_used=len(data), categories=categories,
        rule_results=results, triggered_rules=[r.rule_id for r in triggered],
        contributing_signals=signals, unavailable_inputs=unavailable, warnings=warnings,
        explanation=build_state_explanation(
            level, primary, categories, results, score, unavailable, warnings
        ),
        environment=_environment(data[-1]),
    )