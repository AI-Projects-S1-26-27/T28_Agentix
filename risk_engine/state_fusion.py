"""Fuse rule evidence with the model anomaly score into LOW/MODERATE/HIGH/CRITICAL.

  * Rule evidence per category = sum of triggered rule weights, capped at 1.
  * With a model score:    score = 0.65 * evidence + 0.35 * anomaly_score.
    Without one:           score = evidence, and the level is capped at HIGH.
  * Any triggered rule guarantees at least MODERATE (rules already demand
    context and persistence).
  * CRITICAL needs score >= 0.75 AND a model score AND >= 2 triggered rules in
    the same category (two independent pieces of evidence).
  * A model score with NO triggered rule anywhere is reported as a general
    anomaly, capped at MODERATE (the model alone cannot say why).
"""
from typing import Optional

from reasoning.knowledge_base.knowledge_base import THRESHOLDS
from reasoning.models import Category, RuleResult, RuleStatus
from risk_engine.models import LEVEL_RANK, CategoryAssessment, RiskLevel

RULE_WEIGHT, MODEL_WEIGHT = 0.65, 0.35
CUTOFFS = ((0.75, RiskLevel.CRITICAL), (0.55, RiskLevel.HIGH), (0.30, RiskLevel.MODERATE))


def level_from_score(score: float) -> RiskLevel:
    for cutoff, level in CUTOFFS:
        if score >= cutoff:
            return level
    return RiskLevel.LOW


def _cap(level: RiskLevel, ceiling: RiskLevel) -> RiskLevel:
    return level if LEVEL_RANK[level] <= LEVEL_RANK[ceiling] else ceiling


def _floor(level: RiskLevel, floor: RiskLevel) -> RiskLevel:
    return level if LEVEL_RANK[level] >= LEVEL_RANK[floor] else floor


def fuse(rule_results: list[RuleResult], anomaly_score: Optional[float]) -> list[CategoryAssessment]:
    model_ok = anomaly_score is not None
    any_triggered = any(r.status == RuleStatus.TRIGGERED for r in rule_results)
    out: list[CategoryAssessment] = []

    for category in Category:
        triggered = [
            r for r in rule_results if r.category == category and r.status == RuleStatus.TRIGGERED
        ]
        evidence = min(1.0, sum(r.weight for r in triggered))
        notes: list[str] = []

        if triggered:
            score = RULE_WEIGHT * evidence + MODEL_WEIGHT * anomaly_score if model_ok else evidence
            level = _floor(level_from_score(score), RiskLevel.MODERATE)
            if not (model_ok and score >= 0.75 and len(triggered) >= 2):
                level = _cap(level, RiskLevel.HIGH)
            if not model_ok:
                notes.append("No model score available; level capped at HIGH.")
            elif anomaly_score < 0.2:
                notes.append(
                    f"Conflict: rules triggered but the model anomaly score is low ({anomaly_score:.2f}); "
                    "treated as context-supported but not model-confirmed."
                )
        elif category == Category.GENERAL and model_ok and not any_triggered:
            score = anomaly_score
            level = _cap(level_from_score(score), RiskLevel.MODERATE)
            if anomaly_score >= THRESHOLDS["model_anomaly_high"]:
                notes.append(
                    f"Model flags an unusual pattern (score {anomaly_score:.2f}) but no contextual "
                    "rule explains it; capped at MODERATE."
                )
        else:
            score, level = 0.0, RiskLevel.LOW

        out.append(CategoryAssessment(
            category=category, level=level, score=round(score, 3), evidence=round(evidence, 3),
            triggered_rules=[r.rule_id for r in triggered], notes=notes,
        ))
    return out


def overall(categories: list[CategoryAssessment]) -> tuple[RiskLevel, Optional[Category]]:
    best = max(categories, key=lambda c: (LEVEL_RANK[c.level], c.score))
    return (RiskLevel.LOW, None) if best.level == RiskLevel.LOW else (best.level, best.category)