"""General physiological-anomaly rules."""
from typing import Optional

from reasoning.knowledge_base.knowledge_base import (
    THRESHOLDS, Baseline, CheckResult, Reading, Rule, hr_elevated, num,
)
from reasoning.models import Category


def _gen_1(reading: Reading, baseline: Optional[Baseline]) -> CheckResult:
    hr_hit, signals = hr_elevated(reading, baseline)
    return (num(reading, "activity_level") <= THRESHOLDS["activity_sedentary"] and hr_hit), signals


GENERAL_RULES = (
    Rule(
        "GEN-1", Category.GENERAL, "Elevated heart rate at rest",
        "Heart rate elevated (>= 100 bpm, or personal baseline mean + 1.5 std) while "
        "activity_level <= 0.3.",
        ("heart_rate", "activity_level"), (), 0.30, _gen_1,
        lambda s: (
            f"heart rate {s['heart_rate']:.0f} bpm at low activity ({s['activity_level']:.2f}), "
            f"at or above {s['hr_threshold']:.0f} bpm ({s['hr_threshold_source']})"
        ),
    ),
)