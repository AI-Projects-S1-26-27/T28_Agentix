"""Fatigue rules."""
from typing import Optional

from reasoning.knowledge_base.knowledge_base import (
    THRESHOLDS, Baseline, CheckResult, Reading, Rule, hr_elevated, num,
)
from reasoning.models import Category


def _fat_1(reading: Reading, baseline: Optional[Baseline]) -> CheckResult:
    sleep = num(reading, "sleep_duration_hours")
    hr_hit, signals = hr_elevated(reading, baseline)
    activity = num(reading, "activity_level")
    high_activity = activity is not None and activity >= THRESHOLDS["activity_high"]
    hit = sleep < THRESHOLDS["sleep_short_hours"] and (hr_hit or high_activity)
    return hit, {"sleep_duration_hours": sleep, **signals}


FATIGUE_RULES = (
    Rule(
        "FAT-1", Category.FATIGUE, "Short sleep with physiological or activity strain",
        "Sleep < 6 h AND (heart rate elevated OR activity_level >= 0.8). Needs "
        "sleep_duration_hours, which is NOT in the backend contract yet, so this rule stays "
        "NOT_EVALUABLE until the field is added.",
        ("sleep_duration_hours",), (), 0.40, _fat_1,
        lambda s: (
            f"short sleep ({s['sleep_duration_hours']:.1f} h) together with elevated heart rate "
            f"or high activity"
        ),
    ),
)