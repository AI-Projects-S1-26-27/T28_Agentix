"""Knowledge base: thresholds, helpers and the Rule type.

ALL NUMBERS ARE ILLUSTRATIVE PROTOTYPE VALUES, not clinically validated.
Tune them here, in one place. Field names and units follow
backend/models/measurement.py: heart_rate (bpm), spo2 (%), body_temperature (C),
activity_level (0..1), ambient_temperature (C), humidity (%), aqi, pm25 (ug/m3).
``sleep_duration_hours`` is not in the backend contract yet, so rules that need
it stay NOT_EVALUABLE until it is added.
"""
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

from reasoning.models import Category

THRESHOLDS: dict[str, float] = {
    "heat_index_c": 32.2,  # 90 F, lower bound of the NWS "extreme caution" band
    "heat_ambient_temp_c": 32.0,
    "body_temp_elevated_c": 37.8,
    "hr_fixed_elevated_bpm": 100.0,  # used when no personal baseline is supplied
    "hr_baseline_z": 1.5,  # std-devs above the personal mean
    "hr_min_std_bpm": 1.0,  # floor so a tiny std does not over-trigger
    "aqi_elevated": 101.0,  # start of "unhealthy for sensitive groups"
    "pm25_elevated_ug_m3": 35.0,
    "spo2_fixed_low_pct": 94.0,  # used when no personal baseline is supplied
    "spo2_baseline_drop_pct": 2.0,  # points below the personal mean
    "sleep_short_hours": 6.0,
    "activity_high": 0.8,  # activity_level is 0..1
    "activity_sedentary": 0.3,
    "model_anomaly_high": 0.6,
}

DEFAULT_WINDOW = 5  # look at the last N readings
DEFAULT_MIN_HITS = 3  # condition must hold in at least this many of them

Reading = dict[str, Any]
Baseline = dict[str, dict[str, float]]
CheckResult = tuple[bool, dict[str, Any]]


def num(reading: Reading, name: str) -> Optional[float]:
    """Finite float for ``name``, or None if missing / non-numeric / bool / NaN."""
    value = reading.get(name)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _baseline_stat(baseline: Optional[Baseline], name: str) -> Optional[tuple[float, float]]:
    entry = (baseline or {}).get(name)
    if isinstance(entry, dict):
        mean, std = entry.get("mean"), entry.get("std", 0.0)
        if isinstance(mean, (int, float)) and not isinstance(mean, bool) and math.isfinite(mean):
            ok_std = isinstance(std, (int, float)) and not isinstance(std, bool) and math.isfinite(std)
            return float(mean), float(std) if ok_std else 0.0
    return None


def hr_elevated(reading: Reading, baseline: Optional[Baseline]) -> CheckResult:
    hr = num(reading, "heart_rate")
    stat = _baseline_stat(baseline, "heart_rate")
    if stat:
        threshold = stat[0] + THRESHOLDS["hr_baseline_z"] * max(stat[1], THRESHOLDS["hr_min_std_bpm"])
        source = "personal baseline"
    else:
        threshold, source = THRESHOLDS["hr_fixed_elevated_bpm"], "fixed threshold"
    signals = {"heart_rate": hr, "hr_threshold": round(threshold, 1), "hr_threshold_source": source}
    activity = num(reading, "activity_level")
    if activity is not None:
        signals["activity_level"] = activity
    return (hr is not None and hr >= threshold), signals


def spo2_low(reading: Reading, baseline: Optional[Baseline]) -> CheckResult:
    spo2 = num(reading, "spo2")
    stat = _baseline_stat(baseline, "spo2")
    if stat:
        limit, source = stat[0] - THRESHOLDS["spo2_baseline_drop_pct"], "personal baseline"
    else:
        limit, source = THRESHOLDS["spo2_fixed_low_pct"], "fixed threshold"
    return (spo2 is not None and spo2 <= limit), {
        "spo2": spo2, "spo2_limit": round(limit, 1), "spo2_limit_source": source,
    }


@dataclass(frozen=True)
class Rule:
    rule_id: str
    category: Category
    name: str
    description: str  # shown in the rule table / conditions.json
    required: tuple[str, ...]  # every one of these must be present
    required_any: tuple[tuple[str, ...], ...]  # each group needs at least one present
    weight: float  # evidence strength, 0..1
    check: Callable[[Reading, Optional[Baseline]], CheckResult]
    describe: Callable[[dict[str, Any]], str]
    window: int = DEFAULT_WINDOW
    min_hits: int = DEFAULT_MIN_HITS