"""Heat / environmental-stress rules."""
from typing import Optional

from reasoning.knowledge_base.knowledge_base import (
    THRESHOLDS, Baseline, CheckResult, Reading, Rule, hr_elevated, num,
)
from reasoning.models import Category
from risk_engine.risk_levels import calculate_heat_index_celsius


def _heat_index(reading: Reading) -> Optional[float]:
    try:
        return calculate_heat_index_celsius(
            num(reading, "ambient_temperature"), num(reading, "humidity")
        )
    except (ValueError, TypeError):
        return None


def _heat_1(reading: Reading, baseline: Optional[Baseline]) -> CheckResult:
    heat_index = _heat_index(reading)
    if heat_index is None:
        return False, {}
    hr_hit, hr_signals = hr_elevated(reading, baseline)
    signals = {
        "ambient_temperature": num(reading, "ambient_temperature"),
        "humidity": num(reading, "humidity"),
        "heat_index_c": round(heat_index, 1),
        **hr_signals,
    }
    return (heat_index >= THRESHOLDS["heat_index_c"] and hr_hit), signals


def _heat_2(reading: Reading, baseline: Optional[Baseline]) -> CheckResult:
    body, ambient = num(reading, "body_temperature"), num(reading, "ambient_temperature")
    hit = body >= THRESHOLDS["body_temp_elevated_c"] and ambient >= THRESHOLDS["heat_ambient_temp_c"]
    return hit, {"body_temperature": body, "ambient_temperature": ambient}


HEAT_RULES = (
    Rule(
        "HEAT-1", Category.HEAT, "Hot, humid conditions with elevated heart rate",
        "Heat index >= 32.2 C (from ambient temperature + humidity) AND heart rate elevated "
        "(>= 100 bpm, or personal baseline mean + 1.5 std). No activity allowance: exertion in "
        "dangerous heat counts as heat-strain evidence.",
        ("ambient_temperature", "humidity", "heart_rate"), (), 0.40, _heat_1,
        lambda s: (
            f"heat index {s['heat_index_c']:.0f} C (air {s['ambient_temperature']:.1f} C, humidity "
            f"{s['humidity']:.0f}%) with heart rate {s['heart_rate']:.0f} bpm "
            f"(at or above {s['hr_threshold']:.0f} bpm, {s['hr_threshold_source']})"
        ),
    ),
    Rule(
        "HEAT-2", Category.HEAT, "Elevated body temperature in a hot environment",
        "Body temperature >= 37.8 C AND ambient temperature >= 32 C. Wearable body temperature "
        "may be skin temperature.",
        ("body_temperature", "ambient_temperature"), (), 0.30, _heat_2,
        lambda s: (
            f"body temperature {s['body_temperature']:.1f} C while ambient temperature is "
            f"{s['ambient_temperature']:.1f} C"
        ),
    ),
)