"""Pollution-associated respiratory-risk rules."""
from typing import Optional

from reasoning.knowledge_base.knowledge_base import (
    THRESHOLDS, Baseline, CheckResult, Reading, Rule, num, spo2_low,
)
from reasoning.models import Category


def _poll_1(reading: Reading, baseline: Optional[Baseline]) -> CheckResult:
    aqi, pm25 = num(reading, "aqi"), num(reading, "pm25")
    polluted = (aqi is not None and aqi >= THRESHOLDS["aqi_elevated"]) or (
        pm25 is not None and pm25 >= THRESHOLDS["pm25_elevated_ug_m3"]
    )
    low, signals = spo2_low(reading, baseline)
    signals.update({k: v for k, v in (("aqi", aqi), ("pm25", pm25)) if v is not None})
    return (polluted and low), signals


POLLUTION_RULES = (
    Rule(
        "POLL-1", Category.POLLUTION, "Poor air quality with SpO2 below personal norm",
        "AQI >= 101 OR PM2.5 >= 35 ug/m3, AND SpO2 at or below 94% (or personal baseline mean - 2 "
        "points). Needs SpO2 and at least one of AQI / PM2.5; missing data is never evidence.",
        ("spo2",), (("aqi", "pm25"),), 0.45, _poll_1,
        lambda s: (
            "air quality is elevated ("
            + ", ".join(f"{k.upper()} {s[k]:.0f}" for k in ("aqi", "pm25") if k in s)
            + f") and SpO2 is {s['spo2']:.0f}% (at or below {s['spo2_limit']:.0f}%, "
            f"{s['spo2_limit_source']})"
        ),
    ),
)