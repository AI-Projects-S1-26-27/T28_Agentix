"""Risk fusion / assess_risk tests (LOW / MODERATE / HIGH / CRITICAL)."""
import json
import math
import random

from experiments.risk_rules_simulator_check import run as simulator_check
from reasoning.models import Category, RuleStatus
from risk_engine.assess import assess_risk
from risk_engine.models import RiskLevel
from simulator.data_generator import generate_data

FIELDS = [
    "heart_rate", "spo2", "body_temperature", "activity_level", "ambient_temperature",
    "humidity", "aqi", "pm25", "sleep_duration_hours",
]


def reading(**values):
    base = {"timestamp": "2026-10-19T10:00:00Z", "source": "simulator", **{f: None for f in FIELDS}}
    base.update(values)
    return base


def series(count=5, **values):
    return [reading(**values) for _ in range(count)]


HOT = dict(ambient_temperature=34, humidity=75, heart_rate=115, activity_level=0.5)
VERY_HOT = dict(ambient_temperature=35, humidity=80, heart_rate=120, body_temperature=38.5)


def rule(assessment, rule_id):
    return next(r for r in assessment.rule_results if r.rule_id == rule_id)


# ---------------------------------------------------------------- levels
def test_normal_readings_are_low():
    a = assess_risk(series(ambient_temperature=24, humidity=45, heart_rate=70, activity_level=0.3))
    assert a.level == RiskLevel.LOW and a.primary_category is None and a.triggered_rules == []


def test_single_triggered_rule_without_model_is_moderate():
    a = assess_risk(series(**HOT))
    assert a.level == RiskLevel.MODERATE and a.primary_category == Category.HEAT


def test_model_score_raises_level():
    assert assess_risk(series(**HOT), anomaly_score=0.9).level == RiskLevel.HIGH


def test_two_rules_without_model_are_capped_at_high():
    assert assess_risk(series(**VERY_HOT)).level == RiskLevel.HIGH


def test_critical_needs_two_rules_and_a_model_score():
    assert assess_risk(series(**VERY_HOT), anomaly_score=0.9).level == RiskLevel.CRITICAL
    assert assess_risk(series(**HOT), anomaly_score=1.0).level != RiskLevel.CRITICAL


def test_model_only_anomaly_is_capped_at_moderate():
    a = assess_risk(series(ambient_temperature=22, humidity=40, heart_rate=70), anomaly_score=0.95)
    assert a.level == RiskLevel.MODERATE and a.primary_category == Category.GENERAL
    assert "no contextual rule explains it" in a.explanation


def test_conflict_is_flagged_when_rules_fire_but_model_is_quiet():
    a = assess_risk(series(**HOT), anomaly_score=0.05)
    assert a.level == RiskLevel.MODERATE and "Conflict" in a.explanation


def test_conflicting_categories_pick_strongest():
    readings = series(aqi=200, spo2=90, **VERY_HOT)  # heat (2 rules) beats pollution (1 rule)
    a = assess_risk(readings)
    assert a.primary_category == Category.HEAT
    assert {"HEAT-1", "HEAT-2", "POLL-1"} <= set(a.triggered_rules)


# --------------------------------------------------------- missing data
def test_missing_pollution_inputs_do_not_raise_risk():
    a = assess_risk(series(spo2=85))
    assert a.level == RiskLevel.LOW and "aqi or pm25" in a.unavailable_inputs
    assert "aqi or pm25" in a.explanation


def test_no_readings_is_safe():
    a = assess_risk([])
    assert a.status == "insufficient_data" and a.level == RiskLevel.LOW
    assert a.unavailable_inputs == ["readings"]


def test_insufficient_history_is_warned():
    a = assess_risk(series(2, **HOT))
    assert rule(a, "HEAT-1").status == RuleStatus.NOT_EVALUABLE
    assert any("Not enough history" in w for w in a.warnings)


def test_invalid_anomaly_scores_are_ignored():
    for bad in (2.5, -1, math.nan, "high", True):
        a = assess_risk(series(heart_rate=70), anomaly_score=bad)
        assert a.model_available is False and any("anomaly score" in w for w in a.warnings)


def test_missing_baseline_is_disclosed():
    assert any("No personal baseline" in w for w in assess_risk(series(**HOT)).warnings)
    baseline = {"heart_rate": {"mean": 70, "std": 5}}
    assert not any("No personal baseline" in w for w in assess_risk(series(**HOT), baseline=baseline).warnings)


# ------------------------------------------------------------ input shapes
def test_newest_first_ordering():
    old = series(5, ambient_temperature=22, humidity=40, heart_rate=70)
    recent = series(5, **VERY_HOT)
    api_order = list(reversed(old + recent))  # what GET /api/readings returns
    assert assess_risk(api_order, newest_first=True).primary_category == Category.HEAT
    assert assess_risk(api_order).level == RiskLevel.LOW  # wrong order looks at the old readings


def test_accepts_backend_measurement_objects():
    from datetime import datetime, timezone

    from backend.models.measurement import Measurement

    measurements = [
        Measurement(timestamp=datetime.now(timezone.utc), source="simulator", heart_rate=115,
                    ambient_temperature=34, humidity=75, activity_level=0.5)
        for _ in range(5)
    ]
    assert assess_risk(measurements).primary_category == Category.HEAT


# ------------------------------------------------------------------ output
def test_output_is_json_serialisable_with_required_fields():
    a = assess_risk(series(**HOT), anomaly_score=0.7)
    d = json.loads(json.dumps(a.to_dict()))
    for key in ("level", "primary_category", "triggered_rules", "contributing_signals",
                "unavailable_inputs", "explanation", "disclaimer", "categories", "environment"):
        assert key in d
    assert d["level"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}


def test_explanation_names_signals_unavailable_inputs_and_disclaimer():
    a = assess_risk(series(**HOT), anomaly_score=0.7)
    assert "HEAT-1" in a.explanation and "115" in a.explanation
    assert "aqi or pm25" in a.explanation and "not a medical diagnosis" in a.explanation


def test_environment_context_reuses_environmental_risk_module():
    a = assess_risk(series(aqi=175, ambient_temperature=40, humidity=70, heart_rate=70))
    assert a.environment["risk_level"] == "severe"


def test_environment_context_is_none_for_invalid_values():
    assert assess_risk(series(aqi=5000, heart_rate=70)).environment is None


# -------------------------------------------------- simulator integration
def test_simulator_normal_scenario_has_no_false_alarms():
    summary = simulator_check(windows=200, seed=7)
    assert summary["normal"]["levels"] == {"LOW": 200}


def test_simulator_heatwave_and_pollution_are_detected_in_most_windows():
    summary = simulator_check(windows=200, seed=7)
    assert summary["heatwave"]["categories"].get("heat_environmental_stress", 0) > 150
    assert summary["pollution"]["categories"].get("pollution_respiratory", 0) > 130


def test_simulator_readings_flow_through_assess_risk():
    rng = random.Random(3)
    readings = [generate_data("heatwave", rng) for _ in range(5)]
    assert assess_risk(readings).to_dict()["status"] == "ok"