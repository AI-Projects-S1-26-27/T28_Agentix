"""Rule-engine tests: heat, pollution, fatigue and general rules."""
import json
import math

from reasoning.models import Category, RuleStatus
from reasoning.rule_engine import CONDITIONS_PATH, evaluate_rules, rule_table

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


def result(readings, rule_id, baseline=None):
    return next(r for r in evaluate_rules(readings, baseline) if r.rule_id == rule_id)


# ------------------------------------------------------------------ heat
def test_heat_1_triggers_on_hot_humid_with_elevated_hr():
    r = result(series(ambient_temperature=34, humidity=75, heart_rate=115, activity_level=0.5), "HEAT-1")
    assert r.status == RuleStatus.TRIGGERED
    assert r.category == Category.HEAT
    assert "heat index" in r.explanation and "115" in r.explanation


def test_heat_1_not_triggered_on_normal_day():
    r = result(series(ambient_temperature=24, humidity=45, heart_rate=70), "HEAT-1")
    assert r.status == RuleStatus.NOT_TRIGGERED


def test_heat_1_needs_elevated_hr_even_when_very_hot():
    assert result(series(ambient_temperature=38, humidity=80, heart_rate=72), "HEAT-1").status == RuleStatus.NOT_TRIGGERED


def test_heat_1_needs_hot_conditions_even_when_hr_high():
    assert result(series(ambient_temperature=22, humidity=40, heart_rate=130), "HEAT-1").status == RuleStatus.NOT_TRIGGERED


def test_heat_1_ignores_single_noisy_spike():
    readings = series(4, ambient_temperature=34, humidity=75, heart_rate=70)
    readings.append(reading(ambient_temperature=34, humidity=75, heart_rate=160))
    assert result(readings, "HEAT-1").status == RuleStatus.NOT_TRIGGERED


def test_heat_1_persistence_boundary_three_of_five():
    hot = dict(ambient_temperature=34, humidity=75, heart_rate=115)
    cool = dict(ambient_temperature=34, humidity=75, heart_rate=70)
    assert result([reading(**hot)] * 3 + [reading(**cool)] * 2, "HEAT-1").status == RuleStatus.TRIGGERED
    assert result([reading(**hot)] * 2 + [reading(**cool)] * 3, "HEAT-1").status == RuleStatus.NOT_TRIGGERED


def test_heat_1_hr_threshold_boundary():
    base = dict(ambient_temperature=34, humidity=75)
    assert result(series(heart_rate=100, **base), "HEAT-1").status == RuleStatus.TRIGGERED
    assert result(series(heart_rate=99, **base), "HEAT-1").status == RuleStatus.NOT_TRIGGERED


def test_heat_1_uses_personal_baseline():
    readings = series(ambient_temperature=34, humidity=75, heart_rate=90)
    assert result(readings, "HEAT-1").status == RuleStatus.NOT_TRIGGERED  # fixed threshold 100
    baseline = {"heart_rate": {"mean": 62, "std": 4}}  # threshold 68
    assert result(readings, "HEAT-1", baseline).status == RuleStatus.TRIGGERED


def test_heat_2_body_temperature_in_hot_environment():
    assert result(series(body_temperature=38.2, ambient_temperature=36), "HEAT-2").status == RuleStatus.TRIGGERED
    assert result(series(body_temperature=38.2, ambient_temperature=24), "HEAT-2").status == RuleStatus.NOT_TRIGGERED


# ------------------------------------------------------------- pollution
def test_pollution_triggers_with_aqi_and_low_spo2():
    r = result(series(aqi=160, spo2=92), "POLL-1")
    assert r.status == RuleStatus.TRIGGERED
    assert r.category == Category.POLLUTION


def test_pollution_pm25_alone_is_enough():
    assert result(series(pm25=60, spo2=93), "POLL-1").status == RuleStatus.TRIGGERED


def test_pollution_clean_air_with_low_spo2_does_not_trigger():
    assert result(series(aqi=40, pm25=10, spo2=92), "POLL-1").status == RuleStatus.NOT_TRIGGERED


def test_pollution_dirty_air_with_normal_spo2_does_not_trigger():
    assert result(series(aqi=250, spo2=98), "POLL-1").status == RuleStatus.NOT_TRIGGERED


def test_missing_aqi_and_pm25_is_never_evidence():
    r = result(series(spo2=88), "POLL-1")
    assert r.status == RuleStatus.NOT_EVALUABLE
    assert r.missing_inputs == ["aqi or pm25"]


def test_missing_spo2_is_never_evidence():
    r = result(series(aqi=300), "POLL-1")
    assert r.status == RuleStatus.NOT_EVALUABLE
    assert r.missing_inputs == ["spo2"]


def test_pollution_uses_personal_spo2_baseline():
    readings = series(aqi=150, spo2=95.5)
    assert result(readings, "POLL-1").status == RuleStatus.NOT_TRIGGERED  # fixed limit 94
    assert result(readings, "POLL-1", {"spo2": {"mean": 98, "std": 0.5}}).status == RuleStatus.TRIGGERED


def test_partial_missing_data_still_evaluates_with_remaining_readings():
    readings = series(3, aqi=160, spo2=92) + series(2, aqi=160, spo2=None)
    r = result(readings, "POLL-1")
    assert r.status == RuleStatus.TRIGGERED and r.evaluable == 3


# --------------------------------------------------------------- fatigue
def test_fatigue_triggers_with_short_sleep_and_elevated_hr():
    assert result(series(sleep_duration_hours=4.5, heart_rate=110), "FAT-1").status == RuleStatus.TRIGGERED


def test_fatigue_triggers_with_short_sleep_and_high_activity():
    assert result(series(sleep_duration_hours=4.5, heart_rate=80, activity_level=0.9), "FAT-1").status == RuleStatus.TRIGGERED


def test_fatigue_not_triggered_with_enough_sleep():
    assert result(series(sleep_duration_hours=8, heart_rate=110), "FAT-1").status == RuleStatus.NOT_TRIGGERED


def test_fatigue_not_evaluable_without_sleep_field():
    r = result(series(heart_rate=110), "FAT-1")
    assert r.status == RuleStatus.NOT_EVALUABLE
    assert r.missing_inputs == ["sleep_duration_hours"]


# --------------------------------------------------------------- general
def test_resting_heart_rate_rule():
    assert result(series(heart_rate=110, activity_level=0.1), "GEN-1").status == RuleStatus.TRIGGERED
    assert result(series(heart_rate=110, activity_level=0.7), "GEN-1").status == RuleStatus.NOT_TRIGGERED


# -------------------------------------------------------------- fail-safe
def test_insufficient_history_is_not_evaluable():
    r = result(series(2, ambient_temperature=34, humidity=75, heart_rate=115), "HEAT-1")
    assert r.status == RuleStatus.NOT_EVALUABLE and r.missing_inputs == []
    assert "need 3 readings" in r.explanation


def test_empty_input_does_not_crash():
    assert all(r.status == RuleStatus.NOT_EVALUABLE for r in evaluate_rules([]))


def test_garbage_values_are_treated_as_missing():
    readings = series(heart_rate="abc", ambient_temperature=True, humidity=math.nan, spo2=math.inf)
    assert all(r.status != RuleStatus.TRIGGERED for r in evaluate_rules(readings))


def test_out_of_range_environment_values_fail_safe():
    # calculate_heat_index_celsius rejects these; the rule must not raise.
    assert result(series(ambient_temperature=500, humidity=75, heart_rate=115), "HEAT-1").status == RuleStatus.NOT_TRIGGERED


# ------------------------------------------------------- knowledge base
def test_conditions_json_matches_rule_definitions():
    on_disk = json.loads(CONDITIONS_PATH.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(rule_table())), "Run: python -m reasoning.rule_engine"


def test_every_category_has_a_rule():
    categories = {rule["category"] for rule in rule_table()["rules"]}
    assert categories == {c.value for c in Category}