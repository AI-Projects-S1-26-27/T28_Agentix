from datetime import datetime, timezone
from math import isfinite
from typing import Any, Dict, List, Optional, Tuple

from backend.db.database import MeasurementRepository
from backend.models.measurement import Measurement
from backend.models.risk import RiskAssessment
from risk_engine.risk_fusion import evaluate_environmental_risk

PERSONAL_ANOMALY_THRESHOLD = 3.0
PERSONAL_ANOMALY_SIGNALS = {
    "heart_rate": "heart_rate_bpm",
    "eda": "eda_microsiemens",
    "body_temperature": "body_temperature_c",
}


class RiskService:
    """Service for fusing environmental risks, physiological signals,
    and personal baselines into transparent risk assessments."""

    def __init__(self, repository: Optional[MeasurementRepository] = None):
        self.repository = repository

    def evaluate_reading(
        self,
        measurement: Measurement,
        reading_id: Optional[int] = None,
    ) -> RiskAssessment:
        data = measurement.to_dict()
        ts = measurement.timestamp.astimezone(timezone.utc)

        # 1. Evaluate environmental components via risk_engine
        env_result = evaluate_environmental_risk(data)
        components: Dict[str, Any] = dict(env_result.get("components", {}))
        unavailable_inputs: List[str] = list(env_result.get("unavailable_inputs", []))
        contributing_signals: List[str] = []

        base_severity = env_result.get("severity") or 0
        base_risk_level = env_result.get("risk_level", "unknown")

        # 2. Extract physiological signals
        hr = data.get("heart_rate")
        spo2 = data.get("spo2")
        temp = data.get("body_temperature")
        activity = data.get("activity_level")
        aqi = data.get("aqi")
        pm25 = data.get("pm25")

        phys_components: Dict[str, Any] = {}
        cross_modal_escalation = False
        escalation_reasons: List[str] = []

        # Check unavailable physiological signals
        if hr is None:
            unavailable_inputs.append("heart_rate")
        if spo2 is None:
            unavailable_inputs.append("spo2")
        if temp is None:
            unavailable_inputs.append("body_temperature")

        # 3. Multimodal rule: Heat Stress
        # Elevated heat index + elevated heart rate / body temperature
        heat_comp = components.get("heat")
        if heat_comp and heat_comp.get("severity", 0) >= 1:
            contributing_signals.append("ambient_temperature")
            contributing_signals.append("humidity")
            if hr is not None and hr > 100:
                cross_modal_escalation = True
                contributing_signals.append("heart_rate")
                escalation_reasons.append(
                    f"elevated heart rate ({hr:.0f} bpm) in hot/humid conditions"
                )
            if temp is not None and temp > 37.8:
                cross_modal_escalation = True
                contributing_signals.append("body_temperature")
                escalation_reasons.append(
                    f"elevated skin temperature ({temp:.1f} °C) during heat stress"
                )

        # 4. Multimodal rule: Respiratory Stress
        # High AQI/PM2.5 + decreased SpO2
        aqi_comp = components.get("air_quality")
        if aqi_comp and aqi_comp.get("severity", 0) >= 1:
            contributing_signals.append("aqi")
            if spo2 is not None and spo2 < 95.0:
                cross_modal_escalation = True
                contributing_signals.append("spo2")
                escalation_reasons.append(
                    f"decreased oxygen saturation ({spo2:.0f}%) in degraded air quality"
                )

        # 5. Physiological baseline & extreme vitals screening
        if hr is not None:
            if hr > 130 or hr < 45:
                phys_components["heart_rate"] = {
                    "risk_level": "very_high",
                    "severity": 3,
                    "value": hr,
                    "condition": "tachycardia" if hr > 130 else "bradycardia",
                }
                contributing_signals.append("heart_rate")
                escalation_reasons.append(f"extreme heart rate ({hr:.0f} bpm)")
            elif hr > 105:
                phys_components["heart_rate"] = {
                    "risk_level": "moderate",
                    "severity": 1,
                    "value": hr,
                    "condition": "elevated_heart_rate",
                }
                if "heart_rate" not in contributing_signals:
                    contributing_signals.append("heart_rate")

        if spo2 is not None and spo2 < 90.0:
            phys_components["respiratory"] = {
                "risk_level": "severe",
                "severity": 4,
                "value": spo2,
                "condition": "hypoxemia",
            }
            contributing_signals.append("spo2")
            escalation_reasons.append(f"critically low SpO2 ({spo2:.0f}%)")

        if temp is not None and (temp > 38.5 or temp < 34.0):
            phys_components["temperature"] = {
                "risk_level": "high",
                "severity": 2,
                "value": temp,
                "condition": "hyperthermia" if temp > 38.5 else "hypothermia",
            }
            contributing_signals.append("body_temperature")
            escalation_reasons.append(f"abnormal body temperature ({temp:.1f} °C)")

        if phys_components:
            components["physiological"] = phys_components

        components["personalized_anomaly"] = self._evaluate_personalized_anomaly(data)

        # 6. Fuse final severity and risk level
        final_severity = base_severity
        for p in phys_components.values():
            if p.get("severity", 0) > final_severity:
                final_severity = p["severity"]

        if cross_modal_escalation and final_severity < 4:
            final_severity = min(4, final_severity + 1)

        severity_map = {
            0: "low",
            1: "moderate",
            2: "high",
            3: "very_high",
            4: "severe",
        }
        final_risk_level = severity_map.get(final_severity, base_risk_level)

        # 7. Formulate clear explanation
        base_expl = env_result.get("explanation", "")
        if escalation_reasons:
            explanation = (
                f"{base_expl} Elevated physiological risk detected: {'; '.join(escalation_reasons)}."
            )
        else:
            explanation = base_expl

        # Deduplicate signals
        contributing_signals = list(dict.fromkeys(contributing_signals))
        unavailable_inputs = list(dict.fromkeys(unavailable_inputs))

        return RiskAssessment(
            timestamp=ts,
            reading_id=reading_id,
            risk_level=final_risk_level,
            severity=final_severity,
            components=components,
            unavailable_inputs=unavailable_inputs,
            contributing_signals=contributing_signals,
            explanation=explanation.strip(),
        )

    def _evaluate_personalized_anomaly(self, data: Dict[str, Any]) -> Dict[str, Any]:
        participant_id = data.get("participant_id") or "default"
        baseline_rows = (
            self.repository.get_baselines(participant_id)
            if self.repository is not None
            else []
        )
        baselines = {row["signal"]: row for row in baseline_rows}
        signals: Dict[str, Any] = {}
        anomaly_count = 0
        evaluated_count = 0

        for reading_signal, baseline_signal in PERSONAL_ANOMALY_SIGNALS.items():
            reading_value = data.get(reading_signal)
            baseline = baselines.get(baseline_signal)
            valid_reading = (
                reading_value is None or isfinite(float(reading_value))
            )
            result: Dict[str, Any] = {
                "value": reading_value if valid_reading else None,
                "anomaly": 0,
            }

            if not valid_reading:
                result["evaluation_status"] = "invalid_reading"
            elif baseline is None:
                result["evaluation_status"] = "missing_baseline"
            elif baseline["status"] != "stable":
                result["evaluation_status"] = "baseline_not_stable"
                result["baseline_status"] = baseline["status"]
            elif reading_value is None:
                result["evaluation_status"] = "missing_reading"
            else:
                baseline_mean = float(baseline["mean"])
                baseline_std = float(baseline["std"])

                if not isfinite(baseline_mean) or not isfinite(baseline_std):
                    result["evaluation_status"] = "invalid_baseline"
                elif baseline_std <= 0:
                    result["baseline_mean"] = baseline_mean
                    result["baseline_std"] = baseline_std
                    result["evaluation_status"] = "zero_variance_baseline"
                else:
                    result["baseline_mean"] = baseline_mean
                    result["baseline_std"] = baseline_std
                    z_score = (float(reading_value) - baseline_mean) / baseline_std
                    is_anomaly = abs(z_score) > PERSONAL_ANOMALY_THRESHOLD
                    result.update({
                        "z_score": z_score,
                        "threshold": PERSONAL_ANOMALY_THRESHOLD,
                        "anomaly": int(is_anomaly),
                        "evaluation_status": "evaluated",
                    })
                    evaluated_count += 1
                    anomaly_count += int(is_anomaly)

            signals[baseline_signal] = result

        unevaluable_count = len(PERSONAL_ANOMALY_SIGNALS) - evaluated_count
        if not baseline_rows:
            overall_status = "not_calibrated"
        elif evaluated_count == 0:
            overall_status = "unevaluable"
        elif anomaly_count and unevaluable_count:
            overall_status = "partially_evaluated_anomaly"
        elif anomaly_count:
            overall_status = "anomaly_detected"
        elif unevaluable_count:
            overall_status = "partially_evaluated"
        else:
            overall_status = "normal"

        return {
            "participant_id": participant_id,
            "threshold": PERSONAL_ANOMALY_THRESHOLD,
            "anomaly_count": anomaly_count,
            "unevaluable_count": unevaluable_count,
            "overall_anomaly": int(anomaly_count > 0),
            "overall_status": overall_status,
            "signals": signals,
        }
