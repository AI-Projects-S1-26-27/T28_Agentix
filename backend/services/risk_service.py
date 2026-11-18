from datetime import datetime, timezone
from math import isfinite
from typing import Any, Dict, List, Optional, Tuple

from backend.db.database import MeasurementRepository
from backend.models.measurement import Measurement
from backend.models.risk import RiskAssessment
from risk_engine.risk_fusion import evaluate_environmental_risk
from risk_engine.risk_levels import classify_eda

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

        # 2. Extract physiological signals we actually have (WESAD dataset)
        # spo2 is NOT in our datasets — omitted entirely
        hr = data.get("heart_rate")
        temp = data.get("body_temperature")
        eda = data.get("eda")
        protocol_label = data.get("protocol_label")  # WESAD: 2 = stress protocol
        activity = data.get("activity_level")
        pm25 = data.get("pm25")
        context_flags = data.get("context_flags") or []

        # Context-aware HR thresholds
        if "exercising" in context_flags:
            hr_tachycardia, hr_elevated, hr_bradycardia = 180, 150, 40
        elif "resting" in context_flags:
            hr_tachycardia, hr_elevated, hr_bradycardia = 90, 80, 55
        else:
            hr_tachycardia, hr_elevated, hr_bradycardia = 130, 105, 45

        # Context-aware EDA threshold
        eda_elevated_threshold = 5.0 if "high_stress" in context_flags else 10.0

        phys_components: Dict[str, Any] = {}
        cross_modal_escalation = False
        escalation_reasons: List[str] = []

        # Check unavailable physiological signals (only ones we can have)
        if hr is None:
            unavailable_inputs.append("heart_rate")
        if temp is None:
            unavailable_inputs.append("body_temperature")
        if eda is None:
            unavailable_inputs.append("eda")

        # 3. Multimodal rule: Heat Stress
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

        # 4. Physiological baseline & extreme vitals screening
        if hr is not None:
            if hr > hr_tachycardia or hr < hr_bradycardia:
                phys_components["heart_rate"] = {
                    "risk_level": "very_high",
                    "severity": 3,
                    "value": hr,
                    "condition": "tachycardia" if hr > hr_tachycardia else "bradycardia",
                }
                contributing_signals.append("heart_rate")
                escalation_reasons.append(f"extreme heart rate ({hr:.0f} bpm)")
            elif hr > hr_elevated:
                phys_components["heart_rate"] = {
                    "risk_level": "moderate",
                    "severity": 1,
                    "value": hr,
                    "condition": "elevated_heart_rate",
                }
                if "heart_rate" not in contributing_signals:
                    contributing_signals.append("heart_rate")

        if temp is not None and (temp > 38.5 or temp < 34.0):
            phys_components["temperature"] = {
                "risk_level": "high",
                "severity": 2,
                "value": temp,
                "condition": "hyperthermia" if temp > 38.5 else "hypothermia",
            }
            contributing_signals.append("body_temperature")
            escalation_reasons.append(f"abnormal body temperature ({temp:.1f} °C)")

        # EDA (electrodermal activity) stress proxy
        if eda is not None:
            eda_result = classify_eda(eda)
            if eda_result.get("severity", 0) > 0:
                phys_components["eda"] = eda_result
                contributing_signals.append("eda")
                if eda_result["severity"] >= 2:
                    escalation_reasons.append(
                        f"high skin conductance ({eda:.1f} µS) — stress response"
                    )

        # stress_label from WESAD ground truth (2 = stress protocol)
        if protocol_label is not None:
            try:
                label = int(protocol_label)
            except (ValueError, TypeError):
                label = None
            if label == 2:
                cross_modal_escalation = True
                contributing_signals.append("protocol_label")
                escalation_reasons.append("WESAD stress protocol active (label=2)")
                phys_components["stress_protocol"] = {
                    "risk_level": "high",
                    "severity": 2,
                    "value": label,
                    "condition": "stress_protocol",
                }

        if phys_components:
            components["physiological"] = phys_components

        components["personalized_anomaly"] = self._evaluate_personalized_anomaly(data)

        # 5. Fuse final severity and risk level
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

        # 6. Formulate clear explanation
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
