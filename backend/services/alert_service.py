from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.db.database import MeasurementRepository
from backend.models.alert import Alert, AlertResponse
from backend.models.measurement import Measurement
from backend.models.risk import RiskAssessment


class AlertService:
    """Service to evaluate risk assessments and trigger proactive,
    deduplicated alerts for high-risk environmental and physiological events."""

    def __init__(self, repository: Optional[MeasurementRepository] = None):
        self.repository = repository
        self._last_alert_time: Dict[str, datetime] = {}

    def evaluate_and_create_alert(
        self,
        measurement: Measurement,
        risk: RiskAssessment,
        reading_id: Optional[int] = None,
    ) -> Optional[Alert]:
        severity = risk.severity or 0
        if severity < 2:
            return None

        level = "critical" if severity >= 3 else "warning"

        # Determine alert category
        components = risk.components
        category = "multimodal"
        if "heat" in components and components["heat"].get("severity", 0) >= 2:
            category = "heat_stress"
        elif "air_quality" in components and components["air_quality"].get("severity", 0) >= 2:
            category = "air_quality"
        elif "physiological" in components:
            category = "physiological"

        # Message formulation
        if category == "heat_stress":
            msg = "Heat stress warning: hazardous thermal conditions detected."
        elif category == "air_quality":
            msg = "Air quality alert: elevated pollution levels detected."
        elif category == "physiological":
            msg = "Physiological alert: abnormal vital signs detected."
        else:
            msg = f"Elevated health risk ({risk.risk_level.replace('_', ' ')}) detected."

        alert = Alert(
            timestamp=risk.timestamp,
            reading_id=reading_id,
            level=level,
            category=category,
            message=msg,
            explanation=risk.explanation,
            contributing_signals=risk.contributing_signals,
            acknowledged=False,
        )

        return alert

    def persist_alert_if_eligible(
        self,
        alert: Alert,
    ) -> Optional[AlertResponse]:
        if not self.repository:
            return None

        # Check de-duplication within 30 seconds for the same category
        last_time = self._last_alert_time.get(alert.category)
        if last_time:
            delta = (alert.timestamp - last_time).total_seconds()
            if delta < 30 and alert.level != "critical":
                return None

        saved = self.repository.add_alert(alert)
        self._last_alert_time[alert.category] = alert.timestamp
        return saved

    def acknowledge(self, alert_id: int) -> Optional[AlertResponse]:
        if not self.repository:
            return None
        return self.repository.acknowledge_alert(alert_id)
