from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.db.database import MeasurementRepository
from backend.models.alert import AlertResponse
from backend.models.measurement import Measurement, MeasurementResponse
from backend.models.risk import RiskAssessmentResponse
from backend.services.alert_service import AlertService
from backend.services.risk_service import RiskService
from backend.websocket.manager import ConnectionManager


class DataService:
    """Central orchestrator for reading ingestion, risk evaluation,
    alert dispatch, persistence, and live streaming."""

    def __init__(
        self,
        repository: MeasurementRepository,
        websocket_manager: Optional[ConnectionManager] = None,
    ):
        self.repository = repository
        self.risk_service = RiskService(repository=repository)
        self.alert_service = AlertService(repository=repository)
        self.websocket_manager = websocket_manager

    async def ingest_reading(
        self,
        measurement: Measurement,
    ) -> Dict[str, Any]:
        # 1. Persist the raw measurement
        saved_reading = self.repository.add(measurement)

        # 2. Evaluate comprehensive risk (environmental + physiological)
        risk = self.risk_service.evaluate_reading(
            measurement, reading_id=saved_reading.id
        )

        # 3. Persist the risk assessment
        saved_risk = self.repository.add_risk_assessment(risk)

        # 4. Check if an alert is warranted
        alert = self.alert_service.evaluate_and_create_alert(
            measurement, risk, reading_id=saved_reading.id
        )
        saved_alert: Optional[AlertResponse] = None
        if alert is not None:
            saved_alert = self.alert_service.persist_alert_if_eligible(alert)

        # 5. Broadcast to connected WebSocket clients
        if self.websocket_manager:
            broadcast_payload = {
                "type": "reading_processed",
                "reading": saved_reading.to_dict(),
                "risk": saved_risk.to_dict(),
                "alert": saved_alert.to_dict() if saved_alert else None,
                # Include root fields for backward compatibility with older clients
                **saved_reading.to_dict(),
            }
            await self.websocket_manager.broadcast(broadcast_payload)

        return {
            "reading": saved_reading,
            "risk": saved_risk,
            "alert": saved_alert,
        }
