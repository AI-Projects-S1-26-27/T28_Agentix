from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from backend.models.alert import Alert, AlertResponse
from backend.models.measurement import Measurement, MeasurementResponse
from backend.models.risk import RiskAssessment, RiskAssessmentResponse


class MeasurementRepository:
    """SQLite repository for local health/environmental measurement storage,
    risk assessment records, alerts, and baseline tracking."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            # 1. Measurements table
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    scenario TEXT,
                    heart_rate REAL,
                    spo2 REAL,
                    body_temperature REAL,
                    activity_level REAL,
                    ambient_temperature REAL,
                    humidity REAL,
                    aqi REAL,
                    pm25 REAL,
                    eda REAL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_measurements_timestamp "
                "ON measurements(timestamp)"
            )

            # Ensure 'eda' column exists for existing tables
            try:
                connection.execute("ALTER TABLE measurements ADD COLUMN eda REAL")
            except sqlite3.OperationalError:
                pass

            # 2. Risk assessments table
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reading_id INTEGER,
                    timestamp TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    severity INTEGER,
                    components_json TEXT,
                    unavailable_inputs_json TEXT,
                    contributing_signals_json TEXT,
                    explanation TEXT NOT NULL,
                    FOREIGN KEY (reading_id) REFERENCES measurements(id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_risk_timestamp "
                "ON risk_assessments(timestamp)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_risk_reading_id "
                "ON risk_assessments(reading_id)"
            )

            # 3. Alerts table
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reading_id INTEGER,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    category TEXT NOT NULL,
                    message TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    contributing_signals_json TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    FOREIGN KEY (reading_id) REFERENCES measurements(id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp "
                "ON alerts(timestamp)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged "
                "ON alerts(acknowledged)"
            )

            # 4. Personal Baselines table
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS personal_baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    participant_id TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    mean REAL NOT NULL,
                    std REAL NOT NULL,
                    observation_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(participant_id, signal)
                )
                """
            )

    # -------------------------------------------------------------------------
    # Measurements
    # -------------------------------------------------------------------------

    def add(self, measurement: Measurement) -> MeasurementResponse:
        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO measurements (
                    timestamp, source, scenario, heart_rate, spo2,
                    body_temperature, activity_level, ambient_temperature,
                    humidity, aqi, pm25, eda
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                measurement.as_storage_tuple(),
            )
            measurement_id = cursor.lastrowid
        return MeasurementResponse(id=measurement_id, **measurement.to_dict())

    def recent(self, limit: int = 50) -> list[MeasurementResponse]:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, timestamp, source, scenario, heart_rate, spo2,
                       body_temperature, activity_level, ambient_temperature,
                       humidity, aqi, pm25, eda
                FROM measurements
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [MeasurementResponse(**dict(row)) for row in rows]

    def query_measurements(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MeasurementResponse]:
        query = (
            "SELECT id, timestamp, source, scenario, heart_rate, spo2, "
            "body_temperature, activity_level, ambient_temperature, "
            "humidity, aqi, pm25, eda FROM measurements WHERE 1=1"
        )
        params: list[Any] = []
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time.astimezone(timezone.utc).isoformat())
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time.astimezone(timezone.utc).isoformat())

        query += " ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()
        return [MeasurementResponse(**dict(row)) for row in rows]

    def get_measurement_stats(self) -> Dict[str, Any]:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT
                    COUNT(id) AS total_count,
                    AVG(heart_rate) AS avg_heart_rate,
                    MIN(heart_rate) AS min_heart_rate,
                    MAX(heart_rate) AS max_heart_rate,
                    AVG(spo2) AS avg_spo2,
                    MIN(spo2) AS min_spo2,
                    MAX(spo2) AS max_spo2,
                    AVG(body_temperature) AS avg_body_temperature,
                    MIN(body_temperature) AS min_body_temperature,
                    MAX(body_temperature) AS max_body_temperature,
                    AVG(ambient_temperature) AS avg_ambient_temperature,
                    MIN(ambient_temperature) AS min_ambient_temperature,
                    MAX(ambient_temperature) AS max_ambient_temperature,
                    AVG(humidity) AS avg_humidity,
                    MIN(humidity) AS min_humidity,
                    MAX(humidity) AS max_humidity,
                    AVG(aqi) AS avg_aqi,
                    MIN(aqi) AS min_aqi,
                    MAX(aqi) AS max_aqi,
                    AVG(pm25) AS avg_pm25,
                    MIN(pm25) AS min_pm25,
                    MAX(pm25) AS max_pm25
                FROM measurements
                """
            ).fetchone()
            return dict(row) if row else {}

    # -------------------------------------------------------------------------
    # Risk Assessments
    # -------------------------------------------------------------------------

    def add_risk_assessment(self, assessment: RiskAssessment) -> RiskAssessmentResponse:
        ts = assessment.timestamp.astimezone(timezone.utc).isoformat()
        components_json = json.dumps(assessment.components)
        unavailable_json = json.dumps(assessment.unavailable_inputs)
        contributing_json = json.dumps(assessment.contributing_signals)

        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO risk_assessments (
                    reading_id, timestamp, risk_level, severity,
                    components_json, unavailable_inputs_json,
                    contributing_signals_json, explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assessment.reading_id,
                    ts,
                    assessment.risk_level,
                    assessment.severity,
                    components_json,
                    unavailable_json,
                    contributing_json,
                    assessment.explanation,
                ),
            )
            assessment_id = cursor.lastrowid
        return RiskAssessmentResponse(id=assessment_id, **assessment.to_dict())

    def _row_to_risk_assessment(self, row: sqlite3.Row) -> RiskAssessmentResponse:
        data = dict(row)
        components = json.loads(data.pop("components_json") or "{}")
        unavailable_inputs = json.loads(data.pop("unavailable_inputs_json") or "[]")
        contributing_signals = json.loads(data.pop("contributing_signals_json") or "[]")
        return RiskAssessmentResponse(
            id=data["id"],
            reading_id=data.get("reading_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            risk_level=data["risk_level"],
            severity=data["severity"],
            components=components,
            unavailable_inputs=unavailable_inputs,
            contributing_signals=contributing_signals,
            explanation=data["explanation"],
        )

    def recent_risks(self, limit: int = 50) -> list[RiskAssessmentResponse]:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, reading_id, timestamp, risk_level, severity,
                       components_json, unavailable_inputs_json,
                       contributing_signals_json, explanation
                FROM risk_assessments
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_risk_assessment(row) for row in rows]

    def get_latest_risk(self) -> Optional[RiskAssessmentResponse]:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT id, reading_id, timestamp, risk_level, severity,
                       components_json, unavailable_inputs_json,
                       contributing_signals_json, explanation
                FROM risk_assessments
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            return self._row_to_risk_assessment(row)

    def get_risk_by_reading_id(self, reading_id: int) -> Optional[RiskAssessmentResponse]:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT id, reading_id, timestamp, risk_level, severity,
                       components_json, unavailable_inputs_json,
                       contributing_signals_json, explanation
                FROM risk_assessments
                WHERE reading_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (reading_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_risk_assessment(row)

    # -------------------------------------------------------------------------
    # Alerts
    # -------------------------------------------------------------------------

    def add_alert(self, alert: Alert) -> AlertResponse:
        ts = alert.timestamp.astimezone(timezone.utc).isoformat()
        contributing_json = json.dumps(alert.contributing_signals)
        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO alerts (
                    reading_id, timestamp, level, category,
                    message, explanation, contributing_signals_json,
                    acknowledged
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.reading_id,
                    ts,
                    alert.level,
                    alert.category,
                    alert.message,
                    alert.explanation,
                    contributing_json,
                    1 if alert.acknowledged else 0,
                ),
            )
            alert_id = cursor.lastrowid
        return AlertResponse(id=alert_id, **alert.to_dict())

    def _row_to_alert(self, row: sqlite3.Row) -> AlertResponse:
        data = dict(row)
        contributing_signals = json.loads(data.pop("contributing_signals_json") or "[]")
        return AlertResponse(
            id=data["id"],
            reading_id=data.get("reading_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            level=data["level"],
            category=data["category"],
            message=data["message"],
            explanation=data["explanation"],
            contributing_signals=contributing_signals,
            acknowledged=bool(data["acknowledged"]),
        )

    def recent_alerts(
        self, limit: int = 50, unacknowledged_only: bool = False
    ) -> list[AlertResponse]:
        query = (
            "SELECT id, reading_id, timestamp, level, category, "
            "message, explanation, contributing_signals_json, acknowledged "
            "FROM alerts"
        )
        params: list[Any] = []
        if unacknowledged_only:
            query += " WHERE acknowledged = 0"
        query += " ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_alert(row) for row in rows]

    def acknowledge_alert(self, alert_id: int) -> Optional[AlertResponse]:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,)
            )
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT id, reading_id, timestamp, level, category,
                       message, explanation, contributing_signals_json, acknowledged
                FROM alerts WHERE id = ?
                """,
                (alert_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_alert(row)

    # -------------------------------------------------------------------------
    # Personal Baselines
    # -------------------------------------------------------------------------

    def save_baseline(
        self,
        participant_id: str,
        signal: str,
        mean: float,
        std: float,
        observation_count: int,
        status: str,
    ) -> Dict[str, Any]:
        updated_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO personal_baselines (
                    participant_id, signal, mean, std, observation_count, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(participant_id, signal) DO UPDATE SET
                    mean = excluded.mean,
                    std = excluded.std,
                    observation_count = excluded.observation_count,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (participant_id, signal, mean, std, observation_count, status, updated_at),
            )
        return {
            "participant_id": participant_id,
            "signal": signal,
            "mean": mean,
            "std": std,
            "observation_count": observation_count,
            "status": status,
            "updated_at": updated_at,
        }

    def get_baselines(self, participant_id: str = "default") -> list[Dict[str, Any]]:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT participant_id, signal, mean, std, observation_count, status, updated_at
                FROM personal_baselines
                WHERE participant_id = ?
                ORDER BY signal ASC
                """,
                (participant_id,),
            ).fetchall()
        return [dict(row) for row in rows]
