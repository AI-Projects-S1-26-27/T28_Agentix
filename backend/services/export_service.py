import csv
import hashlib
import io
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.db.database import MeasurementRepository
from backend.models.export import ExportManifest


class ExportService:
    def __init__(self, repository: MeasurementRepository):
        self._repository = repository

    def generate_csv_export(
        self,
        limit: int = 1000,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Tuple[str, ExportManifest]:
        readings = self._repository.recent(limit=limit)
        # Apply time filtering if provided
        filtered = []
        for r in readings:
            ts = r.timestamp.isoformat() if hasattr(r.timestamp, "isoformat") else str(r.timestamp)
            if start_time and ts < start_time:
                continue
            if end_time and ts > end_time:
                continue
            filtered.append(r)

        output = io.StringIO()
        fieldnames = [
            "id",
            "timestamp",
            "source",
            "participant_id",
            "heart_rate",
            "eda",
            "body_temperature",
            "ambient_temperature",
            "humidity",
            "pm25",
            "protocol_label",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for r in filtered:
            writer.writerow({
                "id": r.id,
                "timestamp": r.timestamp.isoformat() if hasattr(r.timestamp, "isoformat") else str(r.timestamp),
                "source": r.source,
                "participant_id": r.participant_id or "",
                "heart_rate": r.heart_rate or "",
                "eda": r.eda or "",
                "body_temperature": r.body_temperature or "",
                "ambient_temperature": r.ambient_temperature or "",
                "humidity": r.humidity or "",
                "pm25": r.pm25 or "",
                "protocol_label": r.protocol_label if r.protocol_label is not None else "",
            })

        csv_content = output.getvalue()
        sha256 = hashlib.sha256(csv_content.encode("utf-8")).hexdigest()

        manifest = ExportManifest(
            export_id=f"exp-{uuid.uuid4().hex[:8]}",
            generated_at=datetime.now(timezone.utc).isoformat(),
            record_count=len(filtered),
            format="csv",
            sha256_checksum=sha256,
            time_range_start=start_time,
            time_range_end=end_time,
        )
        return csv_content, manifest

    def generate_json_export(
        self, limit: int = 1000
    ) -> Tuple[str, ExportManifest]:
        readings = self._repository.recent(limit=limit)
        data = [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in readings]

        # Convert datetime objects to strings
        for row in data:
            if "timestamp" in row and hasattr(row["timestamp"], "isoformat"):
                row["timestamp"] = row["timestamp"].isoformat()

        json_content = json.dumps(data, indent=2)
        sha256 = hashlib.sha256(json_content.encode("utf-8")).hexdigest()

        manifest = ExportManifest(
            export_id=f"exp-{uuid.uuid4().hex[:8]}",
            generated_at=datetime.now(timezone.utc).isoformat(),
            record_count=len(readings),
            format="json",
            sha256_checksum=sha256,
        )
        return json_content, manifest
