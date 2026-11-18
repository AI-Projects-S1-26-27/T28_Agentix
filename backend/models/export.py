from typing import List, Optional
from pydantic import BaseModel, Field


class ExportManifest(BaseModel):
    export_id: str
    generated_at: str
    record_count: int
    format: str
    sha256_checksum: str
    source_filters: Optional[List[str]] = None
    time_range_start: Optional[str] = None
    time_range_end: Optional[str] = None
    clinical_disclaimer: str = (
        "HealthSensing Research Export: Informational and research purposes only. "
        "Not validated as a medical device."
    )
