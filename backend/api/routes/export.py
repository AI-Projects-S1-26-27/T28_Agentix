from typing import Optional
from fastapi import APIRouter, Depends, Query, Response

from backend.api.routes.current import get_repository
from backend.db.database import MeasurementRepository
from backend.models.export import ExportManifest
from backend.services.export_service import ExportService

router = APIRouter(prefix="/export", tags=["export"])


def get_export_service(
    repository: MeasurementRepository = Depends(get_repository),
) -> ExportService:
    return ExportService(repository=repository)


@router.get("/csv")
def export_csv(
    limit: int = Query(default=500, ge=1, le=5000),
    start: Optional[str] = Query(default=None),
    end: Optional[str] = Query(default=None),
    service: ExportService = Depends(get_export_service),
):
    content, manifest = service.generate_csv_export(
        limit=limit, start_time=start, end_time=end
    )
    headers = {
        "Content-Disposition": f'attachment; filename="healthsensing_export_{manifest.export_id}.csv"',
        "X-Checksum-SHA256": manifest.sha256_checksum,
        "X-Record-Count": str(manifest.record_count),
    }
    return Response(content=content, media_type="text/csv", headers=headers)


@router.get("/json")
def export_json(
    limit: int = Query(default=500, ge=1, le=5000),
    service: ExportService = Depends(get_export_service),
):
    content, manifest = service.generate_json_export(limit=limit)
    headers = {
        "Content-Disposition": f'attachment; filename="healthsensing_export_{manifest.export_id}.json"',
        "X-Checksum-SHA256": manifest.sha256_checksum,
        "X-Record-Count": str(manifest.record_count),
    }
    return Response(content=content, media_type="application/json", headers=headers)


@router.get("/manifest", response_model=ExportManifest)
def get_export_manifest(
    limit: int = Query(default=500, ge=1, le=5000),
    service: ExportService = Depends(get_export_service),
) -> ExportManifest:
    _, manifest = service.generate_json_export(limit=limit)
    return manifest
