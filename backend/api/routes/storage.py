from typing import Any, Dict
from fastapi import APIRouter, Depends, Query

from backend.core.config import database_path
from backend.db.retention import RetentionPruneResult, StorageDiagnostics, StorageManager

router = APIRouter(prefix="/storage", tags=["storage"])


def get_storage_manager() -> StorageManager:
    return StorageManager(database_path())


@router.get("/stats", response_model=StorageDiagnostics)
def get_storage_statistics(
    manager: StorageManager = Depends(get_storage_manager),
) -> StorageDiagnostics:
    return manager.get_diagnostics()


@router.post("/checkpoint")
def checkpoint_wal_journal(
    mode: str = Query(default="PASSIVE", regex="^(PASSIVE|FULL|RESTART|TRUNCATE)$"),
    manager: StorageManager = Depends(get_storage_manager),
) -> Dict[str, Any]:
    return manager.checkpoint_wal(mode=mode)


@router.post("/prune", response_model=RetentionPruneResult)
def prune_telemetry(
    keep_latest: int = Query(default=5000, ge=100, le=100000),
    vacuum: bool = Query(default=True),
    manager: StorageManager = Depends(get_storage_manager),
) -> RetentionPruneResult:
    return manager.prune_old_readings(keep_latest=keep_latest, vacuum=vacuum)
