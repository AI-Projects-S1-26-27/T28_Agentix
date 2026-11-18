import os
import sqlite3
from typing import Any, Dict, Optional
from pydantic import BaseModel


class StorageDiagnostics(BaseModel):
    database_file: str
    database_size_bytes: int
    database_size_kb: float
    page_size: int
    total_pages: int
    freelist_pages: int
    wal_journal_mode: str
    table_counts: Dict[str, int]


class RetentionPruneResult(BaseModel):
    deleted_readings: int
    database_size_before_kb: float
    database_size_after_kb: float
    freed_kb: float
    vacuum_executed: bool


class StorageManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_diagnostics(self) -> StorageDiagnostics:
        size_bytes = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        table_counts = {}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA page_size;")
            page_size = cursor.fetchone()[0]

            cursor.execute("PRAGMA page_count;")
            total_pages = cursor.fetchone()[0]

            cursor.execute("PRAGMA freelist_count;")
            freelist_pages = cursor.fetchone()[0]

            cursor.execute("PRAGMA journal_mode;")
            journal_mode = cursor.fetchone()[0]

            for table in ["measurements", "risk_assessments", "alerts", "personal_baselines"]:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table};")
                    table_counts[table] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    table_counts[table] = 0

        return StorageDiagnostics(
            database_file=os.path.basename(self.db_path),
            database_size_bytes=size_bytes,
            database_size_kb=round(size_bytes / 1024.0, 2),
            page_size=page_size,
            total_pages=total_pages,
            freelist_pages=freelist_pages,
            wal_journal_mode=journal_mode,
            table_counts=table_counts,
        )

    def checkpoint_wal(self, mode: str = "PASSIVE") -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA wal_checkpoint({mode});")
            result = cursor.fetchone()
        return {
            "busy": result[0],
            "log": result[1],
            "checkpointed": result[2],
            "mode": mode,
        }

    def prune_old_readings(
        self, keep_latest: int = 5000, vacuum: bool = True
    ) -> RetentionPruneResult:
        size_before = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM measurements;")
            total_rows = cursor.fetchone()[0]

            deleted = 0
            if total_rows > keep_latest:
                rows_to_delete = total_rows - keep_latest
                cursor.execute(
                    "DELETE FROM measurements WHERE id IN (SELECT id FROM measurements ORDER BY id ASC LIMIT ?);",
                    (rows_to_delete,),
                )
                deleted = cursor.rowcount
                conn.commit()

            if vacuum and deleted > 0:
                conn.execute("VACUUM;")

        size_after = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        freed = max(0.0, (size_before - size_after) / 1024.0)

        return RetentionPruneResult(
            deleted_readings=deleted,
            database_size_before_kb=round(size_before / 1024.0, 2),
            database_size_after_kb=round(size_after / 1024.0, 2),
            freed_kb=round(freed, 2),
            vacuum_executed=vacuum and deleted > 0,
        )
