"""Database schema and migration definitions for HealthSensing backend."""

from backend.db.database import MeasurementRepository


def initialize_database(repository: MeasurementRepository) -> None:
    """Initialize SQLite database with all tables and indexes."""
    repository.initialize()
