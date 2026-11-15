"""HealthSensing: Machine Learning Pipeline Interfaces.

This package provides standard, framework-agnostic interfaces for:
1. Typed data contracts and JSON serialization (`ml_contract`).
2. Personalized baseline calibration preparation (`calibration_service`).
3. Multi-signal personalized anomaly detection and RF classification (`inference_service`).
"""

from ai.interfaces.calibration_service import (
    BaselineContext,
    prepare_baseline,
)
from ai.interfaces.inference_service import (
    analyze_readings,
    analyze_rf_windows,
    get_model_metadata,
    run_ml_analysis,
)
from ai.interfaces.ml_contract import (
    BaselineQualityResult,
    MLAnalysisResponse,
    ReadingResult,
    RFWindowResult,
    SignalResult,
    sanitize_for_json,
)

__all__ = [
    "BaselineContext",
    "BaselineQualityResult",
    "MLAnalysisResponse",
    "ReadingResult",
    "RFWindowResult",
    "SignalResult",
    "analyze_readings",
    "analyze_rf_windows",
    "get_model_metadata",
    "prepare_baseline",
    "run_ml_analysis",
    "sanitize_for_json",
]
