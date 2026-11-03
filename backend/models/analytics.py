from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SignalQuantiles(BaseModel):
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    median: float
    q25: float
    q75: float
    sample_count: int


class AutonomicStabilityMetrics(BaseModel):
    stability_index: float = Field(..., ge=0.0, le=100.0, description="Overall vital stability score from 0 (erratic) to 100 (steady)")
    heart_rate_variability_est: float = Field(..., description="Estimated root-mean-square successive difference (RMSSD) surrogate")
    eda_phasic_peaks_est: int = Field(..., ge=0, description="Estimated electrodermal skin conductance responses count")
    drift_velocity_hr: float = Field(..., description="Linear rate of heart rate change in bpm/minute")
    thermal_gradient: float = Field(..., description="Skin temperature change rate in °C/hour")
    homeostatic_status: str = Field(..., description="Classification: optimal, compensatory, high_strain, or uncalibrated")


class TelemetryAnalyticsSummary(BaseModel):
    window_minutes: int
    total_samples: int
    signals: Dict[str, SignalQuantiles]
    stability: AutonomicStabilityMetrics
    calculated_at: str
