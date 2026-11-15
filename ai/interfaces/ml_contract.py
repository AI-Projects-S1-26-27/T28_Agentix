"""HealthSensing: Standardized ML Contract and Typed Data Structures.

This module provides serializable, typed data contracts for the HealthSensing
machine learning pipeline. It defines standard schemas for:
- Per-signal anomaly detection results (SignalResult)
- Single-reading multi-signal results (ReadingResult)
- Signal baseline quality assessments (BaselineQualityResult)
- Supervised Random Forest window predictions (RFWindowResult)
- Top-level unified analysis responses (MLAnalysisResponse)

Design Principles:
------------------
1. Strict type definitions using dataclasses with JSON conversion helpers.
2. Safe serialization: converts numpy types to native Python, NaN/Inf to None (JSON null),
   and timestamps to ISO-8601 strings.
3. Complete separation: raw sensor data arrays and DataFrames are never serialized into metadata.
4. Independent interface: does not depend on FastAPI, SQLite, or frontend frameworks.

Clinical Disclaimer:
--------------------
This module provides experimental algorithmic screening results only.
It does not constitute medical diagnosis, clinical evaluation, or automated
medical risk determination.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import json
import math
from typing import Any

import numpy as np
import pandas as pd


def sanitize_for_json(val: Any) -> Any:
    """Recursively convert data structures into JSON-compliant native Python types.

    Conversions:
    - NaN and +/-inf -> None (serializes to JSON null)
    - np.floating -> float (or None if nan/inf)
    - np.integer -> int
    - np.bool_ -> bool
    - np.ndarray -> list of sanitized elements
    - pd.Timestamp, datetime, date -> ISO-8601 string
    - Dataclass instances -> to_dict() or sanitized asdict()
    - Dict -> dict with string keys and sanitized values
    - List, tuple, set -> list of sanitized values
    """
    if val is None:
        return None

    # Handle float / numpy float NaN and Infinity
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val) or np.isnan(val) or np.isinf(val):
            return None
        return float(val)

    # Handle integers
    if isinstance(val, (int, np.integer)):
        if isinstance(val, bool):  # bool is a subclass of int in Python
            return bool(val)
        return int(val)

    # Handle booleans
    if isinstance(val, (bool, np.bool_)):
        return bool(val)

    # Handle strings
    if isinstance(val, str):
        return val

    # Handle timestamps / dates
    if isinstance(val, (pd.Timestamp, datetime, date)):
        return val.isoformat()

    # Handle NumPy arrays
    if isinstance(val, np.ndarray):
        return [sanitize_for_json(x) for x in val.tolist()]

    # Handle dataclasses with to_dict method
    if hasattr(val, "to_dict") and callable(val.to_dict):
        return val.to_dict()

    # Handle standard dataclasses
    if hasattr(val, "__dataclass_fields__"):
        return sanitize_for_json(asdict(val))

    # Handle dict / Mapping
    if isinstance(val, (dict, Mapping)):
        return {str(k): sanitize_for_json(v) for k, v in val.items()}

    # Handle list / tuple / set / Sequence
    if isinstance(val, (list, tuple, set)):
        return [sanitize_for_json(x) for x in val]

    # Fallback to string representation for unfamiliar types
    return str(val)


@dataclass
class SignalResult:
    """Evaluation result for a single physiological signal channel in one reading.

    Attributes
    ----------
    value : float | None
        Current sensor observation value. None if missing/unrecorded.
    baseline_mean : float | None
        Historical reference baseline mean. None if baseline is unavailable.
    baseline_std : float | None
        Historical reference baseline sample standard deviation (ddof=1).
    difference : float | None
        Deviation from baseline mean (value - baseline_mean).
    z_score : float | None
        Standardized deviation score (difference / baseline_std). None if baseline_std == 0
        or value is missing.
    threshold : float
        Z-score threshold configured for anomaly flagging (e.g. 3.0).
    anomaly : int
        Binary flag indicating whether |z_score| > threshold (1 if anomalous, 0 otherwise).
    evaluation_status : str
        Detailed channel status: 'evaluated', 'missing_reading', 'zero_variance_baseline',
        or 'insufficient_baseline_data'.
    """

    value: float | None
    baseline_mean: float | None
    baseline_std: float | None
    difference: float | None
    z_score: float | None
    threshold: float
    anomaly: int
    evaluation_status: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to sanitized dictionary."""
        return {
            "value": sanitize_for_json(self.value),
            "baseline_mean": sanitize_for_json(self.baseline_mean),
            "baseline_std": sanitize_for_json(self.baseline_std),
            "difference": sanitize_for_json(self.difference),
            "z_score": sanitize_for_json(self.z_score),
            "threshold": float(self.threshold),
            "anomaly": int(self.anomaly),
            "evaluation_status": str(self.evaluation_status),
        }


@dataclass
class ReadingResult:
    """Multi-signal evaluation result for a single temporal observation.

    Attributes
    ----------
    timestamp : str | None
        Observation timestamp formatted as ISO-8601 string.
    signals : dict[str, SignalResult]
        Per-signal evaluation results keyed by signal column name.
    anomaly_count : int
        Total number of evaluated signals exceeding their respective anomaly thresholds.
    overall_anomaly : int
        1 if anomaly_count > 0, else 0.
    unevaluable_count : int
        Total number of configured signals that could not be evaluated (e.g. missing or zero variance).
    overall_status : str
        Categorical row status: 'normal', 'anomaly_detected', 'partially_evaluated_anomaly',
        or 'unevaluable'.
    evaluation_completeness : str
        Completeness of channel evaluations: 'complete' (all channels evaluated),
        'partial' (some evaluated, some unevaluable), or 'none' (zero channels evaluable).
    """

    timestamp: str | None
    signals: dict[str, SignalResult]
    anomaly_count: int
    overall_anomaly: int
    unevaluable_count: int
    overall_status: str
    evaluation_completeness: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to sanitized dictionary."""
        return {
            "timestamp": sanitize_for_json(self.timestamp),
            "signals": {k: v.to_dict() for k, v in self.signals.items()},
            "anomaly_count": int(self.anomaly_count),
            "overall_anomaly": int(self.overall_anomaly),
            "unevaluable_count": int(self.unevaluable_count),
            "overall_status": str(self.overall_status),
            "evaluation_completeness": str(self.evaluation_completeness),
        }


@dataclass
class BaselineQualityResult:
    """Quality diagnostic assessment for a physiological calibration signal.

    Attributes
    ----------
    signal : str
        Physiological signal identifier (e.g. 'eda_microsiemens').
    status : str
        Stationarity categorization: 'stable', 'unstable', or 'insufficient_data'.
    observation_count : int
        Total number of valid calibration observations analyzed.
    num_windows : int
        Number of consecutive non-overlapping windows evaluated.
    mean_drift : float | None
        Absolute difference between initial and final window means.
    pct_mean_drift : float | None
        Percentage drift relative to initial window mean.
    variance_inflation_factor : float | None
        Ratio of pooled variance to mean within-window variance.
    acclimation_detected : bool
        Whether a sustained acclimation trend (e.g. monotonic decrease) was detected.
    measurements : dict[str, Any]
        Detailed statistical measurements supporting the assessment.
    explanation : str
        Human-readable clinical and algorithmic reasoning explaining the status.
    """

    signal: str
    status: str
    observation_count: int
    num_windows: int
    mean_drift: float | None
    pct_mean_drift: float | None
    variance_inflation_factor: float | None
    acclimation_detected: bool
    measurements: dict[str, Any]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to sanitized dictionary."""
        return {
            "signal": str(self.signal),
            "status": str(self.status),
            "observation_count": int(self.observation_count),
            "num_windows": int(self.num_windows),
            "mean_drift": sanitize_for_json(self.mean_drift),
            "pct_mean_drift": sanitize_for_json(self.pct_mean_drift),
            "variance_inflation_factor": sanitize_for_json(self.variance_inflation_factor),
            "acclimation_detected": bool(self.acclimation_detected),
            "measurements": sanitize_for_json(self.measurements),
            "explanation": str(self.explanation),
        }


@dataclass
class RFWindowResult:
    """Supervised Random Forest classification result for a 60-second temporal window.

    Attributes
    ----------
    window_start : str | None
        ISO-8601 timestamp of first observation in the window.
    window_end : str | None
        ISO-8601 timestamp of final observation in the window.
    sample_count : int
        Number of observations contained in the window.
    status : str
        Window processing status: 'classified', 'insufficient_samples', 'model_unavailable',
        or 'timestamp_gap'.
    predicted_class : str | None
        Predicted stress label name: 'baseline' (class 0) or 'stress' (class 1). None if unclassified.
    predicted_class_id : int | None
        Predicted binary integer class: 0 or 1. None if unclassified.
    probabilities : dict[str, float] | None
        Predicted class probability distribution:
        - 'baseline_class_0': P(class = 0)
        - 'stress_class_1': P(class = 1)
        None if unclassified or model unavailable.
    """

    window_start: str | None
    window_end: str | None
    sample_count: int
    status: str
    predicted_class: str | None
    predicted_class_id: int | None
    probabilities: dict[str, float] | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to sanitized dictionary."""
        return {
            "window_start": sanitize_for_json(self.window_start),
            "window_end": sanitize_for_json(self.window_end),
            "sample_count": int(self.sample_count),
            "status": str(self.status),
            "predicted_class": self.predicted_class,
            "predicted_class_id": sanitize_for_json(self.predicted_class_id),
            "probabilities": sanitize_for_json(self.probabilities),
        }


@dataclass
class MLAnalysisResponse:
    """Unified top-level analysis response encapsulating baseline diagnostics and inference outputs.

    Attributes
    ----------
    schema_version : str
        Contract schema version string (e.g. '1.0.0').
    request_id : str | None
        Optional caller-provided tracking identifier.
    generated_at : str
        ISO-8601 UTC timestamp of response creation.
    baseline_metadata : dict[str, Any]
        Metadata describing calibration strategy, parameters, and baseline statistics.
    baseline_quality : dict[str, BaselineQualityResult]
        Per-signal quality assessments for reference baseline data.
    reading_results : list[ReadingResult]
        Per-reading personalized z-score anomaly screening results.
    rf_window_results : list[RFWindowResult]
        Sliding-window supervised Random Forest predictions.
    model_metadata : dict[str, Any]
        Supervised model specification, training protocol, and governance notes.
    """

    schema_version: str = "1.0.0"
    request_id: str | None = None
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    baseline_metadata: dict[str, Any] = field(default_factory=dict)
    baseline_quality: dict[str, BaselineQualityResult] = field(default_factory=dict)
    reading_results: list[ReadingResult] = field(default_factory=list)
    rf_window_results: list[RFWindowResult] = field(default_factory=list)
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize response to sanitized dictionary structure."""
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "generated_at": self.generated_at,
            "baseline_metadata": sanitize_for_json(self.baseline_metadata),
            "baseline_quality": {
                k: v.to_dict() for k, v in self.baseline_quality.items()
            },
            "reading_results": [r.to_dict() for r in self.reading_results],
            "rf_window_results": [w.to_dict() for w in self.rf_window_results],
            "model_metadata": sanitize_for_json(self.model_metadata),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize response to formatted, standard-compliant JSON string.

        Guarantees that no raw NumPy arrays or NaNs leak into the serialized text.
        """
        return json.dumps(self.to_dict(), indent=indent)
