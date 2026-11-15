"""HealthSensing: Multi-Signal and Random Forest Inference Service.

This module provides the core inference service for the HealthSensing pipeline.
It coordinates:
1. Multi-signal personalized z-score anomaly screening per reading (`analyze_readings`).
2. Supervised Random Forest sliding-window stress classification (`analyze_rf_windows`).
3. Unified analysis response generation (`run_ml_analysis`).

Design Principles:
------------------
- Reuses existing tested algorithms: `detect_multisignal_anomalies` and `extract_window_features`.
- Guarantees anti-leakage: raw calibration/sensor arrays are never serialized into responses.
- Respects missing values: missing readings and zero-variance baselines are never classified as normal.
- Strict 1 Hz continuity: windows are never constructed across timestamp gaps.
- RF model is optional and decoupled: missing model results in clean `model_unavailable` status.

Clinical Disclaimer:
--------------------
This service provides experimental anomaly screening and classification metrics.
Outputs do NOT constitute clinical assessment or automated medical risk decisions.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import math
from typing import Any

import numpy as np
import pandas as pd

from ai.baseline.baseline_quality import DEFAULT_FEATURE_COLUMNS
from ai.baseline.multisignal_detector import detect_multisignal_anomalies
from ai.features.feature_engineering import extract_window_features
from ai.interfaces.calibration_service import BaselineContext
from ai.interfaces.ml_contract import (
    MLAnalysisResponse,
    ReadingResult,
    RFWindowResult,
    SignalResult,
    sanitize_for_json,
)
from ai.models.random_forest_model import (
    DEFAULT_PHYSIOLOGICAL_COLUMNS,
    StressRandomForestClassifier,
)
from ai.preprocessing.windowing import create_windows

DEFAULT_Z_SCORE_THRESHOLD: float = 3.0
DEFAULT_WINDOW_SIZE: int = 60
DEFAULT_STRIDE: int = 30
DEFAULT_EXPECTED_SAMPLING_INTERVAL_SEC: float = 1.0
DEFAULT_TIMESTAMP_TOLERANCE_SEC: float = 0.1


def analyze_readings(
    readings: pd.DataFrame,
    baseline_context: BaselineContext,
    feature_columns: Sequence[str] | None = None,
    thresholds: Mapping[str, float] | float | None = None,
    timestamp_column: str = "timestamp",
) -> list[ReadingResult]:
    """Perform personalized multi-signal z-score anomaly screening on current readings.

    Parameters
    ----------
    readings : pd.DataFrame
        Current sensor observations to evaluate.
    baseline_context : BaselineContext
        Validated reference baseline context from calibration service.
    feature_columns : Sequence[str] | None, default=None
        Explicit physiological columns to analyze. Defaults to baseline_context.feature_columns.
    thresholds : Mapping[str, float] | float | None, default=None
        Z-score thresholds per signal or unified threshold. Defaults to 3.0.
    timestamp_column : str, default="timestamp"
        Column name for observation timestamps.

    Returns
    -------
    list[ReadingResult]
        Structured evaluation results for every observation row.

    Raises
    ------
    TypeError
        If readings is not a pandas DataFrame or baseline_context is invalid.
    ValueError
        If readings is empty or configured feature columns are missing or infinite.
    """
    if not isinstance(readings, pd.DataFrame):
        raise TypeError(f"readings must be a pandas DataFrame; received {type(readings).__name__}.")

    if not isinstance(baseline_context, BaselineContext):
        raise TypeError(f"baseline_context must be an instance of BaselineContext; received {type(baseline_context).__name__}.")

    if readings.empty:
        raise ValueError("readings DataFrame is empty.")

    target_columns = list(feature_columns) if feature_columns is not None else list(baseline_context.feature_columns)
    if not target_columns:
        raise ValueError("feature_columns must be specified or present in baseline_context.")

    # Validate target columns exist in readings
    missing_cols = [c for c in target_columns if c not in readings.columns]
    if missing_cols:
        raise ValueError(f"Required feature column(s) missing from readings: {missing_cols}.")

    # Validate no infinite values in target columns
    for col in target_columns:
        num_col = pd.to_numeric(readings[col], errors="coerce")
        if np.isinf(num_col.to_numpy(dtype=float)).any():
            raise ValueError(f"Readings column '{col}' contains infinite (+inf or -inf) values.")

    # Resolve thresholds
    resolved_thresholds: dict[str, float] = {}
    if thresholds is None:
        resolved_thresholds = {col: DEFAULT_Z_SCORE_THRESHOLD for col in target_columns}
    elif isinstance(thresholds, (int, float)):
        resolved_thresholds = {col: float(thresholds) for col in target_columns}
    elif isinstance(thresholds, Mapping):
        resolved_thresholds = {
            col: float(thresholds.get(col, DEFAULT_Z_SCORE_THRESHOLD))
            for col in target_columns
        }
    else:
        raise TypeError("thresholds must be a float, mapping of floats, or None.")

    baseline_df = baseline_context.get_baseline_df()
    total_signals = len(target_columns)

    # Separate channels with sufficient baseline from those with insufficient baseline
    evaluable_cols: list[str] = []
    insufficient_cols: list[str] = []

    for col in target_columns:
        if col in baseline_context.insufficient_signals:
            insufficient_cols.append(col)
        elif col in baseline_df.columns:
            valid_cnt = len(pd.to_numeric(baseline_df[col], errors="coerce").dropna())
            if valid_cnt >= 2:
                evaluable_cols.append(col)
            else:
                insufficient_cols.append(col)
        else:
            insufficient_cols.append(col)

    # Call detect_multisignal_anomalies on evaluable channels
    if evaluable_cols:
        evaluable_thresholds = {col: resolved_thresholds[col] for col in evaluable_cols}
        detector_output = detect_multisignal_anomalies(
            readings=readings,
            baseline_readings=baseline_df,
            feature_columns=evaluable_cols,
            thresholds=evaluable_thresholds,
        )
    else:
        detector_output = readings.copy()
        detector_output["anomaly_count"] = 0
        detector_output["overall_anomaly"] = 0
        detector_output["unevaluable_count"] = total_signals
        detector_output["overall_status"] = "unevaluable"

    # Populate insufficient channels manually with explicit status
    for col in insufficient_cols:
        detector_output[f"{col}_baseline_mean"] = np.nan
        detector_output[f"{col}_baseline_std"] = np.nan
        detector_output[f"{col}_baseline_difference"] = np.nan
        detector_output[f"{col}_baseline_z_score"] = np.nan
        detector_output[f"{col}_detected_anomaly"] = 0
        detector_output[f"{col}_evaluation_status"] = "insufficient_baseline_data"

    # Recalculate row-level summary metrics when insufficient channels are present
    if insufficient_cols:
        anomaly_cols = [f"{col}_detected_anomaly" for col in target_columns]
        detector_output["anomaly_count"] = detector_output[anomaly_cols].sum(axis=1).astype(int)
        detector_output["overall_anomaly"] = (detector_output["anomaly_count"] > 0).astype(int)

        uneval_masks = [
            (detector_output[f"{col}_evaluation_status"] != "evaluated").astype(int)
            for col in target_columns
        ]
        detector_output["unevaluable_count"] = sum(uneval_masks).astype(int)

        status_conditions = [
            detector_output["unevaluable_count"] == total_signals,
            (detector_output["anomaly_count"] > 0) & (detector_output["unevaluable_count"] > 0),
            (detector_output["anomaly_count"] > 0) & (detector_output["unevaluable_count"] == 0),
        ]
        status_choices = [
            "unevaluable",
            "partially_evaluated_anomaly",
            "anomaly_detected",
        ]
        detector_output["overall_status"] = np.select(
            status_conditions, status_choices, default="normal"
        )

    # Format timestamps
    has_timestamps = timestamp_column in detector_output.columns
    if has_timestamps:
        parsed_ts = pd.to_datetime(detector_output[timestamp_column], errors="coerce")
    else:
        parsed_ts = None

    reading_results: list[ReadingResult] = []

    for idx, row in detector_output.iterrows():
        # Timestamp
        ts_str: str | None = None
        if parsed_ts is not None:
            val_ts = parsed_ts.iloc[idx]
            ts_str = val_ts.isoformat() if pd.notna(val_ts) else None
        elif has_timestamps:
            ts_str = str(row[timestamp_column])

        # Per-signal results
        signals_dict: dict[str, SignalResult] = {}
        for col in target_columns:
            cur_val = row[col]
            b_mean = row[f"{col}_baseline_mean"]
            b_std = row[f"{col}_baseline_std"]
            diff = row[f"{col}_baseline_difference"]
            z = row[f"{col}_baseline_z_score"]
            anom = int(row[f"{col}_detected_anomaly"])
            eval_stat = str(row[f"{col}_evaluation_status"])

            signals_dict[col] = SignalResult(
                value=float(cur_val) if pd.notna(cur_val) else None,
                baseline_mean=float(b_mean) if pd.notna(b_mean) else None,
                baseline_std=float(b_std) if pd.notna(b_std) else None,
                difference=float(diff) if pd.notna(diff) else None,
                z_score=float(z) if pd.notna(z) else None,
                threshold=float(resolved_thresholds[col]),
                anomaly=anom,
                evaluation_status=eval_stat,
            )

        uneval = int(row["unevaluable_count"])
        anom_cnt = int(row["anomaly_count"])
        over_anom = int(row["overall_anomaly"])
        over_stat = str(row["overall_status"])

        # Evaluation completeness
        if uneval == 0:
            completeness = "complete"
        elif uneval < total_signals:
            completeness = "partial"
        else:
            completeness = "none"

        reading_results.append(
            ReadingResult(
                timestamp=ts_str,
                signals=signals_dict,
                anomaly_count=anom_cnt,
                overall_anomaly=over_anom,
                unevaluable_count=uneval,
                overall_status=over_stat,
                evaluation_completeness=completeness,
            )
        )

    return reading_results


def analyze_rf_windows(
    readings: pd.DataFrame,
    rf_model: StressRandomForestClassifier | None = None,
    feature_columns: Sequence[str] = DEFAULT_PHYSIOLOGICAL_COLUMNS,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    expected_sampling_interval_sec: float = DEFAULT_EXPECTED_SAMPLING_INTERVAL_SEC,
    timestamp_tolerance_sec: float = DEFAULT_TIMESTAMP_TOLERANCE_SEC,
    timestamp_column: str = "timestamp",
) -> list[RFWindowResult]:
    """Execute sliding-window supervised Random Forest inference.

    Divides readings into contiguous 60-observation windows (advancing by 30 observations,
    50% overlap) ensuring that windows NEVER cross a timestamp gap.

    Parameters
    ----------
    readings : pd.DataFrame
        Continuous sensor readings to classify.
    rf_model : StressRandomForestClassifier | None, default=None
        Fitted Random Forest classifier. If None, windows are processed with 'model_unavailable'.
    feature_columns : Sequence[str], default=DEFAULT_PHYSIOLOGICAL_COLUMNS
        Physiological channels to extract features from.
    window_size : int, default=60
        Window duration in observations (60s at 1 Hz).
    stride : int, default=30
        Stride between consecutive windows in observations (30s at 1 Hz).
    expected_sampling_interval_sec : float, default=1.0
        Expected sampling rate interval.
    timestamp_tolerance_sec : float, default=0.1
        Maximum allowed jitter before declaring a timestamp gap.
    timestamp_column : str, default="timestamp"
        Timestamp column name.

    Returns
    -------
    list[RFWindowResult]
        Classification results for each valid 60-second window.
    """
    if not isinstance(readings, pd.DataFrame):
        raise TypeError(f"readings must be a pandas DataFrame; received {type(readings).__name__}.")

    if readings.empty:
        return []

    target_cols = list(feature_columns)
    for col in target_cols:
        if col not in readings.columns:
            raise ValueError(f"Required physiological column '{col}' missing from readings.")

    # Validate timestamps and continuity
    has_ts = timestamp_column in readings.columns
    if has_ts:
        raw_ts = pd.to_datetime(readings[timestamp_column], errors="coerce")
        if raw_ts.isna().any():
            raise ValueError(f"Column '{timestamp_column}' contains invalid or missing timestamps.")
        sorted_df = readings.assign(_clean_ts=raw_ts).sort_values("_clean_ts").reset_index(drop=True)
    else:
        # Fallback to row index if timestamps omitted
        pseudo_ts = pd.date_range("2026-01-01", periods=len(readings), freq="1s", tz="UTC")
        sorted_df = readings.assign(_clean_ts=pseudo_ts).reset_index(drop=True)

    # Segment into contiguous blocks without timestamp gaps
    time_diff = sorted_df["_clean_ts"].diff().dt.total_seconds()
    time_gap = (time_diff - expected_sampling_interval_sec).abs() > timestamp_tolerance_sec
    time_gap = time_gap.fillna(False)
    sorted_df["_block_id"] = time_gap.cumsum()

    rf_results: list[RFWindowResult] = []
    total_valid_windows = 0

    # Expected 18 feature names in exact order
    expected_feature_order = [
        f"{col}_{stat}"
        for col in target_cols
        for stat in ("mean", "std", "min", "max", "range", "trend")
    ]

    for _, block_df in sorted_df.groupby("_block_id", sort=False):
        if len(block_df) < window_size:
            continue

        windows = create_windows(
            data=block_df,
            window_size=window_size,
            stride=stride,
            timestamp_column="_clean_ts",
        )

        for win in windows:
            # Double check window timestamp continuity
            win_dt = win["_clean_ts"].diff().dropna().dt.total_seconds()
            if ((win_dt - expected_sampling_interval_sec).abs() > timestamp_tolerance_sec).any():
                continue

            total_valid_windows += 1
            win_start = win["_clean_ts"].iloc[0].isoformat()
            win_end = win["_clean_ts"].iloc[-1].isoformat()

            if rf_model is None or not getattr(rf_model, "is_fitted_", False):
                rf_results.append(
                    RFWindowResult(
                        window_start=win_start,
                        window_end=win_end,
                        sample_count=len(win),
                        status="model_unavailable",
                        predicted_class=None,
                        predicted_class_id=None,
                        probabilities=None,
                    )
                )
            else:
                feats = extract_window_features(
                    window=win,
                    feature_columns=target_cols,
                    timestamp_column="_clean_ts",
                )
                X_win = pd.DataFrame([feats])[expected_feature_order]

                pred_class_id = int(rf_model.predict(X_win)[0])
                probs = rf_model.predict_proba(X_win)[0]
                pred_label = "baseline" if pred_class_id == 0 else "stress"

                rf_results.append(
                    RFWindowResult(
                        window_start=win_start,
                        window_end=win_end,
                        sample_count=len(win),
                        status="classified",
                        predicted_class=pred_label,
                        predicted_class_id=pred_class_id,
                        probabilities={
                            "baseline_class_0": round(float(probs[0]), 4),
                            "stress_class_1": round(float(probs[1]), 4),
                        },
                    )
                )

    if total_valid_windows == 0:
        # Report insufficient samples if no 60-observation contiguous window could be formed
        status = "insufficient_samples" if len(readings) < window_size else "timestamp_gap"
        return [
            RFWindowResult(
                window_start=None,
                window_end=None,
                sample_count=len(readings),
                status=status,
                predicted_class=None,
                predicted_class_id=None,
                probabilities=None,
            )
        ]

    return rf_results


def get_model_metadata(
    rf_model: StressRandomForestClassifier | None = None,
    feature_columns: Sequence[str] = DEFAULT_PHYSIOLOGICAL_COLUMNS,
) -> dict[str, Any]:
    """Compile governance metadata for the supervised Random Forest benchmark model.

    Parameters
    ----------
    rf_model : StressRandomForestClassifier | None, default=None
        Optional classifier instance.
    feature_columns : Sequence[str], default=DEFAULT_PHYSIOLOGICAL_COLUMNS
        Configured physiological features.

    Returns
    -------
    dict[str, Any]
        Sanitized metadata dictionary.
    """
    is_avail = bool(rf_model is not None and getattr(rf_model, "is_fitted_", False))
    expected_features = [
        f"{col}_{stat}"
        for col in feature_columns
        for stat in ("mean", "std", "min", "max", "range", "trend")
    ]
    return {
        "model_available": is_avail,
        "model_identifier": "rf_model_all_participants_v1" if is_avail else None,
        "model_type": "RandomForestClassifier (scikit-learn)",
        "window_size_observations": DEFAULT_WINDOW_SIZE,
        "stride_observations": DEFAULT_STRIDE,
        "sampling_rate_hz": DEFAULT_EXPECTED_SAMPLING_INTERVAL_SEC,
        "feature_schema": expected_features,
        "class_mapping": {"baseline_class_0": 0, "stress_class_1": 1},
        "evaluation_protocol": "Leave-One-Participant-Out (LOPO) Cross-Validation on WESAD (N=4)",
        "deployment_note": (
            "Final deployable model retrained on all 4 WESAD participants (228 windows). "
            "Distinguished from cross-validation fold models used strictly to estimate generalization metrics. "
            "Model outputs are experimental statistical classification probabilities and do NOT "
            "represent calibrated medical risk percentages. Not clinically validated for diagnosis."
        ),
    }


def run_ml_analysis(
    readings: pd.DataFrame,
    baseline_context: BaselineContext,
    rf_model: StressRandomForestClassifier | None = None,
    feature_columns: Sequence[str] | None = None,
    thresholds: Mapping[str, float] | float | None = None,
    request_id: str | None = None,
    timestamp_column: str = "timestamp",
) -> MLAnalysisResponse:
    """Execute end-to-end ML analysis producing standardized MLAnalysisResponse.

    Combines:
    1. Multi-signal personalized anomaly detection per observation.
    2. Supervised Random Forest sliding-window classification.
    3. Comprehensive baseline quality diagnostics and model metadata.

    Parameters
    ----------
    readings : pd.DataFrame
        Sensor observations to analyze.
    baseline_context : BaselineContext
        Reference baseline context.
    rf_model : StressRandomForestClassifier | None, default=None
        Optional Random Forest classifier.
    feature_columns : Sequence[str] | None, default=None
        Target physiological columns. Defaults to baseline_context.feature_columns.
    thresholds : Mapping[str, float] | float | None, default=None
        Per-signal or unified anomaly thresholds.
    request_id : str | None, default=None
        Optional caller-provided request tracking identifier.
    timestamp_column : str, default="timestamp"
        Timestamp column name.

    Returns
    -------
    MLAnalysisResponse
        Standardized, serializable analysis response.
    """
    target_cols = list(feature_columns) if feature_columns is not None else list(baseline_context.feature_columns)

    # 1. Multi-signal inference
    reading_results = analyze_readings(
        readings=readings,
        baseline_context=baseline_context,
        feature_columns=target_cols,
        thresholds=thresholds,
        timestamp_column=timestamp_column,
    )

    # 2. Random Forest window inference
    rf_window_results = analyze_rf_windows(
        readings=readings,
        rf_model=rf_model,
        feature_columns=target_cols,
        timestamp_column=timestamp_column,
    )

    # 3. Model metadata
    model_metadata = get_model_metadata(
        rf_model=rf_model,
        feature_columns=target_cols,
    )

    return MLAnalysisResponse(
        schema_version="1.0.0",
        request_id=request_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        baseline_metadata=baseline_context.to_dict(),
        baseline_quality=baseline_context.quality_assessments,
        reading_results=reading_results,
        rf_window_results=rf_window_results,
        model_metadata=model_metadata,
    )
