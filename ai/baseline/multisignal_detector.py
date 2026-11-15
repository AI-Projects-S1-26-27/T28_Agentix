"""HealthSensing: Multi-signal personalized anomaly detection module.

This module provides multi-signal anomaly screening by comparing sensor readings
against an individual's personal reference baseline across multiple physiological signals.

Notes
-----
This is an experimental anomaly screening tool, not a medical diagnostic system.
It must not be used for clinical diagnosis or treatment decisions.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def detect_multisignal_anomalies(
    readings: pd.DataFrame,
    baseline_readings: pd.DataFrame,
    feature_columns: Sequence[str],
    thresholds: Mapping[str, float] | float | None = None,
) -> pd.DataFrame:
    """Flag multi-signal deviations from an individual's reference baseline.

    Calculates baseline mean and sample standard deviation per signal, computes
    differences and z-scores for current readings, and flags readings exceeding
    configured z-score thresholds.

    Parameters
    ----------
    readings : pd.DataFrame
        Current sensor readings to evaluate.
    baseline_readings : pd.DataFrame
        Historical reference readings representing the individual's baseline.
    feature_columns : Sequence[str]
        Explicit list of feature column names to analyze. Numeric columns such
        as stress_label, participant_id, or session_id must not be included.
    thresholds : Mapping[str, float] | float | None, default=None
        Threshold(s) for anomaly detection based on absolute z-scores. Can be:
        - None: defaults to 3.0 for all signals.
        - float/int: applied as threshold to all signals.
        - Mapping (dict): per-signal thresholds; signals not in the mapping default to 3.0.

    Returns
    -------
    pd.DataFrame
        Copy of `readings` with additional columns for each analyzed signal:
        - `{signal}_baseline_mean`: mean of valid baseline readings.
        - `{signal}_baseline_std`: sample standard deviation (ddof=1) of baseline readings.
        - `{signal}_baseline_difference`: current reading minus baseline mean.
        - `{signal}_baseline_z_score`: deviation in standard deviations from baseline mean
          (NaN if baseline standard deviation is zero or current value is missing).
        - `{signal}_detected_anomaly`: 1 if absolute z-score exceeds threshold, else 0.
        - `{signal}_evaluation_status`: evaluation status ('evaluated', 'missing_reading',
          or 'zero_variance_baseline').
        Plus overall summary columns:
        - `anomaly_count`: total number of anomalous signals in the current reading.
        - `overall_anomaly`: 1 if `anomaly_count` > 0, else 0.
        - `unevaluable_count`: number of selected signals that could not be evaluated.
        - `overall_status`: row status ('unevaluable', 'anomaly_detected', 'normal',
          or 'partially_evaluated_anomaly').

    Raises
    ------
    TypeError
        If `readings` or `baseline_readings` is not a pandas DataFrame, or if
        `feature_columns` is not a sequence of strings.
    ValueError
        If DataFrames are empty, if `feature_columns` is empty or missing from either
        DataFrame, if any selected column contains infinite values, if any baseline
        signal contains fewer than two valid observations, or if any threshold
        is non-positive or non-finite.

    Notes
    -----
    - Zero-variance baseline signals: If all baseline readings for a signal are identical
      (sample standard deviation is 0), z-scores are mathematically undefined. To avoid
      silent division by zero or runtime warnings, the baseline standard deviation is set
      to 0.0, the z-score is set to NaN, and the detected anomaly flag is set to 0.
    - Missing data: Handled independently per signal. A missing value (NaN) in one signal
      results in NaN difference and z-score and a 0 anomaly flag for that signal, without
      preventing valid analysis of other signals.
    - This module is intended for experimental anomaly screening only, not medical diagnosis.
    """
    if not isinstance(readings, pd.DataFrame):
        raise TypeError("readings must be a pandas DataFrame.")

    if not isinstance(baseline_readings, pd.DataFrame):
        raise TypeError("baseline_readings must be a pandas DataFrame.")

    if readings.empty:
        raise ValueError("readings DataFrame is empty.")

    if baseline_readings.empty:
        raise ValueError("baseline_readings DataFrame is empty.")

    if feature_columns is None:
        raise ValueError("feature_columns must be explicitly specified.")

    if not isinstance(feature_columns, Sequence) or isinstance(
        feature_columns, (str, bytes)
    ):
        raise TypeError("feature_columns must be a sequence of column names.")

    if len(feature_columns) == 0:
        raise ValueError("feature_columns must not be empty.")

    feature_cols: list[str] = []
    for col in feature_columns:
        if not isinstance(col, str):
            raise TypeError(
                f"All feature column names must be strings, got {type(col).__name__}."
            )
        if col not in feature_cols:
            feature_cols.append(col)

    missing_readings = [
        col for col in feature_cols if col not in readings.columns
    ]
    missing_baseline = [
        col for col in feature_cols if col not in baseline_readings.columns
    ]
    if missing_readings or missing_baseline:
        err_parts = []
        if missing_readings:
            err_parts.append(f"readings missing: {missing_readings}")
        if missing_baseline:
            err_parts.append(f"baseline_readings missing: {missing_baseline}")
        raise ValueError(
            f"Selected feature columns missing: {'; '.join(err_parts)}."
        )

    default_threshold = 3.0
    resolved_thresholds: dict[str, float] = {}

    if thresholds is None:
        resolved_thresholds = {col: default_threshold for col in feature_cols}
    elif isinstance(thresholds, (int, float)) and not isinstance(
        thresholds, bool
    ):
        if not np.isfinite(thresholds) or thresholds <= 0:
            raise ValueError("Threshold must be a positive finite number.")
        resolved_thresholds = {col: float(thresholds) for col in feature_cols}
    elif isinstance(thresholds, Mapping):
        for k, v in thresholds.items():
            if k not in feature_cols:
                raise ValueError(
                    f"Thresholds mapping contains unknown signal '{k}'."
                )
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not np.isfinite(v)
                or v <= 0
            ):
                raise ValueError(
                    f"Threshold for signal '{k}' must be a positive finite number, got {v}."
                )
        resolved_thresholds = {
            col: float(thresholds.get(col, default_threshold))
            for col in feature_cols
        }
    else:
        raise ValueError(
            "thresholds must be None, a positive finite number, or a mapping of signal names to positive finite numbers."
        )

    for col in feature_cols:
        try:
            readings_num = pd.to_numeric(readings[col], errors="raise")
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Column '{col}' in readings must contain numeric data."
            ) from err

        try:
            baseline_num = pd.to_numeric(
                baseline_readings[col], errors="raise"
            )
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Column '{col}' in baseline_readings must contain numeric data."
            ) from err

        readings_vals = readings_num.to_numpy(dtype=float)
        baseline_vals = baseline_num.to_numpy(dtype=float)

        if np.isinf(readings_vals).any():
            raise ValueError(
                f"Readings column '{col}' contains infinite values."
            )

        if np.isinf(baseline_vals).any():
            raise ValueError(
                f"Baseline column '{col}' contains infinite values."
            )

        valid_baseline = baseline_num.dropna()
        if len(valid_baseline) < 2:
            raise ValueError(
                f"Baseline column '{col}' must contain at least two valid observations; "
                f"found {len(valid_baseline)}."
            )

    result = readings.copy()

    for col in feature_cols:
        current_series = pd.to_numeric(readings[col], errors="coerce")
        baseline_series = pd.to_numeric(baseline_readings[col], errors="coerce")
        valid_baseline = baseline_series.dropna()

        baseline_mean = float(valid_baseline.mean())
        baseline_std = float(valid_baseline.std(ddof=1))
        thresh = resolved_thresholds[col]

        result[f"{col}_baseline_mean"] = baseline_mean
        result[f"{col}_baseline_std"] = baseline_std

        diff = current_series - baseline_mean
        result[f"{col}_baseline_difference"] = diff

        if baseline_std == 0.0 or np.isclose(baseline_std, 0.0):
            result[f"{col}_baseline_z_score"] = np.nan
            result[f"{col}_detected_anomaly"] = 0
            eval_status = pd.Series("zero_variance_baseline", index=result.index)
            eval_status[current_series.isna()] = "missing_reading"
            result[f"{col}_evaluation_status"] = eval_status
        else:
            z_score = diff / baseline_std
            result[f"{col}_baseline_z_score"] = z_score
            result[f"{col}_detected_anomaly"] = (
                z_score.abs() > thresh
            ).astype(int)
            eval_status = pd.Series("evaluated", index=result.index)
            eval_status[current_series.isna()] = "missing_reading"
            result[f"{col}_evaluation_status"] = eval_status

    anomaly_cols = [f"{col}_detected_anomaly" for col in feature_cols]
    result["anomaly_count"] = result[anomaly_cols].sum(axis=1).astype(int)
    result["overall_anomaly"] = (result["anomaly_count"] > 0).astype(int)

    unevaluable_masks = [
        (result[f"{col}_evaluation_status"] != "evaluated").astype(int)
        for col in feature_cols
    ]
    result["unevaluable_count"] = sum(unevaluable_masks).astype(int)

    total_signals = len(feature_cols)
    status_conditions = [
        result["unevaluable_count"] == total_signals,
        (result["anomaly_count"] > 0) & (result["unevaluable_count"] > 0),
        (result["anomaly_count"] > 0) & (result["unevaluable_count"] == 0),
    ]
    status_choices = [
        "unevaluable",
        "partially_evaluated_anomaly",
        "anomaly_detected",
    ]
    result["overall_status"] = np.select(
        status_conditions, status_choices, default="normal"
    )

    return result
