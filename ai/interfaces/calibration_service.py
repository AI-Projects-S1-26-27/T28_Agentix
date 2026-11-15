"""HealthSensing: Calibration Service.

This module provides the calibration preparation service for personalized baseline
anomaly detection. It wraps the four baseline calibration strategies:
- Strategy A: Full calibration partition
- Strategy B: Settled baseline (acclimation burn-in exclusion)
- Strategy C: Shared quality-gated suffix selection
- Strategy D: Signal-specific quality-gated selection

It returns a `BaselineContext` object that maintains the selected baseline internally
while exposing sanitized metadata and quality assessments.

Clinical Disclaimer:
--------------------
This module provides experimental baseline preparation for algorithmic screening.
It does NOT make medical risk determinations or clinical diagnoses.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ai.baseline.baseline_quality import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_MIN_SIGNAL_OBS,
    DEFAULT_WINDOW_SEC,
    SignalQualityAssessment,
    assess_calibration_dataset_quality,
    assess_signal_baseline_quality,
    select_quality_gated_calibration_window,
    select_settled_calibration_window,
    select_signal_specific_calibration_windows,
)
from ai.interfaces.ml_contract import (
    BaselineQualityResult,
    sanitize_for_json,
)

SUPPORTED_STRATEGIES: frozenset[str] = frozenset({
    "full",
    "settled",
    "quality_gated",
    "signal_specific",
})

STRATEGY_ALIASES: dict[str, str] = {
    "strategy_a": "full",
    "full_partition": "full",
    "a": "full",
    "strategy_b": "settled",
    "settled_baseline": "settled",
    "b": "settled",
    "strategy_c": "quality_gated",
    "shared_quality_gated": "quality_gated",
    "c": "quality_gated",
    "strategy_d": "signal_specific",
    "signal_specific_quality_gated": "signal_specific",
    "d": "signal_specific",
}


def _assessment_to_quality_result(assessment: SignalQualityAssessment | dict[str, Any]) -> BaselineQualityResult:
    """Convert SignalQualityAssessment instance or dictionary into BaselineQualityResult."""
    if isinstance(assessment, SignalQualityAssessment):
        return BaselineQualityResult(
            signal=assessment.signal,
            status=assessment.status,
            observation_count=assessment.observation_count,
            num_windows=assessment.num_windows,
            mean_drift=assessment.mean_drift,
            pct_mean_drift=assessment.pct_mean_drift,
            variance_inflation_factor=assessment.variance_inflation_factor,
            acclimation_detected=assessment.acclimation_detected,
            measurements=sanitize_for_json(assessment.measurements),
            explanation=assessment.explanation,
        )
    elif isinstance(assessment, dict):
        return BaselineQualityResult(
            signal=str(assessment.get("signal", "unknown")),
            status=str(assessment.get("status", "unknown")),
            observation_count=int(assessment.get("observation_count", 0)),
            num_windows=int(assessment.get("num_windows", 0)),
            mean_drift=sanitize_for_json(assessment.get("mean_drift")),
            pct_mean_drift=sanitize_for_json(assessment.get("pct_mean_drift")),
            variance_inflation_factor=sanitize_for_json(assessment.get("variance_inflation_factor")),
            acclimation_detected=bool(assessment.get("acclimation_detected", False)),
            measurements=sanitize_for_json(assessment.get("measurements", {})),
            explanation=str(assessment.get("explanation", "")),
        )
    else:
        raise TypeError(f"Expected SignalQualityAssessment or dict; received {type(assessment).__name__}.")


class BaselineContext:
    """Encapsulates prepared reference baseline data and quality metadata.

    The internal baseline DataFrame is maintained privately to prevent leakage
    of raw time-series arrays into public JSON serialization contracts.
    """

    def __init__(
        self,
        baseline_df: pd.DataFrame,
        strategy: str,
        feature_columns: Sequence[str],
        quality_assessments: dict[str, BaselineQualityResult],
        selection_metadata: dict[str, Any],
        insufficient_signals: Sequence[str] | None = None,
    ) -> None:
        self._baseline_df = baseline_df.copy()
        self.strategy = strategy
        self.feature_columns = list(feature_columns)
        self.quality_assessments = dict(quality_assessments)
        self.selection_metadata = sanitize_for_json(selection_metadata)
        self.insufficient_signals = list(insufficient_signals or [])

        # Compute summary statistics per signal for transparent monitoring without raw rows
        stats: dict[str, dict[str, Any]] = {}
        for col in self.feature_columns:
            if col in self._baseline_df.columns:
                series = pd.to_numeric(self._baseline_df[col], errors="coerce").dropna()
                n_valid = len(series)
                stats[col] = {
                    "valid_samples": int(n_valid),
                    "mean": float(series.mean()) if n_valid > 0 else None,
                    "std": float(series.std(ddof=1)) if n_valid >= 2 else (0.0 if n_valid == 1 else None),
                    "min": float(series.min()) if n_valid > 0 else None,
                    "max": float(series.max()) if n_valid > 0 else None,
                }
            else:
                stats[col] = {
                    "valid_samples": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                }
        self.baseline_statistics = stats

    def get_baseline_df(self) -> pd.DataFrame:
        """Access internal reference baseline dataframe for inference computations."""
        return self._baseline_df.copy()

    def to_dict(self) -> dict[str, Any]:
        """Serialize public baseline metadata without exposing raw sensor arrays."""
        return {
            "strategy": self.strategy,
            "feature_columns": self.feature_columns,
            "baseline_statistics": sanitize_for_json(self.baseline_statistics),
            "selection_metadata": sanitize_for_json(self.selection_metadata),
            "insufficient_signals": self.insufficient_signals,
        }


def prepare_baseline(
    calibration_readings: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    strategy: str = "signal_specific",
    window_size_sec: int = DEFAULT_WINDOW_SEC,
    min_observations: int = DEFAULT_MIN_SIGNAL_OBS,
    **kwargs: Any,
) -> BaselineContext:
    """Prepare and validate a reference baseline from calibration observations.

    Parameters
    ----------
    calibration_readings : pd.DataFrame
        Candidate calibration observations (e.g. resting baseline period).
    feature_columns : Sequence[str], default=DEFAULT_FEATURE_COLUMNS
        Explicit physiological columns to evaluate.
    strategy : str, default="signal_specific"
        Calibration window selection strategy:
        - "full": complete calibration partition without clipping.
        - "settled": acclimation burn-in exclusion.
        - "quality_gated": shared suffix minimizing joint variance inflation.
        - "signal_specific": independent per-signal quality assessment and window selection.
    window_size_sec : int, default=120
        Window duration in seconds for stability assessment.
    min_observations : int, default=480
        Minimum observations required for suffix selection in Strategies C and D.
    **kwargs : Any
        Additional keyword arguments forwarded to strategy selectors.

    Returns
    -------
    BaselineContext
        Context holding the validated baseline DataFrame and quality metadata.

    Raises
    ------
    TypeError
        If calibration_readings is not a pandas DataFrame.
    ValueError
        If calibration_readings is empty, if feature_columns are missing,
        if infinite values are present, or if strategy is unrecognized.
    """
    if not isinstance(calibration_readings, pd.DataFrame):
        raise TypeError(
            f"calibration_readings must be a pandas DataFrame; received {type(calibration_readings).__name__}."
        )

    if calibration_readings.empty:
        raise ValueError("calibration_readings DataFrame is empty.")

    if not feature_columns:
        raise ValueError("feature_columns must be explicitly specified and non-empty.")

    # Validate feature columns exist in calibration_readings
    missing_cols = [c for c in feature_columns if c not in calibration_readings.columns]
    if missing_cols:
        raise ValueError(
            f"Required feature column(s) missing from calibration_readings: {missing_cols}."
        )

    # Validate no infinite values
    for col in feature_columns:
        num_col = pd.to_numeric(calibration_readings[col], errors="coerce")
        if np.isinf(num_col.to_numpy(dtype=float)).any():
            raise ValueError(f"Calibration column '{col}' contains infinite (+inf or -inf) values.")

    # Normalize strategy name
    clean_strat = strategy.lower().strip()
    norm_strat = STRATEGY_ALIASES.get(clean_strat, clean_strat)
    if norm_strat not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"Unsupported calibration strategy '{strategy}'. "
            f"Supported strategies: {sorted(SUPPORTED_STRATEGIES)}."
        )

    # Initial quality assessment on calibration partition
    initial_assessments = assess_calibration_dataset_quality(
        calibration_df=calibration_readings,
        feature_columns=feature_columns,
        window_size_sec=window_size_sec,
    )

    # Identify any signals with fewer than 2 valid observations
    insufficient_signals: list[str] = []
    for col in feature_columns:
        n_valid = int(pd.to_numeric(calibration_readings[col], errors="coerce").dropna().shape[0])
        if n_valid < 2 or initial_assessments[col].status == "insufficient_data":
            insufficient_signals.append(col)

    # Execute the requested strategy
    if norm_strat == "full":
        selected_df = calibration_readings[list(feature_columns)].copy()
        selection_metadata: dict[str, Any] = {
            "strategy": "Strategy A (Full)",
            "accepted_observations": len(selected_df),
            "decision": "Full calibration partition utilized as reference baseline.",
        }
        quality_results = {
            col: _assessment_to_quality_result(initial_assessments[col])
            for col in feature_columns
        }

    elif norm_strat == "settled":
        min_retained = kwargs.get("min_retained_obs", 240)
        selected_df, raw_meta = select_settled_calibration_window(
            calibration_df=calibration_readings,
            feature_columns=feature_columns,
            window_size_sec=window_size_sec,
            min_retained_obs=min_retained,
        )
        selection_metadata = raw_meta
        # Re-assess selected settled dataframe
        post_assessments = assess_calibration_dataset_quality(
            calibration_df=selected_df,
            feature_columns=feature_columns,
            window_size_sec=window_size_sec,
        )
        quality_results = {
            col: _assessment_to_quality_result(post_assessments[col])
            for col in feature_columns
        }

    elif norm_strat == "quality_gated":
        min_stable = kwargs.get("min_stable_obs", min_observations)
        selected_df, raw_meta = select_quality_gated_calibration_window(
            calibration_df=calibration_readings,
            feature_columns=feature_columns,
            window_size_sec=window_size_sec,
            min_stable_obs=min_stable,
        )
        selection_metadata = raw_meta
        # Re-assess selected gated dataframe
        post_assessments = assess_calibration_dataset_quality(
            calibration_df=selected_df,
            feature_columns=feature_columns,
            window_size_sec=window_size_sec,
        )
        quality_results = {
            col: _assessment_to_quality_result(post_assessments[col])
            for col in feature_columns
        }

    elif norm_strat == "signal_specific":
        step_sec = kwargs.get("step_sec", 60)
        # Strategy D selector: select_signal_specific_calibration_windows
        # Inspect: if any signal has < 2 valid observations, select_signal_specific_calibration_windows
        # will raise ValueError on that signal. If so, handle gracefully.
        if insufficient_signals:
            # Fallback if any channel has < 2 observations: retain available data with explicit status
            padded_dict: dict[str, np.ndarray] = {}
            signal_meta: dict[str, Any] = {}
            max_len = 0
            for col in feature_columns:
                series = pd.to_numeric(calibration_readings[col], errors="coerce").dropna()
                arr = series.to_numpy(dtype=float)
                max_len = max(max_len, len(arr))
                status = "insufficient_data" if len(arr) < 2 else initial_assessments[col].status
                signal_meta[col] = {
                    "signal": col,
                    "status": status,
                    "selection_type": "insufficient_data_fallback",
                    "offset": 0,
                    "observation_count": len(arr),
                    "explanation": f"Signal '{col}' has {len(arr)} valid observations (< 2 required for suffix search).",
                    "assessment": initial_assessments[col].to_dict(),
                }
            for col in feature_columns:
                series = pd.to_numeric(calibration_readings[col], errors="coerce").dropna()
                arr = series.to_numpy(dtype=float)
                padded = np.full(max_len if max_len > 0 else 1, np.nan, dtype=float)
                padded[:len(arr)] = arr
                padded_dict[col] = padded
            selected_df = pd.DataFrame(padded_dict)
            selection_metadata = {
                "strategy": "Strategy D (Signal-Specific Quality-Gated)",
                "signals": signal_meta,
                "decision": "Insufficient valid observations in one or more signals; retained available calibration data.",
            }
            quality_results = {
                col: _assessment_to_quality_result(signal_meta[col]["assessment"])
                for col in feature_columns
            }
        else:
            selected_df, raw_meta = select_signal_specific_calibration_windows(
                calibration_df=calibration_readings,
                feature_columns=feature_columns,
                window_size_sec=window_size_sec,
                min_observations=min_observations,
                step_sec=step_sec,
            )
            selection_metadata = raw_meta
            # Extract the actual individual assessments from Strategy D selection metadata
            quality_results = {}
            for col in feature_columns:
                sig_meta = raw_meta.get("signals", {}).get(col, {})
                sub_assess = sig_meta.get("assessment")
                if sub_assess is not None:
                    q_res = _assessment_to_quality_result(sub_assess)
                    # CRITICAL: Preserve unstable status if fallback was selected!
                    # Do not silently treat unstable_fallback as stable.
                    if sig_meta.get("status") == "unstable":
                        q_res.status = "unstable"
                    quality_results[col] = q_res
                else:
                    quality_results[col] = _assessment_to_quality_result(initial_assessments[col])

    # Re-verify insufficient signals based on final selected baseline
    final_insufficient: list[str] = []
    for col in feature_columns:
        if col in selected_df.columns:
            valid_cnt = int(pd.to_numeric(selected_df[col], errors="coerce").dropna().shape[0])
            if valid_cnt < 2:
                final_insufficient.append(col)
                if col in quality_results:
                    quality_results[col].status = "insufficient_data"
        else:
            final_insufficient.append(col)

    return BaselineContext(
        baseline_df=selected_df,
        strategy=norm_strat,
        feature_columns=feature_columns,
        quality_assessments=quality_results,
        selection_metadata=selection_metadata,
        insufficient_signals=final_insufficient,
    )
