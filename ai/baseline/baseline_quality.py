"""HealthSensing: Baseline Quality Assessment Module.

This module provides pre-flight baseline quality assessment for physiological
signals using calibration data only, prior to personalized anomaly detection.

It evaluates:
- Consecutive-window mean drift.
- Consecutive-window variance changes.
- Sustained directional trends (e.g., acclimation decay).
- Within-window variability vs. pooled variance (variance inflation).
- Missing and invalid readings.
- Sufficiency of calibration observations.

Signals are categorized into:
- `stable`
- `unstable`
- `insufficient_data`

Clinical Disclaimer:
--------------------
This module provides experimental physiological signal quality assessment,
not medical diagnosis or clinical verification.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = (
    "heart_rate_bpm",
    "eda_microsiemens",
    "body_temperature_c",
)
MIN_OBSERVATIONS_THRESHOLD: int = 60
DEFAULT_WINDOW_SEC: int = 120
MIN_WINDOW_SEC: int = 30
DRIFT_PCT_THRESHOLD: float = 20.0
INFLATION_THRESHOLD: float = 3.5


@dataclass
class SignalQualityAssessment:
    """Structured quality assessment result for a physiological signal."""

    signal: str
    status: str  # "stable", "unstable", "insufficient_data"
    observation_count: int
    num_windows: int
    mean_drift: float
    pct_mean_drift: float
    max_consecutive_decreases: int
    max_consecutive_increases: int
    variance_inflation_factor: float
    mean_within_window_variance: float
    pooled_variance: float
    acclimation_detected: bool
    measurements: dict[str, Any]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert assessment to dictionary."""
        return asdict(self)


def assess_signal_baseline_quality(
    calibration_readings: pd.Series | np.ndarray | Sequence[float],
    signal_name: str,
    window_size_sec: int = DEFAULT_WINDOW_SEC,
    min_window_size: int = MIN_WINDOW_SEC,
    drift_pct_threshold: float = DRIFT_PCT_THRESHOLD,
    inflation_threshold: float = INFLATION_THRESHOLD,
    min_observations: int = MIN_OBSERVATIONS_THRESHOLD,
) -> SignalQualityAssessment:
    """Assess calibration quality of a single physiological signal.

    Evaluates whether the signal exhibits stationarity, acclimation drift,
    or insufficient observations using only calibration data.

    Parameters
    ----------
    calibration_readings : pd.Series | np.ndarray | Sequence[float]
        Calibration baseline measurements.
    signal_name : str
        Name of the signal channel.
    window_size_sec : int, default=120
        Window duration in seconds.
    min_window_size : int, default=30
        Minimum observations for the final trailing window.
    drift_pct_threshold : float, default=20.0
        Percentage drift threshold for instability.
    inflation_threshold : float, default=3.5
        Variance inflation factor threshold.
    min_observations : int, default=60
        Minimum valid observations required.

    Returns
    -------
    SignalQualityAssessment
        Structured quality decision with measurements and explanation.
    """
    series = pd.to_numeric(pd.Series(calibration_readings), errors="coerce")
    valid = series.dropna()

    total_obs = len(valid)

    # 1. Check sufficiency of data
    if total_obs < 2:
        return SignalQualityAssessment(
            signal=signal_name,
            status="insufficient_data",
            observation_count=total_obs,
            num_windows=0,
            mean_drift=np.nan,
            pct_mean_drift=np.nan,
            max_consecutive_decreases=0,
            max_consecutive_increases=0,
            variance_inflation_factor=np.nan,
            mean_within_window_variance=np.nan,
            pooled_variance=np.nan,
            acclimation_detected=False,
            measurements={"valid_observations": total_obs, "reason": "fewer_than_2_valid_readings"},
            explanation=(
                f"Signal '{signal_name}' contains only {total_obs} valid observations; "
                f"at least 2 are required to compute baseline statistics."
            ),
        )

    pooled_var = float(valid.var(ddof=1))

    if total_obs < min_observations:
        return SignalQualityAssessment(
            signal=signal_name,
            status="insufficient_data",
            observation_count=total_obs,
            num_windows=0,
            mean_drift=np.nan,
            pct_mean_drift=np.nan,
            max_consecutive_decreases=0,
            max_consecutive_increases=0,
            variance_inflation_factor=np.nan,
            mean_within_window_variance=np.nan,
            pooled_variance=pooled_var,
            acclimation_detected=False,
            measurements={
                "valid_observations": total_obs,
                "required_minimum": min_observations,
                "pooled_variance": pooled_var,
            },
            explanation=(
                f"Signal '{signal_name}' contains {total_obs} valid observations, which is below "
                f"the recommended minimum of {min_observations} for stability analysis."
            ),
        )

    # 2. Divide calibration data into consecutive windows
    n_full_windows = total_obs // window_size_sec
    remainder = total_obs % window_size_sec
    total_windows = n_full_windows + (1 if remainder >= min_window_size else 0)

    if total_windows < 2:
        # Not enough windows for drift assessment, but sufficient for basic baseline
        return SignalQualityAssessment(
            signal=signal_name,
            status="stable",
            observation_count=total_obs,
            num_windows=total_windows,
            mean_drift=0.0,
            pct_mean_drift=0.0,
            max_consecutive_decreases=0,
            max_consecutive_increases=0,
            variance_inflation_factor=1.0,
            mean_within_window_variance=pooled_var,
            pooled_variance=pooled_var,
            acclimation_detected=False,
            measurements={
                "valid_observations": total_obs,
                "windows": total_windows,
                "pooled_variance": pooled_var,
            },
            explanation=(
                f"Signal '{signal_name}' has sufficient observations ({total_obs}) but fewer than 2 full "
                f"analysis windows. Baseline statistics are well-defined."
            ),
        )

    window_means: list[float] = []
    window_vars: list[float] = []

    for w in range(total_windows):
        start = w * window_size_sec
        end = min(start + window_size_sec, total_obs)
        w_vals = valid.iloc[start:end]
        window_means.append(float(w_vals.mean()))
        w_var = float(w_vals.var(ddof=1)) if len(w_vals) >= 2 else 0.0
        window_vars.append(w_var)

    # 3. Calculate window dynamics
    mean_within_var = float(np.mean(window_vars)) if window_vars else 0.0
    var_inflation = (
        float(pooled_var / mean_within_var) if mean_within_var > 0 else 1.0
    )

    deltas = np.diff(window_means)
    total_drift = float(window_means[-1] - window_means[0])
    first_mean = window_means[0]
    pct_drift = (
        float(total_drift / abs(first_mean) * 100) if first_mean != 0 else np.nan
    )

    # Count consecutive directional steps
    consec_decreases = 0
    max_consec_decreases = 0
    consec_increases = 0
    max_consec_increases = 0

    for d in deltas:
        if d < 0:
            consec_decreases += 1
            max_consec_decreases = max(max_consec_decreases, consec_decreases)
            consec_increases = 0
        elif d > 0:
            consec_increases += 1
            max_consec_increases = max(max_consec_increases, consec_increases)
            consec_decreases = 0
        else:
            consec_decreases = 0
            consec_increases = 0

    # 4. Assess acclimation and stability
    # Check physiological noise floor
    is_negligible_variance = (
        (signal_name == "eda_microsiemens" and pooled_var < 0.005)
        or (signal_name == "body_temperature_c" and pooled_var < 0.01)
        or (signal_name == "heart_rate_bpm" and pooled_var < 8.0)
    )

    total_steps = len(deltas)
    sustained_decrease = (
        max_consec_decreases >= max(3, total_steps - 1) and pct_drift < -drift_pct_threshold
    )
    severe_inflation = var_inflation > inflation_threshold

    reasons: list[str] = []
    if is_negligible_variance:
        acclimation_detected = False
    else:
        acclimation_detected = bool(sustained_decrease or (severe_inflation and pct_drift < -15.0))
        if acclimation_detected:
            reasons.append(
                f"sustained acclimation decay ({max_consec_decreases}/{total_steps} decreasing steps, "
                f"net drift: {pct_drift:.1f}%)"
            )
        if severe_inflation:
            reasons.append(
                f"high variance inflation (factor {var_inflation:.2f} > {inflation_threshold})"
            )
        if abs(pct_drift) > drift_pct_threshold * 2.0:
            reasons.append(f"excessive calibration drift ({pct_drift:.1f}%)")

    measurements = {
        "valid_observations": total_obs,
        "num_windows": total_windows,
        "window_means": [round(m, 4) for m in window_means],
        "window_variances": [round(v, 6) for v in window_vars],
        "total_drift": round(total_drift, 4),
        "pct_mean_drift": round(pct_drift, 2),
        "max_consecutive_decreases": max_consec_decreases,
        "max_consecutive_increases": max_consec_increases,
        "variance_inflation_factor": round(var_inflation, 2),
        "mean_within_window_variance": round(mean_within_var, 6),
        "pooled_variance": round(pooled_var, 6),
        "acclimation_detected": acclimation_detected,
        "is_negligible_variance": is_negligible_variance,
    }

    if reasons:
        status = "unstable"
        explanation = (
            f"Signal '{signal_name}' is unstable during calibration due to: {'; '.join(reasons)}. "
            f"Static baseline variance may be inflated."
        )
    elif is_negligible_variance:
        status = "stable"
        explanation = (
            f"Signal '{signal_name}' is stable during calibration. Total variance ({pooled_var:.6f}) "
            f"is within the physiological baseline noise floor."
        )
    else:
        status = "stable"
        explanation = (
            f"Signal '{signal_name}' is stable during calibration. Drift ({pct_drift:.1f}%) and "
            f"variance inflation factor ({var_inflation:.2f}) are within acceptable bounds."
        )

    return SignalQualityAssessment(
        signal=signal_name,
        status=status,
        observation_count=total_obs,
        num_windows=total_windows,
        mean_drift=round(total_drift, 4),
        pct_mean_drift=round(pct_drift, 2),
        max_consecutive_decreases=max_consec_decreases,
        max_consecutive_increases=max_consec_increases,
        variance_inflation_factor=round(var_inflation, 2),
        mean_within_window_variance=round(mean_within_var, 6),
        pooled_variance=round(pooled_var, 6),
        acclimation_detected=acclimation_detected,
        measurements=measurements,
        explanation=explanation,
    )


def assess_calibration_dataset_quality(
    calibration_df: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    window_size_sec: int = DEFAULT_WINDOW_SEC,
    min_window_size: int = MIN_WINDOW_SEC,
    drift_pct_threshold: float = DRIFT_PCT_THRESHOLD,
    inflation_threshold: float = INFLATION_THRESHOLD,
    min_observations: int = MIN_OBSERVATIONS_THRESHOLD,
) -> dict[str, SignalQualityAssessment]:
    """Assess calibration quality for all physiological signals in a dataframe.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        Calibration baseline observations.
    feature_columns : Sequence[str]
        Physiological channels to evaluate.
    window_size_sec : int
        Window duration in seconds.
    min_window_size : int
        Minimum observations for trailing window.
    drift_pct_threshold : float
        Percentage drift threshold for instability.
    inflation_threshold : float
        Variance inflation factor threshold.
    min_observations : int
        Minimum valid observations required.

    Returns
    -------
    dict[str, SignalQualityAssessment]
        Mapping of signal name to assessment object.
    """
    if calibration_df.empty:
        raise ValueError("calibration_df must not be empty.")

    results: dict[str, SignalQualityAssessment] = {}
    for col in feature_columns:
        if col not in calibration_df.columns:
            results[col] = SignalQualityAssessment(
                signal=col,
                status="insufficient_data",
                observation_count=0,
                num_windows=0,
                mean_drift=np.nan,
                pct_mean_drift=np.nan,
                max_consecutive_decreases=0,
                max_consecutive_increases=0,
                variance_inflation_factor=np.nan,
                mean_within_window_variance=np.nan,
                pooled_variance=np.nan,
                acclimation_detected=False,
                measurements={"missing_column": True},
                explanation=f"Signal column '{col}' is missing from calibration data.",
            )
        else:
            results[col] = assess_signal_baseline_quality(
                calibration_readings=calibration_df[col],
                signal_name=col,
                window_size_sec=window_size_sec,
                min_window_size=min_window_size,
                drift_pct_threshold=drift_pct_threshold,
                inflation_threshold=inflation_threshold,
                min_observations=min_observations,
            )

    return results


def select_settled_calibration_window(
    calibration_df: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    window_size_sec: int = DEFAULT_WINDOW_SEC,
    min_retained_obs: int = 240,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Strategy B: Settled baseline selection using calibration data only.

    Excludes initial calibration observations if calibration diagnostics detect
    acclimation in any physiological signal.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        Candidate calibration observations (first 70% of baseline).
    feature_columns : Sequence[str]
        Physiological channels to assess.
    window_size_sec : int
        Window size for acclimation detection.
    min_retained_obs : int
        Minimum calibration observations that must be retained.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        `(selected_calibration_df, decision_metadata)`.
    """
    assessments = assess_calibration_dataset_quality(
        calibration_df=calibration_df,
        feature_columns=feature_columns,
        window_size_sec=window_size_sec,
    )

    any_acclimating = any(a.acclimation_detected for a in assessments.values())
    n_total = len(calibration_df)

    if not any_acclimating or n_total <= min_retained_obs:
        return calibration_df.copy(), {
            "strategy": "Strategy B (Settled)",
            "acclimation_detected": any_acclimating,
            "observations_excluded": 0,
            "accepted_observations": n_total,
            "decision": "All signals stationary; retained full calibration partition.",
        }

    # Find earliest window where acclimating signals have stabilized
    # (i.e. window-to-window drop is no longer steep: drop < 15%)
    # At minimum, drop initial 240 seconds (first 2 windows)
    burn_in_obs = 240
    if n_total - burn_in_obs >= min_retained_obs:
        settled_df = calibration_df.iloc[burn_in_obs:].copy().reset_index(drop=True)
        return settled_df, {
            "strategy": "Strategy B (Settled)",
            "acclimation_detected": True,
            "observations_excluded": burn_in_obs,
            "accepted_observations": len(settled_df),
            "decision": f"Excluded initial {burn_in_obs}s acclimation period; retained {len(settled_df)}s settled baseline.",
        }

    return calibration_df.copy(), {
        "strategy": "Strategy B (Settled)",
        "acclimation_detected": True,
        "observations_excluded": 0,
        "accepted_observations": n_total,
        "decision": "Acclimation detected but retained full data to meet minimum observation threshold.",
    }


def select_quality_gated_calibration_window(
    calibration_df: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    window_size_sec: int = DEFAULT_WINDOW_SEC,
    min_stable_obs: int = 360,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Strategy C: Quality-gated baseline selection using calibration data only.

    Identifies an optimal, acceptable stable calibration window within calibration_df.
    If full calibration is stable, retains all data. If unstable (e.g. EDA decay),
    identifies the contiguous suffix of at least `min_stable_obs` that minimizes
    variance inflation across signals.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        Candidate calibration observations (first 70% of baseline).
    feature_columns : Sequence[str]
        Physiological channels to assess.
    window_size_sec : int
        Window size for quality evaluation.
    min_stable_obs : int
        Minimum observations in the gated window.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        `(selected_calibration_df, decision_metadata)`.
    """
    assessments = assess_calibration_dataset_quality(
        calibration_df=calibration_df,
        feature_columns=feature_columns,
        window_size_sec=window_size_sec,
    )

    n_total = len(calibration_df)
    unstable_signals = [sig for sig, a in assessments.items() if a.status == "unstable"]

    if not unstable_signals or n_total <= min_stable_obs:
        return calibration_df.copy(), {
            "strategy": "Strategy C (Quality-Gated)",
            "gating_status": "accepted_full",
            "accepted_observations": n_total,
            "decision": "Full calibration partition passed stability gating.",
        }

    # Evaluate candidate contiguous suffixes of length >= min_stable_obs
    best_start = 0
    best_score = float("inf")

    # Evaluate candidate start offsets at window boundaries
    candidate_offsets = list(range(0, n_total - min_stable_obs + 1, window_size_sec))
    if (n_total - min_stable_obs) not in candidate_offsets:
        candidate_offsets.append(n_total - min_stable_obs)

    for offset in candidate_offsets:
        candidate_sub = calibration_df.iloc[offset:]
        sub_assessments = assess_calibration_dataset_quality(
            calibration_df=candidate_sub,
            feature_columns=feature_columns,
            window_size_sec=window_size_sec,
            min_observations=min_stable_obs // 2,
        )

        # Composite instability penalty: sum of variance inflation and absolute drift %
        score = sum(
            (a.variance_inflation_factor if not np.isnan(a.variance_inflation_factor) else 10.0)
            + abs(a.pct_mean_drift if not np.isnan(a.pct_mean_drift) else 50.0) / 10.0
            for a in sub_assessments.values()
        )

        if score < best_score:
            best_score = score
            best_start = offset

    gated_df = calibration_df.iloc[best_start:].copy().reset_index(drop=True)
    return gated_df, {
        "strategy": "Strategy C (Quality-Gated)",
        "gating_status": "gated_suffix",
        "observations_excluded": best_start,
        "accepted_observations": len(gated_df),
        "decision": (
            f"Quality gating selected optimal stable window (start={best_start}s, "
            f"length={len(gated_df)}s) minimizing variance inflation across channels."
        ),
    }


DEFAULT_MIN_SIGNAL_OBS: int = 480  # Four 120-second windows


def calculate_single_signal_instability_score(
    assessment: SignalQualityAssessment,
) -> float:
    """Calculate single-signal instability penalty for ranking candidate calibration windows.

    Combines variance inflation factor and absolute percentage mean drift.
    Lower score indicates higher relative stability.

    Parameters
    ----------
    assessment : SignalQualityAssessment
        Quality assessment object for the candidate window.

    Returns
    -------
    float
        Instability score.
    """
    vif = (
        assessment.variance_inflation_factor
        if not np.isnan(assessment.variance_inflation_factor)
        else 10.0
    )
    pct_drift = (
        abs(assessment.pct_mean_drift)
        if not np.isnan(assessment.pct_mean_drift)
        else 50.0
    )
    return float(vif + (pct_drift / 10.0))


def select_single_signal_calibration_window(
    calibration_readings: pd.Series | np.ndarray | Sequence[float],
    signal_name: str,
    window_size_sec: int = DEFAULT_WINDOW_SEC,
    min_observations: int = DEFAULT_MIN_SIGNAL_OBS,
    step_sec: int = 60,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Strategy D: Select calibration window for a single physiological signal independently.

    1. Extract calibration observations in chronological order.
    2. Run existing single-signal quality assessment.
    3. If stable, retain the complete calibration partition.
    4. If unstable, evaluate contiguous chronological suffixes of length >= min_observations.
    5. Select the earliest suffix that passes the existing stability assessment.
    6. If no candidate passes, use fallback selecting the best available candidate using
       single-signal instability score, explicitly recording that the signal remains unstable.

    Parameters
    ----------
    calibration_readings : pd.Series | np.ndarray | Sequence[float]
        Chronological calibration readings for the signal channel.
    signal_name : str
        Name of the signal channel.
    window_size_sec : int, default=120
        Window duration in seconds.
    min_observations : int, default=480
        Minimum observations per candidate suffix (four 120-second windows).
    step_sec : int, default=60
        Step size in seconds for scanning candidate start offsets.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        `(selected_values_array, signal_metadata)`.
    """
    series = pd.to_numeric(pd.Series(calibration_readings), errors="coerce").dropna()
    n_total = len(series)

    if n_total < 2:
        raise ValueError(
            f"Signal '{signal_name}' contains fewer than 2 valid calibration observations ({n_total})."
        )

    # 1. Run initial assessment on full calibration partition
    init_assessment = assess_signal_baseline_quality(
        calibration_readings=series,
        signal_name=signal_name,
        window_size_sec=window_size_sec,
    )

    # 2. If stable or if total observations are below minimum suffix requirement, retain full partition
    if init_assessment.status == "stable" or n_total < min_observations:
        return series.to_numpy(dtype=float), {
            "signal": signal_name,
            "status": init_assessment.status,
            "selection_type": "full_partition",
            "offset": 0,
            "observation_count": n_total,
            "explanation": (
                f"Signal '{signal_name}' is stable across full calibration partition ({n_total} observations)."
                if init_assessment.status == "stable"
                else f"Signal '{signal_name}' retained full partition ({n_total} obs) to meet minimum observation requirements."
            ),
            "assessment": init_assessment.to_dict(),
        }

    # 3. Evaluate contiguous chronological suffixes
    candidate_offsets = sorted(
        set(list(range(0, n_total - min_observations + 1, step_sec)) + [n_total - min_observations])
    )

    best_score = float("inf")
    best_offset = 0
    best_series = series
    best_assessment = init_assessment

    for offset in candidate_offsets:
        if offset == 0:
            # Already evaluated as init_assessment (which was not stable)
            sub_series = series
            sub_assessment = init_assessment
        else:
            sub_series = series.iloc[offset:]
            sub_assessment = assess_signal_baseline_quality(
                calibration_readings=sub_series,
                signal_name=signal_name,
                window_size_sec=window_size_sec,
            )

        # Check for earliest suffix passing stability assessment
        if sub_assessment.status == "stable":
            return sub_series.to_numpy(dtype=float), {
                "signal": signal_name,
                "status": "stable",
                "selection_type": "earliest_stable_suffix",
                "offset": offset,
                "observation_count": len(sub_series),
                "explanation": (
                    f"Signal '{signal_name}' achieved stability at offset {offset}s "
                    f"(retaining {len(sub_series)} observations)."
                ),
                "assessment": sub_assessment.to_dict(),
            }

        # Track best available fallback
        score = calculate_single_signal_instability_score(sub_assessment)
        if score < best_score:
            best_score = score
            best_offset = offset
            best_series = sub_series
            best_assessment = sub_assessment

    # 4. Fallback: select best available candidate, explicitly recording that signal remains unstable
    return best_series.to_numpy(dtype=float), {
        "signal": signal_name,
        "status": "unstable",
        "selection_type": "unstable_fallback",
        "offset": best_offset,
        "observation_count": len(best_series),
        "explanation": (
            f"Signal '{signal_name}' remained unstable across all candidate suffixes >= {min_observations} obs. "
            f"Fallback selected best candidate at offset {best_offset}s (retaining {len(best_series)} obs, "
            f"instability score={best_score:.2f})."
        ),
        "assessment": best_assessment.to_dict(),
    }


def select_signal_specific_calibration_windows(
    calibration_df: pd.DataFrame,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    window_size_sec: int = DEFAULT_WINDOW_SEC,
    min_observations: int = DEFAULT_MIN_SIGNAL_OBS,
    step_sec: int = 60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Strategy D: Construct baseline DataFrame with independently selected signal windows.

    Aligns unequal signal lengths using NaN padding so that each column retains
    its exact valid observations without value duplication or fabrication.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        Candidate calibration observations.
    feature_columns : Sequence[str], default=DEFAULT_FEATURE_COLUMNS
        Physiological channels to assess independently.
    window_size_sec : int, default=120
        Window size for quality assessment.
    min_observations : int, default=480
        Minimum observations per candidate suffix.
    step_sec : int, default=60
        Scan step size in seconds.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        `(aligned_baseline_df, strategy_metadata)`.
    """
    if calibration_df.empty:
        raise ValueError("calibration_df must not be empty.")

    signal_results: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}

    for col in feature_columns:
        if col not in calibration_df.columns:
            raise ValueError(f"Signal column '{col}' is missing from calibration_df.")
        arr, meta = select_single_signal_calibration_window(
            calibration_readings=calibration_df[col],
            signal_name=col,
            window_size_sec=window_size_sec,
            min_observations=min_observations,
            step_sec=step_sec,
        )
        signal_results[col] = (arr, meta)

    # Align columns using NaN padding
    max_len = max(len(arr) for arr, _ in signal_results.values())
    padded_dict: dict[str, np.ndarray] = {}
    for col in feature_columns:
        arr, _ = signal_results[col]
        padded = np.full(max_len, np.nan, dtype=float)
        padded[: len(arr)] = arr
        padded_dict[col] = padded

    baseline_df = pd.DataFrame(padded_dict)

    metadata: dict[str, Any] = {
        "strategy": "Strategy D (Signal-Specific Quality-Gated)",
        "signals": {col: meta for col, (_, meta) in signal_results.items()},
        "signal_sample_counts": {
            col: len(arr) for col, (arr, _) in signal_results.items()
        },
        "signal_statuses": {
            col: meta["status"] for col, (_, meta) in signal_results.items()
        },
        "signal_offsets": {
            col: meta["offset"] for col, (_, meta) in signal_results.items()
        },
        "decision": (
            "Signal-specific calibration windows selected independently: "
            + "; ".join(
                f"{col}: {len(arr)} obs (offset={meta['offset']}s, status={meta['status']})"
                for col, (arr, meta) in signal_results.items()
            )
        ),
    }

    return baseline_df, metadata
