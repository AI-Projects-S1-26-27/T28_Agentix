"""Unit tests for baseline quality assessment module (ai.baseline.baseline_quality)."""

import numpy as np
import pandas as pd
import pytest

from ai.baseline.baseline_quality import (
    SignalQualityAssessment,
    assess_calibration_dataset_quality,
    assess_signal_baseline_quality,
    calculate_single_signal_instability_score,
    select_quality_gated_calibration_window,
    select_settled_calibration_window,
    select_signal_specific_calibration_windows,
    select_single_signal_calibration_window,
)


def test_assess_signal_baseline_quality_fewer_than_two_observations():
    """Verify handling when fewer than 2 valid observations are provided."""
    res = assess_signal_baseline_quality([42.0], signal_name="heart_rate_bpm")
    assert res.status == "insufficient_data"
    assert res.observation_count == 1
    assert res.num_windows == 0
    assert np.isnan(res.mean_drift)
    assert res.acclimation_detected is False


def test_assess_signal_baseline_quality_insufficient_data_threshold():
    """Verify handling when observations are between 2 and min_observations."""
    data = [70.0 + i * 0.1 for i in range(30)]
    res = assess_signal_baseline_quality(
        data, signal_name="heart_rate_bpm", min_observations=60
    )
    assert res.status == "insufficient_data"
    assert res.observation_count == 30
    assert res.acclimation_detected is False
    assert "below the recommended minimum" in res.explanation


def test_assess_signal_baseline_quality_stable_synthetic():
    """Verify that a stationary signal with low noise is assessed as stable."""
    np.random.seed(42)
    # 360 observations (3 full 120s windows) of stable HR around 70 bpm
    stable_hr = 70.0 + np.random.normal(0, 1.0, 360)
    res = assess_signal_baseline_quality(stable_hr, signal_name="heart_rate_bpm")

    assert res.status == "stable"
    assert res.observation_count == 360
    assert res.num_windows == 3
    assert res.acclimation_detected is False
    assert abs(res.pct_mean_drift) < 10.0


def test_assess_signal_baseline_quality_acclimation_decay():
    """Verify that a strong monotonic downward decay is flagged as acclimation."""
    # 720 observations decaying from 5.0 to 0.5 across 6 windows (120s each)
    t = np.linspace(0, 5, 720)
    decaying_eda = 4.5 * np.exp(-t) + 0.5 + np.random.normal(0, 0.02, 720)
    res = assess_signal_baseline_quality(decaying_eda, signal_name="eda_microsiemens")

    assert res.status == "unstable"
    assert res.acclimation_detected is True
    assert res.max_consecutive_decreases >= 4
    assert res.pct_mean_drift < -20.0
    assert "acclimation decay" in res.explanation


def test_assess_signal_baseline_quality_negligible_variance():
    """Verify that signals with variance below physiological noise floor are stable."""
    # Almost constant EDA reading (variance < 0.005)
    np.random.seed(42)
    micro_eda = 0.5 + np.random.normal(0, 0.01, 360)  # var ~ 0.0001
    res = assess_signal_baseline_quality(micro_eda, signal_name="eda_microsiemens")

    assert res.status == "stable"
    assert res.acclimation_detected is False
    assert res.measurements["is_negligible_variance"] is True
    assert "noise floor" in res.explanation


def test_assess_calibration_dataset_quality_dataframe():
    """Verify multi-signal evaluation on a DataFrame."""
    np.random.seed(42)
    df = pd.DataFrame({
        "heart_rate_bpm": 72.0 + np.random.normal(0, 1.0, 360),
        "eda_microsiemens": 1.2 + np.random.normal(0, 0.01, 360),
        "body_temperature_c": 36.5 + np.random.normal(0, 0.02, 360),
    })
    results = assess_calibration_dataset_quality(df)

    assert set(results.keys()) == {"heart_rate_bpm", "eda_microsiemens", "body_temperature_c"}
    for sig, assessment in results.items():
        assert isinstance(assessment, SignalQualityAssessment)
        assert assessment.status == "stable"


def test_assess_calibration_dataset_quality_missing_column():
    """Verify handling when a requested feature column is missing from DataFrame."""
    df = pd.DataFrame({"heart_rate_bpm": [70.0] * 100})
    results = assess_calibration_dataset_quality(
        df, feature_columns=["heart_rate_bpm", "eda_microsiemens"]
    )
    assert results["eda_microsiemens"].status == "insufficient_data"
    assert results["eda_microsiemens"].observation_count == 0


def test_assess_calibration_dataset_quality_empty_df_raises():
    """Verify that an empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="calibration_df must not be empty"):
        assess_calibration_dataset_quality(pd.DataFrame())


def test_select_settled_calibration_window_no_acclimation():
    """Verify Strategy B retains all observations when data is stable."""
    np.random.seed(42)
    df = pd.DataFrame({
        "heart_rate_bpm": 70.0 + np.random.normal(0, 1.0, 500),
        "eda_microsiemens": 0.5 + np.random.normal(0, 0.01, 500),
        "body_temperature_c": 36.6 + np.random.normal(0, 0.02, 500),
    })
    settled_df, meta = select_settled_calibration_window(df)

    assert len(settled_df) == 500
    assert meta["observations_excluded"] == 0
    assert meta["acclimation_detected"] is False


def test_select_settled_calibration_window_acclimation():
    """Verify Strategy B excludes burn-in when acclimation is present."""
    t = np.linspace(0, 5, 600)
    decaying_eda = 4.5 * np.exp(-t) + 0.5
    df = pd.DataFrame({
        "heart_rate_bpm": [70.0] * 600,
        "eda_microsiemens": decaying_eda,
        "body_temperature_c": [36.5] * 600,
    })
    settled_df, meta = select_settled_calibration_window(df, min_retained_obs=240)

    assert meta["acclimation_detected"] is True
    assert meta["observations_excluded"] == 240
    assert len(settled_df) == 360


def test_select_quality_gated_calibration_window():
    """Verify Strategy C identifies the optimal stable window."""
    t = np.linspace(0, 5, 600)
    decaying_eda = 4.5 * np.exp(-t) + 0.5
    df = pd.DataFrame({
        "heart_rate_bpm": [70.0] * 600,
        "eda_microsiemens": decaying_eda,
        "body_temperature_c": [36.5] * 600,
    })
    gated_df, meta = select_quality_gated_calibration_window(df, min_stable_obs=360)

    assert meta["gating_status"] in ("gated_suffix", "accepted_full")
    assert len(gated_df) >= 360
    assert meta["accepted_observations"] == len(gated_df)


def test_signal_quality_assessment_to_dict():
    """Verify that to_dict returns serializable dictionary with all required fields."""
    res = assess_signal_baseline_quality([70.0] * 120, signal_name="heart_rate_bpm")
    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["signal"] == "heart_rate_bpm"
    assert "status" in d
    assert "measurements" in d
    assert "explanation" in d


def test_select_single_signal_calibration_window_stable():
    """Verify Strategy D retains complete partition when signal is stable."""
    np.random.seed(42)
    stable_signal = 70.0 + np.random.normal(0, 1.0, 600)
    arr, meta = select_single_signal_calibration_window(
        stable_signal, signal_name="heart_rate_bpm", min_observations=480
    )
    assert meta["status"] == "stable"
    assert meta["selection_type"] == "full_partition"
    assert meta["offset"] == 0
    assert len(arr) == 600
    assert meta["observation_count"] == 600


def test_select_single_signal_calibration_window_unstable_pass():
    """Verify Strategy D selects earliest stable suffix when initial signal acclimates."""
    # 240s early decay followed by 480s stable signal
    t = np.linspace(0, 4, 240)
    decay_part = 4.0 * np.exp(-t) + 1.0
    stable_part = [1.0] * 480
    signal = np.concatenate([decay_part, stable_part])

    arr, meta = select_single_signal_calibration_window(
        signal, signal_name="eda_microsiemens", min_observations=480, step_sec=60
    )
    assert meta["status"] == "stable"
    assert meta["selection_type"] == "earliest_stable_suffix"
    assert meta["offset"] > 0
    assert len(arr) >= 480
    assert meta["observation_count"] == len(arr)


def test_select_single_signal_calibration_window_fallback():
    """Verify Strategy D fallback when all suffixes remain unstable."""
    # Continuous steep decay across all 720 observations
    t = np.linspace(0, 5, 720)
    decay = 10.0 * np.exp(-t / 2.0)

    arr, meta = select_single_signal_calibration_window(
        decay, signal_name="eda_microsiemens", min_observations=480, step_sec=60
    )
    # Must NOT be silently marked stable
    assert meta["status"] == "unstable"
    assert meta["selection_type"] == "unstable_fallback"
    assert len(arr) >= 480
    assert "remained unstable" in meta["explanation"]


def test_select_signal_specific_calibration_windows_nan_padding():
    """Verify Strategy D multi-signal alignment using NaN padding without data fabrication."""
    np.random.seed(42)
    # Signal A: 600 stable samples
    # Signal B: 720 decaying samples (suffix selected)
    # Signal C: 600 stable samples
    t = np.linspace(0, 4, 240)
    decay = 4.0 * np.exp(-t) + 1.0
    sig_b = np.concatenate([decay, [1.0] * 480])

    df = pd.DataFrame({
        "heart_rate_bpm": [70.0] * 720,
        "eda_microsiemens": sig_b,
        "body_temperature_c": [36.5] * 720,
    })

    baseline_df, meta = select_signal_specific_calibration_windows(
        df, min_observations=480, step_sec=60
    )

    assert isinstance(baseline_df, pd.DataFrame)
    assert set(baseline_df.columns) == {"heart_rate_bpm", "eda_microsiemens", "body_temperature_c"}
    assert meta["strategy"] == "Strategy D (Signal-Specific Quality-Gated)"

    # Check that each column drops NaNs to its exact selected length
    for col in ["heart_rate_bpm", "eda_microsiemens", "body_temperature_c"]:
        valid_vals = baseline_df[col].dropna()
        expected_len = meta["signal_sample_counts"][col]
        assert len(valid_vals) == expected_len
        assert expected_len >= 480


def test_select_signal_specific_calibration_windows_empty_or_missing_raises():
    """Verify error handling on empty DataFrame or missing column."""
    with pytest.raises(ValueError, match="calibration_df must not be empty"):
        select_signal_specific_calibration_windows(pd.DataFrame())

    df = pd.DataFrame({"heart_rate_bpm": [70.0] * 500})
    with pytest.raises(ValueError, match="missing from calibration_df"):
        select_signal_specific_calibration_windows(
            df, feature_columns=["heart_rate_bpm", "eda_microsiemens"]
        )
