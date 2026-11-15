"""Tests for personal baseline stability investigation module."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from ai.baseline.baseline_stability import (
    compute_calibration_vs_heldout_comparison,
    compute_windowed_baseline_statistics,
    run_baseline_stability_investigation,
)


@pytest.fixture
def synthetic_baseline():
    """Create synthetic baseline dataframe of 1140 seconds (similar to S3)."""
    timestamps = pd.date_range("2026-01-01 08:00:00", periods=1140, freq="1s", tz="UTC")

    np.random.seed(42)
    # Stationary heart rate
    hr = np.random.normal(70.0, 2.0, size=1140)

    # Decaying non-stationary EDA (acclimation from 6.0 down to 0.5)
    t = np.linspace(0, 1, 1140)
    eda = 5.5 * np.exp(-4 * t) + 0.5 + np.random.normal(0, 0.05, size=1140)

    # Stationary body temperature
    temp = np.random.normal(33.0, 0.1, size=1140)

    return pd.DataFrame({
        "participant_id": ["PTest"] * 1140,
        "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stress_label": [1] * 1140,
        "heart_rate_bpm": hr,
        "eda_microsiemens": eda,
        "body_temperature_c": temp,
    })


def test_windowed_statistics_structure(synthetic_baseline):
    res = compute_windowed_baseline_statistics(
        baseline_df=synthetic_baseline,
        participant_id="PTest",
        feature_columns=["heart_rate_bpm", "eda_microsiemens", "body_temperature_c"],
        window_size_sec=120,
        min_window_size=30,
        split_ratio=0.70,
    )

    # 1140 observations // 120 = 9 full windows + 60s remainder = 10 windows
    # 10 windows * 3 signals = 30 rows
    assert len(res) == 30
    assert set(res["signal"].unique()) == {
        "heart_rate_bpm",
        "eda_microsiemens",
        "body_temperature_c",
    }
    assert res["window_index"].max() == 10

    # Required column checks
    required_cols = [
        "participant_id",
        "window_index",
        "partition",
        "start_seconds",
        "end_seconds",
        "observation_count",
        "signal",
        "mean",
        "std",
        "variance",
        "median",
        "iqr",
        "delta_mean_prev_window",
        "pct_change_mean_prev_window",
        "delta_var_prev_window",
        "variance_ratio_prev_window",
    ]
    for col in required_cols:
        assert col in res.columns

    # First window delta should be NaN, second window delta should be difference of means
    hr_res = res[res["signal"] == "heart_rate_bpm"].reset_index(drop=True)
    assert np.isnan(hr_res.loc[0, "delta_mean_prev_window"])
    expected_delta = hr_res.loc[1, "mean"] - hr_res.loc[0, "mean"]
    assert hr_res.loc[1, "delta_mean_prev_window"] == pytest.approx(expected_delta)


def test_partition_assignment(synthetic_baseline):
    res = compute_windowed_baseline_statistics(
        baseline_df=synthetic_baseline,
        participant_id="PTest",
        split_ratio=0.70,
    )
    # n_cal = floor(1140 * 0.70) = 798
    # W1-W6: 0-720 <= 798 -> calibration
    # W7: 720-840 spans 798 -> boundary
    # W8-W10: >= 840 -> heldout
    w_partitions = (
        res[res["signal"] == "heart_rate_bpm"]
        .set_index("window_index")["partition"]
        .to_dict()
    )

    for w in range(1, 7):
        assert w_partitions[w] == "calibration"
    assert w_partitions[7] == "boundary"
    for w in range(8, 11):
        assert w_partitions[w] == "heldout"


def test_acclimation_detection_on_synthetic_drift(synthetic_baseline):
    comp = compute_calibration_vs_heldout_comparison(
        baseline_df=synthetic_baseline,
        participant_id="PTest",
        split_ratio=0.70,
    )

    eda_comp = comp[comp["signal"] == "eda_microsiemens"].iloc[0]
    hr_comp = comp[comp["signal"] == "heart_rate_bpm"].iloc[0]

    # Decaying EDA should trigger acclimation detection
    assert bool(eda_comp["acclimation_detected"]) is True
    assert eda_comp["variance_inflation_factor"] > 3.0
    assert eda_comp["pct_change_mean"] < -20.0

    # Stationary HR should not trigger acclimation
    assert bool(hr_comp["acclimation_detected"]) is False
    assert hr_comp["variance_inflation_factor"] < 3.0


def test_error_handling_empty_or_missing_columns(synthetic_baseline):
    # Empty dataframe
    with pytest.raises(ValueError, match="must not be empty"):
        compute_windowed_baseline_statistics(
            baseline_df=pd.DataFrame(),
            participant_id="PTest",
        )

    # Missing required feature columns
    incomplete_df = synthetic_baseline.drop(columns=["eda_microsiemens"])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        compute_windowed_baseline_statistics(
            baseline_df=incomplete_df,
            participant_id="PTest",
        )


def test_end_to_end_baseline_stability_investigation(tmp_path):
    w_df, c_df, report = run_baseline_stability_investigation(output_dir=tmp_path)

    # 4 participants * 10 windows * 3 signals = 120 rows
    assert len(w_df) == 120
    # 4 participants * 3 signals = 12 comparison rows
    assert len(c_df) == 12

    # Check files created
    assert (tmp_path / "baseline_window_diagnostics.csv").exists()
    assert (tmp_path / "baseline_stability_report.md").exists()

    # S3 EDA variance inflation verification
    s3_eda_comp = c_df[
        (c_df["participant_id"] == "S3") & (c_df["signal"] == "eda_microsiemens")
    ].iloc[0]

    # Verify S3's extreme variance ratio (> 400x) and inflation factor (> 15x)
    assert s3_eda_comp["variance_ratio_cal_to_heldout"] > 400.0
    assert s3_eda_comp["variance_inflation_factor"] > 15.0
    assert bool(s3_eda_comp["acclimation_detected"]) is True

    # Report text verification
    assert "442.3x variance ratio" in report
    assert "18.99x variance inflation factor" in report
