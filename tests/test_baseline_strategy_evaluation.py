"""Unit tests for baseline strategy evaluation (ai.experiments.evaluate_baseline_strategies)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ai.baseline.baseline_quality import (
    select_quality_gated_calibration_window,
    select_settled_calibration_window,
    select_signal_specific_calibration_windows,
)
from ai.experiments.evaluate_baseline_strategies import (
    evaluate_participant_strategies,
    run_baseline_strategy_experiment,
)
from ai.experiments.evaluate_wesad_anomalies import split_baseline_chronologically


def test_zero_data_leakage_in_strategy_selection():
    """Verify that held-out baseline and stress data cannot leak into calibration selection."""
    # Create baseline (1000s) and stress (300s)
    timestamps = pd.date_range("2026-01-01", periods=1300, freq="1s", tz="UTC")
    df_clean = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": [70.0] * 1300,
        "eda_microsiemens": [1.0] * 1300,
        "body_temperature_c": [36.5] * 1300,
        "stress_label": [1] * 1000 + [2] * 300,
    })

    # Poison the held-out baseline and stress with extreme outlier values
    df_poisoned = df_clean.copy()
    df_poisoned.loc[700:, "eda_microsiemens"] = 9999.0
    df_poisoned.loc[700:, "heart_rate_bpm"] = 500.0

    cal_clean, _ = split_baseline_chronologically(df_clean[df_clean["stress_label"] == 1], 0.70)
    cal_poisoned, _ = split_baseline_chronologically(df_poisoned[df_poisoned["stress_label"] == 1], 0.70)

    # Strategy B on clean vs poisoned
    cal_b_clean, _ = select_settled_calibration_window(cal_clean)
    cal_b_poisoned, _ = select_settled_calibration_window(cal_poisoned)
    pd.testing.assert_frame_equal(cal_b_clean, cal_b_poisoned)

    # Strategy C on clean vs poisoned
    cal_c_clean, _ = select_quality_gated_calibration_window(cal_clean)
    cal_c_poisoned, _ = select_quality_gated_calibration_window(cal_poisoned)
    pd.testing.assert_frame_equal(cal_c_clean, cal_c_poisoned)

    # Strategy D on clean vs poisoned
    cal_d_clean, _ = select_signal_specific_calibration_windows(cal_clean)
    cal_d_poisoned, _ = select_signal_specific_calibration_windows(cal_poisoned)
    pd.testing.assert_frame_equal(cal_d_clean, cal_d_poisoned)


def test_evaluate_participant_strategies_structure():
    """Verify that evaluate_participant_strategies returns structured metrics for all 4 strategies."""
    timestamps = pd.date_range("2026-01-01", periods=1000, freq="1s", tz="UTC")
    np.random.seed(42)
    raw_df = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": 70.0 + np.random.normal(0, 1.0, 1000),
        "eda_microsiemens": 1.0 + np.random.normal(0, 0.05, 1000),
        "body_temperature_c": 36.5 + np.random.normal(0, 0.02, 1000),
        "stress_label": [1] * 700 + [2] * 300,
    })

    results = evaluate_participant_strategies(raw_df, participant_id="TEST_P")

    assert len(results) == 4
    strategy_names = [r["strategy"] for r in results]
    assert "Strategy A (Existing)" in strategy_names
    assert "Strategy B (Settled)" in strategy_names
    assert "Strategy C (Quality-Gated)" in strategy_names
    assert "Strategy D (Signal-Specific)" in strategy_names

    for r in results:
        assert r["participant_id"] == "TEST_P"
        assert "precision" in r
        assert "recall" in r
        assert "f1_score" in r
        assert "false_positive_rate" in r
        assert "calibration_samples" in r
        assert "hr_calibration_samples" in r
        assert "eda_calibration_samples" in r
        assert "temp_calibration_samples" in r
        assert "hr_selection_offset" in r
        assert "hr_assessment_status" in r
        assert "fully_unevaluable_samples" in r
        assert "partially_evaluated_samples" in r
        assert "decision_info" in r


def test_evaluate_participant_strategies_duplicate_timestamps_raises():
    """Verify that duplicate timestamps trigger ValueError."""
    ts = pd.date_range("2026-01-01", periods=10, freq="1s", tz="UTC").tolist()
    ts[1] = ts[0]  # Introduce duplicate
    df = pd.DataFrame({
        "timestamp": ts,
        "heart_rate_bpm": [70.0] * 10,
        "eda_microsiemens": [1.0] * 10,
        "body_temperature_c": [36.5] * 10,
        "stress_label": [1] * 7 + [2] * 3,
    })
    with pytest.raises(ValueError, match="duplicate timestamps"):
        evaluate_participant_strategies(df, participant_id="S2")


def test_evaluate_participant_strategies_empty_raises():
    """Verify that empty dataframe triggers ValueError."""
    with pytest.raises(ValueError, match="is empty"):
        evaluate_participant_strategies(pd.DataFrame(), participant_id="S2")


def test_evaluate_participant_strategies_missing_labels_raises():
    """Verify that missing baseline or stress observations trigger ValueError."""
    ts = pd.date_range("2026-01-01", periods=10, freq="1s", tz="UTC")
    df_no_stress = pd.DataFrame({
        "timestamp": ts,
        "heart_rate_bpm": [70.0] * 10,
        "eda_microsiemens": [1.0] * 10,
        "body_temperature_c": [36.5] * 10,
        "stress_label": [1] * 10,
    })
    with pytest.raises(ValueError, match="has no stress observations"):
        evaluate_participant_strategies(df_no_stress, participant_id="S2")


def test_run_baseline_strategy_experiment_results(tmp_path: Path):
    """Verify full experiment execution and key clinical findings."""
    comparison_df, csv_path, report_path = run_baseline_strategy_experiment(
        output_dir=tmp_path
    )

    assert csv_path.exists()
    assert report_path.exists()
    assert len(comparison_df) == 16  # 4 participants x 4 strategies

    # Check S2 Strategy D preserves temperature variance
    s2_a = comparison_df[(comparison_df["participant_id"] == "S2") & (comparison_df["strategy"] == "Strategy A (Existing)")].iloc[0]
    s2_d = comparison_df[(comparison_df["participant_id"] == "S2") & (comparison_df["strategy"] == "Strategy D (Signal-Specific)")].iloc[0]
    assert s2_d["false_positives"] == s2_a["false_positives"]
    assert s2_d["false_positive_rate"] == pytest.approx(s2_a["false_positive_rate"])

    # Check S3 improvement
    s3_a = comparison_df[(comparison_df["participant_id"] == "S3") & (comparison_df["strategy"] == "Strategy A (Existing)")].iloc[0]
    s3_b = comparison_df[(comparison_df["participant_id"] == "S3") & (comparison_df["strategy"] == "Strategy B (Settled)")].iloc[0]
    s3_c = comparison_df[(comparison_df["participant_id"] == "S3") & (comparison_df["strategy"] == "Strategy C (Quality-Gated)")].iloc[0]
    s3_d = comparison_df[(comparison_df["participant_id"] == "S3") & (comparison_df["strategy"] == "Strategy D (Signal-Specific)")].iloc[0]

    assert s3_b["recall"] > s3_a["recall"]
    assert s3_c["recall"] >= s3_b["recall"]
    assert s3_d["recall"] > s3_a["recall"]
    assert s3_d["precision"] == 1.0

    # Check S4 stability
    s4_a = comparison_df[(comparison_df["participant_id"] == "S4") & (comparison_df["strategy"] == "Strategy A (Existing)")].iloc[0]
    s4_b = comparison_df[(comparison_df["participant_id"] == "S4") & (comparison_df["strategy"] == "Strategy B (Settled)")].iloc[0]
    s4_d = comparison_df[(comparison_df["participant_id"] == "S4") & (comparison_df["strategy"] == "Strategy D (Signal-Specific)")].iloc[0]
    assert s4_a["calibration_samples"] == s4_b["calibration_samples"] == s4_d["calibration_samples"]
    assert s4_a["recall"] == s4_b["recall"] == s4_d["recall"]
