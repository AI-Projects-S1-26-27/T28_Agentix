"""Unit tests for calibration duration sensitivity experiment (ai.experiments.evaluate_calibration_sensitivity)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ai.experiments.evaluate_calibration_sensitivity import (
    evaluate_participant_sensitivity,
    run_calibration_duration_sensitivity_experiment,
)


def test_evaluate_participant_sensitivity_monotonic_partitions():
    """Verify that calibration size grows and held-out size shrinks as ratio increases."""
    timestamps = pd.date_range("2026-01-01", periods=1000, freq="1s", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": [70.0] * 1000,
        "eda_microsiemens": [1.0] * 1000,
        "body_temperature_c": [36.5] * 1000,
        "stress_label": [1] * 700 + [2] * 300,
    })

    ratios = [0.70, 0.75, 0.80, 0.85]
    results = evaluate_participant_sensitivity(df, participant_id="TEST_P", ratios=ratios, min_observations=200)

    assert len(results) == 4
    cal_sizes = [r["calibration_partition_size"] for r in results]
    heldout_sizes = [r["heldout_baseline_size"] for r in results]

    # Calibration size strictly increases; held-out size strictly decreases
    assert cal_sizes == sorted(cal_sizes)
    assert heldout_sizes == sorted(heldout_sizes, reverse=True)
    for c, h in zip(cal_sizes, heldout_sizes):
        assert c + h == 700


def test_evaluate_participant_sensitivity_zero_data_leakage():
    """Verify that held-out baseline and stress data cannot leak into calibration selection."""
    timestamps = pd.date_range("2026-01-01", periods=1200, freq="1s", tz="UTC")
    df_clean = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": [70.0] * 1200,
        "eda_microsiemens": [1.0] * 1200,
        "body_temperature_c": [36.5] * 1200,
        "stress_label": [1] * 900 + [2] * 300,
    })

    df_poisoned = df_clean.copy()
    # Poison only the held-out portion (indices >= 800) and stress
    df_poisoned.loc[800:, "heart_rate_bpm"] = 999.0
    df_poisoned.loc[800:, "eda_microsiemens"] = 555.0

    res_clean = evaluate_participant_sensitivity(df_clean, "TP", ratios=[0.70, 0.75], min_observations=480)
    res_poisoned = evaluate_participant_sensitivity(df_poisoned, "TP", ratios=[0.70, 0.75], min_observations=480)

    for rc, rp in zip(res_clean, res_poisoned):
        assert rc["hr_baseline_mean"] == pytest.approx(rp["hr_baseline_mean"])
        assert rc["hr_baseline_std"] == pytest.approx(rp["hr_baseline_std"])
        assert rc["eda_baseline_mean"] == pytest.approx(rp["eda_baseline_mean"])
        assert rc["eda_baseline_std"] == pytest.approx(rp["eda_baseline_std"])


def test_run_calibration_duration_sensitivity_experiment_results(tmp_path: Path):
    """Verify full execution across all 4 participants and 4 ratios."""
    df_res, csv_path, report_path = run_calibration_duration_sensitivity_experiment(output_dir=tmp_path)

    assert csv_path.exists()
    assert report_path.exists()
    assert len(df_res) == 16  # 4 participants x 4 ratios

    # Verify S3 recall progression and zero false positives
    s3_rows = df_res[df_res["participant_id"] == "S3"].sort_values("calibration_ratio").reset_index(drop=True)
    assert len(s3_rows) == 4

    recalls = list(s3_rows["recall"])
    assert recalls[0] == pytest.approx(0.5578, abs=0.01)  # 70%
    assert recalls[1] >= 0.80  # 75%
    assert recalls[2] >= 0.80  # 80%
    assert recalls[3] == pytest.approx(1.0, abs=0.01)     # 85%

    # S3 false positives remain 0 across all ratios
    assert (s3_rows["false_positives"] == 0).all()
    assert (s3_rows["precision"] == 1.0).all()

    # S4 retains 100% recall and near-zero/zero FPR across all ratios
    s4_rows = df_res[df_res["participant_id"] == "S4"]
    assert (s4_rows["recall"] == 1.0).all()
    assert (s4_rows["false_positive_rate"] <= 0.01).all()
