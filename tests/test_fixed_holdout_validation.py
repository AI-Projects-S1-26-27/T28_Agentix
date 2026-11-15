"""Unit tests for confirmatory fixed-holdout validation experiment (ai.experiments.evaluate_fixed_holdout)."""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ai.experiments.evaluate_fixed_holdout import (
    evaluate_participant_fixed_holdout,
    run_fixed_holdout_experiment,
)


def test_identical_heldout_baseline_and_stress_across_configurations():
    """Verify that held-out baseline and stress evaluation sets are identical across all configurations."""
    df_s3 = pd.read_csv("data/processed/wesad_s3_real.csv")
    df_s3["timestamp"] = pd.to_datetime(df_s3["timestamp"], utc=True)
    df_s3 = df_s3.sort_values("timestamp").reset_index(drop=True)

    base = df_s3[df_s3["stress_label"] == 1].reset_index(drop=True)
    stress = df_s3[df_s3["stress_label"] == 2].reset_index(drop=True)

    n_test_start = math.floor(len(base) * 0.85)
    expected_heldout = base.iloc[n_test_start:].reset_index(drop=True)

    results = evaluate_participant_fixed_holdout(df_s3, participant_id="S3")

    for r in results:
        assert r["heldout_baseline_observations"] == len(expected_heldout)
        assert r["stress_observations"] == len(stress)
        assert r["total_evaluation_observations"] == len(expected_heldout) + len(stress)


def test_no_calibration_test_overlap():
    """Verify that calibration observations never overlap with the reserved final 15% test set."""
    timestamps = pd.date_range("2026-01-01", periods=1000, freq="1s", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": [70.0] * 1000,
        "eda_microsiemens": [1.0] * 1000,
        "body_temperature_c": [36.5] * 1000,
        "stress_label": [1] * 700 + [2] * 300,
    })

    n_test_start = math.floor(700 * 0.85)
    test_timestamps = set(df["timestamp"].iloc[n_test_start:700])

    ratios = [0.70, 0.75, 0.80, 0.85]
    for r in ratios:
        n_cal = math.floor(700 * r)
        cal_timestamps = set(df["timestamp"].iloc[:n_cal])
        # Intersection between calibration and test timestamps must be strictly empty
        assert cal_timestamps.isdisjoint(test_timestamps)


def test_no_stress_data_leakage():
    """Verify that modifying or poisoning stress data has zero effect on calibration statistics."""
    timestamps = pd.date_range("2026-01-01", periods=1200, freq="1s", tz="UTC")
    df_clean = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": [70.0] * 1200,
        "eda_microsiemens": [1.0] * 1200,
        "body_temperature_c": [36.5] * 1200,
        "stress_label": [1] * 900 + [2] * 300,
    })

    df_poisoned = df_clean.copy()
    # Poison stress period only (indices 900..1200)
    df_poisoned.loc[900:, "heart_rate_bpm"] = 999.0
    df_poisoned.loc[900:, "eda_microsiemens"] = 999.0

    res_clean = evaluate_participant_fixed_holdout(df_clean, "TEST_P", ratios=[0.70, 0.80], min_observations=480)
    res_poisoned = evaluate_participant_fixed_holdout(df_poisoned, "TEST_P", ratios=[0.70, 0.80], min_observations=480)

    for rc, rp in zip(res_clean, res_poisoned):
        assert rc["hr_baseline_mean"] == pytest.approx(rp["hr_baseline_mean"])
        assert rc["hr_baseline_std"] == pytest.approx(rp["hr_baseline_std"])
        assert rc["eda_baseline_mean"] == pytest.approx(rp["eda_baseline_mean"])
        assert rc["eda_baseline_std"] == pytest.approx(rp["eda_baseline_std"])


def test_no_leakage_from_unused_intermediate_observations():
    """Verify that poisoning unused intermediate gap observations does not affect shorter calibrations."""
    timestamps = pd.date_range("2026-01-01", periods=1200, freq="1s", tz="UTC")
    df_clean = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": [70.0] * 1200,
        "eda_microsiemens": [1.0] * 1200,
        "body_temperature_c": [36.5] * 1200,
        "stress_label": [1] * 900 + [2] * 300,
    })

    # For ratio 0.70, cal ends at 900 * 0.70 = 630. Fixed test starts at 900 * 0.85 = 765.
    # Intermediate unused gap is 630..765. Poison that gap.
    df_gap_poisoned = df_clean.copy()
    df_gap_poisoned.loc[630:765, "eda_microsiemens"] = 8888.0

    res_clean = evaluate_participant_fixed_holdout(df_clean, "TEST_P", ratios=[0.70], min_observations=480)
    res_gap_poisoned = evaluate_participant_fixed_holdout(df_gap_poisoned, "TEST_P", ratios=[0.70], min_observations=480)

    # 70% configuration must be completely invariant to gap corruption
    assert res_clean[0]["eda_baseline_mean"] == pytest.approx(res_gap_poisoned[0]["eda_baseline_mean"])
    assert res_clean[0]["eda_baseline_std"] == pytest.approx(res_gap_poisoned[0]["eda_baseline_std"])
    assert res_clean[0]["eda_calibration_samples"] == res_gap_poisoned[0]["eda_calibration_samples"]


def test_correct_chronological_ordering():
    """Verify that non-monotonic timestamps raise ValueError."""
    ts = pd.to_datetime(["2026-01-01 10:00:02", "2026-01-01 10:00:01", "2026-01-01 10:00:03"], utc=True)
    df = pd.DataFrame({
        "timestamp": ts,
        "heart_rate_bpm": [70.0, 72.0, 71.0],
        "eda_microsiemens": [1.0, 1.1, 1.0],
        "body_temperature_c": [36.5, 36.6, 36.5],
        "stress_label": [1, 1, 2],
    })
    with pytest.raises(ValueError, match="not monotonic increasing"):
        evaluate_participant_fixed_holdout(df, participant_id="S2")


def test_reproducible_results():
    """Verify that running the fixed-holdout experiment produces deterministic results."""
    df_s4 = pd.read_csv("data/processed/wesad_s4_real.csv")
    res1 = evaluate_participant_fixed_holdout(df_s4, "S4", ratios=[0.70, 0.85])
    res2 = evaluate_participant_fixed_holdout(df_s4, "S4", ratios=[0.70, 0.85])

    assert res1 == res2


def test_handling_of_missing_and_zero_variance():
    """Verify safe handling of zero-variance baseline and missing values without crashes."""
    timestamps = pd.date_range("2026-01-01", periods=1000, freq="1s", tz="UTC")
    # Base dataframe with varying HR and EDA/temp
    df = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": [70.0 + (i % 5) for i in range(1000)],
        "eda_microsiemens": [1.0 + (i % 3) * 0.1 for i in range(1000)],
        "body_temperature_c": [36.5 + (i % 4) * 0.1 for i in range(1000)],
        "stress_label": [1] * 700 + [2] * 300,
    })
    # Inject missing value in stress evaluation
    df.loc[750, "eda_microsiemens"] = np.nan

    results = evaluate_participant_fixed_holdout(df, "TEST_P", ratios=[0.70], min_observations=200)
    assert len(results) == 1
    # Exactly one observation in evaluation has a missing signal, so partially evaluated count is 1
    assert results[0]["partially_evaluated_samples"] == 1

    # Also verify zero-variance baseline handling does not crash
    df_zero_var = df.copy()
    df_zero_var["heart_rate_bpm"] = 72.0  # perfectly constant
    res_zero = evaluate_participant_fixed_holdout(df_zero_var, "TEST_ZERO", ratios=[0.70], min_observations=200)
    assert res_zero[0]["hr_baseline_std"] == 0.0


def test_run_fixed_holdout_experiment_outputs(tmp_path: Path):
    """Verify complete execution and generation of CSV and Markdown outputs."""
    df_res, csv_p, rep_p = run_fixed_holdout_experiment(output_dir=tmp_path)

    assert csv_p.exists()
    assert rep_p.exists()
    assert len(df_res) == 16  # 4 participants x 4 ratios

    # S3 confirmation
    s3_rows = df_res[df_res["participant_id"] == "S3"].sort_values("calibration_ratio").reset_index(drop=True)
    recalls = list(s3_rows["recall"])
    assert recalls[0] == pytest.approx(0.5578, abs=0.01)
    assert recalls[1] >= 0.80
    assert recalls[2] >= 0.80
    assert recalls[3] == pytest.approx(1.0, abs=0.01)
    assert (s3_rows["false_positives"] == 0).all()
