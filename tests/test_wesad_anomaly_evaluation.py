"""Tests for WESAD personalized multi-signal anomaly evaluation experiment."""

from collections.abc import Sequence
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from ai.experiments.evaluate_wesad_anomalies import (
    CALIBRATION_RATIO,
    DEFAULT_FEATURE_COLUMNS,
    calculate_classification_metrics,
    evaluate_participant,
    run_wesad_anomaly_evaluation,
    split_baseline_chronologically,
)


@pytest.fixture
def synthetic_wesad_participant():
    """Create a realistic synthetic participant dataframe matching WESAD schema."""
    timestamps = pd.date_range("2026-01-01 08:00:00", periods=2000, freq="1s", tz="UTC")
    # Labels:
    # 0: first 200 obs
    # 1: next 1000 obs (baseline)
    # 0: next 200 obs
    # 2: next 500 obs (stress)
    # 4: remaining 100 obs
    labels = (
        [0] * 200
        + [1] * 1000
        + [0] * 200
        + [2] * 500
        + [4] * 100
    )

    np.random.seed(42)
    hr = np.random.normal(70.0, 2.0, size=2000)
    eda = np.random.normal(0.5, 0.05, size=2000)
    temp = np.random.normal(32.0, 0.2, size=2000)

    # Induce acute stress anomalies during label 2
    eda[1400:1900] += 2.0
    hr[1400:1900] += 25.0

    return pd.DataFrame({
        "participant_id": ["PTest"] * 2000,
        "session_id": ["Session1"] * 2000,
        "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "heart_rate_bpm": hr,
        "eda_microsiemens": eda,
        "body_temperature_c": temp,
        "stress_label": labels,
        "source": ["SensorE4"] * 2000,
    })


def test_chronological_baseline_splitting():
    # Test with exactly 1144 baseline observations
    timestamps = pd.date_range("2026-01-01", periods=1144, freq="1s", tz="UTC")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "stress_label": [1] * 1144,
        "heart_rate_bpm": np.random.normal(70, 1, 1144),
    })

    cal_df, heldout_df = split_baseline_chronologically(df, split_ratio=0.7)

    # 1144 * 0.7 = 800.8 -> floor = 800
    assert len(cal_df) == 800
    assert len(heldout_df) == 344
    assert len(cal_df) + len(heldout_df) == 1144

    # Confirm chronological ordering: calibration strictly precedes heldout
    assert cal_df["timestamp"].max() < heldout_df["timestamp"].min()


def test_no_overlap_between_calibration_and_evaluation(synthetic_wesad_participant):
    p_metric, s_metrics, pred_df = evaluate_participant(
        raw_df=synthetic_wesad_participant,
        participant_id="PTest",
    )

    baseline_obs = synthetic_wesad_participant[
        synthetic_wesad_participant["stress_label"] == 1
    ]
    n_cal = int(np.floor(len(baseline_obs) * CALIBRATION_RATIO))
    cal_timestamps = set(baseline_obs["timestamp"].iloc[:n_cal])
    eval_timestamps = set(pred_df["timestamp"].astype(str))

    # Calibration and evaluation must be mutually exclusive
    assert cal_timestamps.isdisjoint(eval_timestamps)


def test_baseline_calibration_uses_only_label_1(synthetic_wesad_participant):
    baseline_obs = synthetic_wesad_participant[
        synthetic_wesad_participant["stress_label"] == 1
    ]
    cal_df, heldout_df = split_baseline_chronologically(baseline_obs)

    assert (cal_df["stress_label"] == 1).all()
    assert (heldout_df["stress_label"] == 1).all()


def test_stress_evaluation_uses_only_label_2(synthetic_wesad_participant):
    p_metric, s_metrics, pred_df = evaluate_participant(
        raw_df=synthetic_wesad_participant,
        participant_id="PTest",
    )

    # Evaluation set must contain only heldout baseline (1) and stress (2)
    eval_labels = set(pred_df["stress_label"].unique())
    assert eval_labels == {1, 2}

    # Labels 0, 4 must be completely excluded
    assert 0 not in eval_labels
    assert 4 not in eval_labels

    # Ground truth mapping: 0 for label 1, 1 for label 2
    assert ((pred_df["stress_label"] == 1) == (pred_df["ground_truth"] == 0)).all()
    assert ((pred_df["stress_label"] == 2) == (pred_df["ground_truth"] == 1)).all()


def test_each_participant_receives_independent_baseline(synthetic_wesad_participant):
    # Participant A: baseline HR ~ 60 bpm
    df_a = synthetic_wesad_participant.copy()
    df_a["participant_id"] = "PA"
    df_a.loc[df_a["stress_label"] == 1, "heart_rate_bpm"] = 60.0 + np.random.normal(
        0, 0.5, size=(df_a["stress_label"] == 1).sum()
    )

    # Participant B: baseline HR ~ 95 bpm
    df_b = synthetic_wesad_participant.copy()
    df_b["participant_id"] = "PB"
    df_b.loc[df_b["stress_label"] == 1, "heart_rate_bpm"] = 95.0 + np.random.normal(
        0, 0.5, size=(df_b["stress_label"] == 1).sum()
    )

    _, _, preds_a = evaluate_participant(df_a, "PA")
    _, _, preds_b = evaluate_participant(df_b, "PB")

    mean_a = preds_a["heart_rate_bpm_baseline_mean"].iloc[0]
    mean_b = preds_b["heart_rate_bpm_baseline_mean"].iloc[0]

    assert mean_a == pytest.approx(60.0, abs=0.5)
    assert mean_b == pytest.approx(95.0, abs=0.5)
    assert abs(mean_a - mean_b) > 30.0


def test_no_metadata_or_target_leakage_into_features(monkeypatch, synthetic_wesad_participant):
    passed_feature_columns: list[str] = []

    import ai.experiments.evaluate_wesad_anomalies as eval_module
    original_detector = eval_module.detect_multisignal_anomalies

    def mock_detector(readings, baseline_readings, feature_columns, thresholds=None):
        passed_feature_columns.extend(feature_columns)
        return original_detector(readings, baseline_readings, feature_columns, thresholds)

    monkeypatch.setattr(eval_module, "detect_multisignal_anomalies", mock_detector)

    evaluate_participant(synthetic_wesad_participant, "PTest")

    # Strictly physiological features
    assert set(passed_feature_columns) == set(DEFAULT_FEATURE_COLUMNS)

    forbidden_cols = [
        "stress_label",
        "participant_id",
        "session_id",
        "timestamp",
        "source",
        "ground_truth",
    ]
    for col in forbidden_cols:
        assert col not in passed_feature_columns


def test_correct_confusion_matrix_and_metric_calculations():
    # Known confusion matrix:
    # y_true: 4 stress (1), 4 baseline (0)
    # y_pred: 3 anomalies (1), 5 normal (0)
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 0, 1, 0, 0, 0]
    # TP=2, FP=1, TN=3, FN=2

    metrics = calculate_classification_metrics(y_true, y_pred)

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 3
    assert metrics["false_negatives"] == 2
    assert metrics["total_evaluation_samples"] == 8

    expected_prec = 2 / 3
    expected_rec = 2 / 4
    expected_f1 = 2 * (expected_prec * expected_rec) / (expected_prec + expected_rec)
    expected_acc = 5 / 8
    expected_fpr = 1 / 4

    assert metrics["precision"] == pytest.approx(expected_prec)
    assert metrics["recall"] == pytest.approx(expected_rec)
    assert metrics["f1_score"] == pytest.approx(expected_f1)
    assert metrics["accuracy"] == pytest.approx(expected_acc)
    assert metrics["false_positive_rate"] == pytest.approx(expected_fpr)
    assert metrics["stress_detection_rate"] == pytest.approx(expected_rec)
    assert not metrics["zero_division_precision"]

    # Zero division case: No positive predictions (TP=0, FP=0)
    zero_pred_metrics = calculate_classification_metrics([1, 0], [0, 0])
    assert zero_pred_metrics["true_positives"] == 0
    assert zero_pred_metrics["false_positives"] == 0
    assert zero_pred_metrics["precision"] == 0.0
    assert zero_pred_metrics["zero_division_precision"] is True


def test_missing_hr_handling(synthetic_wesad_participant):
    df = synthetic_wesad_participant.copy()
    # Introduce missing HR readings in the evaluation portion
    df.loc[1450:1455, "heart_rate_bpm"] = np.nan

    p_metric, s_metrics, pred_df = evaluate_participant(df, "PTest")

    # Missing readings should be flagged as missing_reading
    missing_eval_status = pred_df.loc[
        pred_df["heart_rate_bpm"].isna(), "heart_rate_bpm_evaluation_status"
    ]
    assert (missing_eval_status == "missing_reading").all()

    # Other signals should remain evaluated
    eda_eval_status = pred_df.loc[
        pred_df["heart_rate_bpm"].isna(), "eda_microsiemens_evaluation_status"
    ]
    assert (eda_eval_status == "evaluated").all()


def test_invalid_or_insufficient_calibration_data(synthetic_wesad_participant):
    df = synthetic_wesad_participant.copy()

    # 1. Insufficient baseline observations (< 2 rows)
    few_baseline = df[df["stress_label"] != 1].copy()
    few_baseline = pd.concat([few_baseline, df[df["stress_label"] == 1].iloc[:1]])
    with pytest.raises(ValueError, match="Calibration partition contains"):
        evaluate_participant(few_baseline, "PTest")

    # 2. Calibration data contains all NaNs for a signal
    df_nan_cal = df.copy()
    # Wipe HR during baseline
    df_nan_cal.loc[df_nan_cal["stress_label"] == 1, "heart_rate_bpm"] = np.nan
    with pytest.raises(ValueError, match="at least 2 required"):
        evaluate_participant(df_nan_cal, "PTest")

    # 3. Duplicate timestamps
    df_dup = df.copy()
    df_dup.loc[1, "timestamp"] = df_dup.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="duplicate timestamps"):
        evaluate_participant(df_dup, "PTest")


def test_reproducible_results(synthetic_wesad_participant):
    # Run evaluation twice on identical data
    p_metric1, s_metrics1, pred_df1 = evaluate_participant(synthetic_wesad_participant, "PTest")
    p_metric2, s_metrics2, pred_df2 = evaluate_participant(synthetic_wesad_participant, "PTest")

    assert p_metric1 == p_metric2
    assert s_metrics1 == s_metrics2
    pd.testing.assert_frame_equal(pred_df1, pred_df2)


def test_end_to_end_evaluation_execution(tmp_path):
    # Run end-to-end evaluation with output directed to tmp_path
    summary = run_wesad_anomaly_evaluation(output_dir=tmp_path)

    assert "aggregate_metrics" in summary
    assert "macro_average" in summary["aggregate_metrics"]
    assert "pooled" in summary["aggregate_metrics"]

    # Verify all 5 required files exist
    assert (tmp_path / "participant_metrics.csv").exists()
    assert (tmp_path / "signal_metrics.csv").exists()
    assert (tmp_path / "combined_predictions.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "evaluation_report.md").exists()

    # Validate participant_metrics.csv has exactly 4 rows (S2, S3, S4, S5)
    df_pm = pd.read_csv(tmp_path / "participant_metrics.csv")
    assert len(df_pm) == 4
    assert df_pm["participant_id"].tolist() == ["S2", "S3", "S4", "S5"]


def test_evaluation_report_interpretation(tmp_path):
    run_wesad_anomaly_evaluation(output_dir=tmp_path)

    report_text = (tmp_path / "evaluation_report.md").read_text(encoding="utf-8")
    df_sm = pd.read_csv(tmp_path / "signal_metrics.csv")

    # (1) EDA observations
    assert (
        "S4 and S5 achieved 100% EDA precision and recall; S2 and S3 produced no EDA anomaly predictions."
        in report_text
    )

    # Verify actual metrics in signal_metrics.csv
    s4_eda = df_sm[(df_sm["participant_id"] == "S4") & (df_sm["signal"] == "eda_microsiemens")].iloc[0]
    s5_eda = df_sm[(df_sm["participant_id"] == "S5") & (df_sm["signal"] == "eda_microsiemens")].iloc[0]
    s2_eda = df_sm[(df_sm["participant_id"] == "S2") & (df_sm["signal"] == "eda_microsiemens")].iloc[0]
    s3_eda = df_sm[(df_sm["participant_id"] == "S3") & (df_sm["signal"] == "eda_microsiemens")].iloc[0]

    assert s4_eda["precision"] == 1.0 and s4_eda["recall"] == 1.0
    assert s5_eda["precision"] == 1.0 and s5_eda["recall"] == 1.0
    assert s2_eda["true_positives"] == 0 and s2_eda["false_positives"] == 0
    assert s3_eda["true_positives"] == 0 and s3_eda["false_positives"] == 0

    # (2) Heart-rate observation
    assert "S4 HR produced 253 true positives and 3 false positives" in report_text
    s4_hr = df_sm[(df_sm["participant_id"] == "S4") & (df_sm["signal"] == "heart_rate_bpm")].iloc[0]
    assert s4_hr["true_positives"] == 253
    assert s4_hr["false_positives"] == 3

    # (3) Unsupported claim removed
    assert "substantially higher recall" not in report_text
    assert "substantially improves recall" not in report_text
    assert "Multi-Signal Synergy" not in report_text

    # (4) Dominant signal variation finding present
    assert "dominant anomaly-detection signal varies across participants" in report_text
