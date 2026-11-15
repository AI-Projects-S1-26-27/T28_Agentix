"""Tests for multi-signal personalized anomaly detection."""

import numpy as np
import pandas as pd
import pytest

from ai.baseline.multisignal_detector import detect_multisignal_anomalies


@pytest.fixture
def baseline_data():
    return pd.DataFrame({
        "heart_rate": [70.0, 72.0, 71.0, 69.0, 70.0, 71.0, 72.0, 70.0],
        "eda": [0.50, 0.52, 0.49, 0.51, 0.50, 0.52, 0.48, 0.50],
        "stress_label": [0, 0, 0, 0, 0, 0, 0, 0],
        "participant_id": ["P1"] * 8,
    })


@pytest.fixture
def normal_readings():
    return pd.DataFrame({
        "heart_rate": [70.5, 71.0],
        "eda": [0.50, 0.51],
        "stress_label": [0, 0],
        "participant_id": ["P1", "P1"],
    })


def test_normal_readings(baseline_data, normal_readings):
    result = detect_multisignal_anomalies(
        readings=normal_readings,
        baseline_readings=baseline_data,
        feature_columns=["heart_rate", "eda"],
    )

    # Check that original columns are preserved
    assert "heart_rate" in result.columns
    assert "eda" in result.columns
    assert "stress_label" in result.columns
    assert "participant_id" in result.columns

    # Check generated column names
    for sig in ["heart_rate", "eda"]:
        assert f"{sig}_baseline_mean" in result.columns
        assert f"{sig}_baseline_std" in result.columns
        assert f"{sig}_baseline_difference" in result.columns
        assert f"{sig}_baseline_z_score" in result.columns
        assert f"{sig}_detected_anomaly" in result.columns
        assert f"{sig}_evaluation_status" in result.columns

    assert "anomaly_count" in result.columns
    assert "overall_anomaly" in result.columns
    assert "unevaluable_count" in result.columns
    assert "overall_status" in result.columns

    # Normal readings should not trigger anomalies
    assert (result["heart_rate_detected_anomaly"] == 0).all()
    assert (result["eda_detected_anomaly"] == 0).all()
    assert (result["anomaly_count"] == 0).all()
    assert (result["overall_anomaly"] == 0).all()
    assert (result["unevaluable_count"] == 0).all()
    assert (result["overall_status"] == "normal").all()


def test_anomalous_heart_rate(baseline_data):
    readings = pd.DataFrame({
        "heart_rate": [110.0, 40.0],  # high and low heart rate anomalies
        "eda": [0.50, 0.51],          # normal eda
    })

    result = detect_multisignal_anomalies(
        readings=readings,
        baseline_readings=baseline_data,
        feature_columns=["heart_rate", "eda"],
        thresholds=3.0,
    )

    assert result["heart_rate_detected_anomaly"].tolist() == [1, 1]
    assert result["eda_detected_anomaly"].tolist() == [0, 0]
    assert result["anomaly_count"].tolist() == [1, 1]
    assert result["overall_anomaly"].tolist() == [1, 1]


def test_anomalous_eda(baseline_data):
    readings = pd.DataFrame({
        "heart_rate": [70.5, 71.0],  # normal heart rate
        "eda": [2.5, 3.0],           # highly anomalous eda
    })

    result = detect_multisignal_anomalies(
        readings=readings,
        baseline_readings=baseline_data,
        feature_columns=["heart_rate", "eda"],
        thresholds=3.0,
    )

    assert result["heart_rate_detected_anomaly"].tolist() == [0, 0]
    assert result["eda_detected_anomaly"].tolist() == [1, 1]
    assert result["anomaly_count"].tolist() == [1, 1]
    assert result["overall_anomaly"].tolist() == [1, 1]


def test_multiple_simultaneous_anomalies(baseline_data):
    readings = pd.DataFrame({
        "heart_rate": [120.0, 71.0],
        "eda": [3.5, 0.50],
    })

    result = detect_multisignal_anomalies(
        readings=readings,
        baseline_readings=baseline_data,
        feature_columns=["heart_rate", "eda"],
    )

    # First row has 2 anomalies, second has 0
    assert result.loc[0, "heart_rate_detected_anomaly"] == 1
    assert result.loc[0, "eda_detected_anomaly"] == 1
    assert result.loc[0, "anomaly_count"] == 2
    assert result.loc[0, "overall_anomaly"] == 1

    assert result.loc[1, "heart_rate_detected_anomaly"] == 0
    assert result.loc[1, "eda_detected_anomaly"] == 0
    assert result.loc[1, "anomaly_count"] == 0
    assert result.loc[1, "overall_anomaly"] == 0


def test_missing_values_in_one_signal(baseline_data):
    # Readings with missing values in EDA or heart_rate
    readings = pd.DataFrame({
        "heart_rate": [115.0, np.nan, 71.0],
        "eda": [np.nan, 3.5, 0.50],
    })

    result = detect_multisignal_anomalies(
        readings=readings,
        baseline_readings=baseline_data,
        feature_columns=["heart_rate", "eda"],
    )

    # Row 0: anomalous heart rate, missing eda
    assert result.loc[0, "heart_rate_detected_anomaly"] == 1
    assert np.isnan(result.loc[0, "eda_baseline_difference"])
    assert np.isnan(result.loc[0, "eda_baseline_z_score"])
    assert result.loc[0, "eda_detected_anomaly"] == 0
    assert result.loc[0, "anomaly_count"] == 1
    assert result.loc[0, "overall_anomaly"] == 1

    # Row 1: missing heart rate, anomalous eda
    assert np.isnan(result.loc[1, "heart_rate_baseline_difference"])
    assert np.isnan(result.loc[1, "heart_rate_baseline_z_score"])
    assert result.loc[1, "heart_rate_detected_anomaly"] == 0
    assert result.loc[1, "eda_detected_anomaly"] == 1
    assert result.loc[1, "anomaly_count"] == 1
    assert result.loc[1, "overall_anomaly"] == 1

    # Row 2: both normal
    assert result.loc[2, "anomaly_count"] == 0
    assert result.loc[2, "overall_anomaly"] == 0


def test_missing_values_in_baseline_handled_independently():
    # Baseline has missing EDA in several rows, but still >= 2 valid observations
    baseline = pd.DataFrame({
        "heart_rate": [70.0, 72.0, 71.0, 69.0],
        "eda": [0.50, np.nan, np.nan, 0.52],
    })
    readings = pd.DataFrame({
        "heart_rate": [71.0],
        "eda": [0.51],
    })

    result = detect_multisignal_anomalies(
        readings=readings,
        baseline_readings=baseline,
        feature_columns=["heart_rate", "eda"],
    )

    # Heart rate baseline mean should use all 4 readings
    assert result.loc[0, "heart_rate_baseline_mean"] == pytest.approx(70.5)
    # EDA baseline mean should use the 2 valid readings: (0.50 + 0.52)/2 = 0.51
    assert result.loc[0, "eda_baseline_mean"] == pytest.approx(0.51)
    assert result.loc[0, "anomaly_count"] == 0


def test_missing_feature_columns(baseline_data, normal_readings):
    # Column missing from readings
    with pytest.raises(ValueError, match="Selected feature columns missing"):
        detect_multisignal_anomalies(
            readings=normal_readings,
            baseline_readings=baseline_data,
            feature_columns=["heart_rate", "nonexistent_signal"],
        )

    # Column missing from baseline
    incomplete_baseline = baseline_data.drop(columns=["eda"])
    with pytest.raises(ValueError, match="Selected feature columns missing"):
        detect_multisignal_anomalies(
            readings=normal_readings,
            baseline_readings=incomplete_baseline,
            feature_columns=["heart_rate", "eda"],
        )

    # Empty feature_columns
    with pytest.raises(ValueError, match="feature_columns must not be empty"):
        detect_multisignal_anomalies(
            readings=normal_readings,
            baseline_readings=baseline_data,
            feature_columns=[],
        )

    # None feature_columns
    with pytest.raises(ValueError, match="explicitly specified"):
        detect_multisignal_anomalies(
            readings=normal_readings,
            baseline_readings=baseline_data,
            feature_columns=None,  # type: ignore
        )


def test_insufficient_baseline_observations(normal_readings):
    # Baseline with only 1 observation
    insufficient_baseline = pd.DataFrame({
        "heart_rate": [70.0],
        "eda": [0.50],
    })

    with pytest.raises(ValueError, match="at least two valid observations"):
        detect_multisignal_anomalies(
            readings=normal_readings,
            baseline_readings=insufficient_baseline,
            feature_columns=["heart_rate", "eda"],
        )

    # Baseline with 2 rows, but one signal only has 1 non-null observation
    sparse_baseline = pd.DataFrame({
        "heart_rate": [70.0, 72.0],
        "eda": [0.50, np.nan],
    })

    with pytest.raises(ValueError, match="at least two valid observations"):
        detect_multisignal_anomalies(
            readings=normal_readings,
            baseline_readings=sparse_baseline,
            feature_columns=["heart_rate", "eda"],
        )

    # Empty DataFrames
    with pytest.raises(ValueError, match="DataFrame is empty"):
        detect_multisignal_anomalies(
            readings=pd.DataFrame(),
            baseline_readings=insufficient_baseline,
            feature_columns=["heart_rate"],
        )

    with pytest.raises(ValueError, match="DataFrame is empty"):
        detect_multisignal_anomalies(
            readings=normal_readings,
            baseline_readings=pd.DataFrame(),
            feature_columns=["heart_rate"],
        )


def test_infinite_values(baseline_data, normal_readings):
    # Infinite value in readings
    inf_readings = pd.DataFrame({
        "heart_rate": [70.0, np.inf],
        "eda": [0.50, 0.51],
    })
    with pytest.raises(ValueError, match="contains infinite values"):
        detect_multisignal_anomalies(
            readings=inf_readings,
            baseline_readings=baseline_data,
            feature_columns=["heart_rate", "eda"],
        )

    neg_inf_readings = pd.DataFrame({
        "heart_rate": [70.0, -np.inf],
        "eda": [0.50, 0.51],
    })
    with pytest.raises(ValueError, match="contains infinite values"):
        detect_multisignal_anomalies(
            readings=neg_inf_readings,
            baseline_readings=baseline_data,
            feature_columns=["heart_rate", "eda"],
        )

    # Infinite value in baseline
    inf_baseline = pd.DataFrame({
        "heart_rate": [70.0, 72.0, np.inf],
        "eda": [0.50, 0.51, 0.52],
    })
    with pytest.raises(ValueError, match="contains infinite values"):
        detect_multisignal_anomalies(
            readings=normal_readings,
            baseline_readings=inf_baseline,
            feature_columns=["heart_rate", "eda"],
        )


def test_invalid_thresholds(baseline_data, normal_readings):
    # Non-positive thresholds
    with pytest.raises(ValueError, match="positive finite number"):
        detect_multisignal_anomalies(
            normal_readings, baseline_data, ["heart_rate"], thresholds=0
        )

    with pytest.raises(ValueError, match="positive finite number"):
        detect_multisignal_anomalies(
            normal_readings, baseline_data, ["heart_rate"], thresholds=-1.5
        )

    # Non-finite thresholds
    with pytest.raises(ValueError, match="positive finite number"):
        detect_multisignal_anomalies(
            normal_readings, baseline_data, ["heart_rate"], thresholds=np.nan
        )

    with pytest.raises(ValueError, match="positive finite number"):
        detect_multisignal_anomalies(
            normal_readings, baseline_data, ["heart_rate"], thresholds=np.inf
        )

    # Invalid dictionary thresholds
    with pytest.raises(ValueError, match="positive finite number"):
        detect_multisignal_anomalies(
            normal_readings,
            baseline_data,
            ["heart_rate", "eda"],
            thresholds={"heart_rate": 2.5, "eda": -0.5},
        )

    # Unknown signal in threshold mapping
    with pytest.raises(ValueError, match="unknown signal"):
        detect_multisignal_anomalies(
            normal_readings,
            baseline_data,
            ["heart_rate"],
            thresholds={"unknown_signal": 2.0},
        )

    # Invalid type for threshold
    with pytest.raises(ValueError, match="thresholds must be None"):
        detect_multisignal_anomalies(
            normal_readings, baseline_data, ["heart_rate"], thresholds="high"
        )


def test_zero_variance_baseline():
    # Signal with zero variance in baseline
    baseline = pd.DataFrame({
        "heart_rate": [70.0, 70.0, 70.0, 70.0],  # zero variance
        "eda": [0.50, 0.52, 0.48, 0.50],        # normal variance
    })
    readings = pd.DataFrame({
        "heart_rate": [70.0, 85.0],
        "eda": [0.50, 2.5],  # second row has eda anomaly
    })

    result = detect_multisignal_anomalies(
        readings=readings,
        baseline_readings=baseline,
        feature_columns=["heart_rate", "eda"],
    )

    # For zero-variance signal:
    assert result["heart_rate_baseline_std"].iloc[0] == 0.0
    assert result["heart_rate_baseline_difference"].tolist() == [0.0, 15.0]
    assert result["heart_rate_baseline_z_score"].isna().all()
    assert (result["heart_rate_detected_anomaly"] == 0).all()

    # The other signal is unaffected
    assert result.loc[0, "eda_detected_anomaly"] == 0
    assert result.loc[1, "eda_detected_anomaly"] == 1
    assert result.loc[0, "anomaly_count"] == 0
    assert result.loc[1, "anomaly_count"] == 1


def test_custom_per_signal_thresholds(baseline_data):
    # heart_rate z-score for 73.0 is ~2.24 (mean=70.625, std=1.06)
    readings = pd.DataFrame({
        "heart_rate": [73.0],
        "eda": [0.50],
    })

    # With threshold 3.0: not flagged
    res_default = detect_multisignal_anomalies(
        readings, baseline_data, ["heart_rate", "eda"]
    )
    assert res_default.loc[0, "heart_rate_detected_anomaly"] == 0

    # With custom threshold 2.0: flagged
    res_custom = detect_multisignal_anomalies(
        readings,
        baseline_data,
        ["heart_rate", "eda"],
        thresholds={"heart_rate": 2.0, "eda": 3.0},
    )
    assert res_custom.loc[0, "heart_rate_detected_anomaly"] == 1
    assert res_custom.loc[0, "eda_detected_anomaly"] == 0
    assert res_custom.loc[0, "anomaly_count"] == 1


def test_input_type_validation(normal_readings, baseline_data):
    # Not DataFrames
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        detect_multisignal_anomalies(
            readings=[1, 2, 3],  # type: ignore
            baseline_readings=baseline_data,
            feature_columns=["heart_rate"],
        )

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        detect_multisignal_anomalies(
            readings=normal_readings,
            baseline_readings={"heart_rate": [70, 72]},  # type: ignore
            feature_columns=["heart_rate"],
        )

    # Non-sequence feature_columns
    with pytest.raises(TypeError, match="must be a sequence"):
        detect_multisignal_anomalies(
            readings=normal_readings,
            baseline_readings=baseline_data,
            feature_columns=123,  # type: ignore
        )

    # String as feature_columns
    with pytest.raises(TypeError, match="must be a sequence"):
        detect_multisignal_anomalies(
            readings=normal_readings,
            baseline_readings=baseline_data,
            feature_columns="heart_rate",  # type: ignore
        )


def test_zero_variance_baseline_with_large_difference():
    baseline = pd.DataFrame({
        "heart_rate": [70.0, 70.0, 70.0, 70.0],  # zero variance
        "eda": [0.50, 0.52, 0.48, 0.50],
    })
    readings = pd.DataFrame({
        "heart_rate": [120.0],  # large baseline difference of +50 bpm
        "eda": [0.51],
    })

    result = detect_multisignal_anomalies(
        readings=readings,
        baseline_readings=baseline,
        feature_columns=["heart_rate", "eda"],
    )

    assert result.loc[0, "heart_rate_baseline_mean"] == pytest.approx(70.0)
    assert result.loc[0, "heart_rate_baseline_std"] == 0.0
    assert result.loc[0, "heart_rate_baseline_difference"] == pytest.approx(50.0)
    assert np.isnan(result.loc[0, "heart_rate_baseline_z_score"])
    assert result.loc[0, "heart_rate_detected_anomaly"] == 0
    assert result.loc[0, "heart_rate_evaluation_status"] == "zero_variance_baseline"

    assert result.loc[0, "eda_evaluation_status"] == "evaluated"
    assert result.loc[0, "eda_detected_anomaly"] == 0

    assert result.loc[0, "unevaluable_count"] == 1
    assert result.loc[0, "overall_status"] == "normal"


def test_missing_current_reading(baseline_data):
    readings = pd.DataFrame({
        "heart_rate": [np.nan],
        "eda": [0.51],
    })

    result = detect_multisignal_anomalies(
        readings=readings,
        baseline_readings=baseline_data,
        feature_columns=["heart_rate", "eda"],
    )

    assert np.isnan(result.loc[0, "heart_rate_baseline_difference"])
    assert np.isnan(result.loc[0, "heart_rate_baseline_z_score"])
    assert result.loc[0, "heart_rate_detected_anomaly"] == 0
    assert result.loc[0, "heart_rate_evaluation_status"] == "missing_reading"

    assert result.loc[0, "eda_evaluation_status"] == "evaluated"
    assert result.loc[0, "eda_detected_anomaly"] == 0

    assert result.loc[0, "unevaluable_count"] == 1
    assert result.loc[0, "overall_status"] == "normal"


def test_all_selected_signals_unevaluable():
    baseline = pd.DataFrame({
        "heart_rate": [70.0, 70.0, 70.0, 70.0],  # zero variance
        "eda": [0.50, 0.52, 0.48, 0.50],
    })
    readings = pd.DataFrame({
        "heart_rate": [80.0],    # zero variance baseline
        "eda": [np.nan],         # missing reading
    })

    result = detect_multisignal_anomalies(
        readings=readings,
        baseline_readings=baseline,
        feature_columns=["heart_rate", "eda"],
    )

    assert result.loc[0, "heart_rate_evaluation_status"] == "zero_variance_baseline"
    assert result.loc[0, "eda_evaluation_status"] == "missing_reading"
    assert result.loc[0, "heart_rate_detected_anomaly"] == 0
    assert result.loc[0, "eda_detected_anomaly"] == 0
    assert result.loc[0, "anomaly_count"] == 0
    assert result.loc[0, "overall_anomaly"] == 0
    assert result.loc[0, "unevaluable_count"] == 2
    assert result.loc[0, "overall_status"] == "unevaluable"


def test_one_signal_anomalous_and_another_unevaluable(baseline_data):
    readings = pd.DataFrame({
        "heart_rate": [115.0],  # anomalous
        "eda": [np.nan],        # unevaluable (missing)
    })

    result = detect_multisignal_anomalies(
        readings=readings,
        baseline_readings=baseline_data,
        feature_columns=["heart_rate", "eda"],
    )

    assert result.loc[0, "heart_rate_evaluation_status"] == "evaluated"
    assert result.loc[0, "heart_rate_detected_anomaly"] == 1
    assert result.loc[0, "eda_evaluation_status"] == "missing_reading"
    assert result.loc[0, "eda_detected_anomaly"] == 0
    assert result.loc[0, "anomaly_count"] == 1
    assert result.loc[0, "overall_anomaly"] == 1
    assert result.loc[0, "unevaluable_count"] == 1
    assert result.loc[0, "overall_status"] == "partially_evaluated_anomaly"


def test_all_signals_evaluated_normally(baseline_data, normal_readings):
    result = detect_multisignal_anomalies(
        readings=normal_readings,
        baseline_readings=baseline_data,
        feature_columns=["heart_rate", "eda"],
    )

    for sig in ["heart_rate", "eda"]:
        assert (result[f"{sig}_evaluation_status"] == "evaluated").all()
        assert (result[f"{sig}_detected_anomaly"] == 0).all()

    assert (result["unevaluable_count"] == 0).all()
    assert (result["anomaly_count"] == 0).all()
    assert (result["overall_anomaly"] == 0).all()
    assert (result["overall_status"] == "normal").all()


def test_anomaly_detected_when_all_evaluated(baseline_data):
    readings = pd.DataFrame({
        "heart_rate": [110.0],
        "eda": [0.50],
    })

    result = detect_multisignal_anomalies(
        readings=readings,
        baseline_readings=baseline_data,
        feature_columns=["heart_rate", "eda"],
    )

    assert result.loc[0, "heart_rate_evaluation_status"] == "evaluated"
    assert result.loc[0, "eda_evaluation_status"] == "evaluated"
    assert result.loc[0, "heart_rate_detected_anomaly"] == 1
    assert result.loc[0, "eda_detected_anomaly"] == 0
    assert result.loc[0, "unevaluable_count"] == 0
    assert result.loc[0, "anomaly_count"] == 1
    assert result.loc[0, "overall_anomaly"] == 1
    assert result.loc[0, "overall_status"] == "anomaly_detected"
