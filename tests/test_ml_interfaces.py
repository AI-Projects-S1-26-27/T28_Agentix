"""Unit tests for the HealthSensing ML integration interface.

Tests cover:
- Calibration service with all supported strategies (full, settled, quality_gated, signal_specific)
- Unstable calibration status preservation (no silent promotion to stable)
- Insufficient calibration data handling (< 2 observations and < 60 observations)
- Multi-signal inference: normal readings, anomaly detection, missing sensor values
- Zero-variance baseline handling without division by zero
- Evaluation completeness: complete, partial, and unevaluable
- Random Forest window inference: model unavailable, insufficient samples, timestamp gap, probability ordering
- JSON serialization safety: NaN -> null, no raw arrays leaking into metadata
- Stable feature names and order
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai.interfaces import (
    BaselineContext,
    BaselineQualityResult,
    MLAnalysisResponse,
    ReadingResult,
    RFWindowResult,
    SignalResult,
    analyze_readings,
    analyze_rf_windows,
    get_model_metadata,
    prepare_baseline,
    run_ml_analysis,
    sanitize_for_json,
)
from ai.models.random_forest_model import (
    DEFAULT_PHYSIOLOGICAL_COLUMNS,
    StressRandomForestClassifier,
)


@pytest.fixture
def sample_calibration_df() -> pd.DataFrame:
    """Create a stationary 600-observation calibration dataframe."""
    np.random.seed(42)
    n = 600
    timestamps = pd.date_range("2026-01-01 08:00:00", periods=n, freq="1s", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": 70.0 + np.random.normal(0, 1.5, n),
        "eda_microsiemens": 1.2 + np.random.normal(0, 0.05, n),
        "body_temperature_c": 34.0 + np.random.normal(0, 0.08, n),
    })


@pytest.fixture
def acclimating_calibration_df() -> pd.DataFrame:
    """Create an acclimating calibration dataset where EDA has a steep initial decay."""
    np.random.seed(42)
    n = 600
    timestamps = pd.date_range("2026-01-01 08:00:00", periods=n, freq="1s", tz="UTC")
    # EDA drops from 5.0 to 1.0 over first 300 samples, then stabilizes around 1.0
    eda = np.zeros(n)
    eda[:300] = np.linspace(5.0, 1.0, 300) + np.random.normal(0, 0.05, 300)
    eda[300:] = 1.0 + np.random.normal(0, 0.05, 300)

    return pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": 70.0 + np.random.normal(0, 1.5, n),
        "eda_microsiemens": eda,
        "body_temperature_c": 34.0 + np.random.normal(0, 0.08, n),
    })


@pytest.fixture
def current_readings_df() -> pd.DataFrame:
    """Create 120 current sensor observations at 1 Hz within resting bounds."""
    np.random.seed(123)
    n = 120
    timestamps = pd.date_range("2026-01-01 10:00:00", periods=n, freq="1s", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": 70.2 + np.random.normal(0, 0.4, n),
        "eda_microsiemens": 1.20 + np.random.normal(0, 0.015, n),
        "body_temperature_c": 34.02 + np.random.normal(0, 0.02, n),
    })


# ============================================================================
# Calibration Service Tests
# ============================================================================


def test_prepare_baseline_valid_strategies(sample_calibration_df: pd.DataFrame):
    """Verify that all four strategies execute and return a valid BaselineContext."""
    for strat in ("full", "settled", "quality_gated", "signal_specific"):
        ctx = prepare_baseline(
            calibration_readings=sample_calibration_df,
            feature_columns=list(DEFAULT_PHYSIOLOGICAL_COLUMNS),
            strategy=strat,
        )
        assert isinstance(ctx, BaselineContext)
        assert ctx.strategy == strat
        assert len(ctx.get_baseline_df()) > 0
        assert set(ctx.feature_columns) == set(DEFAULT_PHYSIOLOGICAL_COLUMNS)

        for col in DEFAULT_PHYSIOLOGICAL_COLUMNS:
            assert col in ctx.quality_assessments
            res = ctx.quality_assessments[col]
            assert isinstance(res, BaselineQualityResult)
            assert res.status in ("stable", "unstable", "insufficient_data")

        # Verify no raw sensor dataframe leaks into to_dict()
        ctx_dict = ctx.to_dict()
        assert "_baseline_df" not in ctx_dict
        assert "baseline_statistics" in ctx_dict
        assert "selection_metadata" in ctx_dict


def test_unstable_calibration_status_preservation(acclimating_calibration_df: pd.DataFrame):
    """Verify that if a signal remains unstable or triggers fallback, unstable is preserved."""
    # Under Strategy D with acclimating data
    ctx = prepare_baseline(
        calibration_readings=acclimating_calibration_df,
        feature_columns=["heart_rate_bpm", "eda_microsiemens", "body_temperature_c"],
        strategy="signal_specific",
        min_observations=480,  # Total len is 600, so suffix can be at most 120s from start
    )
    # The EDA channel acclimated from 5.0 to 1.0 across first 300s.
    # An offset >= 300 leaves only 300s, which is < 480 min_observations.
    # Therefore, Strategy D must select fallback and keep status 'unstable'.
    eda_qual = ctx.quality_assessments["eda_microsiemens"]
    assert eda_qual.status == "unstable"
    assert eda_qual.acclimation_detected is True
    # Confirm it was not silently converted to stable
    assert ctx.selection_metadata["signals"]["eda_microsiemens"]["status"] == "unstable"


def test_insufficient_calibration_data():
    """Verify handling when calibration data contains fewer than 60 observations or < 2 observations."""
    np.random.seed(42)
    # 1. Fewer than 60 observations (e.g. 30 samples)
    df_short = pd.DataFrame({
        "heart_rate_bpm": [70.0] * 30,
        "eda_microsiemens": [1.0] * 30,
        "body_temperature_c": [34.0] * 30,
    })
    ctx_short = prepare_baseline(
        calibration_readings=df_short,
        feature_columns=list(DEFAULT_PHYSIOLOGICAL_COLUMNS),
        strategy="signal_specific",
    )
    # When < 60 observations, quality assessment flags insufficient_data
    for col in DEFAULT_PHYSIOLOGICAL_COLUMNS:
        assert ctx_short.quality_assessments[col].status == "insufficient_data"

    # 2. Fewer than 2 observations (e.g. 1 sample)
    df_one = pd.DataFrame({
        "heart_rate_bpm": [70.0],
        "eda_microsiemens": [1.0],
        "body_temperature_c": [34.0],
    })
    ctx_one = prepare_baseline(
        calibration_readings=df_one,
        feature_columns=list(DEFAULT_PHYSIOLOGICAL_COLUMNS),
        strategy="signal_specific",
    )
    assert len(ctx_one.insufficient_signals) == 3
    for col in DEFAULT_PHYSIOLOGICAL_COLUMNS:
        assert ctx_one.quality_assessments[col].status == "insufficient_data"

    # 3. Empty dataframe raises ValueError
    with pytest.raises(ValueError, match="empty"):
        prepare_baseline(
            calibration_readings=pd.DataFrame(),
            feature_columns=list(DEFAULT_PHYSIOLOGICAL_COLUMNS),
        )


# ============================================================================
# Inference Service Tests
# ============================================================================


def test_normal_multisignal_readings(
    sample_calibration_df: pd.DataFrame,
    current_readings_df: pd.DataFrame,
):
    """Verify that normal sensor readings produce normal status and complete evaluation."""
    ctx = prepare_baseline(sample_calibration_df, strategy="signal_specific")
    results = analyze_readings(readings=current_readings_df, baseline_context=ctx)

    assert len(results) == len(current_readings_df)
    for r in results:
        assert isinstance(r, ReadingResult)
        assert r.overall_anomaly == 0
        assert r.anomaly_count == 0
        assert r.unevaluable_count == 0
        assert r.overall_status == "normal"
        assert r.evaluation_completeness == "complete"
        for col in DEFAULT_PHYSIOLOGICAL_COLUMNS:
            sig = r.signals[col]
            assert isinstance(sig, SignalResult)
            assert sig.anomaly == 0
            assert sig.evaluation_status == "evaluated"
            assert sig.z_score is not None
            assert abs(sig.z_score) <= sig.threshold


def test_anomaly_detection(sample_calibration_df: pd.DataFrame):
    """Verify that physiological spikes exceeding threshold are correctly flagged as anomalies."""
    ctx = prepare_baseline(sample_calibration_df, strategy="signal_specific")

    # HR baseline is ~70 +/- 1.5 bpm. A reading of 120 bpm is > 30 std devs away.
    df_anom = pd.DataFrame({
        "timestamp": ["2026-01-01T12:00:00Z"],
        "heart_rate_bpm": [120.0],
        "eda_microsiemens": [10.0],
        "body_temperature_c": [34.0],
    })
    results = analyze_readings(readings=df_anom, baseline_context=ctx)
    assert len(results) == 1
    r = results[0]
    assert r.overall_anomaly == 1
    assert r.anomaly_count >= 2
    assert r.overall_status == "anomaly_detected"
    assert r.evaluation_completeness == "complete"
    assert r.signals["heart_rate_bpm"].anomaly == 1
    assert r.signals["eda_microsiemens"].anomaly == 1


def test_missing_sensor_values(sample_calibration_df: pd.DataFrame):
    """Verify that NaN sensor readings do not crash the detector and are marked missing_reading."""
    ctx = prepare_baseline(sample_calibration_df, strategy="signal_specific")

    df_missing = pd.DataFrame({
        "timestamp": ["2026-01-01T12:00:00Z"],
        "heart_rate_bpm": [70.0],
        "eda_microsiemens": [np.nan],  # Missing EDA
        "body_temperature_c": [34.0],
    })
    results = analyze_readings(readings=df_missing, baseline_context=ctx)
    r = results[0]
    assert r.unevaluable_count == 1
    assert r.evaluation_completeness == "partial"
    assert r.signals["eda_microsiemens"].evaluation_status == "missing_reading"
    assert r.signals["eda_microsiemens"].value is None
    assert r.signals["eda_microsiemens"].z_score is None
    assert r.signals["eda_microsiemens"].anomaly == 0


def test_zero_variance_baseline():
    """Verify that constant calibration data (zero variance) produces zero_variance_baseline status."""
    n = 200
    df_const = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="1s", tz="UTC"),
        "heart_rate_bpm": [72.0] * n,  # Zero variance
        "eda_microsiemens": [1.5] * n,
        "body_temperature_c": [34.0] * n,
    })
    ctx = prepare_baseline(df_const, strategy="full")

    current = pd.DataFrame({
        "timestamp": ["2026-01-01T12:00:00Z"],
        "heart_rate_bpm": [75.0],
        "eda_microsiemens": [1.5],
        "body_temperature_c": [34.0],
    })
    results = analyze_readings(readings=current, baseline_context=ctx)
    r = results[0]
    for col in DEFAULT_PHYSIOLOGICAL_COLUMNS:
        sig = r.signals[col]
        assert sig.evaluation_status == "zero_variance_baseline"
        assert sig.baseline_std == 0.0
        assert sig.z_score is None
        assert sig.anomaly == 0
    assert r.evaluation_completeness == "none"
    assert r.overall_status == "unevaluable"


def test_insufficient_baseline_data_propagation():
    """Verify that a channel with insufficient calibration data propagates cleanly to inference."""
    df_cal = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=100, freq="1s", tz="UTC"),
        "heart_rate_bpm": [70.0 + np.random.normal(0, 1) for _ in range(100)],
        "eda_microsiemens": [np.nan] * 100,  # Completely missing in calibration
        "body_temperature_c": [34.0 + np.random.normal(0, 0.1) for _ in range(100)],
    })
    ctx = prepare_baseline(df_cal, strategy="signal_specific")
    assert "eda_microsiemens" in ctx.insufficient_signals

    current = pd.DataFrame({
        "timestamp": ["2026-01-01T12:00:00Z"],
        "heart_rate_bpm": [71.0],
        "eda_microsiemens": [1.5],
        "body_temperature_c": [34.0],
    })
    results = analyze_readings(readings=current, baseline_context=ctx)
    r = results[0]
    assert r.signals["eda_microsiemens"].evaluation_status == "insufficient_baseline_data"
    assert r.signals["eda_microsiemens"].baseline_mean is None
    assert r.signals["eda_microsiemens"].anomaly == 0
    assert r.unevaluable_count == 1
    assert r.evaluation_completeness == "partial"


# ============================================================================
# Random Forest Window Inference Tests
# ============================================================================


def test_rf_model_unavailable(current_readings_df: pd.DataFrame):
    """Verify that omitting the RF model reports model_unavailable without error."""
    results = analyze_rf_windows(
        readings=current_readings_df,
        rf_model=None,
    )
    assert len(results) > 0
    for w in results:
        assert isinstance(w, RFWindowResult)
        assert w.status == "model_unavailable"
        assert w.predicted_class is None
        assert w.probabilities is None


def test_rf_insufficient_samples():
    """Verify that recordings with fewer than 60 observations report insufficient_samples."""
    df_short = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=45, freq="1s", tz="UTC"),
        "heart_rate_bpm": [70.0] * 45,
        "eda_microsiemens": [1.0] * 45,
        "body_temperature_c": [34.0] * 45,
    })
    results = analyze_rf_windows(readings=df_short, rf_model=None)
    assert len(results) == 1
    assert results[0].status == "insufficient_samples"
    assert results[0].sample_count == 45


def test_rf_timestamp_gap_handling():
    """Verify that windows are never constructed across a timestamp gap."""
    # 50 samples, then 10-second gap, then 50 samples (neither block reaches 60s)
    ts1 = pd.date_range("2026-01-01 10:00:00", periods=50, freq="1s", tz="UTC")
    ts2 = pd.date_range("2026-01-01 10:01:00", periods=50, freq="1s", tz="UTC")
    df_gap = pd.DataFrame({
        "timestamp": ts1.append(ts2),
        "heart_rate_bpm": [70.0] * 100,
        "eda_microsiemens": [1.0] * 100,
        "body_temperature_c": [34.0] * 100,
    })
    results = analyze_rf_windows(readings=df_gap, rf_model=None)
    assert len(results) == 1
    assert results[0].status == "timestamp_gap"


def test_rf_window_classification_and_probability_ordering():
    """Verify that a fitted model classifies 60s windows with guaranteed class column ordering."""
    # Train dummy model on 18 statistical features
    np.random.seed(42)
    expected_cols = [
        f"{col}_{stat}"
        for col in DEFAULT_PHYSIOLOGICAL_COLUMNS
        for stat in ("mean", "std", "min", "max", "range", "trend")
    ]
    X_train = pd.DataFrame(np.random.randn(20, 18), columns=expected_cols)
    y_train = pd.Series([0] * 10 + [1] * 10)

    model = StressRandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # 120 continuous observations -> (120 - 60)//30 + 1 = 3 windows
    ts = pd.date_range("2026-01-01 10:00:00", periods=120, freq="1s", tz="UTC")
    readings = pd.DataFrame({
        "timestamp": ts,
        "heart_rate_bpm": 70.0 + np.random.normal(0, 1, 120),
        "eda_microsiemens": 1.0 + np.random.normal(0, 0.1, 120),
        "body_temperature_c": 34.0 + np.random.normal(0, 0.05, 120),
    })

    windows = analyze_rf_windows(readings=readings, rf_model=model)
    assert len(windows) == 3

    for w in windows:
        assert w.status == "classified"
        assert w.sample_count == 60
        assert w.predicted_class in ("baseline", "stress")
        assert w.predicted_class_id in (0, 1)
        assert w.probabilities is not None
        assert "baseline_class_0" in w.probabilities
        assert "stress_class_1" in w.probabilities
        # Sum of probabilities should be approximately 1.0
        prob_sum = w.probabilities["baseline_class_0"] + w.probabilities["stress_class_1"]
        assert prob_sum == pytest.approx(1.0, abs=1e-3)


# ============================================================================
# JSON Serialization Safety Tests
# ============================================================================


def test_sanitize_for_json_conversions():
    """Verify that NaN and Inf are converted to None, and numpy types to native Python."""
    data = {
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "neg_inf": float("-inf"),
        "np_nan": np.nan,
        "np_int": np.int64(42),
        "np_float": np.float64(3.1415),
        "np_bool": np.bool_(True),
        "np_arr": np.array([1.0, np.nan, 2.0]),
        "timestamp": pd.Timestamp("2026-01-01 12:00:00", tz="UTC"),
    }
    sanitized = sanitize_for_json(data)
    assert sanitized["nan_val"] is None
    assert sanitized["inf_val"] is None
    assert sanitized["neg_inf"] is None
    assert sanitized["np_nan"] is None
    assert sanitized["np_int"] == 42
    assert isinstance(sanitized["np_int"], int)
    assert sanitized["np_float"] == pytest.approx(3.1415)
    assert isinstance(sanitized["np_float"], float)
    assert sanitized["np_bool"] is True
    assert isinstance(sanitized["np_bool"], bool)
    assert sanitized["np_arr"] == [1.0, None, 2.0]
    assert sanitized["timestamp"] == "2026-01-01T12:00:00+00:00"

    # Verify standard json.dumps succeeds without error
    json_str = json.dumps(sanitized)
    assert "NaN" not in json_str
    assert "Infinity" not in json_str
    roundtrip = json.loads(json_str)
    assert roundtrip["nan_val"] is None


def test_end_to_end_ml_analysis_response_serialization(
    sample_calibration_df: pd.DataFrame,
    current_readings_df: pd.DataFrame,
):
    """Verify that run_ml_analysis produces a valid, complete, and serializable MLAnalysisResponse."""
    ctx = prepare_baseline(sample_calibration_df, strategy="signal_specific")
    response = run_ml_analysis(
        readings=current_readings_df,
        baseline_context=ctx,
        rf_model=None,
        request_id="test_req_001",
    )

    assert isinstance(response, MLAnalysisResponse)
    assert response.schema_version == "1.0.0"
    assert response.request_id == "test_req_001"
    assert len(response.reading_results) == len(current_readings_df)
    assert len(response.rf_window_results) > 0

    # Test dictionary export
    resp_dict = response.to_dict()
    assert "baseline_metadata" in resp_dict
    assert "baseline_quality" in resp_dict
    assert "reading_results" in resp_dict
    assert "rf_window_results" in resp_dict
    assert "model_metadata" in resp_dict

    # Test JSON string serialization
    json_text = response.to_json(indent=2)
    assert isinstance(json_text, str)
    assert "NaN" not in json_text
    assert "Infinity" not in json_text

    # Validate json.loads roundtrip
    parsed = json.loads(json_text)
    assert parsed["schema_version"] == "1.0.0"
    assert parsed["request_id"] == "test_req_001"
    assert len(parsed["reading_results"]) == len(current_readings_df)


def test_model_metadata_structure():
    """Verify that get_model_metadata exposes documented governance fields without claims of clinical validation."""
    meta = get_model_metadata(rf_model=None)
    assert meta["model_available"] is False
    assert meta["model_identifier"] is None
    assert meta["window_size_observations"] == 60
    assert meta["stride_observations"] == 30
    assert len(meta["feature_schema"]) == 18
    assert meta["class_mapping"] == {"baseline_class_0": 0, "stress_class_1": 1}
    assert "LOPO" in meta["evaluation_protocol"]
    assert "Not clinically validated" in meta["deployment_note"]
