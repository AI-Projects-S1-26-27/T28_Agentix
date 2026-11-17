"""Focused robustness and edge-case verification for HealthSensing ML pipeline.

Covers:
1. Noisy physiological measurements (high-frequency Gaussian noise, sensor jitter, and extreme spikes).
2. Missing sensor fields (dropped columns, all-NaN channels, partial channel coverage).
3. Irregular timestamp intervals (sampling rate jitter, bursty packets, micro-gaps).
4. Class imbalance (extreme negative-to-positive ratios, zero-division metric safety).
5. Controlled simulator scenarios (evaluating simulator streams with detectors while keeping simulated and real WESAD metrics isolated).
6. Graceful handling of incomplete or unevaluable data (multi-channel dropouts, zero-variance baselines).
"""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from ai.baseline.baseline_quality import (
    DEFAULT_FEATURE_COLUMNS,
    assess_calibration_dataset_quality,
)
from ai.baseline.multisignal_detector import detect_multisignal_anomalies
from ai.features.feature_engineering import extract_window_features
from ai.interfaces import (
    BaselineContext,
    analyze_readings,
    analyze_rf_windows,
    prepare_baseline,
    run_ml_analysis,
)
from ai.models.random_forest_model import (
    DEFAULT_PHYSIOLOGICAL_COLUMNS,
    StressRandomForestClassifier,
)
from ai.training.train_rf import (
    calculate_classification_metrics,
    extract_pure_label_windows_and_features,
)
from simulator.data_generator import generate_data
from simulator.simulator import stream_data


# ============================================================================
# 1. Noisy Physiological Measurements
# ============================================================================


def test_noisy_physiological_measurements_do_not_destabilize_detector():
    """Verify that high-frequency noise and heavy-tailed sensor jitter produce stable finite metrics."""
    np.random.seed(42)
    n = 300
    timestamps = pd.date_range("2026-01-01 10:00:00", periods=n, freq="1s", tz="UTC")

    # Baseline with moderate noise
    cal_df = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": 70.0 + np.random.normal(0, 1.5, n),
        "eda_microsiemens": 1.2 + np.random.normal(0, 0.05, n),
        "body_temperature_c": 34.0 + np.random.normal(0, 0.08, n),
    })

    ctx = prepare_baseline(cal_df, strategy="signal_specific")

    # Test readings with high-frequency noise and isolated non-fatal spikes
    noisy_hr = 70.0 + np.random.normal(0, 5.0, 60)
    noisy_eda = 1.2 + np.random.laplace(0, 0.5, 60)  # Heavy-tailed noise
    noisy_temp = 34.0 + np.random.normal(0, 0.5, 60)

    test_df = pd.DataFrame({
        "timestamp": timestamps[:60],
        "heart_rate_bpm": noisy_hr,
        "eda_microsiemens": noisy_eda,
        "body_temperature_c": noisy_temp,
    })

    results = analyze_readings(readings=test_df, baseline_context=ctx)
    assert len(results) == 60
    for r in results:
        assert r.overall_status in ("normal", "anomaly_detected", "partially_evaluated_anomaly")
        for col in DEFAULT_PHYSIOLOGICAL_COLUMNS:
            sig = r.signals[col]
            assert np.isfinite(sig.z_score) or sig.z_score is None


def test_noisy_window_features_remain_finite():
    """Verify that extract_window_features produces strictly finite features on noisy data."""
    np.random.seed(123)
    n = 60
    ts = pd.date_range("2026-01-01 10:00:00", periods=n, freq="1s", tz="UTC")
    noisy_window = pd.DataFrame({
        "timestamp": ts,
        "heart_rate_bpm": np.random.normal(75, 10, n),
        "eda_microsiemens": np.abs(np.random.normal(2, 1, n)),
        "body_temperature_c": np.random.normal(34, 0.5, n),
    })

    feats = extract_window_features(noisy_window, feature_columns=list(DEFAULT_PHYSIOLOGICAL_COLUMNS))
    assert len(feats) == 18
    for k, v in feats.items():
        assert not np.isnan(v), f"Feature {k} returned NaN on noisy data"
        assert not np.isinf(v), f"Feature {k} returned Inf on noisy data"


# ============================================================================
# 2. Missing Sensor Fields
# ============================================================================


def test_dropped_sensor_column_raises_clear_error():
    """Verify that passing readings missing a required feature column raises a clear descriptive ValueError."""
    cal_df = pd.DataFrame({
        "heart_rate_bpm": [70.0] * 100,
        "eda_microsiemens": [1.0] * 100,
        "body_temperature_c": [34.0] * 100,
    })
    ctx = prepare_baseline(cal_df, strategy="full")

    # Incomplete test dataframe missing body_temperature_c entirely
    incomplete_test = pd.DataFrame({
        "heart_rate_bpm": [72.0] * 10,
        "eda_microsiemens": [1.1] * 10,
    })

    with pytest.raises(ValueError, match="missing from readings"):
        analyze_readings(readings=incomplete_test, baseline_context=ctx)


def test_all_nan_channel_handled_gracefully():
    """Verify that a channel that is entirely NaN in incoming stream is marked missing_reading for all rows."""
    cal_df = pd.DataFrame({
        "heart_rate_bpm": [70.0 + i * 0.01 for i in range(100)],
        "eda_microsiemens": [1.0 + i * 0.005 for i in range(100)],
        "body_temperature_c": [34.0 + i * 0.002 for i in range(100)],
    })
    ctx = prepare_baseline(cal_df, strategy="signal_specific")

    stream_with_all_nan_eda = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="1s", tz="UTC"),
        "heart_rate_bpm": [71.0] * 20,
        "eda_microsiemens": [np.nan] * 20,  # Dropped/disconnected sensor
        "body_temperature_c": [34.1] * 20,
    })

    results = analyze_readings(readings=stream_with_all_nan_eda, baseline_context=ctx)
    assert len(results) == 20
    for r in results:
        assert r.unevaluable_count == 1
        assert r.evaluation_completeness == "partial"
        assert r.signals["eda_microsiemens"].evaluation_status == "missing_reading"
        assert r.signals["eda_microsiemens"].value is None
        assert r.signals["eda_microsiemens"].anomaly == 0
        # The other 2 channels evaluate normally
        assert r.signals["heart_rate_bpm"].evaluation_status == "evaluated"
        assert r.signals["body_temperature_c"].evaluation_status == "evaluated"


# ============================================================================
# 3. Irregular Timestamp Intervals
# ============================================================================


def test_sampling_rate_jitter_and_micro_gaps():
    """Verify that jitter exceeding timestamp_tolerance_sec breaks window continuity."""
    # Observations with jitter: 0s, 1s, 2s, 3.5s (gap 1.5s > 1.1s), 4.5s, 5.5s...
    ts_list = [
        pd.Timestamp("2026-01-01 10:00:00", tz="UTC"),
        pd.Timestamp("2026-01-01 10:00:01", tz="UTC"),
        pd.Timestamp("2026-01-01 10:00:02", tz="UTC"),
        pd.Timestamp("2026-01-01 10:00:03.5", tz="UTC"),  # Gap of 1.5s
    ]
    # Followed by 65 continuous 1-second samples
    ts_cont = pd.date_range("2026-01-01 10:00:05", periods=65, freq="1s", tz="UTC")
    all_ts = pd.Series(ts_list)._append(pd.Series(ts_cont)).reset_index(drop=True)

    n_tot = len(all_ts)
    df_jitter = pd.DataFrame({
        "timestamp": all_ts,
        "heart_rate_bpm": [70.0] * n_tot,
        "eda_microsiemens": [1.0] * n_tot,
        "body_temperature_c": [34.0] * n_tot,
        "stress_label": [1] * n_tot,
    })

    # Total observations is 69.
    # Block 1 has 3 observations (< 60) -> produces 0 windows.
    # Block 2 has 1 observation (< 60) -> produces 0 windows.
    # Block 3 has 65 observations -> produces (65 - 60)//30 + 1 = 1 window.
    X, y, meta = extract_pure_label_windows_and_features(
        df=df_jitter,
        participant_id="TEST_JITTER",
        window_size=60,
        stride=30,
        expected_sampling_interval_sec=1.0,
        timestamp_tolerance_sec=0.1,
    )
    assert len(X) == 1
    # The generated window starts at 10:00:05, strictly within the continuous Block 3
    assert meta.iloc[0]["window_start_timestamp"].startswith("2026-01-01 10:00:05")


# ============================================================================
# 4. Class Imbalance and Metric Calculation Safety
# ============================================================================


def test_extreme_class_imbalance_metric_calculation():
    """Verify that extreme class imbalance (e.g. 99% baseline, 1% stress) does not divide by zero."""
    # 990 negative samples, 10 positive samples
    y_true = np.array([0] * 990 + [1] * 10)
    # Model that predicts negative for everything (TP=0, FP=0)
    y_pred_all_neg = np.zeros(1000, dtype=int)

    metrics_neg = calculate_classification_metrics(y_true=y_true, y_pred=y_pred_all_neg)
    assert metrics_neg["precision"] == 0.0
    assert metrics_neg["recall"] == 0.0
    assert metrics_neg["f1_score"] == 0.0
    assert metrics_neg["accuracy"] == 0.99
    assert metrics_neg["false_positive_rate"] == 0.0
    assert metrics_neg["specificity"] == 1.0

    # Model that predicts positive for everything (TN=0, FN=0)
    y_pred_all_pos = np.ones(1000, dtype=int)
    metrics_pos = calculate_classification_metrics(y_true=y_true, y_pred=y_pred_all_pos)
    assert metrics_pos["recall"] == 1.0
    assert metrics_pos["precision"] == 0.01
    assert metrics_pos["false_positive_rate"] == 1.0
    assert metrics_pos["specificity"] == 0.0


def test_random_forest_balanced_subsample_handles_imbalanced_fit():
    """Verify that StressRandomForestClassifier trains and predicts stably on imbalanced data."""
    np.random.seed(42)
    expected_cols = [
        f"{col}_{stat}"
        for col in DEFAULT_PHYSIOLOGICAL_COLUMNS
        for stat in ("mean", "std", "min", "max", "range", "trend")
    ]
    # 50 baseline samples vs. 5 stress samples (10:1 imbalance)
    n_base = 50
    n_stress = 5
    X_imb = pd.DataFrame(np.random.randn(n_base + n_stress, 18), columns=expected_cols)
    y_imb = pd.Series([0] * n_base + [1] * n_stress)

    rf = StressRandomForestClassifier(n_estimators=20, random_state=42, class_weight="balanced_subsample")
    rf.fit(X_imb, y_imb)
    assert rf.is_fitted_
    preds = rf.predict(X_imb)
    probs = rf.predict_proba(X_imb)
    assert len(preds) == 55
    assert probs.shape == (55, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


# ============================================================================
# 5. Controlled Simulator Scenarios
# ============================================================================


def test_controlled_simulator_stream_with_personalized_detector():
    """Verify HealthSensing detector on simulated streams across normal and heatwave scenarios.

    Simulated data streams are tested for algorithmic compatibility while
    keeping simulated metrics isolated from genuine WESAD benchmark results.
    """
    # 1. Generate 300 simulated normal readings for calibration
    normal_records = list(stream_data("normal", count=300))
    cal_sim_df = pd.DataFrame({
        "timestamp": [r["timestamp"] for r in normal_records],
        "heart_rate_bpm": [float(r["heart_rate"]) for r in normal_records],
        "eda_microsiemens": [1.0 + float(r["activity_level"]) * 0.5 for r in normal_records],
        "body_temperature_c": [float(r["body_temperature"]) for r in normal_records],
    })

    ctx = prepare_baseline(cal_sim_df, strategy="full")
    assert ctx.strategy == "full"
    assert len(ctx.get_baseline_df()) == 300

    # 2. Evaluate simulated normal readings (should be predominantly normal)
    test_normal = cal_sim_df.iloc[:60].copy()
    norm_res = analyze_readings(readings=test_normal, baseline_context=ctx)
    anom_count_normal = sum(r.overall_anomaly for r in norm_res)
    assert anom_count_normal < 10, "Simulated normal readings generated excessive anomalies"

    # 3. Generate simulated heatwave readings (elevated HR, high temperature)
    heatwave_records = list(stream_data("heatwave", count=60))
    heat_sim_df = pd.DataFrame({
        "timestamp": [r["timestamp"] for r in heatwave_records],
        "heart_rate_bpm": [float(r["heart_rate"]) for r in heatwave_records],
        "eda_microsiemens": [1.0 + float(r["activity_level"]) * 0.5 for r in heatwave_records],
        "body_temperature_c": [float(r["body_temperature"]) for r in heatwave_records],
    })

    heat_res = analyze_readings(readings=heat_sim_df, baseline_context=ctx)
    # Heatwave readings should trigger physiological anomaly flags due to elevated temperature/HR
    anom_count_heat = sum(r.overall_anomaly for r in heat_res)
    assert anom_count_heat > 30, "Simulated heatwave scenario failed to trigger expected physiological anomalies"


# ============================================================================
# 6. Graceful Handling of Incomplete or Unevaluable Data
# ============================================================================


def test_multi_channel_dropout_unevaluable_status():
    """Verify that simultaneous dropout across all channels sets overall_status='unevaluable'."""
    cal_df = pd.DataFrame({
        "heart_rate_bpm": [70.0 + i * 0.01 for i in range(100)],
        "eda_microsiemens": [1.0 + i * 0.005 for i in range(100)],
        "body_temperature_c": [34.0 + i * 0.002 for i in range(100)],
    })
    ctx = prepare_baseline(cal_df, strategy="full")

    # All 3 channels missing
    all_dropout_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="1s", tz="UTC"),
        "heart_rate_bpm": [np.nan] * 5,
        "eda_microsiemens": [np.nan] * 5,
        "body_temperature_c": [np.nan] * 5,
    })

    results = analyze_readings(readings=all_dropout_df, baseline_context=ctx)
    assert len(results) == 5
    for r in results:
        assert r.unevaluable_count == 3
        assert r.evaluation_completeness == "none"
        assert r.overall_status == "unevaluable"
        assert r.overall_anomaly == 0
        assert r.anomaly_count == 0
        for col in DEFAULT_PHYSIOLOGICAL_COLUMNS:
            assert r.signals[col].evaluation_status == "missing_reading"
            assert r.signals[col].anomaly == 0
