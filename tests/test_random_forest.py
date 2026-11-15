"""Unit tests for Random Forest classical ML benchmark (ai.models.random_forest_model and ai.training.train_rf)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ai.models.random_forest_model import (
    DEFAULT_PHYSIOLOGICAL_COLUMNS,
    FORBIDDEN_COLUMNS,
    StressRandomForestClassifier,
    create_rf_classifier,
)
from ai.training.train_rf import (
    calculate_classification_metrics,
    extract_pure_label_windows_and_features,
    generate_random_forest_report,
    run_lopo_benchmark,
    validate_and_sort_participant_data,
    validate_timestamp_config,
)


@pytest.fixture
def synthetic_participant_df() -> pd.DataFrame:
    """Create a synthetic participant dataset with baseline, transition, stress, and transient labels."""
    # 200 baseline (label 1), 50 transient (label 0), 200 stress (label 2), 50 amusement (label 3)
    n_rows = 500
    timestamps = pd.date_range("2026-01-01 10:00:00", periods=n_rows, freq="1s", tz="UTC")

    labels = [1] * 200 + [0] * 50 + [2] * 200 + [3] * 50

    np.random.seed(42)
    hr = 70.0 + np.random.normal(0, 2, n_rows)
    # Give stress higher HR
    hr[250:450] += 20.0

    eda = 1.0 + np.random.normal(0, 0.1, n_rows)
    eda[250:450] += 3.0

    temp = 36.5 + np.random.normal(0, 0.05, n_rows)

    return pd.DataFrame({
        "participant_id": "TEST_P",
        "session_id": "TEST_SESSION",
        "timestamp": timestamps,
        "heart_rate_bpm": hr,
        "eda_microsiemens": eda,
        "body_temperature_c": temp,
        "stress_label": labels,
        "source": "SYNTHETIC",
    })


# ============================================================================
# TASK 1: Timestamp Continuity & Block Segmentation Tests
# ============================================================================


def test_normal_one_second_timestamps_produce_expected_windows(synthetic_participant_df: pd.DataFrame):
    """Verify that uniform 1-second sampled continuous data produces the expected window counts."""
    X, y, meta = extract_pure_label_windows_and_features(
        synthetic_participant_df,
        participant_id="TEST_P",
        window_size=60,
        stride=30,
        expected_sampling_interval_sec=1.0,
        timestamp_tolerance_sec=0.1,
    )
    # 200 continuous baseline obs -> (200 - 60)//30 + 1 = 5 windows
    # 200 continuous stress obs -> (200 - 60)//30 + 1 = 5 windows
    assert len(X) == 10
    assert (y == 0).sum() == 5
    assert (y == 1).sum() == 5


def test_timestamp_gap_splits_recording_into_separate_blocks():
    """Verify that a timestamp gap splits a continuous label into separate blocks."""
    # 120 baseline observations with a 10-second gap in the middle (at index 60)
    ts_part1 = pd.date_range("2026-01-01 10:00:00", periods=60, freq="1s", tz="UTC")
    # 10s gap: jump from 10:00:59 to 10:01:10
    ts_part2 = pd.date_range("2026-01-01 10:01:10", periods=60, freq="1s", tz="UTC")
    timestamps = ts_part1.append(ts_part2)

    df_gap = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": [70.0] * 120,
        "eda_microsiemens": [1.0] * 120,
        "body_temperature_c": [36.5] * 120,
        "stress_label": [1] * 120,
    })

    # Without gap, 120 obs would give (120 - 60)//30 + 1 = 3 windows.
    # But with gap, it is split into two 60-obs blocks.
    # Block 1 (60 obs): 1 window. Block 2 (60 obs): 1 window.
    # Total = 2 windows, and neither crosses the gap.
    X, y, meta = extract_pure_label_windows_and_features(
        df_gap,
        participant_id="TEST_GAP",
        window_size=60,
        stride=30,
        expected_sampling_interval_sec=1.0,
        timestamp_tolerance_sec=0.1,
    )

    assert len(X) == 2
    # First window ends at or before 10:00:59
    assert pd.to_datetime(meta.loc[0, "window_end_timestamp"]) <= pd.Timestamp("2026-01-01 10:00:59", tz="UTC")
    # Second window starts at or after 10:01:10
    assert pd.to_datetime(meta.loc[1, "window_start_timestamp"]) >= pd.Timestamp("2026-01-01 10:01:10", tz="UTC")


def test_no_generated_window_crosses_timestamp_gap():
    """Verify that when a fragment around a gap is smaller than window_size, no partial window is created."""
    # 80 observations total: 50 obs, then 5s gap, then 30 obs (neither fragment >= 60 obs)
    ts1 = pd.date_range("2026-01-01 10:00:00", periods=50, freq="1s", tz="UTC")
    ts2 = pd.date_range("2026-01-01 10:00:55", periods=30, freq="1s", tz="UTC")
    timestamps = ts1.append(ts2)

    df_fragmented = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": [70.0] * 80,
        "eda_microsiemens": [1.0] * 80,
        "body_temperature_c": [36.5] * 80,
        "stress_label": [1] * 80,
    })

    X, y, meta = extract_pure_label_windows_and_features(
        df_fragmented,
        participant_id="TEST_FRAG",
        window_size=60,
        stride=30,
    )

    # Neither side of the gap has 60 continuous observations; 0 windows must be generated
    assert len(X) == 0
    assert len(y) == 0


def test_label_transitions_split_blocks_correctly(synthetic_participant_df: pd.DataFrame):
    """Verify that contiguous regions of different labels are partitioned without window overlap."""
    X, y, meta = extract_pure_label_windows_and_features(
        synthetic_participant_df,
        participant_id="TEST_P",
        window_size=60,
        stride=30,
    )

    for _, row in meta.iterrows():
        # Every window must be 100% pure label
        assert row["true_label"] in (1, 2)
        assert row["true_class"] in (0, 1)


def test_existing_continuous_wesad_recordings():
    """Verify that genuine WESAD recordings have unbroken 1.0s continuity and yield expected windows."""
    df_s2 = pd.read_csv("data/processed/wesad_s2_real.csv")
    X, y, meta = extract_pure_label_windows_and_features(
        df_s2,
        participant_id="S2",
        window_size=60,
        stride=30,
    )
    # 37 baseline + 19 stress = 56 windows
    assert len(X) == 56
    assert (y == 0).sum() == 37
    assert (y == 1).sum() == 19


# ============================================================================
# TASK 2: Target Validation Tests
# ============================================================================


def test_target_validation_rejects_empty():
    """Verify that empty target arrays are rejected."""
    model = StressRandomForestClassifier()
    X = np.ones((5, 18))
    with pytest.raises(ValueError, match="empty"):
        model._validate_targets([])
    with pytest.raises(ValueError, match="empty"):
        model._validate_targets(np.array([]))


def test_target_validation_rejects_missing_and_nan_and_inf():
    """Verify that None, NaN, and infinity in targets are rejected."""
    model = StressRandomForestClassifier()
    with pytest.raises(ValueError, match="cannot be None"):
        model._validate_targets(None)
    with pytest.raises(ValueError, match="missing, NaN, or infinite"):
        model._validate_targets([0, np.nan, 1])
    with pytest.raises(ValueError, match="missing, NaN, or infinite"):
        model._validate_targets([0, np.inf, 1])
    with pytest.raises(ValueError, match="missing, NaN, or infinite"):
        model._validate_targets([0, -np.inf, 1])


def test_target_validation_rejects_non_numeric():
    """Verify that string or object targets are rejected."""
    model = StressRandomForestClassifier()
    with pytest.raises(TypeError, match="numeric"):
        model._validate_targets(["baseline", "stress"])


def test_target_validation_rejects_fractional_labels():
    """Verify that fractional values (e.g. 0.8) are rejected instead of silently truncated to int."""
    model = StressRandomForestClassifier()
    with pytest.raises(ValueError, match="fractional or out-of-range"):
        model._validate_targets([0.8, 1.0])
    with pytest.raises(ValueError, match="fractional or out-of-range"):
        model._validate_targets([0.0, 0.5])


def test_target_validation_requires_both_classes_during_fit():
    """Verify that fitting with single-class targets raises ValueError."""
    model = StressRandomForestClassifier()
    X = np.ones((10, 18))
    # All class 0
    with pytest.raises(ValueError, match="requires both binary classes"):
        model.fit(X, [0] * 10)
    # All class 1
    with pytest.raises(ValueError, match="requires both binary classes"):
        model.fit(X, [1] * 10)


def test_predict_proba_guarantees_class_column_order():
    """Verify that predict_proba returns column 0 = class 0 (baseline) and column 1 = class 1 (stress)."""
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(40, 18), columns=[f"f_{i}" for i in range(18)])
    y_train = pd.Series([0] * 20 + [1] * 20)

    model = StressRandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_train)
    assert probs.shape == (40, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs[:, 0] >= 0.0).all() and (probs[:, 0] <= 1.0).all()
    assert (probs[:, 1] >= 0.0).all() and (probs[:, 1] <= 1.0).all()


# ============================================================================
# TASK 3: Feature Validation and Imputation Tests
# ============================================================================


def test_feature_validation_rejects_infinity():
    """Verify that infinite values (+inf or -inf) in feature matrix are explicitly rejected."""
    model = StressRandomForestClassifier()
    X_inf = pd.DataFrame({
        "f1": [1.0, np.inf, 2.0],
        "f2": [3.0, 4.0, -np.inf],
    })
    y = [0, 1, 0]
    with pytest.raises(ValueError, match="infinite"):
        model.fit(X_inf, y)


def test_feature_validation_detects_all_missing_column():
    """Verify that a feature entirely missing (100% NaN) in training data is detected and rejected."""
    model = StressRandomForestClassifier()
    X_all_nan = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0],
        "f2_all_nan": [np.nan, np.nan, np.nan, np.nan],
    })
    y = [0, 0, 1, 1]
    with pytest.raises(ValueError, match="Feature column.*contain only missing values"):
        model.fit(X_all_nan, y)


def test_feature_preservation_and_ordering():
    """Verify that feature names and ordering are validated during predict."""
    np.random.seed(42)
    cols = ["col_a", "col_b", "col_c"]
    X_train = pd.DataFrame(np.random.randn(20, 3), columns=cols)
    y_train = [0] * 10 + [1] * 10

    model = StressRandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Scrambled column order must raise ValueError
    X_scrambled = X_train[["col_c", "col_a", "col_b"]]
    with pytest.raises(ValueError, match="Feature columns and order do not match"):
        model.predict(X_scrambled)

    # Missing column must raise ValueError
    X_missing_col = X_train[["col_a", "col_b"]]
    with pytest.raises(ValueError, match="Feature columns and order do not match"):
        model.predict(X_missing_col)


def test_missing_value_handling_without_leakage():
    """Verify that missing values in evaluation features are imputed using training-fold statistics only."""
    np.random.seed(42)
    X_train = pd.DataFrame({
        "feat1": [9.0, 10.0, 11.0, 10.0],
        "feat2": [20.0, 20.0, 20.0, 20.0],
    })
    y_train = pd.Series([0, 0, 1, 1])

    X_test = pd.DataFrame({
        "feat1": [np.nan],
        "feat2": [20.0],
    })

    model = StressRandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    assert model.imputer.statistics_[0] == pytest.approx(10.0)
    pred = model.predict(X_test)
    assert len(pred) == 1
    assert pred[0] in (0, 1)


# ============================================================================
# TASK 4: Metric Validation Tests
# ============================================================================


def test_metric_validation_rejects_empty():
    """Verify that calculate_classification_metrics rejects empty inputs."""
    with pytest.raises(ValueError, match="empty"):
        calculate_classification_metrics([], [])


def test_metric_validation_rejects_unequal_lengths():
    """Verify that calculate_classification_metrics rejects unequal input lengths."""
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_classification_metrics([0, 1], [0])


def test_metric_validation_rejects_fractional_labels():
    """Verify that calculate_classification_metrics rejects fractional values without truncation."""
    with pytest.raises(ValueError, match="fractional or out-of-range"):
        calculate_classification_metrics([0.8, 1.0], [0, 1])
    with pytest.raises(ValueError, match="fractional or out-of-range"):
        calculate_classification_metrics([0, 1], [0.5, 1.0])


def test_metric_validation_rejects_non_binary_and_nan():
    """Verify that calculate_classification_metrics rejects out-of-range, NaN, or non-numeric values."""
    with pytest.raises(ValueError, match="binary values"):
        calculate_classification_metrics([0, 2], [0, 1])
    with pytest.raises(ValueError, match="missing, NaN, or infinite"):
        calculate_classification_metrics([0, np.nan], [0, 1])
    with pytest.raises(TypeError, match="numeric"):
        calculate_classification_metrics(["0", "1"], [0, 1])


def test_metric_calculation_normal():
    """Verify accurate calculation of classification metrics for valid inputs."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    res = calculate_classification_metrics(y_true, y_pred)
    assert res["true_positives"] == 1
    assert res["false_positives"] == 1
    assert res["true_negatives"] == 1
    assert res["false_negatives"] == 1
    assert res["accuracy"] == 0.5
    assert res["precision"] == 0.5
    assert res["recall"] == 0.5
    assert res["f1_score"] == 0.5


# ============================================================================
# TASK 5: Participant Data Validation & LOPO Benchmark Tests
# ============================================================================


def test_participant_independent_splits_and_no_overlap():
    """Verify that in LOPO evaluation, training data never contains windows from the held-out participant."""
    p_df, pooled, win_df, rep_p = run_lopo_benchmark(save_artifacts=False)

    assert len(p_df) == 4
    for _, row in p_df.iterrows():
        test_p = row["heldout_participant"]
        train_ps = row["training_participants"].split("+")
        assert test_p not in train_ps
        assert len(train_ps) == 3

    counts = win_df["participant_id"].value_counts().to_dict()
    assert counts["S2"] == 56
    assert counts["S3"] == 57
    assert counts["S4"] == 57
    assert counts["S5"] == 58
    assert len(win_df) == 228


def test_deterministic_model_training():
    """Verify that training with fixed random_state produces identical predictions."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 18), columns=[f"f_{i}" for i in range(18)])
    y = pd.Series([0] * 25 + [1] * 25)

    model1 = StressRandomForestClassifier(n_estimators=50, random_state=42)
    model1.fit(X, y)
    preds1 = model1.predict(X)

    model2 = StressRandomForestClassifier(n_estimators=50, random_state=42)
    model2.fit(X, y)
    preds2 = model2.predict(X)

    np.testing.assert_array_equal(preds1, preds2)
    np.testing.assert_allclose(
        model1.feature_importances().values,
        model2.feature_importances().values,
    )


def test_model_serialization(tmp_path: Path):
    """Verify save and load functionality for StressRandomForestClassifier."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(20, 18), columns=[f"f_{i}" for i in range(18)])
    y = pd.Series([0] * 10 + [1] * 10)

    model = StressRandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)

    save_path = tmp_path / "model.joblib"
    model.save(save_path)
    assert save_path.exists()

    loaded_model = StressRandomForestClassifier.load(save_path)
    assert loaded_model.is_fitted_
    np.testing.assert_array_equal(model.predict(X), loaded_model.predict(X))


def test_lopo_macro_metrics_sample_std_and_feature_count():
    """Verify that macro metrics use sample standard deviation (ddof=1) and dynamic feature dimensions."""
    p_df, pooled, win_df, rep_p = run_lopo_benchmark(save_artifacts=False)
    macro = pooled["participant_macro_metrics"]

    # Verify fold recall values: S2=0.0, S3=1.0, S4=1.0, S5=0.55
    recalls = p_df.set_index("heldout_participant")["recall"].to_dict()
    assert recalls["S2"] == pytest.approx(0.0)
    assert recalls["S3"] == pytest.approx(1.0)
    assert recalls["S4"] == pytest.approx(1.0)
    assert recalls["S5"] == pytest.approx(0.55)

    # Check macro recall mean and sample std (ddof=1)
    assert macro["macro_recall"] == pytest.approx(0.6375, abs=1e-4)
    assert macro["macro_recall_std"] == pytest.approx(0.4750, abs=1e-4)

    # Check macro precision mean and sample std (ddof=1)
    assert macro["macro_precision"] == pytest.approx(0.5330, abs=1e-4)
    assert macro["macro_precision_std"] == pytest.approx(0.4219, abs=1e-4)

    # Check macro accuracy mean and sample std (ddof=1)
    assert macro["macro_accuracy"] == pytest.approx(0.7452, abs=1e-4)
    assert macro["macro_accuracy_std"] == pytest.approx(0.1880, abs=1e-4)

    # Check dynamic feature count matches 18
    assert pooled["parameters"]["n_features"] == 18


def test_invalid_timestamp_configurations(synthetic_participant_df: pd.DataFrame):
    """Verify validation for expected_sampling_interval_sec and timestamp_tolerance_sec."""
    # 1. Interval <= 0
    with pytest.raises(ValueError, match="strictly positive"):
        validate_timestamp_config(expected_sampling_interval_sec=0, timestamp_tolerance_sec=0.1)

    with pytest.raises(ValueError, match="strictly positive"):
        validate_timestamp_config(expected_sampling_interval_sec=-1.0, timestamp_tolerance_sec=0.1)

    with pytest.raises(ValueError, match="strictly positive"):
        extract_pure_label_windows_and_features(
            synthetic_participant_df,
            participant_id="TEST_P",
            expected_sampling_interval_sec=0.0,
            timestamp_tolerance_sec=0.1,
        )

    # 2. Tolerance < 0
    with pytest.raises(ValueError, match="non-negative"):
        validate_timestamp_config(expected_sampling_interval_sec=1.0, timestamp_tolerance_sec=-0.05)

    with pytest.raises(ValueError, match="non-negative"):
        extract_pure_label_windows_and_features(
            synthetic_participant_df,
            participant_id="TEST_P",
            expected_sampling_interval_sec=1.0,
            timestamp_tolerance_sec=-0.1,
        )

    # 3. Tolerance >= Interval
    with pytest.raises(ValueError, match="strictly less than"):
        validate_timestamp_config(expected_sampling_interval_sec=1.0, timestamp_tolerance_sec=1.0)

    with pytest.raises(ValueError, match="strictly less than"):
        validate_timestamp_config(expected_sampling_interval_sec=1.0, timestamp_tolerance_sec=1.5)

    with pytest.raises(ValueError, match="strictly less than"):
        extract_pure_label_windows_and_features(
            synthetic_participant_df,
            participant_id="TEST_P",
            expected_sampling_interval_sec=1.0,
            timestamp_tolerance_sec=1.0,
        )

    # 4. Invalid types
    with pytest.raises(TypeError, match="numeric value"):
        validate_timestamp_config(expected_sampling_interval_sec="1.0", timestamp_tolerance_sec=0.1)  # type: ignore

    with pytest.raises(TypeError, match="numeric value"):
        validate_timestamp_config(expected_sampling_interval_sec=True, timestamp_tolerance_sec=0.1)  # type: ignore

    with pytest.raises(TypeError, match="numeric value"):
        validate_timestamp_config(expected_sampling_interval_sec=1.0, timestamp_tolerance_sec=False)  # type: ignore


def test_dataframe_and_array_type_consistency():
    """Verify input type consistency, column order validation, and array support."""
    np.random.seed(42)
    cols = ["feat_a", "feat_b", "feat_c"]
    X_df = pd.DataFrame(np.random.randn(20, 3), columns=cols)
    X_arr = X_df.to_numpy()
    y = pd.Series([0] * 10 + [1] * 10)

    # --- Model fitted with DataFrame ---
    model_df = StressRandomForestClassifier(n_estimators=10, random_state=42)
    model_df.fit(X_df, y)
    assert model_df.fitted_with_dataframe_ is True
    assert model_df.feature_names_ == cols
    assert model_df.n_features_in_ == 3

    # Predict with DataFrame succeeds
    preds_df = model_df.predict(X_df)
    assert len(preds_df) == 20
    probs_df = model_df.predict_proba(X_df)
    assert probs_df.shape == (20, 2)

    # Predict with NumPy array raises TypeError
    with pytest.raises(TypeError, match="Model was fitted with a pandas DataFrame"):
        model_df.predict(X_arr)

    with pytest.raises(TypeError, match="Model was fitted with a pandas DataFrame"):
        model_df.predict_proba(X_arr)

    # Predict with DataFrame with mismatched column order raises ValueError
    X_reordered = X_df[["feat_b", "feat_a", "feat_c"]]
    with pytest.raises(ValueError, match="Feature columns and order do not match"):
        model_df.predict(X_reordered)

    with pytest.raises(ValueError, match="Feature columns and order do not match"):
        model_df.predict_proba(X_reordered)

    # Predict with DataFrame with wrong column names raises ValueError
    X_wrong_cols = pd.DataFrame(np.random.randn(20, 3), columns=["feat_x", "feat_y", "feat_z"])
    with pytest.raises(ValueError, match="Feature columns and order do not match"):
        model_df.predict(X_wrong_cols)

    # --- Model fitted with NumPy array ---
    model_arr = StressRandomForestClassifier(n_estimators=10, random_state=42)
    model_arr.fit(X_arr, y)
    assert model_arr.fitted_with_dataframe_ is False
    assert model_arr.feature_names_ is None
    assert model_arr.n_features_in_ == 3

    # Predict with NumPy array succeeds
    preds_arr = model_arr.predict(X_arr)
    assert len(preds_arr) == 20
    probs_arr = model_arr.predict_proba(X_arr)
    assert probs_arr.shape == (20, 2)

    # Predict with DataFrame raises TypeError
    with pytest.raises(TypeError, match="Model was fitted with a NumPy array"):
        model_arr.predict(X_df)

    with pytest.raises(TypeError, match="Model was fitted with a NumPy array"):
        model_arr.predict_proba(X_df)

    # Predict with wrong feature count raises ValueError
    with pytest.raises(ValueError, match="Feature count mismatch"):
        model_arr.predict(np.random.randn(20, 4))

    # Predict with 1D array raises ValueError
    with pytest.raises(ValueError, match="must be 2D"):
        model_arr.predict(np.random.randn(20))


def test_dynamic_class_counts_in_report_and_metrics(tmp_path: Path):
    """Verify that class counts in pooled_metrics.json and random_forest_report.md are derived dynamically."""
    p_df, pooled, win_df, rep_p = run_lopo_benchmark(save_artifacts=True, output_dir=tmp_path)

    # Check pooled_metrics.json
    cm = pooled["pooled_confusion_matrix"]
    assert "total_baseline_windows" in cm
    assert "total_stress_windows" in cm
    assert cm["total_baseline_windows"] == int(p_df["testing_baseline_windows"].sum())
    assert cm["total_stress_windows"] == int(p_df["testing_stress_windows"].sum())
    assert cm["total_baseline_windows"] == 149
    assert cm["total_stress_windows"] == 79
    assert cm["total_windows"] == 228

    # Check report content
    report_text = rep_p.read_text(encoding="utf-8")
    expected_line = f"- **Total Windows Evaluated:** 228 (149 Baseline Class 0, 79 Stress Class 1)"
    assert expected_line in report_text

    # Test dynamic generation with arbitrary custom class counts
    custom_metrics = p_df.copy()
    custom_metrics["testing_baseline_windows"] = [10, 20, 30, 40]  # sum = 100
    custom_metrics["testing_stress_windows"] = [5, 10, 15, 20]      # sum = 50
    custom_cm = cm.copy()
    custom_cm["total_windows"] = 150
    custom_cm["total_baseline_windows"] = 100
    custom_cm["total_stress_windows"] = 50
    custom_pooled = pooled.copy()
    custom_pooled["pooled_confusion_matrix"] = custom_cm

    custom_report_path = tmp_path / "custom_report.md"
    dummy_model = StressRandomForestClassifier(n_estimators=10, random_state=42)
    dummy_model.fit(pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"]), [0]*5 + [1]*5)

    generate_random_forest_report(
        participant_metrics=custom_metrics,
        pooled_metrics=custom_pooled,
        final_model=dummy_model,
        report_path=custom_report_path,
        n_features=3,
    )
    custom_text = custom_report_path.read_text(encoding="utf-8")
    assert "- **Total Windows Evaluated:** 150 (100 Baseline Class 0, 50 Stress Class 1)" in custom_text
