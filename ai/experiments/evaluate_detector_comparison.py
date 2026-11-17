"""HealthSensing: Comprehensive Detector Comparison and Visualization.

This module builds a rigorous, fair, and transparent comparison between:
1. Fixed Statistical Threshold / Baseline Detector (Strategy A: un-gated personal baseline).
2. Personalized Multi-Signal Anomaly Detector (Strategy D: signal-specific quality-gated).
3. Random Forest Classical ML Benchmark (Leave-One-Participant-Out cross-validation).
4. Fixed Population Statistical Threshold (population mean +/- 3*std from training cohort).

Protocol Alignment & Fair Comparison:
-------------------------------------
- Reading-Level Evaluation: Evaluates personalized anomaly detectors at 1 Hz on held-out baseline
  (remaining 30%) and acute stress induction.
- Window-Level Evaluation: Evaluates Random Forest on 60-observation windows with 30-observation stride.
- Common Held-Out Evaluation Subset: Evaluates all detectors on the exact same temporal test period
  (the held-out 30% baseline partition and the stress period), without data leakage.
- Metrics Reported: Precision, Recall, F1-score, Accuracy, Specificity, False Positive Rate (FPR),
  AUROC (where continuous scores/probabilities are available), and Detection Latency (seconds from
  stress onset to first alert).
- Separates participant-level metrics, macro-averages (unweighted fold mean +/- sample SD, ddof=1),
  and pooled metrics.

Clinical Disclaimer:
--------------------
This module provides experimental algorithmic research metrics and comparisons.
It does NOT constitute medical diagnosis or clinical verification.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ai.baseline.baseline_quality import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_MIN_SIGNAL_OBS,
    DEFAULT_WINDOW_SEC,
    assess_calibration_dataset_quality,
    select_signal_specific_calibration_windows,
)
from ai.baseline.multisignal_detector import detect_multisignal_anomalies
from ai.features.feature_engineering import extract_window_features
from ai.models.random_forest_model import (
    DEFAULT_PHYSIOLOGICAL_COLUMNS,
    StressRandomForestClassifier,
)
from ai.preprocessing.windowing import create_windows

DEFAULT_PARTICIPANT_IDS: tuple[str, ...] = ("S2", "S3", "S4", "S5")
DATA_DIR: Path = Path("data/processed")
OUTPUT_DIR: Path = Path("experiments/results/detector_comparison")
PLOTS_DIR: Path = Path("experiments/plots")
BASELINE_LABEL: int = 1
STRESS_LABEL: int = 2
CALIBRATION_RATIO: float = 0.70
WINDOW_SIZE: int = 60
STRIDE: int = 30


def calculate_metrics_dict(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    y_score: Sequence[float] | np.ndarray | pd.Series | None = None,
) -> dict[str, Any]:
    """Calculate standard classification performance metrics."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    n_samples = int(len(yt))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    acc = float((tp + tn) / n_samples) if n_samples > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    auroc = None
    if y_score is not None:
        try:
            scores = np.asarray(y_score, dtype=float)
            if len(np.unique(yt)) == 2 and not np.isnan(scores).all():
                # Replace NaNs with min score for robust AUC calculation
                valid_scores = np.nan_to_num(scores, nan=np.nanmin(scores) if not np.isnan(scores).all() else 0.0)
                auroc = float(roc_auc_score(yt, valid_scores))
        except Exception:
            auroc = None

    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_samples": n_samples,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "specificity": round(spec, 4),
        "false_positive_rate": round(fpr, 4),
        "auroc": round(auroc, 4) if auroc is not None else None,
    }


def compute_macro_averages(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute unweighted macro-averages with sample standard deviation (ddof=1)."""
    df = pd.DataFrame(records)
    metric_keys = ["precision", "recall", "f1_score", "accuracy", "specificity", "false_positive_rate"]
    macro = {}
    for k in metric_keys:
        if k in df.columns:
            vals = df[k].dropna()
            macro[f"macro_{k}"] = round(float(vals.mean()), 4)
            macro[f"macro_{k}_std"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0

    if "auroc" in df.columns and not df["auroc"].isna().all():
        vals = df["auroc"].dropna()
        macro["macro_auroc"] = round(float(vals.mean()), 4)
        macro["macro_auroc_std"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0
    else:
        macro["macro_auroc"] = None
        macro["macro_auroc_std"] = None

    macro["std_convention"] = "sample_standard_deviation_ddof_1"
    return macro


def run_comprehensive_detector_comparison(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    plots_dir: Path = PLOTS_DIR,
    participant_ids: Sequence[str] = DEFAULT_PARTICIPANT_IDS,
) -> dict[str, Any]:
    """Execute complete detector comparison, generate tables and plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_data: dict[str, pd.DataFrame] = {}
    for pid in participant_ids:
        csv_file = data_dir / f"wesad_{pid.lower()}_real.csv"
        df = pd.read_csv(csv_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        raw_data[pid] = df.sort_values("timestamp").reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 1. Reading-Level Personalized Anomaly Detectors (Strategy A and Strategy D)
    # -------------------------------------------------------------------------
    strat_a_records = []
    strat_d_records = []
    pop_thresh_records = []

    # Reading-level continuous scores and latencies
    detector_latencies: dict[str, dict[str, float]] = {
        "Strategy A (Baseline)": {},
        "Strategy D (Personalized)": {},
        "Population Threshold": {},
        "Random Forest (LOPO)": {},
    }
    detector_roc_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        "Strategy A (Baseline)": {},
        "Strategy D (Personalized)": {},
        "Random Forest (LOPO)": {},
    }

    # First collect training cohort statistics for Leave-One-Participant-Out Population Threshold
    pop_baselines: dict[str, dict[str, tuple[float, float]]] = {}
    for test_p in participant_ids:
        train_ps = [p for p in participant_ids if p != test_p]
        train_dfs = [raw_data[p][raw_data[p]["stress_label"] == BASELINE_LABEL] for p in train_ps]
        train_baseline_combined = pd.concat(train_dfs, axis=0)
        pop_baselines[test_p] = {}
        for col in DEFAULT_PHYSIOLOGICAL_COLUMNS:
            col_series = pd.to_numeric(train_baseline_combined[col], errors="coerce").dropna()
            pop_baselines[test_p][col] = (float(col_series.mean()), float(col_series.std(ddof=1)))

    for pid in participant_ids:
        df = raw_data[pid]
        b_df = df[df["stress_label"] == BASELINE_LABEL].copy()
        s_df = df[df["stress_label"] == STRESS_LABEL].copy()

        n_cal = math.floor(len(b_df) * CALIBRATION_RATIO)
        cal_full = b_df.iloc[:n_cal].copy()
        heldout_b = b_df.iloc[n_cal:].copy()

        eval_df = pd.concat([heldout_b, s_df], axis=0).sort_values("timestamp").reset_index(drop=True)
        y_true = (eval_df["stress_label"] == STRESS_LABEL).astype(int).to_numpy()

        # Strategy A: Full 70% baseline
        det_a = detect_multisignal_anomalies(
            readings=eval_df,
            baseline_readings=cal_full,
            feature_columns=DEFAULT_PHYSIOLOGICAL_COLUMNS,
            thresholds=3.0,
        )
        z_cols = [f"{col}_baseline_z_score" for col in DEFAULT_PHYSIOLOGICAL_COLUMNS]
        # Maximum absolute z-score as continuous anomaly score
        z_max_a = det_a[z_cols].abs().max(axis=1).to_numpy()

        res_a = calculate_metrics_dict(y_true, det_a["overall_anomaly"].to_numpy(), z_max_a)
        res_a["participant_id"] = pid
        res_a["detector"] = "Strategy A (Fixed Baseline)"
        strat_a_records.append(res_a)
        detector_roc_data["Strategy A (Baseline)"][pid] = (y_true, z_max_a)

        # Detection Latency for Strategy A (time to first anomaly in stress period)
        stress_eval_a = det_a[det_a["stress_label"] == STRESS_LABEL].reset_index(drop=True)
        first_idx_a = (stress_eval_a["overall_anomaly"] == 1).idxmax() if (stress_eval_a["overall_anomaly"] == 1).any() else None
        latency_a = float(first_idx_a) if first_idx_a is not None and stress_eval_a.loc[first_idx_a, "overall_anomaly"] == 1 else float(len(stress_eval_a))
        detector_latencies["Strategy A (Baseline)"][pid] = latency_a

        # Strategy D: Signal-specific quality-gated baseline
        cal_d_df, meta_d = select_signal_specific_calibration_windows(
            calibration_df=cal_full,
            feature_columns=DEFAULT_PHYSIOLOGICAL_COLUMNS,
        )
        det_d = detect_multisignal_anomalies(
            readings=eval_df,
            baseline_readings=cal_d_df,
            feature_columns=DEFAULT_PHYSIOLOGICAL_COLUMNS,
            thresholds=3.0,
        )
        z_max_d = det_d[z_cols].abs().max(axis=1).to_numpy()
        res_d = calculate_metrics_dict(y_true, det_d["overall_anomaly"].to_numpy(), z_max_d)
        res_d["participant_id"] = pid
        res_d["detector"] = "Strategy D (Personalized Quality-Gated)"
        strat_d_records.append(res_d)
        detector_roc_data["Strategy D (Personalized)"][pid] = (y_true, z_max_d)

        # Detection Latency for Strategy D
        stress_eval_d = det_d[det_d["stress_label"] == STRESS_LABEL].reset_index(drop=True)
        first_idx_d = (stress_eval_d["overall_anomaly"] == 1).idxmax() if (stress_eval_d["overall_anomaly"] == 1).any() else None
        latency_d = float(first_idx_d) if first_idx_d is not None and stress_eval_d.loc[first_idx_d, "overall_anomaly"] == 1 else float(len(stress_eval_d))
        detector_latencies["Strategy D (Personalized)"][pid] = latency_d

        # Fixed Population Statistical Threshold: apply population mean +/- 3*pop_std
        pop_z_scores = []
        for col in DEFAULT_PHYSIOLOGICAL_COLUMNS:
            m, s = pop_baselines[pid][col]
            pop_z = (pd.to_numeric(eval_df[col], errors="coerce") - m) / s
            pop_z_scores.append(pop_z.abs())
        pop_z_df = pd.concat(pop_z_scores, axis=1)
        pop_anom = (pop_z_df > 3.0).any(axis=1).astype(int).to_numpy()
        pop_z_max = pop_z_df.max(axis=1).to_numpy()

        res_pop = calculate_metrics_dict(y_true, pop_anom, pop_z_max)
        res_pop["participant_id"] = pid
        res_pop["detector"] = "Fixed Population Threshold"
        pop_thresh_records.append(res_pop)

        stress_eval_pop = eval_df[eval_df["stress_label"] == STRESS_LABEL].reset_index(drop=True)
        stress_pop_anom = pop_anom[eval_df["stress_label"] == STRESS_LABEL]
        first_idx_pop = int(np.argmax(stress_pop_anom == 1)) if (stress_pop_anom == 1).any() else None
        latency_pop = float(first_idx_pop) if first_idx_pop is not None and stress_pop_anom[first_idx_pop] == 1 else float(len(stress_eval_pop))
        detector_latencies["Population Threshold"][pid] = latency_pop

    # -------------------------------------------------------------------------
    # 2. Window-Level Supervised Random Forest (LOPO Cross-Validation)
    # -------------------------------------------------------------------------
    rf_win_preds_file = Path("experiments/results/random_forest/window_predictions.csv")
    rf_part_metrics_file = Path("experiments/results/random_forest/participant_metrics.csv")

    rf_win_df = pd.read_csv(rf_win_preds_file)
    rf_part_df = pd.read_csv(rf_part_metrics_file)

    rf_records = []
    for pid in participant_ids:
        p_sub = rf_win_df[rf_win_df["participant_id"] == pid].reset_index(drop=True)
        y_true_rf = p_sub["true_class"].to_numpy()
        y_pred_rf = p_sub["predicted_class"].to_numpy()
        y_prob_rf = p_sub["prob_stress_class_1"].to_numpy()

        m_rf = calculate_metrics_dict(y_true_rf, y_pred_rf, y_prob_rf)
        m_rf["participant_id"] = pid
        m_rf["detector"] = "Random Forest (LOPO Windows)"
        rf_records.append(m_rf)
        detector_roc_data["Random Forest (LOPO)"][pid] = (y_true_rf, y_prob_rf)

        # RF Latency: stride is 30s; first stress window start offset
        stress_windows = p_sub[p_sub["true_class"] == 1].reset_index(drop=True)
        stress_detected = stress_windows["predicted_class"] == 1
        if stress_detected.any():
            first_win_idx = int(stress_detected.idxmax())
            # Each window advances by stride=30 seconds
            latency_rf = float(first_win_idx * STRIDE)
        else:
            latency_rf = float(len(stress_windows) * STRIDE)
        detector_latencies["Random Forest (LOPO)"][pid] = latency_rf

    # -------------------------------------------------------------------------
    # 3. Common Held-Out Evaluation Subset Comparison
    # -------------------------------------------------------------------------
    # Extract common test windows corresponding to held-out baseline + stress
    common_heldout_records = []
    common_window_records = []

    for pid in participant_ids:
        df = raw_data[pid]
        b_df = df[df["stress_label"] == BASELINE_LABEL].copy()
        s_df = df[df["stress_label"] == STRESS_LABEL].copy()

        n_cal = math.floor(len(b_df) * CALIBRATION_RATIO)
        heldout_b = b_df.iloc[n_cal:].copy().reset_index(drop=True)
        stress_part = s_df.copy().reset_index(drop=True)

        # Create windows strictly from held-out baseline (Class 0)
        heldout_b_windows = create_windows(heldout_b, window_size=WINDOW_SIZE, stride=STRIDE, timestamp_column="timestamp")
        # Create windows strictly from stress (Class 1)
        stress_windows = create_windows(stress_part, window_size=WINDOW_SIZE, stride=STRIDE, timestamp_column="timestamp")

        # Feature matrix for common windows
        common_windows = heldout_b_windows + stress_windows
        y_common = np.array([0] * len(heldout_b_windows) + [1] * len(stress_windows))

        # Train fold RF model on the 3 other participants
        train_ps = [p for p in participant_ids if p != pid]
        # Reconstruct train windows strictly from train_ps
        train_wins = []
        train_labels = []
        for tp in train_ps:
            tp_df = raw_data[tp]
            tp_b = tp_df[tp_df["stress_label"] == BASELINE_LABEL].reset_index(drop=True)
            tp_s = tp_df[tp_df["stress_label"] == STRESS_LABEL].reset_index(drop=True)
            w_b = create_windows(tp_b, window_size=WINDOW_SIZE, stride=STRIDE, timestamp_column="timestamp")
            w_s = create_windows(tp_s, window_size=WINDOW_SIZE, stride=STRIDE, timestamp_column="timestamp")
            train_wins.extend(w_b)
            train_labels.extend([0] * len(w_b))
            train_wins.extend(w_s)
            train_labels.extend([1] * len(w_s))

        # Extract features
        expected_feat_names = [f"{col}_{stat}" for col in DEFAULT_PHYSIOLOGICAL_COLUMNS for stat in ("mean", "std", "min", "max", "range", "trend")]
        X_train_rows = [extract_window_features(w, feature_columns=list(DEFAULT_PHYSIOLOGICAL_COLUMNS), timestamp_column="timestamp") for w in train_wins]
        X_train = pd.DataFrame(X_train_rows)[expected_feat_names]
        y_train = pd.Series(train_labels)

        rf_fold_model = StressRandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample")
        rf_fold_model.fit(X_train, y_train)

        X_common_rows = [extract_window_features(w, feature_columns=list(DEFAULT_PHYSIOLOGICAL_COLUMNS), timestamp_column="timestamp") for w in common_windows]
        X_common = pd.DataFrame(X_common_rows)[expected_feat_names]

        y_pred_rf = rf_fold_model.predict(X_common)
        y_prob_rf = rf_fold_model.predict_proba(X_common)[:, 1]

        # For personalized detectors: evaluate window-level anomaly detection
        # (Window flagged as anomaly if >= 50% of observations in the window are flagged)
        cal_full = b_df.iloc[:n_cal].copy()
        cal_d_df, _ = select_signal_specific_calibration_windows(calibration_df=cal_full, feature_columns=DEFAULT_PHYSIOLOGICAL_COLUMNS)

        y_pred_a_win = []
        y_score_a_win = []
        y_pred_d_win = []
        y_score_d_win = []

        for w_idx, win in enumerate(common_windows):
            det_a_w = detect_multisignal_anomalies(readings=win, baseline_readings=cal_full, feature_columns=DEFAULT_PHYSIOLOGICAL_COLUMNS, thresholds=3.0)
            det_d_w = detect_multisignal_anomalies(readings=win, baseline_readings=cal_d_df, feature_columns=DEFAULT_PHYSIOLOGICAL_COLUMNS, thresholds=3.0)

            # Window anomaly decision: >= 50% samples anomalous (majority rule)
            pred_a = 1 if (det_a_w["overall_anomaly"].mean() >= 0.5) else 0
            score_a = float(det_a_w[[f"{c}_baseline_z_score" for c in DEFAULT_PHYSIOLOGICAL_COLUMNS]].abs().max().max())

            pred_d = 1 if (det_d_w["overall_anomaly"].mean() >= 0.5) else 0
            score_d = float(det_d_w[[f"{c}_baseline_z_score" for c in DEFAULT_PHYSIOLOGICAL_COLUMNS]].abs().max().max())

            y_pred_a_win.append(pred_a)
            y_score_a_win.append(score_a)
            y_pred_d_win.append(pred_d)
            y_score_d_win.append(score_d)

            # Record granular window comparison entry
            win_start = str(win["timestamp"].iloc[0])
            win_end = str(win["timestamp"].iloc[-1])
            gt_label = int(y_common[w_idx])
            common_window_records.append({
                "participant_id": pid,
                "window_index": w_idx,
                "window_start": win_start,
                "window_end": win_end,
                "ground_truth_label": gt_label,
                "label_name": "stress" if gt_label == 1 else "baseline_heldout",
                "rf_predicted_class": int(y_pred_rf[w_idx]),
                "rf_stress_probability": round(float(y_prob_rf[w_idx]), 4),
                "strat_a_predicted_class": int(pred_a),
                "strat_a_max_zscore": round(float(score_a), 4),
                "strat_d_predicted_class": int(pred_d),
                "strat_d_max_zscore": round(float(score_d), 4),
                "aggregation_rule": "majority_vote_ge_50pct_reading_anomalies",
            })

        # Calculate common metrics
        m_rf_com = calculate_metrics_dict(y_common, y_pred_rf, y_prob_rf)
        m_rf_com["participant_id"] = pid
        m_rf_com["detector"] = "Random Forest (Common Windows)"
        common_heldout_records.append(m_rf_com)

        m_d_com = calculate_metrics_dict(y_common, np.array(y_pred_d_win), np.array(y_score_d_win))
        m_d_com["participant_id"] = pid
        m_d_com["detector"] = "Strategy D (Common Windows)"
        common_heldout_records.append(m_d_com)

        m_a_com = calculate_metrics_dict(y_common, np.array(y_pred_a_win), np.array(y_score_a_win))
        m_a_com["participant_id"] = pid
        m_a_com["detector"] = "Strategy A (Common Windows)"
        common_heldout_records.append(m_a_com)

    # -------------------------------------------------------------------------
    # 4. Compile Tables and Macro Aggregates
    # -------------------------------------------------------------------------
    all_comparison_records = strat_a_records + strat_d_records + pop_thresh_records + rf_records
    comparison_df = pd.DataFrame(all_comparison_records)
    common_df = pd.DataFrame(common_heldout_records)
    common_windows_df = pd.DataFrame(common_window_records)

    # Compute macro summaries
    detectors = [
        "Strategy A (Fixed Baseline)",
        "Strategy D (Personalized Quality-Gated)",
        "Fixed Population Threshold",
        "Random Forest (LOPO Windows)",
    ]
    macro_summaries = {}
    for d in detectors:
        sub = [r for r in all_comparison_records if r["detector"] == d]
        macro_summaries[d] = compute_macro_averages(sub)

    common_detectors = [
        "Strategy A (Common Windows)",
        "Strategy D (Common Windows)",
        "Random Forest (Common Windows)",
    ]
    common_macro_summaries = {}
    for d in common_detectors:
        sub = [r for r in common_heldout_records if r["detector"] == d]
        common_macro_summaries[d] = compute_macro_averages(sub)

    # Save CSVs
    comparison_df.to_csv(output_dir / "comparison_metrics.csv", index=False)
    common_df.to_csv(output_dir / "common_heldout_comparison.csv", index=False)
    common_windows_df.to_csv(output_dir / "common_window_predictions.csv", index=False)

    summary_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology_documentation": {
            "window_aggregation_rule": (
                "For reading-level personalized detectors evaluated on 60s windows, a window is "
                "classified as anomalous (1) if >= 50% of the 1 Hz readings within that window are "
                "flagged as overall anomalies (majority vote). The continuous score is the maximum "
                "absolute z-score across all physiological channels observed during that window."
            ),
            "alert_latency_policy": (
                "Alert latency is measured as elapsed seconds from genuine stress onset (t=0) "
                "to the earliest timestamp of a flagged anomaly (1 Hz) or positive window (60s). "
                "If a detector fails to trigger any alert during the entire stress condition (missed detection), "
                "the latency is right-censored at the total duration of the stress evaluation period."
            ),
            "cross_validation_leakage_control": (
                "For Strategy A and D, calibration windows are selected exclusively from the initial 70% "
                "baseline partition; held-out 30% baseline and stress observations are strictly excluded. "
                "For Random Forest (LOPO), training windows are constructed exclusively from the three other "
                "participants; the held-out test participant is completely unseen during model fitting."
            ),
        },
        "established_protocols_macro_summary": macro_summaries,
        "common_heldout_subset_macro_summary": common_macro_summaries,
        "detection_latencies_seconds": detector_latencies,
    }
    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    # -------------------------------------------------------------------------
    # 5. Generate Visualizations (Matplotlib)
    # -------------------------------------------------------------------------
    generate_comparison_plots(
        comparison_df=comparison_df,
        common_df=common_df,
        macro_summaries=macro_summaries,
        detector_latencies=detector_latencies,
        detector_roc_data=detector_roc_data,
        plots_dir=plots_dir,
    )

    return summary_json


def generate_comparison_plots(
    comparison_df: pd.DataFrame,
    common_df: pd.DataFrame,
    macro_summaries: dict[str, Any],
    detector_latencies: dict[str, dict[str, float]],
    detector_roc_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    plots_dir: Path,
) -> None:
    """Generate high-quality comparison visualization charts using actual experiment outputs."""
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # Plot 1: Macro-Average Detector Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    metrics_to_plot = ["precision", "recall", "f1_score", "accuracy", "specificity", "false_positive_rate"]
    metric_labels = ["Precision", "Recall", "F1-Score", "Accuracy", "Specificity", "FPR"]

    detectors = [
        "Strategy A (Fixed Baseline)",
        "Strategy D (Personalized Quality-Gated)",
        "Fixed Population Threshold",
        "Random Forest (LOPO Windows)",
    ]
    short_names = ["Strategy A\n(Fixed Baseline)", "Strategy D\n(Personalized)", "Population\nThreshold", "Random Forest\n(LOPO)"]
    colors = ["#4A90E2", "#2ECC71", "#E67E22", "#9B59B6"]

    x = np.arange(len(metric_labels))
    width = 0.2

    for i, (det, color) in enumerate(zip(detectors, colors)):
        vals = [macro_summaries[det][f"macro_{m}"] for m in metrics_to_plot]
        errs = [macro_summaries[det][f"macro_{m}_std"] for m in metrics_to_plot]
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, capsize=3, label=short_names[i], color=color, alpha=0.9, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Metric Score (0.0 to 1.0)", fontweight="bold")
    ax.set_title("HealthSensing: Macro-Average Detector Performance Comparison (Mean +/- 1 SD)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_ylim(0.0, 1.15)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.savefig(plots_dir / "detector_metrics_comparison.png")
    plt.close(fig)

    # Plot 2: Participant-Wise Performance (Recall & FPR across S2, S3, S4, S5)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    pids = ["S2", "S3", "S4", "S5"]

    det_keys = [
        "Strategy A (Fixed Baseline)",
        "Strategy D (Personalized Quality-Gated)",
        "Fixed Population Threshold",
        "Random Forest (LOPO Windows)",
    ]
    det_short = ["Strat A", "Strat D", "Pop Thresh", "RF (LOPO)"]

    for idx, pid in enumerate(pids):
        ax = axes[idx // 2, idx % 2]
        p_sub = comparison_df[comparison_df["participant_id"] == pid].set_index("detector")

        recalls = [p_sub.loc[k, "recall"] if k in p_sub.index else 0.0 for k in det_keys]
        fprs = [p_sub.loc[k, "false_positive_rate"] if k in p_sub.index else 0.0 for k in det_keys]

        ind = np.arange(len(det_keys))
        w = 0.35
        b1 = ax.bar(ind - w/2, recalls, w, label="Recall (Sensitivity)", color="#2ECC71", edgecolor="black", linewidth=0.8)
        b2 = ax.bar(ind + w/2, fprs, w, label="False Positive Rate", color="#E74C3C", edgecolor="black", linewidth=0.8)

        ax.set_title(f"Participant {pid} Performance", fontweight="bold")
        ax.set_xticks(ind)
        ax.set_xticklabels(det_short, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.15)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.95)

        # Annotate values
        for bar in b1:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        for bar in b2:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Participant-Wise Recall vs. False-Positive Rate Across Detectors", fontweight="bold", fontsize=14)
    fig.savefig(plots_dir / "participant_performance_comparison.png")
    plt.close(fig)

    # Plot 3: Calibration Strategy Comparison (Strategies A, B, C, D)
    strat_comp_file = Path("experiments/results/wesad_anomaly_evaluation/baseline_strategy_comparison.csv")
    if strat_comp_file.exists():
        sc_df = pd.read_csv(strat_comp_file)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

        strategies = ["Strategy A (Existing)", "Strategy B (Settled)", "Strategy C (Quality-Gated)", "Strategy D (Signal-Specific)"]
        labels = ["A (Full)", "B (Settled)", "C (Quality-Gated)", "D (Signal-Specific)"]
        x_strat = np.arange(len(strategies))

        # Precision & Recall
        for pid, marker, col in zip(["S2", "S3", "S4", "S5"], ["o", "s", "^", "D"], ["#4A90E2", "#E74C3C", "#2ECC71", "#9B59B6"]):
            p_data = sc_df[sc_df["participant_id"] == pid].set_index("strategy")
            rec_vals = [p_data.loc[s, "recall"] if s in p_data.index else 0.0 for s in strategies]
            fpr_vals = [p_data.loc[s, "false_positive_rate"] if s in p_data.index else 0.0 for s in strategies]

            ax1.plot(x_strat, rec_vals, marker=marker, linewidth=2, label=f"Participant {pid}", color=col)
            ax2.plot(x_strat, fpr_vals, marker=marker, linewidth=2, label=f"Participant {pid}", color=col)

        ax1.set_title("Stress Recall Across Calibration Strategies", fontweight="bold")
        ax1.set_xticks(x_strat)
        ax1.set_xticklabels(labels, rotation=15, ha="right")
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_ylabel("Recall")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="lower right")

        ax2.set_title("False Positive Rate Across Calibration Strategies", fontweight="bold")
        ax2.set_xticks(x_strat)
        ax2.set_xticklabels(labels, rotation=15, ha="right")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("FPR")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right")

        fig.suptitle("Impact of Calibration Strategy on Anomaly Detection Sensitivity and False Alarms", fontweight="bold", fontsize=13)
        fig.savefig(plots_dir / "calibration_strategy_comparison.png")
        plt.close(fig)

    # Plot 4: ROC Curves Comparison (AUROC)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    # Plot representative curves for participant S4 and pooled
    for det_name, color, style in zip(["Strategy A (Baseline)", "Strategy D (Personalized)", "Random Forest (LOPO)"], ["#4A90E2", "#2ECC71", "#9B59B6"], ["-", "--", "-."]):
        # Combine all participants' test ground truth and continuous scores
        all_yt = []
        all_ys = []
        for pid in pids:
            if pid in detector_roc_data[det_name]:
                yt, ys = detector_roc_data[det_name][pid]
                all_yt.extend(yt)
                all_ys.extend(ys)
        if len(all_yt) > 0 and len(np.unique(all_yt)) == 2:
            fprs, tprs, _ = roc_curve(all_yt, all_ys)
            auc_val = roc_auc_score(all_yt, all_ys)
            ax.plot(fprs, tprs, label=f"{det_name} (AUC = {auc_val:.3f})", color=color, linestyle=style, linewidth=2.2)

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Chance (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontweight="bold", pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.savefig(plots_dir / "roc_curves_comparison.png")
    plt.close(fig)

    # Plot 5: Detection Latency Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    lat_detectors = ["Strategy A (Baseline)", "Strategy D (Personalized)", "Population Threshold", "Random Forest (LOPO)"]
    lat_colors = ["#4A90E2", "#2ECC71", "#E67E22", "#9B59B6"]

    x_lat = np.arange(len(pids))
    width_lat = 0.2

    for i, (det, color) in enumerate(zip(lat_detectors, lat_colors)):
        lats = [detector_latencies[det].get(p, 0.0) for p in pids]
        offset = (i - 1.5) * width_lat
        bars = ax.bar(x_lat + offset, lats, width_lat, label=det, color=color, alpha=0.9, edgecolor="black", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{int(h)}s", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Detection Latency from Stress Onset (Seconds)", fontweight="bold")
    ax.set_title("Stress Detection Alert Latency Across Participants (Lower is Faster)", fontweight="bold", pad=12)
    ax.set_xticks(x_lat)
    ax.set_xticklabels([f"Participant {p}" for p in pids], fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.savefig(plots_dir / "detection_latency_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    print("Executing comprehensive detector comparison...")
    summary = run_comprehensive_detector_comparison()
    print("Comparison execution completed.")
    print("Macro Summaries:")
    print(json.dumps(summary["established_protocols_macro_summary"], indent=2))
    print("\nCommon Held-out Subset Macro Summaries:")
    print(json.dumps(summary["common_heldout_subset_macro_summary"], indent=2))
    print("\nDetection Latencies (seconds):")
    print(json.dumps(summary["detection_latencies_seconds"], indent=2))
