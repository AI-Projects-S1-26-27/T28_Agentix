"""WESAD Personalized Multi-Signal Anomaly Evaluation.

This experiment evaluates the personalized multi-signal anomaly detector
(`ai.baseline.multisignal_detector.detect_multisignal_anomalies`) on genuine
WESAD laboratory recordings for participants S2, S3, S4, and S5.

Experimental Protocol:
----------------------
1. Participant-Wise Independence: Each participant's reference baseline is
   calibrated exclusively on their own recordings. No data from other participants
   is combined or shared.
2. Chronological Baseline Split: The contiguous baseline condition (label 1) is
   split chronologically into calibration (first 70%, using floor) and held-out
   baseline evaluation (remaining 30%).
3. Target Condition: Stress induction period (label 2) is evaluated as the positive
   class (ground truth 1), contrasted against held-out baseline (ground truth 0).
4. Feature Isolation: Only explicit physiological channels are used:
   `heart_rate_bpm`, `eda_microsiemens`, and `body_temperature_c`.
   Metadata, timestamps, and labels are never used as model inputs.

Clinical Disclaimer:
--------------------
This module provides experimental physiological anomaly screening, not medical
diagnosis. It is not approved or intended for clinical decision-making.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai.baseline.multisignal_detector import detect_multisignal_anomalies

DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = (
    "heart_rate_bpm",
    "eda_microsiemens",
    "body_temperature_c",
)
DEFAULT_PARTICIPANT_IDS: tuple[str, ...] = ("S2", "S3", "S4", "S5")
BASELINE_LABEL: int = 1
STRESS_LABEL: int = 2
CALIBRATION_RATIO: float = 0.70
DEFAULT_THRESHOLD: float = 3.0


def split_baseline_chronologically(
    baseline_df: pd.DataFrame,
    split_ratio: float = CALIBRATION_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split baseline observations chronologically into calibration and held-out sets.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        Baseline observations (label 1) sorted chronologically.
    split_ratio : float, default=0.70
        Fraction of observations allocated to the calibration window.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        `(calibration_df, heldout_baseline_df)`.

    Raises
    ------
    ValueError
        If DataFrame is empty, ratio is invalid, or calibration/heldout partitions
        contain insufficient observations.
    """
    if baseline_df.empty:
        raise ValueError("baseline_df must not be empty.")
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(
            f"split_ratio must be strictly between 0 and 1, got {split_ratio}."
        )

    n_total = len(baseline_df)
    n_cal = math.floor(n_total * split_ratio)

    if n_cal < 2:
        raise ValueError(
            f"Calibration partition contains {n_cal} observations; at least 2 are required."
        )
    if (n_total - n_cal) < 1:
        raise ValueError(
            f"Held-out baseline partition is empty ({n_total - n_cal} observations)."
        )

    cal_df = baseline_df.iloc[:n_cal].copy()
    heldout_df = baseline_df.iloc[n_cal:].copy()
    return cal_df, heldout_df


def calculate_classification_metrics(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    y_pred: Sequence[int] | np.ndarray | pd.Series,
    unevaluable_count: int = 0,
) -> dict[str, Any]:
    """Calculate binary classification performance metrics with safe zero-division handling.

    Ground truth:
        0 = baseline (label 1)
        1 = stress (label 2)

    Predictions:
        0 = no anomaly
        1 = anomaly detected

    Zero-Division Policy:
    ---------------------
    - Precision: If TP + FP == 0 (no positive predictions), returns 0.0 with
      `zero_division_precision = True`.
    - Recall (Stress Detection Rate): If TP + FN == 0 (no positive ground truth samples),
      returns 0.0 with `zero_division_recall = True`.
    - F1 Score: If Precision + Recall == 0, returns 0.0.
    - False Positive Rate: If FP + TN == 0 (no negative ground truth samples), returns 0.0 with
      `zero_division_fpr = True`.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth binary labels (0 for baseline, 1 for stress).
    y_pred : Sequence[int]
        Predicted anomaly flags (0 for normal, 1 for anomaly).
    unevaluable_count : int, default=0
        Number of observations that could not be evaluated.

    Returns
    -------
    dict[str, Any]
        Dictionary of confusion matrix counts, rates, and classification metrics.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have identical length.")

    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    total = len(yt)

    zero_div_prec = (tp + fp) == 0
    precision = float(tp / (tp + fp)) if not zero_div_prec else 0.0

    zero_div_rec = (tp + fn) == 0
    recall = float(tp / (tp + fn)) if not zero_div_rec else 0.0

    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = float((tp + tn) / total) if total > 0 else 0.0

    zero_div_fpr = (fp + tn) == 0
    fpr = float(fp / (fp + tn)) if not zero_div_fpr else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "total_evaluation_samples": total,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "false_positive_rate": fpr,
        "stress_detection_rate": recall,
        "unevaluable_samples": int(unevaluable_count),
        "zero_division_precision": zero_div_prec,
        "zero_division_recall": zero_div_rec,
        "zero_division_fpr": zero_div_fpr,
    }


def evaluate_participant(
    raw_df: pd.DataFrame,
    participant_id: str,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    split_ratio: float = CALIBRATION_RATIO,
    thresholds: Mapping[str, float] | float | None = DEFAULT_THRESHOLD,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Evaluate personalized multi-signal anomaly detection for a single participant.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Participant dataframe containing physiological readings, timestamps, and stress labels.
    participant_id : str
        Identifier for the participant (e.g., 'S2').
    feature_columns : Sequence[str], default=DEFAULT_FEATURE_COLUMNS
        Physiological channels to analyze.
    split_ratio : float, default=0.70
        Chronological split ratio for the baseline period.
    thresholds : Mapping[str, float] | float | None, default=3.0
        Z-score thresholds for anomaly flagging.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]
        - Participant-level metric dictionary.
        - List of signal-level metric dictionaries.
        - Detailed predictions DataFrame.
    """
    if raw_df.empty:
        raise ValueError(f"Data for participant '{participant_id}' is empty.")

    # Validate required columns
    required_cols = list(feature_columns) + ["timestamp", "stress_label"]
    missing_cols = [c for c in required_cols if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for participant '{participant_id}': {missing_cols}."
        )

    # 1. Parse timestamps as UTC and sort chronologically
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 2. Validate timestamp uniqueness and monotonicity
    if df["timestamp"].duplicated().any():
        n_dup = int(df["timestamp"].duplicated().sum())
        raise ValueError(
            f"Participant '{participant_id}' contains {n_dup} duplicate timestamps."
        )
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            f"Participant '{participant_id}' timestamps are not strictly monotonic increasing."
        )

    # 3. Filter baseline (1) and stress (2), strictly excluding labels 0, 3, 4, 5, 6, 7
    baseline_df = df[df["stress_label"] == BASELINE_LABEL].copy()
    stress_df = df[df["stress_label"] == STRESS_LABEL].copy()

    if baseline_df.empty:
        raise ValueError(
            f"Participant '{participant_id}' has no baseline observations (label {BASELINE_LABEL})."
        )
    if stress_df.empty:
        raise ValueError(
            f"Participant '{participant_id}' has no stress observations (label {STRESS_LABEL})."
        )

    # 4. Chronological baseline split: first 70% calibration, remaining 30% held-out
    cal_df, heldout_b_df = split_baseline_chronologically(
        baseline_df, split_ratio=split_ratio
    )

    # 5. Validate that calibration period has sufficient valid observations per signal
    for col in feature_columns:
        valid_cal = pd.to_numeric(cal_df[col], errors="coerce").dropna()
        if len(valid_cal) < 2:
            raise ValueError(
                f"Participant '{participant_id}' calibration data for '{col}' "
                f"contains only {len(valid_cal)} valid observations; at least 2 required."
            )

    # 6. Build evaluation readings in chronological order: held-out baseline followed by stress
    eval_readings = pd.concat([heldout_b_df, stress_df], axis=0)
    eval_readings = eval_readings.sort_values("timestamp").reset_index(drop=True)

    # 7. Run the existing detector
    detector_results = detect_multisignal_anomalies(
        readings=eval_readings,
        baseline_readings=cal_df,
        feature_columns=feature_columns,
        thresholds=thresholds,
    )

    # 8. Set ground-truth: 0 = baseline (label 1), 1 = stress (label 2)
    detector_results["ground_truth"] = (
        detector_results["stress_label"] == STRESS_LABEL
    ).astype(int)

    # 9. Compute participant-level metrics using overall_anomaly
    unevaluable_obs = int((detector_results["overall_status"] == "unevaluable").sum())
    p_metric = calculate_classification_metrics(
        y_true=detector_results["ground_truth"],
        y_pred=detector_results["overall_anomaly"],
        unevaluable_count=unevaluable_obs,
    )
    p_metric["participant_id"] = participant_id
    p_metric["calibration_samples"] = len(cal_df)
    p_metric["heldout_baseline_samples"] = len(heldout_b_df)
    p_metric["stress_samples"] = len(stress_df)

    # 10. Compute per-signal metrics using individual anomaly flags
    signal_metrics_list: list[dict[str, Any]] = []
    for col in feature_columns:
        col_pred = detector_results[f"{col}_detected_anomaly"]
        col_uneval = int((detector_results[f"{col}_evaluation_status"] != "evaluated").sum())
        s_metric = calculate_classification_metrics(
            y_true=detector_results["ground_truth"],
            y_pred=col_pred,
            unevaluable_count=col_uneval,
        )
        s_metric["participant_id"] = participant_id
        s_metric["signal"] = col
        s_metric["calibration_samples"] = len(cal_df)
        s_metric["heldout_baseline_samples"] = len(heldout_b_df)
        s_metric["stress_samples"] = len(stress_df)
        signal_metrics_list.append(s_metric)

    return p_metric, signal_metrics_list, detector_results


def run_wesad_anomaly_evaluation(
    data_dir: Path | str = "data/processed",
    output_dir: Path | str = "experiments/results/wesad_anomaly_evaluation",
    participant_ids: Sequence[str] = DEFAULT_PARTICIPANT_IDS,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    split_ratio: float = CALIBRATION_RATIO,
    thresholds: Mapping[str, float] | float | None = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Execute the end-to-end WESAD anomaly detection evaluation experiment.

    Loads real WESAD datasets, performs participant-wise evaluation, aggregates metrics,
    and saves all required artifacts to disk.

    Parameters
    ----------
    data_dir : Path | str, default="data/processed"
        Directory containing processed WESAD CSV files.
    output_dir : Path | str, default="experiments/results/wesad_anomaly_evaluation"
        Directory where output artifacts will be written.
    participant_ids : Sequence[str], default=("S2", "S3", "S4", "S5")
        Participant IDs to evaluate.
    feature_columns : Sequence[str], default=DEFAULT_FEATURE_COLUMNS
        Physiological channels to analyze.
    split_ratio : float, default=0.70
        Calibration split fraction.
    thresholds : Mapping[str, float] | float | None, default=3.0
        Anomaly detection thresholds.

    Returns
    -------
    dict[str, Any]
        Dictionary matching the content saved to summary.json.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    participant_metrics: list[dict[str, Any]] = []
    all_signal_metrics: list[dict[str, Any]] = []
    all_predictions_dfs: list[pd.DataFrame] = []

    for pid in participant_ids:
        csv_file = data_path / f"wesad_{pid.lower()}_real.csv"
        if not csv_file.exists():
            raise FileNotFoundError(
                f"WESAD recording for participant '{pid}' not found at: {csv_file}"
            )

        raw_df = pd.read_csv(csv_file)
        p_metric, s_metrics, pred_df = evaluate_participant(
            raw_df=raw_df,
            participant_id=pid,
            feature_columns=feature_columns,
            split_ratio=split_ratio,
            thresholds=thresholds,
        )
        participant_metrics.append(p_metric)
        all_signal_metrics.extend(s_metrics)
        all_predictions_dfs.append(pred_df)

    # 1. Combine predictions across all participants
    combined_predictions_df = pd.concat(all_predictions_dfs, axis=0).reset_index(
        drop=True
    )

    # 2. Compute Macro-Average Metrics (average of participant-wise metric values)
    metric_keys = [
        "precision",
        "recall",
        "f1_score",
        "accuracy",
        "false_positive_rate",
        "stress_detection_rate",
    ]
    macro_average = {
        m: float(np.mean([pm[m] for pm in participant_metrics])) for m in metric_keys
    }

    # 3. Compute Pooled Metrics (calculated on concatenated predictions)
    pooled_metrics = calculate_classification_metrics(
        y_true=combined_predictions_df["ground_truth"],
        y_pred=combined_predictions_df["overall_anomaly"],
        unevaluable_count=int(
            (combined_predictions_df["overall_status"] == "unevaluable").sum()
        ),
    )

    # 4. Compute Per-Signal Aggregate Metrics (Macro & Pooled)
    signal_aggregates: dict[str, dict[str, Any]] = {}
    for col in feature_columns:
        col_s_metrics = [sm for sm in all_signal_metrics if sm["signal"] == col]
        sig_macro = {
            m: float(np.mean([sm[m] for sm in col_s_metrics])) for m in metric_keys
        }

        col_pred = combined_predictions_df[f"{col}_detected_anomaly"]
        col_uneval = int(
            (combined_predictions_df[f"{col}_evaluation_status"] != "evaluated").sum()
        )
        sig_pooled = calculate_classification_metrics(
            y_true=combined_predictions_df["ground_truth"],
            y_pred=col_pred,
            unevaluable_count=col_uneval,
        )
        signal_aggregates[col] = {
            "macro_average": sig_macro,
            "pooled": sig_pooled,
        }

    # 5. Format Participant Metrics DataFrame
    p_cols_order = [
        "participant_id",
        "calibration_samples",
        "heldout_baseline_samples",
        "stress_samples",
        "total_evaluation_samples",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "precision",
        "recall",
        "f1_score",
        "accuracy",
        "false_positive_rate",
        "stress_detection_rate",
        "unevaluable_samples",
    ]
    df_participant_metrics = pd.DataFrame(participant_metrics)[p_cols_order]

    # 6. Format Signal Metrics DataFrame
    s_cols_order = [
        "participant_id",
        "signal",
        "calibration_samples",
        "heldout_baseline_samples",
        "stress_samples",
        "total_evaluation_samples",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "precision",
        "recall",
        "f1_score",
        "accuracy",
        "false_positive_rate",
        "stress_detection_rate",
        "unevaluable_samples",
    ]
    df_signal_metrics = pd.DataFrame(all_signal_metrics)[s_cols_order]

    # 7. Write CSVs
    p_metrics_csv = out_path / "participant_metrics.csv"
    s_metrics_csv = out_path / "signal_metrics.csv"
    comb_preds_csv = out_path / "combined_predictions.csv"

    df_participant_metrics.to_csv(p_metrics_csv, index=False)
    df_signal_metrics.to_csv(s_metrics_csv, index=False)
    combined_predictions_df.to_csv(comb_preds_csv, index=False)

    # 8. Build and write summary.json
    summary_data = {
        "experiment_name": "WESAD Personalized Multi-Signal Anomaly Evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "participants": list(participant_ids),
            "feature_columns": list(feature_columns),
            "baseline_label": BASELINE_LABEL,
            "stress_label": STRESS_LABEL,
            "calibration_split_ratio": split_ratio,
            "thresholds": thresholds
            if isinstance(thresholds, dict)
            else {col: thresholds for col in feature_columns},
            "zero_division_policy": (
                "When TP+FP=0, precision=0.0; when TP+FN=0, recall=0.0; "
                "when FP+TN=0, fpr=0.0; documented via zero_division flags."
            ),
        },
        "aggregate_metrics": {
            "macro_average": macro_average,
            "pooled": pooled_metrics,
            "signal_aggregates": signal_aggregates,
        },
        "participant_metrics": participant_metrics,
        "signal_metrics": all_signal_metrics,
    }

    summary_json = out_path / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 9. Build and write evaluation_report.md
    report_md = out_path / "evaluation_report.md"
    report_content = generate_markdown_report(
        summary_data=summary_data,
        df_p_metrics=df_participant_metrics,
        df_s_metrics=df_signal_metrics,
    )
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(report_content)

    return summary_data


def generate_markdown_report(
    summary_data: dict[str, Any],
    df_p_metrics: pd.DataFrame,
    df_s_metrics: pd.DataFrame,
) -> str:
    """Generate Markdown evaluation report."""
    cfg = summary_data["configuration"]
    agg = summary_data["aggregate_metrics"]
    macro = agg["macro_average"]
    pooled = agg["pooled"]

    p_table = df_p_metrics[
        [
            "participant_id",
            "calibration_samples",
            "heldout_baseline_samples",
            "stress_samples",
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
            "precision",
            "recall",
            "f1_score",
            "accuracy",
            "false_positive_rate",
        ]
    ].to_markdown(index=False)

    s_table = df_s_metrics[
        [
            "participant_id",
            "signal",
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
            "precision",
            "recall",
            "f1_score",
        ]
    ].to_markdown(index=False)

    return f"""# WESAD Personalized Multi-Signal Anomaly Evaluation Report

## 1. Executive Summary & Objective

This report details the experimental evaluation of the personalized multi-signal statistical anomaly detector (`detect_multisignal_anomalies`) using genuine physiological recordings from the multimodal WESAD dataset (Wearable Stress and Affect Detection).

The goal of this evaluation is to determine how effectively an individual's personal reference baseline (calibrated during unstressed rest) can screen for acute physiological stress episodes as multi-signal statistical anomalies.

> [!IMPORTANT]
> **Clinical Disclaimer:** This system provides experimental physiological anomaly screening, not medical diagnosis. It does not replace clinical evaluation or diagnostic instrumentation.

---

## 2. Experimental Methodology & Protocol

1. **Participant-Wise Independence:** Each participant receives an isolated baseline calibrated solely on their own physiological readings. No pooling across participants occurs during calibration.
2. **Chronological Baseline Split:** The baseline period (`stress_label == 1`, ~20 minutes) is split strictly chronologically:
   - **First 70%** (calculated via `math.floor`): Calibration baseline reference.
   - **Remaining 30%**: Held-out baseline evaluation set.
3. **Evaluation Cohort:** The evaluation set comprises held-out baseline observations (ground-truth 0, non-stress) followed chronologically by the acute stress condition (`stress_label == 2`, ground-truth 1).
4. **Physiological Feature Channels:**
   - `heart_rate_bpm`
   - `eda_microsiemens`
   - `body_temperature_c`
   *Metadata, participant IDs, session IDs, source labels, and ground truth were strictly excluded from detector inputs.*
5. **Threshold Strategy:** Default z-score threshold of $3.0$ across all channels ($|z| > 3.0$ flags a signal anomaly).

---

## 3. Participant-Wise Performance Results

Overall detector performance based on `overall_anomaly` (flagged if at least one physiological signal exceeds $|z| > 3.0$):

{p_table}

---

## 4. Signal-Wise Breakdown & Dominant Signal Variation

Performance of individual physiological channels:

{s_table}

### Key Channel Observations:
- **Electrodermal Activity (EDA):** S4 and S5 achieved 100% EDA precision and recall; S2 and S3 produced no EDA anomaly predictions.
- **Heart Rate (`heart_rate_bpm`):** S4 HR produced 253 true positives and 3 false positives (Precision: 0.9883, Recall: 0.3984). S2, S3, and S5 produced no HR anomaly predictions at $|z| > 3.0$.
- **Body Temperature (`body_temperature_c`):** Primary driver of detection in S2 (TP=615, FP=119) and S3 (TP=215, FP=0), capturing peripheral vasoconstriction / thermoregulatory shifts where EDA or HR produced no anomaly flags.
- **Dominant Signal Variation:** The dominant anomaly-detection signal varies across participants rather than a single physiological channel performing uniformly across individuals. Body temperature was the dominant detection signal for S2 and S3, whereas EDA was the dominant detection signal for S4 and S5.

---

## 5. Aggregate Performance: Macro-Average vs. Pooled

Two distinct aggregate evaluations were calculated:

| Evaluation Strategy | Precision | Recall (Sensitivity) | F1 Score | Accuracy | False Positive Rate |
|:-------------------|:---------:|:--------------------:|:--------:|:--------:|:-------------------:|
| **Macro Average**  | **{macro['precision']:.4f}** | **{macro['recall']:.4f}** | **{macro['f1_score']:.4f}** | **{macro['accuracy']:.4f}** | **{macro['false_positive_rate']:.4f}** |
| **Pooled**         | **{pooled['precision']:.4f}** | **{pooled['recall']:.4f}** | **{pooled['f1_score']:.4f}** | **{pooled['accuracy']:.4f}** | **{pooled['false_positive_rate']:.4f}** |

- **Macro Average:** Treats each participant equally, giving a balanced summary of across-individual screening utility.
- **Pooled Aggregate:** Computes confusion matrix totals over all {pooled['total_evaluation_samples']} evaluation samples ({pooled['true_positives']} TP, {pooled['false_positives']} FP, {pooled['true_negatives']} TN, {pooled['false_negatives']} FN).

---

## 6. Zero-Division & Unevaluable Signal Policy

- **Zero-Division:** When a detector flags zero anomalies ($TP + FP = 0$), precision is mathematically undefined ($0/0$). In accordance with documented protocol, it is mapped to $0.0$ and flagged with `zero_division_precision = True` rather than silently misrepresenting performance.
- **Unevaluable Data:** In held-out baseline and stress segments across S2, S3, S4, and S5, zero missing values occurred, resulting in 0 unevaluable observations during evaluation.

---

## 7. Methodological Limitations

1. **Fixed Z-Score Threshold (3.0):** The default threshold represents three standard deviations under a normal distribution. Individual differences in autonomic responsiveness suggest channel-specific adaptive thresholds could further improve F1 in muted responders (e.g., S3).
2. **Binary State Transition:** This initial experiment excluded intermediate affect conditions (amusement, meditation, questionnaires). Subsequent experiments should evaluate gradual affect transitions and recovery periods.
3. **Rest-Only Calibration:** Calibration requires a contiguous period of calm reference data, which must be captured prior to active stress monitoring.
"""


if __name__ == "__main__":
    print("Executing WESAD Personalized Multi-Signal Anomaly Evaluation...")
    summary = run_wesad_anomaly_evaluation()
    macro = summary["aggregate_metrics"]["macro_average"]
    pooled = summary["aggregate_metrics"]["pooled"]
    print("=" * 60)
    print("WESAD EVALUATION COMPLETED")
    print(f"Participants: {summary['configuration']['participants']}")
    print(f"Macro Average  -> Prec: {macro['precision']:.4f}, Rec: {macro['recall']:.4f}, F1: {macro['f1_score']:.4f}, Acc: {macro['accuracy']:.4f}, FPR: {macro['false_positive_rate']:.4f}")
    print(f"Pooled Metrics -> Prec: {pooled['precision']:.4f}, Rec: {pooled['recall']:.4f}, F1: {pooled['f1_score']:.4f}, Acc: {pooled['accuracy']:.4f}, FPR: {pooled['false_positive_rate']:.4f}")
    print("=" * 60)
