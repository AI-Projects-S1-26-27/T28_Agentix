# HealthSensing: Consolidated Machine Learning Experiment & Reproducibility Log

> [!IMPORTANT]
> **Clinical Disclaimer:** This document logs experimental research benchmarks and algorithmic evaluation
> for physiological anomaly screening. It does NOT constitute medical device validation, clinical verification,
> or diagnostic claims.

---

## 1. Provenance and Hardware Specifications

- **Dataset Source:** Multimodal WESAD (Wearable Stress and Affect Detection) public benchmark dataset (Schmidt et al., 2018).
- **Physical Sensor Device:** Empatica E4 wristband.
- **Physical Measurement Modalities:**
  1. Optical photoplethysmography (PPG) -> Heart Rate (`heart_rate_bpm`, resampled to 1 Hz).
  2. Electrodermal skin conductance -> Galvanic Skin Response (`eda_microsiemens`, 4 Hz resampled to 1 Hz).
  3. Infrared thermopile -> Peripheral skin temperature (`body_temperature_c`, 4 Hz resampled to 1 Hz).
- **Sampling Frequency:** Nominal 1.0 Hz (1 observation per second). Continuous temporal blocks verified with `expected_sampling_interval_sec = 1.0` and `timestamp_tolerance_sec = 0.1`.
- **Participants Evaluated ($N=4$):**
  - **S2:** 6,080 total observations (Baseline: 1,144; Stress: 615; Amusement/Transient/Meditation: 4,321).
  - **S3:** 6,491 total observations (Baseline: 1,140; Stress: 640; Amusement/Transient/Meditation: 4,711).
  - **S4:** 6,416 total observations (Baseline: 1,158; Stress: 635; Amusement/Transient/Meditation: 4,623).
  - **S5:** 6,267 total observations (Baseline: 1,198; Stress: 645; Amusement/Transient/Meditation: 4,424).
- **Label Schema & Inclusions:**
  - Label 0: Transient / non-study periods (excluded from evaluation).
  - Label 1: Baseline resting condition (used for calibration and negative test evaluation).
  - Label 2: Trier Social Stress Test (TSST) and mental arithmetic acute stress induction (evaluated as positive stress condition).
  - Label 3: Amusement (excluded from binary stress evaluation).
  - Label 4: Meditation / recovery (excluded from binary stress evaluation).
  - Labels 5, 6, 7: Study-specific questionnaires / intervals (excluded).

---

## 2. Physiological Feature Extraction Specifications

- **Excluded Attributes:** Timestamps (`timestamp`), participant identifiers (`participant_id`), session IDs (`session_id`), source labels (`source`), and ground-truth labels (`stress_label`, `true_label`, `true_class`, `predicted_class`) are strictly barred from feature vectors to prevent data leakage.
- **Physiological Channels Analyzed (3):**
  1. `heart_rate_bpm`
  2. `eda_microsiemens`
  3. `body_temperature_c`
- **Feature Extraction for Windowed Machine Learning (18 features total):**
  For each 60-second window, 6 statistical features are calculated per channel:
  - Mean: $\mu = \frac{1}{n} \sum x_i$
  - Sample Standard Deviation: $s = \sqrt{\frac{1}{n-1} \sum (x_i - \mu)^2}$ ($\text{ddof}=1$)
  - Minimum: $\min(x)$
  - Maximum: $\max(x)$
  - Range: $\max(x) - \min(x)$
  - Linear Trend: First-order polynomial slope ($\beta_1$ from ordinary least squares regression over time index)
- **Feature Column Order:**
  ```python
  [
      "heart_rate_bpm_mean", "heart_rate_bpm_std", "heart_rate_bpm_min", "heart_rate_bpm_max", "heart_rate_bpm_range", "heart_rate_bpm_trend",
      "eda_microsiemens_mean", "eda_microsiemens_std", "eda_microsiemens_min", "eda_microsiemens_max", "eda_microsiemens_range", "eda_microsiemens_trend",
      "body_temperature_c_mean", "body_temperature_c_std", "body_temperature_c_min", "body_temperature_c_max", "body_temperature_c_range", "body_temperature_c_trend"
  ]
  ```

---

## 3. Evaluation Split Protocols

### Protocol A: Personalized Multi-Signal Anomaly Detection (Reading-Level)
1. **Participant Isolation:** Each participant's reference baseline is fitted strictly on their own resting data. No pooling occurs across subjects.
2. **Chronological 70/30 Baseline Split:**
   - Calibration Partition: First 70% of Label 1 (using `math.floor`).
     - S2: 800 observations.
     - S3: 798 observations.
     - S4: 810 observations.
     - S5: 838 observations.
   - Held-Out Baseline Evaluation Partition: Remaining 30% of Label 1 (Ground Truth 0).
     - S2: 344 observations.
     - S3: 342 observations.
     - S4: 348 observations.
     - S5: 360 observations.
   - Target Stress Partition: Entire Label 2 period (Ground Truth 1).
     - S2: 615 observations.
     - S3: 640 observations.
     - S4: 635 observations.
     - S5: 645 observations.
   - Total Evaluation Readings: 3,929 observations across the 4 participants.

### Protocol B: Supervised Random Forest Benchmark (Window-Level LOPO)
1. **Leave-One-Participant-Out (LOPO) Cross-Validation:**
   - 4 cross-validation folds. In each fold, 3 participants form the training set; the remaining 4th participant serves strictly as the unseen test evaluation set.
2. **Windowing Parameters:**
   - Window duration: 60 observations (60s at 1 Hz).
   - Advance stride: 30 observations (30s at 1 Hz, 50% overlap).
   - Window purity: Windows are extracted strictly within contiguous, single-label blocks. Windows crossing label transitions or timestamp gaps $> 1.1$s are excluded.
3. **Total Valid Windows (228):**
   - S2: 37 baseline windows (Class 0), 19 stress windows (Class 1) -> 56 test windows.
   - S3: 37 baseline windows (Class 0), 20 stress windows (Class 1) -> 57 test windows.
   - S4: 37 baseline windows (Class 0), 20 stress windows (Class 1) -> 57 test windows.
   - S5: 38 baseline windows (Class 0), 20 stress windows (Class 1) -> 58 test windows.
   - Aggregate: 149 Baseline Windows (Class 0) + 79 Stress Windows (Class 1) = 228 Windows.

### Protocol C: Common Held-Out Evaluation Subset (Identical Evaluation Periods)
- **Evaluation Partition:** Evaluates all models on identical temporal intervals: the held-out 30% baseline partition (Class 0) and the stress partition (Class 1).
- **Windows per Participant:** 10–11 held-out baseline windows (S2: 10, S3: 10, S4: 10, S5: 11; total 41) + 19–20 stress windows (S2: 19, S3: 20, S4: 20, S5: 20; total 79) = 120 common windows across the 4 participants.

---

## 4. Algorithmic Configurations and Random Seeds

### 4.1 Statistical Baseline Detectors (Strategies A, B, C, D)
- **Fixed Z-Score Anomaly Threshold:** $|z| > 3.0$ across all channels.
- **Zero-Division Policy:** When $\text{TP} + \text{FP} = 0$, precision is defined as $0.0$; when $\text{TP} + \text{FN} = 0$, recall is defined as $0.0$; when $\text{FP} + \text{TN} = 0$, false positive rate is defined as $0.0$.
- **Stationarity Diagnostic Parameters:**
  - Window size: 120 seconds.
  - Variance inflation threshold: 3.5.
  - Mean drift threshold: 20.0%.
  - Minimum candidate suffix observations: 480 observations (4 consecutive 120s windows).
  - Offset scan step: 60 seconds.

### 4.2 Random Forest Classifier
- **Implementation:** `sklearn.ensemble.RandomForestClassifier` wrapped with anti-leakage input validation.
- **Number of Estimators:** `n_estimators = 300`
- **Class Weight Strategy:** `class_weight = "balanced_subsample"`
- **Random State Seed:** `random_state = 42` (fixed across all folds for deterministic reproducibility)
- **Missing Value Imputation:** `sklearn.impute.SimpleImputer(strategy="median")` fitted strictly on training fold data.
- **Parallel Execution:** `n_jobs = -1`

---

## 5. Experiment Outputs and Artifact Inventory

| Experiment Module | Output Path / Artifact | Purpose |
|:---|:---|:---|
| `evaluate_wesad_anomalies.py` | `experiments/results/wesad_anomaly_evaluation/participant_metrics.csv` | Primary 70/30 personalized anomaly evaluation on S2–S5 |
| `evaluate_wesad_anomalies.py` | `experiments/results/wesad_anomaly_evaluation/summary.json` | Complete aggregate and signal-wise metric breakdown |
| `evaluate_baseline_strategies.py` | `experiments/results/wesad_anomaly_evaluation/baseline_strategy_comparison.csv` | Head-to-head comparison of Strategies A, B, C, and D |
| `evaluate_calibration_sensitivity.py` | `experiments/results/wesad_anomaly_evaluation/calibration_duration_sensitivity.csv` | Strategy D sensitivity across 70%, 75%, 80%, 85% partitions |
| `evaluate_fixed_holdout.py` | `experiments/results/wesad_anomaly_evaluation/fixed_holdout_comparison.csv` | Confirmatory validation on fixed 15% holdout partition |
| `diagnose_s3_strategy_d.py` | `experiments/results/wesad_anomaly_evaluation/s3_strategy_d_signal_metrics.csv` | Detailed channel-level diagnostic of S3 acclimation |
| `train_rf.py` | `experiments/results/random_forest/participant_metrics.csv` | LOPO cross-validation fold metrics (S2, S3, S4, S5) |
| `train_rf.py` | `experiments/results/random_forest/pooled_metrics.json` | Pooled confusion matrix and macro metrics ($\text{ddof}=1$) |
| `train_rf.py` | `experiments/results/random_forest/rf_model_all_participants.joblib` | Final retrained inference model (strictly for deployment) |
| `evaluate_detector_comparison.py` | `experiments/results/detector_comparison/comparison_metrics.csv` | Cross-detector comparison table across all participants |
| `evaluate_detector_comparison.py` | `experiments/results/detector_comparison/common_heldout_comparison.csv` | Direct apples-to-apples common subset comparison |
| `evaluate_detector_comparison.py` | `experiments/results/detector_comparison/comparison_summary.json` | Macro summaries, AUROC, and alert latencies |
| `evaluate_detector_comparison.py` | `experiments/plots/*.png` | 5 publication-quality comparison charts |

---

## 6. Execution Runtime & Environment

- **Execution Environment:** Windows 11, x86_64, Python 3.12.4.
- **Core Dependencies:**
  - `numpy`: 1.26.4
  - `pandas`: 2.2.2
  - `scikit-learn`: 1.4.2
  - `scipy`: 1.13.1
  - `matplotlib`: 3.8.4
  - `pytest`: 7.4.4
- **Benchmark Execution Runtimes:**
  - `evaluate_wesad_anomalies.py`: ~1.8 seconds.
  - `evaluate_baseline_strategies.py`: ~3.2 seconds.
  - `train_rf.py` (LOPO 4-fold cross-validation + retrain): ~8.4 seconds.
  - `evaluate_detector_comparison.py`: ~4.6 seconds.
  - Complete repository test suite (203 tests): 27.17 seconds.

---

## 7. Step-by-Step Reproducibility Instructions

To reproduce all reported metrics, tables, and visualization plots from scratch:

```bash
# 1. Verify environment and execute full test suite
python -m pytest -q

# 2. Run WESAD personalized anomaly evaluation (Strategy A baseline)
python -m ai.experiments.evaluate_wesad_anomalies

# 3. Run baseline calibration strategies evaluation (Strategies A, B, C, D)
python -m ai.experiments.evaluate_baseline_strategies

# 4. Run Random Forest LOPO benchmark and generate model artifacts
python -m ai.training.train_rf

# 5. Run comprehensive cross-detector comparison and generate plots
python -m ai.experiments.evaluate_detector_comparison
```

---

## 8. Explicit Metadata Limitations

- **Unavailable Data:**
  - Environmental sensor channels (ambient temperature, humidity, AQI, PM2.5) are NOT present in the laboratory WESAD dataset (only physiological PPG, EDA, and skin temperature were collected on the E4). Environmental responses are tested exclusively via synthetic simulator streams.
  - Continuous blood pressure and SpO2 were not recorded on the Empatica E4 wristband in WESAD.
  - Latency in real-world human perception was not measured; alert latency is defined strictly as elapsed seconds from protocol stress onset to first algorithmic detection.
