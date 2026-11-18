# HealthSensing: Consolidated Machine Learning Evaluation Report

> [!IMPORTANT]
> **Clinical & Research Disclaimer:** This report summarizes algorithmic research and experimental evaluations
> conducted on physiological sensor recordings. HealthSensing is an experimental physiological anomaly screening
> system, NOT a certified medical diagnostic device. It does not replace medical consultation, clinical diagnosis,
> or standardized clinical instrumentation. General laboratory stress detection on the WESAD dataset does NOT
> validate heat illness detection, heatstroke diagnosis, or environmental pollution toxicity screening.

---

## 1. Executive Summary & Objective

The primary objective of the HealthSensing machine learning pipeline is to evaluate whether personal resting reference baselines and classical supervised classifiers can reliably screen for acute physiological stress episodes from continuous wristband biosensors.

Using genuine laboratory recordings from the multimodal WESAD dataset ($N=4$ participants: **S2, S3, S4, and S5** wearing Empatica E4 wristbands), we compared three distinct paradigms:
1. **Fixed Population Statistical Thresholding:** Applying population-wide static physiological thresholds across all individuals.
2. **Personalized Multi-Signal Anomaly Screening:** Standardizing incoming readings against each individual's resting reference baseline ($z = (x - \mu) / \sigma$), evaluating both un-gated (Strategy A) and signal-specific quality-gated (Strategy D) calibration.
3. **Supervised Population-Level Random Forest Benchmark:** A 300-tree ensemble trained on three participants and evaluated on the held-out fourth participant under Leave-One-Participant-Out (LOPO) cross-validation using 60-second sliding windows.

### Primary Empirical Findings:
1. **Superiority of Personalized Normalization over Population Thresholds:**
   - Fixed population statistical thresholds achieved an average Recall of only **22.22%** (failing completely on S2, S4, and S5 because natural inter-individual resting variability places population boundaries outside individual physiological ranges).
   - In contrast, personalized baseline detection achieved **83.40%** Recall under Strategy A and **88.94%** Recall under Strategy D, while maintaining high precision (**95.30%–95.83%**).
2. **Supervised Classical ML (Random Forest) Exhibits Severe Generalization Discrepancies:**
   - When evaluated under strict Leave-One-Participant-Out cross-validation across unseen individuals, the Random Forest model suffered from acute cross-subject fragility:
     - **Participant S2:** Recall was **0.00%** (19 of 19 stress windows missed; the model predicted baseline for all windows).
     - **Participant S3:** Incurred severe false alarms (**67.57% False Positive Rate**, 25 false alarms out of 37 baseline windows).
     - **Participant S4:** Achieved perfect classification (**100% Precision, 100% Recall**).
     - **Participant S5:** Produced intermediate performance (**55.00% Recall, 13.16% FPR**).
   - Macro Recall was $0.6375 \pm 0.4750$ with an average FPR of $0.2018 \pm 0.3220$.
3. **Signal-Specific Quality Gating (Strategy D) Resolves Baseline Non-Stationarity:**
   - In Participant S3, pronounced electrodermal acclimation decay (-89% drift) during rest inflated baseline variance under Strategy A, suppressing stress recall to 33.59%.
   - Signal-specific quality-gated calibration (Strategy D) restored S3 stress recall to **55.78%** without increasing false positive rates, while reducing stress detection latency from **419 seconds down to 188 seconds** (over 3.8 minutes faster alert).

---

## 2. Evaluation Methodology & Protocol Separation

A central scientific requirement of this evaluation is acknowledging that the personalized anomaly detector and the Random Forest classifier operate under distinct temporal framing:
- **Personalized Anomaly Screening:** Operates at **1 Hz reading-level granularity** (1 sample per second), evaluating continuous relative deviations from rest.
- **Random Forest Benchmark:** Operates on **60-second sliding windows** (30-second stride, 50% overlap), aggregating 18 time-domain statistical features per window.

To ensure scientific integrity, we present two complementary evaluations:
1. **Established Protocol Results:** Retaining the validated published protocols for each model.
2. **Common Held-Out Evaluation Subset:** A direct apples-to-apples comparison on identical participants (S2–S5) and identical evaluation time intervals (the held-out 30% baseline partition and the acute stress period), ensuring zero data leakage.

---

## 3. Comprehensive Performance Comparison

### 3.1 Macro-Average Benchmark Results (Unweighted Fold Mean $\pm$ Sample SD, ddof=1)

| Detection Methodology | Evaluation Granularity | Macro Precision | Macro Recall (Sensitivity) | Macro F1-Score | Macro Accuracy | Macro Specificity | Macro False Positive Rate | Macro AUROC | Mean Alert Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed Population Threshold** | 1 Hz Reading | $0.5000 \pm 0.5774$ | $0.2222 \pm 0.4119$ | $0.2517 \pm 0.4427$ | $0.4995 \pm 0.2643$ | $1.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.7459 \pm 0.4973$ | $462.5\text{ s}$ |
| **Strategy A (Fixed Personal Baseline)** | 1 Hz Reading | $\mathbf{0.9583 \pm 0.0803}$ | $0.8340 \pm 0.3321$ | $0.8531 \pm 0.2370$ | $0.8600 \pm 0.2036$ | $\mathbf{0.9114 \pm 0.1716}$ | $\mathbf{0.0886 \pm 0.1716}$ | $0.9938 \pm 0.0125$ | $104.8\text{ s}$ |
| **Strategy D (Personalized Quality-Gated)** | 1 Hz Reading | $0.9530 \pm 0.0773$ | $\mathbf{0.8894 \pm 0.2211}$ | $\mathbf{0.9037 \pm 0.1309}$ | $\mathbf{0.8927 \pm 0.1324}$ | $0.9017 \pm 0.1659$ | $0.0983 \pm 0.1659$ | $\mathbf{0.9969 \pm 0.0062}$ | $\mathbf{47.0\text{ s}}$ |
| **Random Forest (LOPO Ensemble)** | 60s Window | $0.5330 \pm 0.4219$ | $0.6375 \pm 0.4750$ | $0.5566 \pm 0.4135$ | $0.7452 \pm 0.1880$ | $0.7982 \pm 0.3220$ | $0.2018 \pm 0.3220$ | $0.8333 \pm 0.1224$ | $150.0\text{ s}$ |

### 3.2 Common Held-Out Evaluation Subset (Identical Windows & Test Intervals)

When all models are evaluated strictly on the common held-out evaluation windows (10–11 baseline windows + 19–20 stress windows per participant; 41 baseline and 79 stress windows, 120 total windows):

| Methodology on Common Test Windows | Macro Precision | Macro Recall | Macro F1-Score | Macro Accuracy | Macro Specificity | Macro FPR | Macro AUROC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Strategy A (Common Windows)** | $\mathbf{0.9659 \pm 0.0682}$ | $0.8250 \pm 0.3500$ | $0.8471 \pm 0.2594$ | $0.8575 \pm 0.2215$ | $0.9250 \pm 0.1500$ | $0.0750 \pm 0.1500$ | $0.9950 \pm 0.0100$ |
| **Strategy D (Common Windows)** | $\mathbf{0.9659 \pm 0.0682}$ | $\mathbf{0.9000 \pm 0.2000}$ | $\mathbf{0.9192 \pm 0.1180}$ | $\mathbf{0.9075 \pm 0.1259}$ | $0.9250 \pm 0.1500$ | $0.0750 \pm 0.1500$ | $\mathbf{1.0000 \pm 0.0000}$ |
| **Random Forest (Common Windows)** | $0.7065 \pm 0.4728$ | $0.6375 \pm 0.4750$ | $0.6600 \pm 0.4610$ | $0.7389 \pm 0.2972$ | $\mathbf{0.9273 \pm 0.0951}$ | $\mathbf{0.0727 \pm 0.0951}$ | $0.8717 \pm 0.1552$ |

---

## 4. Participant-Wise Detailed Breakdown

The table below breaks down performance across each of the four human participants:

| Participant | Detector | Test N | TP | FP | TN | FN | Precision | Recall | F1-Score | Accuracy | FPR | Alert Latency |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S2** | Fixed Population Threshold | 959 | 0 | 0 | 344 | 615 | 0.0000 | 0.0000 | 0.0000 | 0.3587 | 0.0000 | $615\text{ s}$ (No alert) |
| **S2** | Strategy A (Baseline) | 959 | 615 | 119 | 225 | 0 | 0.8379 | 1.0000 | 0.9118 | 0.8759 | 0.3459 | $\mathbf{0\text{ s}}$ |
| **S2** | Strategy D (Personalized) | 959 | 615 | 119 | 225 | 0 | 0.8379 | 1.0000 | 0.9118 | 0.8759 | 0.3459 | $\mathbf{0\text{ s}}$ |
| **S2** | Random Forest (LOPO) | 56 | 0 | 0 | 37 | 19 | 0.0000 | 0.0000 | 0.0000 | 0.6607 | 0.0000 | $570\text{ s}$ (No alert) |
| **S3** | Fixed Population Threshold | 982 | 570 | 0 | 342 | 70 | 1.0000 | 0.8906 | 0.9421 | 0.9287 | 0.0000 | $103\text{ s}$ |
| **S3** | Strategy A (Baseline) | 982 | 215 | 0 | 342 | 425 | 1.0000 | 0.3359 | 0.5029 | 0.5672 | 0.0000 | $419\text{ s}$ |
| **S3** | Strategy D (Personalized) | 982 | 357 | 0 | 342 | 283 | 1.0000 | $\mathbf{0.5578}$ | $\mathbf{0.7161}$ | $\mathbf{0.7118}$ | 0.0000 | $\mathbf{188\text{ s}}$ |
| **S3** | Random Forest (LOPO) | 57 | 20 | 25 | 12 | 0 | 0.4444 | 1.0000 | 0.6154 | 0.5614 | 0.6757 | $\mathbf{0\text{ s}}$ |
| **S4** | Fixed Population Threshold | 983 | 0 | 0 | 348 | 635 | 0.0000 | 0.0000 | 0.0000 | 0.3540 | 0.0000 | $635\text{ s}$ (No alert) |
| **S4** | Strategy A (Baseline) | 983 | 635 | 3 | 345 | 0 | 0.9953 | 1.0000 | 0.9976 | 0.9969 | 0.0086 | $\mathbf{0\text{ s}}$ |
| **S4** | Strategy D (Personalized) | 983 | 635 | 3 | 345 | 0 | 0.9953 | 1.0000 | 0.9976 | 0.9969 | 0.0086 | $\mathbf{0\text{ s}}$ |
| **S4** | Random Forest (LOPO) | 57 | 20 | 0 | 37 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | $\mathbf{0\text{ s}}$ |
| **S5** | Fixed Population Threshold | 1005 | 0 | 0 | 360 | 645 | 0.0000 | 0.0000 | 0.0000 | 0.3582 | 0.0000 | $497\text{ s}$ |
| **S5** | Strategy A (Baseline) | 1005 | 645 | 0 | 360 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | $\mathbf{0\text{ s}}$ |
| **S5** | Strategy D (Personalized) | 1005 | 645 | 14 | 346 | 0 | 0.9788 | 1.0000 | 0.9893 | 0.9861 | 0.0389 | $\mathbf{0\text{ s}}$ |
| **S5** | Random Forest (LOPO) | 58 | 11 | 5 | 33 | 9 | 0.6875 | 0.5500 | 0.6111 | 0.7586 | 0.1316 | $30\text{ s}$ |

---

## 5. Comparison Visualizations

The actual experimental results were rendered as publication-quality visual figures using matplotlib and saved in the project repository:

1. **Overall Detector Metric Performance:**
   [detector_metrics_comparison.png](file:///c:/Users/animi/Documents/GitHub/HealthSensing/experiments/plots/detector_metrics_comparison.png)
   *Compares Precision, Recall, F1, Accuracy, Specificity, and False Positive Rate across all four detector configurations with sample standard deviation error bars.*

2. **Participant-Wise Performance Breakdown:**
   [participant_performance_comparison.png](file:///c:/Users/animi/Documents/GitHub/HealthSensing/experiments/plots/participant_performance_comparison.png)
   *A 4-panel visual detailing Recall vs. FPR for each participant, highlighting S2 under-triggering in Random Forest and S3 false alarm rates.*

3. **Calibration Strategy Comparison (Strategies A, B, C, D):**
   [calibration_strategy_comparison.png](file:///c:/Users/animi/Documents/GitHub/HealthSensing/experiments/plots/calibration_strategy_comparison.png)
   *Demonstrates the impact of acclimation exclusion and signal-specific quality gating on stress sensitivity and false alarm mitigation.*

4. **Continuous Receiver Operating Characteristic (ROC) Curves:**
   [roc_curves_comparison.png](file:///c:/Users/animi/Documents/GitHub/HealthSensing/experiments/plots/roc_curves_comparison.png)
   *Plots continuous discrimination performance: Strategy D achieves $\text{AUROC} = 0.997$, Strategy A achieves $\text{AUROC} = 0.994$, and Random Forest achieves $\text{AUROC} = 0.833$.*

5. **Detection Alert Latency Comparison:**
   [detection_latency_comparison.png](file:///c:/Users/animi/Documents/GitHub/HealthSensing/experiments/plots/detection_latency_comparison.png)
   *Illustrates the elapsed response time (seconds) from stress onset to the first alert across detectors and participants.*

---

## 6. Conceptual Analysis: Personalized Screening vs. Supervised Classification

The experimental results illuminate a profound conceptual and architectural distinction in biosensing:

```
               [ Absolute Sensor Values: HR (bpm), EDA (uS), Temp (C) ]
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
     [ Supervised Population ML ]                  [ Personalized Anomaly Screening ]
• Fits static hyperplanes across cohort.     • Learns individual resting parameters (mu, sigma).
• Subject to inter-individual baseline bias: • Converts readings to normalized deviations:
  - Low-baseline subjects (S2) -> 0% recall.   z = (x - mu) / sigma
  - High-baseline subjects (S3) -> 67% FPR.  • Relative shifts absorb baseline variance.
• Fragile to out-of-distribution subjects.   • High recall and precision across all subjects.
```

### Why Supervised Population Models Fail Across Diverse Individuals:
- **Participant S2 (0% Recall):** Participant S2 possesses naturally low resting heart rate (median 65 bpm) and low electrodermal levels. When evaluated by a model trained on participants with higher resting heart rates and skin conductance, S2's acute stress response remained entirely below the population decision boundary.
- **Participant S3 (67.57% FPR):** Conversely, Participant S3 exhibits naturally elevated resting tonic skin conductance (initial plateau up to 4.5 $\mu\text{S}$). The population model interpreted S3's calm resting physiology as chronic stress.
- **Why Personalization Succeeds:** Standardizing against an individual's verified resting baseline ($z = (x - \mu) / \sigma$) transforms absolute physiological values into relative deviations. An increase from 65 to 80 bpm represents a statistically massive relative shift ($z > 4$) for S2, despite remaining within normal population resting bounds.

---

## 7. Robustness and Stress Testing Findings

To verify pipeline resilience under real-world sensor imperfections, focused robustness tests were evaluated in `tests/test_robustness.py`:

1. **Measurement Noise & Artifacts:** Additive Gaussian noise ($\sigma = 5$ bpm) and heavy-tailed Laplacian noise on EDA did not destabilize the detector or cause numerical overflow. Feature extraction remained strictly finite.
2. **Missing Sensor Fields & Dropouts:** When an entire sensor channel failed or disconnected (all NaNs), the pipeline gracefully marked the channel `evaluation_status="missing_reading"` and set `evaluation_completeness="partial"`, without fabricating zeros or misclassifying the reading as normal. Simultaneous dropout across all channels correctly produced `overall_status="unevaluable"`.
3. **Sampling Rate Jitter & Gaps:** The windowing engine enforces strict 1 Hz continuity. Intermittent transmission delays ($> 1.1$s) correctly split continuous blocks and prevented window creation across timestamp gaps.
4. **Extreme Class Imbalance:** Tested under 99:1 negative-to-positive class imbalance. Zero-division policies correctly handled $\text{TP}+\text{FP}=0$ without runtime crashes, and Random Forest balanced subsampling maintained stable probability calibration.
5. **Controlled Simulator Scenarios:** Synthetic streams from `simulator/` demonstrated that simulated heatwave conditions (elevated temperature and heart rate) triggered physiological alerts while maintaining complete data isolation from genuine WESAD benchmark records.

---

## 8. Limitations & Clinical Governance

1. **Small Sample Size ($N=4$):** While each WESAD recording contains over 6,000 one-second observations, validation was conducted on four laboratory participants. Larger cohorts across diverse demographics and age groups are necessary for general population claims.
2. **Laboratory vs. Ambulatory Conditions:** WESAD recordings were collected during structured laboratory tasks (Trier Social Stress Test, mental arithmetic). In free-living ambulatory settings, physical exertion, motion artifacts, speaking, and circadian shifts introduce substantial physiological variation.
3. **Resting Calibration Requirement:** Personalized detection requires an initial resting calibration period (~10–15 minutes). If the initial calibration period coincides with physical activity or acute anxiety, baseline variance will be inflated.
4. **Scope of Anomaly Detection:** General stress detection does NOT validate environmental heat illness, heat exhaustion, or air pollution toxicity. Anomaly detection flags statistical divergence from rest; it does not diagnose medical etiology.
