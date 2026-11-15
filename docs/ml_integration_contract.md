# HealthSensing: Machine Learning Integration Contract & Interface Specification

> [!IMPORTANT]
> **Clinical Disclaimer:** This integration interface provides experimental algorithmic screening results only.
> It does NOT constitute medical diagnosis, clinical evaluation, or automated medical decision-making.
> Outputs must be combined with clinical expertise and must not be used as standalone medical diagnostics.

---

## 1. Overview and Architecture

The HealthSensing ML integration interface provides a clean, decoupled, framework-independent contract between the AI/ML pipeline and downstream integration systems (FastAPI backend routes, SQLite persistence, WebSocket event streams, and React visualizations).

The architecture separates physiological analysis into two complementary temporal granularities:
1. **Per-Observation Personalized Anomaly Screening (`ReadingResult`):** Instantaneous z-score comparison against an individual's personal resting reference baseline ($z = (x - \mu) / \sigma$). Accommodates resting physiological differences across individuals.
2. **Sliding-Window Supervised Stress Classification (`RFWindowResult`):** 60-second sliding-window statistical analysis (30-second stride) classified by an experimental population-level Random Forest model.

```
                              [ Incoming 1 Hz Sensor Data ]
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
       [ Calibration Service ]                           [ Inference Service ]
   • Validates Stationarity                         • Personalized Anomaly Screening
   • Quality-Gating Strategies (A, B, C, D)           (Per-observation z-score deviation)
   • Generates BaselineContext                      • Supervised Random Forest Classifier
                     │                                (60s contiguous windows, 30s stride)
                     └──────────────────────┬──────────────────────┘
                                            ▼
                               [ MLAnalysisResponse (JSON) ]
                                            │
                       ┌────────────────────┴────────────────────┐
                       ▼                                         ▼
            [ Avani: Risk Fusion ]                    [ Vedant: FastAPI & WS ]
            • Knowledge Base Rules                    • WebSocket Streaming
            • Multi-factor Explanations               • Persistence & API Endpoints
```

---

## 2. Input Specifications

### 2.1 Sensor Stream Requirements
- **Hardware Source:** Empatica E4 wristband (or compatible wearable sensors).
- **Nominal Sampling Rate:** 1 Hz (1 sample per second).
- **Sampling Interval:** `expected_sampling_interval_sec = 1.0`, `timestamp_tolerance_sec = 0.1`.
- **Timestamp Format:** ISO-8601 UTC string (e.g., `"2026-01-01T10:00:00+00:00"`).

### 2.2 Physiological Channels and Units
| Field Name | Physical Quantity | Units | Typical Resting Range | Missing Representation |
|:---|:---|:---:|:---:|:---:|
| `heart_rate_bpm` | Optical photoplethysmography (PPG) heart rate | Beats per minute (bpm) | 50 – 95 bpm | `null` / `NaN` |
| `eda_microsiemens` | Tonic and phasic electrodermal skin conductance | Microsiemens ($\mu\text{S}$) | 0.1 – 15.0 $\mu\text{S}$ | `null` / `NaN` |
| `body_temperature_c` | Peripheral skin temperature | Degrees Celsius ($^\circ\text{C}$) | 31.0 – 36.5 $^\circ\text{C}$ | `null` / `NaN` |

### 2.3 Input JSON Schema Example

```json
{
  "request_id": "req-20261115-001",
  "participant_id": "S2",
  "readings": [
    {
      "timestamp": "2026-01-01T10:00:00Z",
      "heart_rate_bpm": 72.4,
      "eda_microsiemens": 1.25,
      "body_temperature_c": 34.12
    },
    {
      "timestamp": "2026-01-01T10:00:01Z",
      "heart_rate_bpm": 73.1,
      "eda_microsiemens": null,
      "body_temperature_c": 34.14
    }
  ],
  "calibration_readings": [
    {
      "timestamp": "2026-01-01T09:00:00Z",
      "heart_rate_bpm": 68.2,
      "eda_microsiemens": 1.10,
      "body_temperature_c": 33.95
    }
  ],
  "strategy": "signal_specific",
  "thresholds": {
    "heart_rate_bpm": 3.0,
    "eda_microsiemens": 3.0,
    "body_temperature_c": 3.0
  }
}
```

---

## 3. Calibration Service Contract

The calibration service prepares reference baselines using one of four strategies:
- `"full"` (Strategy A): Complete calibration partition without truncation.
- `"settled"` (Strategy B): Excludes initial acclimation burn-in (drops initial 240 seconds if acclimation detected).
- `"quality_gated"` (Strategy C): Shared contiguous suffix (minimum 480s) minimizing joint variance inflation.
- `"signal_specific"` (Strategy D - Recommended): Independently evaluates each channel, identifying earliest stable suffix per signal.

### Baseline Quality Statuses
| Status | Meaning | Downstream Action |
|:---|:---|:---|
| `stable` | Stationarity verified; variance inflation factor $< 3.5$, mean drift $< 20\%$. | Reliable baseline for anomaly screening. |
| `unstable` | Non-stationarity or acclimation trend detected; fallback suffix selected. | Flagged in metadata; baseline std may be inflated. |
| `insufficient_data` | Fewer than 60 observations or $< 2$ valid numeric samples. | Signal cannot be calibrated; downstream readings marked `insufficient_baseline_data`. |

### Calibration Metadata Example (Extracted from BaselineContext)
```json
{
  "strategy": "signal_specific",
  "feature_columns": ["heart_rate_bpm", "eda_microsiemens", "body_temperature_c"],
  "baseline_statistics": {
    "heart_rate_bpm": {
      "valid_samples": 798,
      "mean": 71.32,
      "std": 3.41,
      "min": 64.0,
      "max": 82.5
    },
    "eda_microsiemens": {
      "valid_samples": 480,
      "mean": 1.15,
      "std": 0.18,
      "min": 0.85,
      "max": 1.82
    },
    "body_temperature_c": {
      "valid_samples": 798,
      "mean": 34.05,
      "std": 0.22,
      "min": 33.5,
      "max": 34.6
    }
  },
  "selection_metadata": {
    "strategy": "Strategy D (Signal-Specific Quality-Gated)",
    "signal_offsets": {
      "heart_rate_bpm": 0,
      "eda_microsiemens": 318,
      "body_temperature_c": 0
    },
    "decision": "Signal-specific calibration windows selected independently."
  },
  "insufficient_signals": []
}
```

---

## 4. Inference Output Specifications

### 4.1 Per-Signal Result (`SignalResult`)
| Field | Type | Description | Nullable? |
|:---|:---|:---|:---:|
| `value` | `float` | Sensor observation value | Yes (null if missing) |
| `baseline_mean` | `float` | Individual's reference baseline mean | Yes (null if uncalibrated) |
| `baseline_std` | `float` | Sample standard deviation (ddof=1) | Yes (null if uncalibrated) |
| `difference` | `float` | Raw deviation ($x - \mu$) | Yes (null if missing value/baseline) |
| `z_score` | `float` | Normalized deviation ($z = (x - \mu) / \sigma$) | Yes (null if missing or $\sigma = 0$) |
| `threshold` | `float` | Configured absolute z-score anomaly threshold | No |
| `anomaly` | `int` | Binary anomaly flag ($1$ if $\|z\| > \text{threshold}$, else $0$) | No |
| `evaluation_status` | `string` | Categorical evaluation state (see below) | No |

#### Supported `evaluation_status` Values:
- `"evaluated"`: Reading is finite and baseline is valid; z-score computed successfully.
- `"missing_reading"`: Sensor reading was missing (`NaN` or `null`); anomaly flag is 0.
- `"zero_variance_baseline"`: Baseline standard deviation is exactly 0.0 (z-score mathematically undefined); anomaly flag is 0.
- `"insufficient_baseline_data"`: Fewer than 2 baseline observations were available; anomaly flag is 0.

> [!WARNING]
> An anomaly flag of `0` when `evaluation_status != "evaluated"` must NEVER be interpreted as "definitely normal". It reflects an unevaluable sensor channel.

### 4.2 Multi-Signal Reading Result (`ReadingResult`)
| Field | Type | Description |
|:---|:---|:---|
| `timestamp` | `string` / `null` | ISO-8601 timestamp string |
| `signals` | `object` | Map of `SignalResult` objects keyed by column name |
| `anomaly_count` | `int` | Total anomalous channels in this reading |
| `overall_anomaly` | `int` | $1$ if `anomaly_count` $> 0$, else $0$ |
| `unevaluable_count` | `int` | Number of channels that could not be evaluated |
| `overall_status` | `string` | Categorical reading state (see below) |
| `evaluation_completeness` | `string` | `"complete"`, `"partial"`, or `"none"` |

#### Supported `overall_status` Values:
- `"normal"`: All evaluable channels are within reference thresholds ($\|z\| \le \text{threshold}$).
- `"anomaly_detected"`: One or more channels exceed threshold with all channels evaluable.
- `"partially_evaluated_anomaly"`: One or more channels exceed threshold while some channels are unevaluable.
- `"unevaluable"`: Zero configured channels could be evaluated.

---

## 5. Random Forest Window Inference (`RFWindowResult`)

### 5.1 Protocol and Windowing Requirements
- **Window Length:** 60 consecutive observations (60 seconds at 1 Hz).
- **Stride:** 30 observations (30 seconds, 50% temporal overlap).
- **Continuity Gating:** Consecutive samples must be $1.0 \pm 0.1$ seconds apart. Windows are NEVER formed across timestamp gaps.
- **Features Extracted (18 total):** Mean, Standard Deviation, Minimum, Maximum, Range, and Linear Trend for each of the 3 channels.
- **Probabilities:** Column 0 represents Class 0 (Baseline), Column 1 represents Class 1 (Stress).

### 5.2 Supported `status` Values:
- `"classified"`: Window successfully processed by fitted classifier.
- `"insufficient_samples"`: Fewer than 60 consecutive observations available in recording.
- `"timestamp_gap"`: Recording duration exceeds 60s, but internal timestamp discontinuity broke continuous blocks.
- `"model_unavailable"`: Random Forest model was omitted (`null`), allowing rule-based anomaly detection to operate independently.

---

## 6. Complete Standardized Analysis Payload (`MLAnalysisResponse`)

```json
{
  "schema_version": "1.0.0",
  "request_id": "req-20261115-001",
  "generated_at": "2026-11-15T15:30:00.000000+00:00",
  "baseline_metadata": {
    "strategy": "signal_specific",
    "feature_columns": ["heart_rate_bpm", "eda_microsiemens", "body_temperature_c"],
    "baseline_statistics": {
      "heart_rate_bpm": {
        "valid_samples": 798,
        "mean": 71.32,
        "std": 3.41,
        "min": 64.0,
        "max": 82.5
      },
      "eda_microsiemens": {
        "valid_samples": 480,
        "mean": 1.15,
        "std": 0.18,
        "min": 0.85,
        "max": 1.82
      },
      "body_temperature_c": {
        "valid_samples": 798,
        "mean": 34.05,
        "std": 0.22,
        "min": 33.5,
        "max": 34.6
      }
    },
    "selection_metadata": {
      "strategy": "Strategy D (Signal-Specific Quality-Gated)",
      "signal_offsets": {
        "heart_rate_bpm": 0,
        "eda_microsiemens": 318,
        "body_temperature_c": 0
      },
      "decision": "Signal-specific calibration windows selected independently."
    },
    "insufficient_signals": []
  },
  "baseline_quality": {
    "heart_rate_bpm": {
      "signal": "heart_rate_bpm",
      "status": "stable",
      "observation_count": 798,
      "num_windows": 6,
      "mean_drift": 1.24,
      "pct_mean_drift": 1.74,
      "variance_inflation_factor": 1.12,
      "acclimation_detected": false,
      "measurements": {
        "window_means": [71.1, 71.4, 71.2, 71.5, 71.3, 71.4]
      },
      "explanation": "Signal 'heart_rate_bpm' exhibits stable baseline characteristics."
    },
    "eda_microsiemens": {
      "signal": "eda_microsiemens",
      "status": "stable",
      "observation_count": 480,
      "num_windows": 4,
      "mean_drift": 0.05,
      "pct_mean_drift": 4.35,
      "variance_inflation_factor": 1.25,
      "acclimation_detected": false,
      "measurements": {
        "window_means": [1.18, 1.16, 1.14, 1.13]
      },
      "explanation": "Signal 'eda_microsiemens' achieved stability at offset 318s."
    },
    "body_temperature_c": {
      "signal": "body_temperature_c",
      "status": "stable",
      "observation_count": 798,
      "num_windows": 6,
      "mean_drift": 0.08,
      "pct_mean_drift": 0.24,
      "variance_inflation_factor": 1.05,
      "acclimation_detected": false,
      "measurements": {
        "window_means": [34.02, 34.04, 34.05, 34.06, 34.08, 34.07]
      },
      "explanation": "Signal 'body_temperature_c' exhibits stable baseline characteristics."
    }
  },
  "reading_results": [
    {
      "timestamp": "2026-01-01T10:00:00+00:00",
      "signals": {
        "heart_rate_bpm": {
          "value": 85.2,
          "baseline_mean": 71.32,
          "baseline_std": 3.41,
          "difference": 13.88,
          "z_score": 4.07,
          "threshold": 3.0,
          "anomaly": 1,
          "evaluation_status": "evaluated"
        },
        "eda_microsiemens": {
          "value": 2.45,
          "baseline_mean": 1.15,
          "baseline_std": 0.18,
          "difference": 1.30,
          "z_score": 7.22,
          "threshold": 3.0,
          "anomaly": 1,
          "evaluation_status": "evaluated"
        },
        "body_temperature_c": {
          "value": 34.10,
          "baseline_mean": 34.05,
          "baseline_std": 0.22,
          "difference": 0.05,
          "z_score": 0.23,
          "threshold": 3.0,
          "anomaly": 0,
          "evaluation_status": "evaluated"
        }
      },
      "anomaly_count": 2,
      "overall_anomaly": 1,
      "unevaluable_count": 0,
      "overall_status": "anomaly_detected",
      "evaluation_completeness": "complete"
    },
    {
      "timestamp": "2026-01-01T10:00:01+00:00",
      "signals": {
        "heart_rate_bpm": {
          "value": 72.0,
          "baseline_mean": 71.32,
          "baseline_std": 3.41,
          "difference": 0.68,
          "z_score": 0.20,
          "threshold": 3.0,
          "anomaly": 0,
          "evaluation_status": "evaluated"
        },
        "eda_microsiemens": {
          "value": null,
          "baseline_mean": 1.15,
          "baseline_std": 0.18,
          "difference": null,
          "z_score": null,
          "threshold": 3.0,
          "anomaly": 0,
          "evaluation_status": "missing_reading"
        },
        "body_temperature_c": {
          "value": 34.08,
          "baseline_mean": 34.05,
          "baseline_std": 0.22,
          "difference": 0.03,
          "z_score": 0.14,
          "threshold": 3.0,
          "anomaly": 0,
          "evaluation_status": "evaluated"
        }
      },
      "anomaly_count": 0,
      "overall_anomaly": 0,
      "unevaluable_count": 1,
      "overall_status": "normal",
      "evaluation_completeness": "partial"
    }
  ],
  "rf_window_results": [
    {
      "window_start": "2026-01-01T10:00:00+00:00",
      "window_end": "2026-01-01T10:00:59+00:00",
      "sample_count": 60,
      "status": "classified",
      "predicted_class": "stress",
      "predicted_class_id": 1,
      "probabilities": {
        "baseline_class_0": 0.1523,
        "stress_class_1": 0.8477
      }
    }
  ],
  "model_metadata": {
    "model_available": true,
    "model_identifier": "rf_model_all_participants_v1",
    "model_type": "RandomForestClassifier (scikit-learn)",
    "window_size_observations": 60,
    "stride_observations": 30,
    "sampling_rate_hz": 1.0,
    "feature_schema": [
      "heart_rate_bpm_mean",
      "heart_rate_bpm_std",
      "heart_rate_bpm_min",
      "heart_rate_bpm_max",
      "heart_rate_bpm_range",
      "heart_rate_bpm_trend",
      "eda_microsiemens_mean",
      "eda_microsiemens_std",
      "eda_microsiemens_min",
      "eda_microsiemens_max",
      "eda_microsiemens_range",
      "eda_microsiemens_trend",
      "body_temperature_c_mean",
      "body_temperature_c_std",
      "body_temperature_c_min",
      "body_temperature_c_max",
      "body_temperature_c_range",
      "body_temperature_c_trend"
    ],
    "class_mapping": {
      "baseline_class_0": 0,
      "stress_class_1": 1
    },
    "evaluation_protocol": "Leave-One-Participant-Out (LOPO) Cross-Validation on WESAD (N=4)",
    "deployment_note": "Final deployable model retrained on all 4 WESAD participants (228 windows). Distinguished from cross-validation fold models used strictly to estimate generalization metrics. Model outputs are experimental statistical classification probabilities and do NOT represent calibrated medical risk percentages. Not clinically validated for diagnosis."
  }
}
```

---

## 7. Python Service Usage Example

```python
import pandas as pd
from ai.interfaces import (
    prepare_baseline,
    run_ml_analysis,
)
from ai.models.random_forest_model import StressRandomForestClassifier

# 1. Load calibration data (e.g. resting baseline period)
calib_df = pd.read_csv("calibration_data.csv")

# 2. Prepare reference baseline with Strategy D (signal-specific gating)
baseline_ctx = prepare_baseline(
    calibration_readings=calib_df,
    feature_columns=["heart_rate_bpm", "eda_microsiemens", "body_temperature_c"],
    strategy="signal_specific",
)

# 3. Load optional deployable Random Forest model (once, outside request loop)
rf_model = StressRandomForestClassifier.load("experiments/results/random_forest/rf_model_all_participants.joblib")

# 4. Stream current readings
current_readings_df = pd.read_csv("live_stream.csv")

# 5. Execute unified analysis
response = run_ml_analysis(
    readings=current_readings_df,
    baseline_context=baseline_ctx,
    rf_model=rf_model,
    request_id="req-live-001",
)

# 6. Safe JSON output for FastAPI / WebSocket transmission
json_payload = response.to_json(indent=2)
```
