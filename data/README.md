# HealthSensing Data Engineering & Preprocessing Documentation

**Owner:** Kartikeya Singh (Member 1)  
**Deliverables:** Ingestion, Multi-Frequency Synchronization, Quality Assurance, and Normalization  

## 1. Multi-Frequency Sensor Synchronization Methodology
* **Chest RespiBAN (ECG, Respiration, Temp):** 700 Hz
* **Wrist Empatica E4 (BVP):** 64 Hz
* **Wrist Empatica E4 (EDA & Skin Temp):** 4 Hz
* **Empatica E4 Precomputed Heart Rate:** 1 Hz
* **Ground-Truth Protocol Labels:** 700 Hz

### Synchronization Strategy:
1. **Target Epoch:** All streams are resampled and aligned into **1-second non-overlapping epochs (1 Hz)**.
2. **Sub-Second Downsampling:** For 4 Hz signals (EDA and Skin Temp), we compute the 4-sample window mean over each second.
3. **Heart Rate Integration:** Empatica E4 1 Hz pulse rate is synchronized directly to the second epoch.
4. **Label Alignment:** Ground-truth labels are sampled at the epoch midpoint (s * 700 + 350) to reflect the active experimental condition.

## 2. WESAD Protocol & Ground-Truth Labels
* **0 = Transient / Undefined:** Unannotated intervals between experimental blocks.
* **1 = Baseline:** 20-minute relaxed resting condition while reading neutral magazines.
* **2 = Stress:** Trier Social Stress Test (TSST) consisting of public speaking and mental arithmetic under evaluation.
* **3 = Amusement:** Watching humorous video clips.
* **4 = Meditation:** Guided relaxation period.
* The prepared participant CSVs also contain protocol codes 5, 6, and 7. Their meanings have not been verified from the dataset documentation available to this project, so the replay preserves them without assigning meanings.

> **Crucial Rule on Personal Baselines:**  
> Personal baseline means, standard deviations, and normalization parameters are fitted **strictly on stress_label == 1 (Baseline)**. Observations under stress (label == 2) are strictly evaluated relative to this calibrated baseline.

## 3. Scope & Participant Coverage
* **Processed WESAD participants:** Subjects S2, S3, S4, and S5 are included in the current data pipeline. WESAD protocol labels describe experimental conditions; they are not clinical diagnoses or health-risk ground truth.
* **Extensibility:** The shared parser in `data/src/parse_real_wesad_full.py` supports WESAD participants S2 through S17. The single-participant entry point in `data/src/parse_real_wesad.py` delegates to it for S2.
* **Timestamp handling:** The parser uses available E4 EDA/HR timestamp metadata. If neither source is available, it warns and leaves timestamps empty rather than assigning a fabricated date.

WESAD citation: Schmidt, P. & Reiss, A. (2018), [WESAD (Wearable Stress and Affect Detection)](https://doi.org/10.24432/C57K5T), UCI Machine Learning Repository. UCI directs users to the original dataset site for licensing information and does not specify a license on its record; confirm the applicable terms before redistributing raw or derived data.

## 4. Replaying WESAD Data Through the Backend
Start the local API, then replay the prepared files:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
python -m data.src.replay_wesad_to_backend --csv data/processed/wesad_s2_real.csv --max-readings 300
```

The replay tool calibrates each participant from the first 70% of their WESAD baseline-label (`1`) rows, then sends only later readings to `POST /api/readings`. Calibration rows and earlier rows are skipped so the tool does not evaluate training data using a baseline calculated from those same rows. It preserves participant IDs and all non-negative integer protocol labels; labels are not treated as health-risk labels. The backend stores readings and risk results locally. When a stable baseline is available, `components.personalized_anomaly` flags a signal more than 3 standard deviations from that participant's baseline. Missing, unstable, or zero-variance signals are reported as unevaluable, not normal. An anomaly flag alone does not trigger a health alert.

Pass multiple CSV paths to replay more participants. Omit `--max-readings` to replay every row. Replaying the same file again inserts duplicate readings.

## 5. Ljubljana Environmental Data Provenance
The Ljubljana data in `processed/ljubljana_real.csv` is retrieved from the [Open-Meteo Air Quality API](https://air-quality-api.open-meteo.com/v1/air-quality) and [Weather Forecast API](https://api.open-meteo.com/v1/forecast). These are model-based API products, not measurements from an official monitoring station. Their timestamps are API time-series timestamps and should not be described as in-situ recordings.

Attribute the API data to [Open-Meteo](https://open-meteo.com/) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Open-Meteo's free API terms are for non-commercial use; verify the current [licence and terms](https://open-meteo.com/en/licence) before deployment or other use.

## 6. End-to-End Preprocessing Sequence Execution
The complete sequence is demonstrated in `run_complete_genuine_pipeline.py`:
Real Dataset -> Loading -> Cleaning -> Feature Extraction -> Normalization -> Processed Output

Run verification:
```bash
python tests/test_preprocessing_pipeline.py
python run_complete_genuine_pipeline.py
