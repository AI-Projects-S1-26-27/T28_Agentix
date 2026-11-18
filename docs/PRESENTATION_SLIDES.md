# HealthSensing: Final Presentation Slides (6 Slides)

**Course:** CS F407 / U407 Artificial Intelligence  
**Track:** Application Track  
**Team Name:** AGENTIX (Team 28)  
**GitHub Repository:** [https://github.com/animish-k/HealthSensing](https://github.com/animish-k/HealthSensing)

---

## Slide 1: Title & Project Overview

- **Project Title:** HealthSensing: Personal Health Companion
- **Subtitle:** Privacy-Preserving Edge-AI Analytics and Early Warning for Environmental Health Risks
- **Team Name & Number:** AGENTIX — Team 28
- **Project Track & Type:** Application Track — Local/Edge-First Health Analytics and Early-Warning Prototype
- **GitHub Repository:** `https://github.com/animish-k/HealthSensing`
- **Live Production Dashboard:** `https://healthsensing.vercel.app`
- **Live Backend Web Service:** `https://healthsensing.onrender.com`
- **Team Members:**
  - Animish Kumar (AI/ML Pipeline Lead)
  - Aryan Saini (Simulator & Scenarios Lead)
  - Avani Asati (Reasoning & Risk Engine Lead)
  - Vedant Nichal (Backend & Data Streaming Lead)
  - Kartikeya Singh (Data Acquisition & Preprocessing)
  - Chaitanya Chalith (Frontend Dashboard Lead)
  - Siddharth Venkata Arava Reddy (ML Evaluation & Reproducibility)

---

## Slide 2: Problem Statement & Motivation

- **Core Limitation of Commercial Wearables:** Current smartwatches display disconnected raw numbers or rely on static population thresholds (e.g. alert if $HR > 100\text{ bpm}$).
- **The Baseline Problem:** Resting vitals vary drastically between individuals. A reading of 85 bpm represents normal rest for one person, but acute tachycardia for an endurance athlete (resting baseline 52 bpm). Fixed thresholds fail to detect meaningful individual deviations.
- **The Context Isolation Problem:** Vitals are evaluated in a vacuum without environmental context. Elevated heart rate indoors during light exercise is harmless; the exact same elevation at 42 °C ambient heat and 80% humidity indicates imminent heat stroke.
- **Privacy & Connectivity Risks:** Continuous transmission of biometrics to centralized cloud servers introduces severe privacy liabilities and fails during disaster-induced network blackouts.
- **Central Research Question:** *Does personalized multimodal anomaly detection, augmented with environmental context on edge hardware, provide more reliable early risk detection than fixed population thresholds?*

---

## Slide 3: Background & Related Work

- **Multimodal Wearable Biosensing (WESAD Benchmark):** Schmidt et al. (2018) proved combining PPG heart rate, electrodermal activity (EDA), and skin temperature outperforms single-channel monitoring. However, standard models pool subjects, hiding cross-subject generalization failures.
- **Personalized Anomaly Detection:** Prior work (Sensors 2022; Nature Flex. Electron. 2026) demonstrated subject-normalized $z$-scores ($z = (x - \mu)/\sigma$). However, existing approaches overlook **sensor acclimation drift** during initial wear.
- **Environmental Health Formulations:**
  - NOAA Rothfusz Heat Index regression (apparent temperature from dry-bulb temp and relative humidity).
  - US EPA Air Quality Index breakpoints for particulate matter ($\text{PM}_{2.5}$).
- **The HealthSensing Opportunity:** Bridging localized personal baseline calibration with deterministic meteorological risk equations in an entirely offline, edge-native architecture.

---

## Slide 4: Proposed Methodology & Technology Stack

- **End-to-End Edge Pipeline:**
  $$\text{Preprocessed WESAD Data / Simulator} \longrightarrow \text{FastAPI Ingestion} \longrightarrow \text{SQLite Local DB} \longrightarrow \text{Risk Fusion} \longrightarrow \text{WebSocket Live Stream} \longrightarrow \text{Dashboard}$$
- **18-Feature Sliding Window Extraction:** 60-second windows (30s stride) extracting Mean, Std, Min, Max, Range, and Linear Trend Slope ($\beta_1$) across $HR$, $EDA$, and $Temp$.
- **Personalized Baseline Calibration Strategies (A vs. D):**
  - *Strategy A (Full):* Standard 70% chronological resting baseline.
  - *Strategy D (Signal-Specific Quality-Gated):* Quantifies Variance Inflation Factor ($\text{VIF} = \sigma_{\text{full}}^2 / \sigma_{\text{suffix}}^2$) and mean drift to isolate the earliest stable resting suffix per sensor, eliminating acclimation artifacts.
- **Deterministic Multimodal Risk Fusion (65% Rules / 35% Model):**
  - HEAT-1: Heat Index $\ge 32.2\text{ }^\circ\text{C}$ ($90\text{ }^\circ\text{F}$) + $HR \ge 100\text{ bpm}$ or $+1.5\sigma$.
  - POLLUTION-1: $\text{AQI} \ge 101$ / $\text{PM}_{2.5} \ge 35\text{ }\mu\text{g/m}^3$ + $\text{SpO}_2 \le 94\%$.
  - Categorical tiers: **LOW, MODERATE, HIGH, CRITICAL** with deterministic signal attribution.
- **Technology Stack:** Python/FastAPI backend, SQLite3 (WAL mode), WebSockets, React 18/Vite/Tailwind, Scikit-Learn; deployable via Render persistent SSD & Vercel edge proxy.

---

## Slide 5: Experimental Results & Key Findings

- **Evaluated on WESAD Dataset ($N=4$ Human Subjects, Empatica E4 Research Wristband):**
  - **Fixed Population Thresholds:** Failed completely on S2 and S4 (0.0% recall), detected only 4.96% on S5, and detected stress effectively only on S3 (83.91% recall; macro recall: **22.22%**).
  - **Supervised Random Forest (LOPO Cross-Validation):** Severe cross-subject generalization collapse (macro recall $63.75\% \pm 47.50\%$; **0% recall** on S2, **67.57% false alarm rate** on S3).
  - **Personalized Baseline (Strategy D):** Achieved **88.94% Macro Recall, 95.30% Precision, 0.997 AUROC**, and cut alert latency to **47.0 seconds**.
- **Acclimation Resolution in Participant S3:**
  - S3 exhibited an -89% electrodermal baseline drift. Strategy A's inflated variance suppressed stress recall to 33.59% (latency 419s).
  - Strategy D automatically stabilized at offset 318s, restoring recall to **55.78%** and reducing detection latency to **188 seconds** (3.8 minutes faster).
- **Core Conclusion:** Personalized reference standardization is non-negotiable for wearable anomaly detection; population-level models fail across human biological diversity.
- **Live System Demonstration:** Live dashboard deployed at `https://healthsensing.vercel.app` backed by persistent Python ASGI service on Render (`https://healthsensing.onrender.com`).

---

## Slide 6: Project Management, Work Division & Accountability

- **Modular Responsibility Breakdown:**
  - *Animish Kumar:* Statistical anomaly detector, Random Forest benchmark, ML interfaces.
  - *Aryan Saini:* Simulator engine, seedable scenario generation (`heatwave`, `pollution`, `fatigue`, `normal`).
  - *Avani Asati:* Knowledge base, environmental equations, rule engine, risk fusion logic.
  - *Vedant Nichal:* FastAPI backend architecture, SQLite schema, REST APIs, WebSocket streaming, integration testing, Render deployment config.
  - *Kartikeya Singh:* WESAD data inspection, cleaning, timestamp alignment, Open-Meteo environmental data processing.
  - *Chaitanya Chalith:* React dashboard components, UI layout, telemetry cards, Vercel edge deployment config.
  - *Siddharth Venkata Arava Reddy:* LOPO evaluation protocols, reproducibility runner, comparison plots, experiment logging.
- **Repository Health & Reproducibility:**
  - Full test suite verified: **288 passed, 0 failed** in 20.73 seconds.
  - Transparent AI disclosure (ChatGPT and Antigravity) and thorough limitations discussion.
