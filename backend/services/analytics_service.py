import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.db.database import MeasurementRepository
from backend.models.analytics import (
    AutonomicStabilityMetrics,
    SignalQuantiles,
    TelemetryAnalyticsSummary,
)


class AnalyticsService:
    def __init__(self, repository: MeasurementRepository):
        self._repository = repository

    @staticmethod
    def _compute_quantiles(values: List[float]) -> Optional[SignalQuantiles]:
        if not values:
            return None
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean_val = sum(sorted_vals) / n
        variance = sum((x - mean_val) ** 2 for x in sorted_vals) / n
        std_val = math.sqrt(variance)

        def percentile(p: float) -> float:
            k = (n - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return float(sorted_vals[int(k)])
            d0 = sorted_vals[int(f)] * (c - k)
            d1 = sorted_vals[int(c)] * (k - f)
            return float(d0 + d1)

        return SignalQuantiles(
            mean=round(mean_val, 2),
            std_dev=round(std_val, 2),
            min_value=round(sorted_vals[0], 2),
            max_value=round(sorted_vals[-1], 2),
            median=round(percentile(0.50), 2),
            q25=round(percentile(0.25), 2),
            q75=round(percentile(0.75), 2),
            sample_count=n,
        )

    def compute_summary(
        self, window_minutes: int = 60, participant_id: Optional[str] = None
    ) -> TelemetryAnalyticsSummary:
        # Fetch recent readings (assumes ~1 reading every 5 seconds = 12/min)
        sample_limit = max(10, min(window_minutes * 12, 1000))
        readings = self._repository.recent(limit=sample_limit)

        hr_vals = [r.heart_rate for r in readings if r.heart_rate is not None]
        eda_vals = [r.eda for r in readings if r.eda is not None]
        temp_vals = [r.body_temperature for r in readings if r.body_temperature is not None]
        pm25_vals = [r.pm25 for r in readings if r.pm25 is not None]
        amb_temp_vals = [r.ambient_temperature for r in readings if r.ambient_temperature is not None]

        signals: Dict[str, SignalQuantiles] = {}
        if hr_vals:
            signals["heart_rate"] = self._compute_quantiles(hr_vals)
        if eda_vals:
            signals["eda"] = self._compute_quantiles(eda_vals)
        if temp_vals:
            signals["body_temperature"] = self._compute_quantiles(temp_vals)
        if pm25_vals:
            signals["pm25"] = self._compute_quantiles(pm25_vals)
        if amb_temp_vals:
            signals["ambient_temperature"] = self._compute_quantiles(amb_temp_vals)

        # Compute Autonomic Stability Index
        stability = self._compute_stability(hr_vals, eda_vals, temp_vals)

        return TelemetryAnalyticsSummary(
            window_minutes=window_minutes,
            total_samples=len(readings),
            signals=signals,
            stability=stability,
            calculated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _compute_stability(
        self, hr_vals: List[float], eda_vals: List[float], temp_vals: List[float]
    ) -> AutonomicStabilityMetrics:
        if len(hr_vals) < 2:
            return AutonomicStabilityMetrics(
                stability_index=100.0,
                heart_rate_variability_est=0.0,
                eda_phasic_peaks_est=0,
                drift_velocity_hr=0.0,
                thermal_gradient=0.0,
                homeostatic_status="uncalibrated",
            )

        # Successive differences for HRV RMSSD proxy
        successive_diffs = [
            (hr_vals[i] - hr_vals[i - 1]) ** 2 for i in range(1, len(hr_vals))
        ]
        rmssd_proxy = math.sqrt(sum(successive_diffs) / len(successive_diffs))

        # EDA peak counter (simple threshold derivative)
        phasic_peaks = 0
        if len(eda_vals) > 2:
            for i in range(1, len(eda_vals) - 1):
                if eda_vals[i] > eda_vals[i - 1] and eda_vals[i] > eda_vals[i + 1] and eda_vals[i] > 1.5:
                    phasic_peaks += 1

        # Drift rate: linear delta between first and last over window
        drift_hr = (hr_vals[0] - hr_vals[-1])  # recent - oldest in slice
        thermal_gradient = (temp_vals[0] - temp_vals[-1]) if len(temp_vals) >= 2 else 0.0

        # Stability penalty score (100 is best)
        penalty = 0.0
        if rmssd_proxy > 15.0:
            penalty += min(35.0, (rmssd_proxy - 15.0) * 2.0)
        if abs(drift_hr) > 10.0:
            penalty += min(30.0, (abs(drift_hr) - 10.0) * 1.5)
        if phasic_peaks > 5:
            penalty += min(25.0, (phasic_peaks - 5) * 3.0)

        stability_score = max(0.0, min(100.0, 100.0 - penalty))

        if stability_score >= 80.0:
            status = "optimal"
        elif stability_score >= 55.0:
            status = "compensatory"
        else:
            status = "high_strain"

        return AutonomicStabilityMetrics(
            stability_index=round(stability_score, 1),
            heart_rate_variability_est=round(rmssd_proxy, 2),
            eda_phasic_peaks_est=phasic_peaks,
            drift_velocity_hr=round(drift_hr, 2),
            thermal_gradient=round(thermal_gradient, 2),
            homeostatic_status=status,
        )
