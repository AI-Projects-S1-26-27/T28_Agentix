"""Replay prepared WESAD recordings through the local HealthSensing API."""

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import math
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from pydantic import ValidationError

from ai.interfaces.calibration_service import prepare_baseline
from backend.models.measurement import Measurement

BASELINE_LABEL = 1
CALIBRATION_RATIO = 0.70
FEATURE_COLUMNS = (
    "heart_rate_bpm",
    "eda_microsiemens",
    "body_temperature_c",
)
BACKEND_SIGNALS = {
    "heart_rate_bpm": "heart_rate_bpm",
    "eda_microsiemens": "eda_microsiemens",
    "body_temperature_c": "body_temperature_c",
}
REQUIRED_COLUMNS = {
    "participant_id",
    "timestamp",
    "stress_label",
    *FEATURE_COLUMNS,
}
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


class ReplayError(RuntimeError):
    """Raised when prepared WESAD data cannot be replayed safely."""


def _post_json(
    endpoint: str,
    payload: Mapping[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ReplayError(
            f"Backend rejected {endpoint} with HTTP {error.code}: {detail}"
        ) from error
    except URLError as error:
        raise ReplayError(f"Backend is unavailable: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise ReplayError(f"Backend returned invalid JSON for {endpoint}.") from error


def _optional_float(value: Any, column: str) -> float | None:
    if pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{column} contains a non-finite value.")
    return number


def build_reading_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Map one prepared WESAD row to the backend's reading schema."""
    participant_id = str(row.get("participant_id", "")).strip()
    if not participant_id or participant_id.lower() == "nan":
        raise ValueError("participant_id must be present in every WESAD row.")

    raw_timestamp = row.get("timestamp")
    if pd.isna(raw_timestamp):
        raise ValueError("WESAD rows must have a timestamp.")
    timestamp = pd.Timestamp(raw_timestamp)
    if timestamp.tzinfo is None:
        raise ValueError("WESAD timestamps must include a timezone.")

    payload: dict[str, Any] = {
        "timestamp": timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
        "source": "wesad",
        "participant_id": participant_id,
        "heart_rate": _optional_float(row.get("heart_rate_bpm"), "heart_rate_bpm"),
        "eda": _optional_float(row.get("eda_microsiemens"), "eda_microsiemens"),
        "body_temperature": _optional_float(
            row.get("body_temperature_c"), "body_temperature_c"
        ),
    }

    label = _optional_float(row.get("stress_label"), "stress_label")
    if label is not None:
        if not label.is_integer() or label < 0:
            raise ValueError("WESAD protocol labels must be non-negative integers.")
        payload["protocol_label"] = int(label)

    return payload


def _select_calibration_rows(
    participant_id: str,
    participant_rows: pd.DataFrame,
) -> pd.DataFrame:
    baseline_rows = participant_rows.loc[
        participant_rows["stress_label"] == BASELINE_LABEL
    ].sort_values("_parsed_timestamp")
    if baseline_rows.empty:
        raise ReplayError(
            f"Participant {participant_id} has no WESAD baseline-label rows."
        )

    calibration_count = math.floor(len(baseline_rows) * CALIBRATION_RATIO)
    if calibration_count < 2:
        raise ReplayError(
            f"Participant {participant_id} has too few baseline rows to calibrate."
        )

    return baseline_rows.iloc[:calibration_count]


def _prepare_calibration_payloads(
    participant_id: str,
    calibration_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    context = prepare_baseline(
        calibration_readings=calibration_rows,
        feature_columns=FEATURE_COLUMNS,
        strategy="signal_specific",
    )

    payloads: list[dict[str, Any]] = []
    for signal in FEATURE_COLUMNS:
        statistics = context.baseline_statistics[signal]
        mean = statistics["mean"]
        std = statistics["std"]
        observation_count = statistics["valid_samples"]
        if (
            mean is None
            or std is None
            or observation_count < 2
            or not math.isfinite(float(mean))
            or not math.isfinite(float(std))
        ):
            continue

        payloads.append({
            "participant_id": participant_id,
            "signal": BACKEND_SIGNALS[signal],
            "mean": float(mean),
            "std": float(std),
            "observation_count": int(observation_count),
            "status": context.quality_assessments[signal].status,
        })

    if not payloads:
        raise ReplayError(
            f"Participant {participant_id} has no usable signals for baseline calibration."
        )
    return payloads


def _load_wesad_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ReplayError(
            f"{path} is missing required columns: {', '.join(missing_columns)}."
        )
    if frame.empty:
        raise ReplayError(f"{path} contains no WESAD rows.")

    timestamps: list[pd.Timestamp] = []
    for value in frame["timestamp"]:
        if pd.isna(value):
            raise ReplayError(f"{path} contains a row without a timestamp.")
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            raise ReplayError(f"{path} contains a timestamp without a timezone.")
        timestamps.append(timestamp.tz_convert("UTC"))
    frame["_parsed_timestamp"] = timestamps

    labels = pd.to_numeric(frame["stress_label"], errors="raise")
    if labels.isna().any():
        raise ReplayError(f"{path} contains a row without a WESAD protocol label.")
    if ((labels < 0) | (labels % 1 != 0)).any():
        raise ReplayError(
            f"{path} contains a WESAD protocol label that is not a non-negative integer."
        )
    frame["stress_label"] = labels.astype(int)

    if frame["participant_id"].isna().any():
        raise ReplayError(f"{path} contains a row without participant_id.")
    frame["participant_id"] = frame["participant_id"].astype(str).str.strip()
    if (frame["participant_id"] == "").any():
        raise ReplayError(f"{path} contains a blank participant_id.")
    return frame.sort_values("_parsed_timestamp").reset_index(drop=True)


def replay_wesad_files(
    csv_paths: Sequence[str | Path],
    base_url: str = DEFAULT_BASE_URL,
    max_readings: int | None = None,
    delay_seconds: float = 0.0,
    timeout: float = 5.0,
    request_sender: Callable[
        [str, Mapping[str, Any], float], Mapping[str, Any]
    ] | None = None,
) -> dict[str, Any]:
    """Calibrate participant baselines and replay CSV rows through the API."""
    if not csv_paths:
        raise ValueError("At least one WESAD CSV path is required.")
    if max_readings is not None and max_readings < 1:
        raise ValueError("max_readings must be a positive integer.")
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("delay_seconds must be finite and non-negative.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive.")

    sender = request_sender or _post_json
    frames = [_load_wesad_csv(Path(path)) for path in csv_paths]
    readings = pd.concat(frames, ignore_index=True).sort_values(
        "_parsed_timestamp"
    ).reset_index(drop=True)
    all_participant_ids = sorted(
        readings["participant_id"].astype(str).unique().tolist()
    )
    base = base_url.rstrip("/")
    calibration_endpoint = f"{base}/api/baseline/calibrate"
    readings_endpoint = f"{base}/api/readings"

    replay_parts: list[pd.DataFrame] = []
    calibration_payloads: dict[str, list[dict[str, Any]]] = {}
    for participant_id in all_participant_ids:
        participant_rows = readings.loc[
            readings["participant_id"].astype(str) == participant_id
        ]
        calibration_rows = _select_calibration_rows(
            participant_id, participant_rows
        )
        replay_parts.append(
            participant_rows.loc[
                participant_rows["_parsed_timestamp"]
                > calibration_rows["_parsed_timestamp"].max()
            ]
        )
        calibration_payloads[participant_id] = _prepare_calibration_payloads(
            participant_id, calibration_rows
        )

    eligible_readings = pd.concat(replay_parts, ignore_index=True).sort_values(
        "_parsed_timestamp"
    ).reset_index(drop=True)
    if eligible_readings.empty:
        raise ReplayError("No readings remain after the baseline calibration period.")
    selected_readings = (
        eligible_readings.head(max_readings)
        if max_readings is not None
        else eligible_readings
    )
    participant_ids = sorted(
        selected_readings["participant_id"].astype(str).unique().tolist()
    )
    reading_payloads: list[dict[str, Any]] = []
    for _, row in selected_readings.iterrows():
        payload = build_reading_payload(row.to_dict())
        try:
            Measurement(**payload)
        except ValidationError as error:
            raise ReplayError(
                f"Reading at {payload['timestamp']} does not match the backend schema: {error}"
            ) from error
        reading_payloads.append(payload)

    calibrations_saved = 0
    baseline_statuses: list[dict[str, str]] = []
    for participant_id in participant_ids:
        for payload in calibration_payloads[participant_id]:
            try:
                sender(calibration_endpoint, payload, timeout)
            except ReplayError as error:
                raise ReplayError(
                    f"Failed to calibrate {participant_id} signal "
                    f"{payload['signal']}: {error}"
                ) from error
            calibrations_saved += 1
            baseline_statuses.append({
                "participant_id": participant_id,
                "signal": payload["signal"],
                "status": str(payload["status"]),
            })

    for index, payload in enumerate(reading_payloads, start=1):
        try:
            sender(readings_endpoint, payload, timeout)
        except ReplayError as error:
            raise ReplayError(
                f"Failed to replay reading {index} at {payload['timestamp']}: {error}"
            ) from error
        if delay_seconds and index < len(selected_readings):
            time.sleep(delay_seconds)

    return {
        "participants": participant_ids,
        "baselines_saved": calibrations_saved,
        "baseline_statuses": baseline_statuses,
        "readings_sent": len(selected_readings),
        "eligible_readings": len(eligible_readings),
        "input_readings": len(readings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate and replay prepared WESAD CSV data into the local API."
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or more prepared WESAD CSV files.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--max-readings",
        type=int,
        help="Optional limit across all CSV files; omit to replay every row.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional pause between API requests.",
    )
    args = parser.parse_args()

    summary = replay_wesad_files(
        csv_paths=args.csv,
        base_url=args.base_url,
        max_readings=args.max_readings,
        delay_seconds=args.delay_seconds,
    )
    print(
        f"Replayed {summary['readings_sent']} of "
        f"{summary['eligible_readings']} eligible rows "
        f"from {summary['input_readings']} input rows "
        f"for {', '.join(summary['participants'])}; "
        f"saved {summary['baselines_saved']} baseline signal(s)."
    )
    print(
        "Baseline quality: "
        + ", ".join(
            f"{item['participant_id']}.{item['signal']}={item['status']}"
            for item in summary["baseline_statuses"]
        )
    )
    print("Inspect /api/risks/recent for personalized_anomaly results.")


if __name__ == "__main__":
    main()
