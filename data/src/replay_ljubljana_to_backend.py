"""Replay Ljubljana environmental CSV data into the local HealthSensing API."""

import argparse
import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_CSV = "data/processed/ljubljana_real.csv"


def _post_json(endpoint: str, payload: dict, timeout: float = 5.0) -> dict:
    req = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Backend rejected {endpoint} HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"Backend unavailable: {e.reason}") from e


def _optional_float(val) -> float | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    f = float(val)
    return f if math.isfinite(f) else None


def build_payload(row: dict, timestamp_override: str | None = None) -> dict:
    raw_ts = timestamp_override or str(row.get("timestamp", ""))
    # Ensure ISO format with Z suffix
    if "T" not in raw_ts:
        raw_ts = raw_ts.replace(" ", "T")
    if not raw_ts.endswith("Z") and "+" not in raw_ts:
        raw_ts += ":00Z" if raw_ts.count(":") == 1 else "Z"

    return {
        "source": "ljubljana",
        "timestamp": raw_ts,
        "pm25":                _optional_float(row.get("pm25_ug_m3")),
        "pm10":                _optional_float(row.get("pm10_ug_m3")),
        "ambient_temperature": _optional_float(row.get("ambient_temperature_c")),
        "humidity":            _optional_float(row.get("humidity_percent")),
    }


def replay_ljubljana(
    csv_path: str | Path = DEFAULT_CSV,
    base_url: str = DEFAULT_BASE_URL,
    delay_seconds: float = 2.0,
    loop: bool = False,
) -> int:
    """Send Ljubljana rows to the backend. Returns number of rows sent."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Ljubljana CSV not found: {path}")

    df = pd.read_csv(path).dropna(subset=["timestamp"])
    endpoint = f"{base_url.rstrip('/')}/api/readings"
    sent = 0

    while True:
        for _, row in df.iterrows():
            payload = build_payload(row.to_dict())
            try:
                _post_json(endpoint, payload)
                sent += 1
                if sent % 10 == 0:
                    print(f"[ljubljana] Sent {sent} env readings...")
            except RuntimeError as e:
                print(f"[ljubljana] Warning: {e}")
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        print(f"[ljubljana] Finished pass ({sent} total). {'Looping...' if loop else 'Done.'}")
        if not loop:
            break

    return sent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay Ljubljana environmental CSV data into the HealthSensing API."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to ljubljana_real.csv")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--delay-seconds", type=float, default=2.0,
                        help="Seconds between readings (default: 2)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop forever after finishing the CSV")
    args = parser.parse_args()

    print(f"[ljubljana] Replaying {args.csv} → {args.base_url} (delay={args.delay_seconds}s, loop={args.loop})")
    total = replay_ljubljana(
        csv_path=args.csv,
        base_url=args.base_url,
        delay_seconds=args.delay_seconds,
        loop=args.loop,
    )
    print(f"[ljubljana] Done. Sent {total} readings.")


if __name__ == "__main__":
    main()
