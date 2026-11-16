import time
from typing import Iterator, Optional

from simulator.config import DEFAULT_SCENARIO, SIMULATION_INTERVAL
from simulator.data_generator import generate_data
from simulator.client import post_reading


def _fetch_active_scenario(base_url: str, fallback: str) -> str:
    """Poll the backend for the currently active scenario."""
    try:
        import urllib.request, json as _json
        api_base = base_url.split("/api/")[0] + "/api"
        with urllib.request.urlopen(f"{api_base}/scenarios/active", timeout=1) as r:
            return _json.loads(r.read()).get("scenario", fallback)
    except Exception:
        return fallback


def stream_data(scenario: str, count: Optional[int] = None, endpoint: Optional[str] = None) -> Iterator[dict]:
    """Yield simulated records. In continuous mode, checks backend for active scenario each tick."""
    generated = 0
    current = scenario
    while count is None or generated < count:
        # In continuous mode, respect scenario changes made via the UI
        if count is None and endpoint:
            current = _fetch_active_scenario(endpoint, current)
        yield generate_data(current)
        generated += 1


def run_simulator(
    scenario: str,
    count: Optional[int] = None,
    endpoint: Optional[str] = None,
) -> None:
    """Print simulated data for a finite run or continuously."""
    print("----------------------------------------")
    print("Personal Health Companion")
    print("Health Data Simulator")
    print("----------------------------------------")
    print(f"Scenario : {scenario} (will follow UI changes in continuous mode)")
    print(f"Interval : {SIMULATION_INTERVAL} seconds")
    print("Press CTRL+C to stop.")
    print("----------------------------------------")

    try:
        for data in stream_data(scenario, count, endpoint):
            print(data)
            if endpoint:
                print(post_reading(data, endpoint=endpoint))
            if count is None:
                time.sleep(SIMULATION_INTERVAL)
    except KeyboardInterrupt:
        print("\nSimulator stopped.")


if __name__ == "__main__":
    from simulator.client import post_reading as _client
    import inspect as _inspect
    _default_endpoint = _inspect.signature(_client).parameters["endpoint"].default
    run_simulator(DEFAULT_SCENARIO, endpoint=_default_endpoint)
