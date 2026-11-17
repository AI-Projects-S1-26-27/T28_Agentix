"""Unit tests for HealthSensing data simulator."""

import random
import pytest

from simulator.config import DEFAULT_SCENARIO, SIMULATION_INTERVAL
from simulator.data_generator import generate_data
from simulator.scenarios import SCENARIOS
from simulator.simulator import stream_data


def test_default_scenario_configuration():
    """Verify that default scenario is registered and valid."""
    assert DEFAULT_SCENARIO in SCENARIOS
    assert SIMULATION_INTERVAL > 0


@pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
def test_all_scenarios_generate_valid_ranges(scenario_name: str):
    """Verify that each scenario generates readings within defined bounds."""
    rng = random.Random(42)
    ranges = SCENARIOS[scenario_name]
    for _ in range(10):
        data = generate_data(scenario_name, rng=rng)
        assert data["scenario"] == scenario_name
        assert data["source"] == "simulator"
        assert ranges["heart_rate"][0] <= data["heart_rate"] <= ranges["heart_rate"][1]
        assert ranges["body_temperature"][0] <= data["body_temperature"] <= ranges["body_temperature"][1]
        assert ranges["ambient_temperature"][0] <= data["ambient_temperature"] <= ranges["ambient_temperature"][1]
        assert ranges["humidity"][0] <= data["humidity"] <= ranges["humidity"][1]
        assert ranges["aqi"][0] <= data["aqi"] <= ranges["aqi"][1]
        assert ranges["pm25"][0] <= data["pm25"] <= ranges["pm25"][1]


def test_stream_data_finite_count():
    """Verify that stream_data yields exactly the requested count of items."""
    stream = list(stream_data("normal", count=5))
    assert len(stream) == 5
    for item in stream:
        assert item["scenario"] == "normal"
