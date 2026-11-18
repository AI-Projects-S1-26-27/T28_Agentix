"""Prototype sanity check: run the risk engine on windows of simulator output.

Usage (from repo root):  python -m experiments.risk_rules_simulator_check
This checks the rules against *simulated* data only. It is NOT a model
evaluation and says nothing about real-world or clinical accuracy.
"""
import random
from collections import Counter

from risk_engine.assess import assess_risk
from simulator.data_generator import generate_data
from simulator.scenarios import SCENARIOS

WINDOWS = 1000
WINDOW_SIZE = 5
SEED = 42


def run(windows: int = WINDOWS, seed: int = SEED, anomaly_score=None) -> dict:
    results = {}
    for scenario in SCENARIOS:
        rng = random.Random(seed)
        levels, categories = Counter(), Counter()
        for _ in range(windows):
            readings = [generate_data(scenario, rng) for _ in range(WINDOW_SIZE)]
            assessment = assess_risk(readings, anomaly_score=anomaly_score)
            levels[assessment.level.value] += 1
            categories[assessment.primary_category.value if assessment.primary_category else "none"] += 1
        results[scenario] = {"levels": dict(levels), "categories": dict(categories)}
    return results


if __name__ == "__main__":
    for scenario, summary in run().items():
        print(f"{scenario:10s} levels={summary['levels']}  categories={summary['categories']}")