from math import isfinite
from typing import Mapping, Optional

from risk_engine.explanation import build_explanation
from risk_engine.risk_levels import (
	calculate_heat_index_celsius,
	classify_aqi,
	classify_heat_index,
	compute_aqi_from_pm25,
)


def _optional_number(reading: Mapping[str, object], field: str) -> Optional[float]:
	value = reading.get(field)
	if value is None:
		return None
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		raise ValueError(f"{field} must be a number or null")
	number = float(value)
	if not isfinite(number):
		raise ValueError(f"{field} must be finite")
	return number


def evaluate_environmental_risk(reading: Mapping[str, object]) -> dict:
	"""Assess environmental signals without inferring an individual's diagnosis."""
	components = {}
	unavailable_inputs = []

	# AQI derived from PM2.5 (we have pm25_ug_m3; compute AQI via EPA formula) or direct aqi
	pm25 = _optional_number(reading, "pm25")
	if pm25 is not None:
		aqi_value = compute_aqi_from_pm25(pm25)
		if aqi_value is not None:
			aqi_result = classify_aqi(aqi_value)
			if aqi_result:
				components["air_quality"] = aqi_result
				components["aqi"] = aqi_result
		else:
			unavailable_inputs.append("aqi")
	else:
		aqi_direct = _optional_number(reading, "aqi")
		if aqi_direct is not None:
			aqi_result = classify_aqi(aqi_direct)
			if aqi_result:
				components["air_quality"] = aqi_result
				components["aqi"] = aqi_result
		else:
			unavailable_inputs.append("aqi")

	ambient_temperature = _optional_number(reading, "ambient_temperature")
	humidity = _optional_number(reading, "humidity")
	if ambient_temperature is None or humidity is None:
		unavailable_inputs.append("heat_index")
	else:
		heat_index = calculate_heat_index_celsius(ambient_temperature, humidity)
		components["heat"] = classify_heat_index(heat_index)

	if not components:
		risk_level = "unknown"
		severity = None
	else:
		highest = max(
			components.values(),
			key=lambda result: result["severity"],
		)
		risk_level = highest["risk_level"]
		severity = highest["severity"]

	return {
		"risk_level": risk_level,
		"severity": severity,
		"components": components,
		"unavailable_inputs": unavailable_inputs,
		"explanation": build_explanation(
			risk_level,
			components,
			unavailable_inputs,
		),
	}
