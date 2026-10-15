"""LAYER 0 -- hard, deterministic safety limits.

This module has NO dependency on the ML model, on scikit-learn, or on the
network. It is a pure function of the incoming reading, so it keeps working
even if every model file is missing or completely wrong.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config as C


@dataclass(frozen=True)
class SafetyViolation:
    joint: str
    sensor: str
    value: float
    threshold: float

    def describe(self) -> str:
        return (f"{self.joint} ({C.JOINT_LOCATION[self.joint]}) {self.sensor} = "
                f"{self.value:.2f} {C.UNITS[self.sensor]} exceeds hard limit "
                f"{self.threshold} {C.UNITS[self.sensor]}")


def check_hard_limits(reading: dict) -> list[SafetyViolation]:
    """Pure, deterministic, model-free. Runs in microseconds."""
    out: list[SafetyViolation] = []
    for j in C.JOINTS:
        for s in C.SENSORS:
            v = reading.get(C.sensor_key(j, s))
            if v is None:
                continue
            limit = C.SAFETY_THRESHOLDS[s]
            triggered = abs(float(v)) > limit if s == "torque" else float(v) > limit
            if triggered:
                out.append(SafetyViolation(j, s, float(v), limit))
    return out