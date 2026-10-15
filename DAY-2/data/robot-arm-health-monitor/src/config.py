"""
Central configuration for the robotic-arm health-monitoring pipeline.

Everything an engineer might want to tune lives here: fleet layout, hard
safety limits, the sensor plausibility envelope, spoof-detection limits,
the maintenance-log vocabulary and the final decision policy.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Fleet layout
# ---------------------------------------------------------------------------
JOINTS = ["J1", "J2", "J3", "J4", "J5", "J6"]

JOINT_LOCATION = {
    "J1": "shoulder", "J2": "shoulder",
    "J3": "elbow",
    "J4": "wrist", "J5": "wrist", "J6": "wrist",
}

SENSORS = ["torque", "temperature", "vibration"]

UNITS = {"torque": "Nm", "temperature": "degC", "vibration": "mm/s"}


def sensor_key(joint: str, sensor: str) -> str:
    """Column name for one (joint, sensor) pair, e.g. 'J3_torque'."""
    return f"{joint}_{sensor}"


READING_COLUMNS = [sensor_key(j, s) for j in JOINTS for s in SENSORS]


# ---------------------------------------------------------------------------
# 2. LAYER 0 -- hard deterministic safety limits. NEVER touched by ML.
# ---------------------------------------------------------------------------
#   torque      120 Nm  = 120 % of the 100 Nm continuous rating (mechanical
#                         overload / gear-tooth shear territory)
#   temperature  85 C   = class-F winding insulation limit, and the point where
#                         lithium-complex bearing grease breaks down quickly
#   vibration    12 mm/s RMS ~ ISO 10816 zone D ("damage occurring")
SAFETY_THRESHOLDS = {
    "torque": 120.0,
    "temperature": 85.0,
    "vibration": 12.0,
}


# ---------------------------------------------------------------------------
# 3. LAYER 1 -- spoofed-sensor defence
# ---------------------------------------------------------------------------
# Absolute physical envelope a genuine sensor can never leave.
PLAUSIBLE_RANGE = {
    "torque": (0.0, 160.0),
    "temperature": (5.0, 120.0),
    "vibration": (0.0, 25.0),
}

MAX_TORQUE_STEP = 45.0        # Nm per cycle; a real joint cannot teleport
FROZEN_WINDOW = 5             # cycles used for the "stuck sensor" test
FROZEN_STD_EPS = 0.05         # Nm; below this the torque signal is frozen
FROZEN_OTHER_STD_MIN = 0.05   # other sensors must be moving for this to fire

MASK_RESIDUAL_SIGMA = 2.5     # residual must be < -2.5 sigma of the physics prior
MIN_MASK_DELTA = 12.0         # ...and at least this many Nm below expectation
MASK_TEMP_MARGIN = 7.0        # corroborating heat (degC above median)
MASK_VIB_MARGIN = 2.0         # corroborating vibration (mm/s above median)

SPOOF_SCORE_THRESHOLD = 1.0


# ---------------------------------------------------------------------------
# 4. LAYER 2 -- risk fusion policy
# ---------------------------------------------------------------------------
RISK_WEIGHT_MODEL = 0.7
RISK_WEIGHT_LOG = 0.3

RISK_MONITOR = 0.30           # risk >= 0.30 -> MONITOR
RISK_SCHEDULE = 0.65          # risk >= 0.65 -> SCHEDULE_MAINTENANCE

ML_CONFIDENT_PROB = 0.85
RISK_FLOOR_ML_CONFIDENT = 0.70
LOG_CRITICAL_MIN_MODEL_PROB = 0.45
RISK_FLOOR_LOG_CRITICAL = 0.70

DECISIONS = ["SAFE", "MONITOR", "SCHEDULE_MAINTENANCE", "EMERGENCY_STOP"]