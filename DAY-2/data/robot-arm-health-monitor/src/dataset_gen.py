"""Synthetic historical dataset of arm inspections with wear labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import JOINTS, READING_COLUMNS, sensor_key

WEAR_LABEL_CUTOFF = 0.55
FAILURE_MODES = ["bearing_wear", "gear_backlash", "lubrication_loss", "motor_overheat"]


def _make_row(rng, wear_by_joint, reading_id, arm_id, noise_scale=1.0):
    row = {"reading_id": reading_id, "arm_id": arm_id}
    for j in JOINTS:
        w = float(wear_by_joint[j])
        row[sensor_key(j, "torque")] = round(
            float(np.clip(38 + 62 * w + rng.normal(0, 6 * noise_scale), 0, 200)), 2)
        row[sensor_key(j, "temperature")] = round(
            float(np.clip(42 + 45 * w + rng.normal(0, 3 * noise_scale), 0, 150)), 2)
        row[sensor_key(j, "vibration")] = round(
            float(np.clip(2.2 + 10 * w + rng.normal(0, 0.7 * noise_scale), 0, 30)), 2)
    return row


def generate_dataset(n_rows: int = 240, seed: int = 7,
                     out_csv: str = "data/historical_readings.csv") -> pd.DataFrame:
    """Create a labelled historical dataset of past joint failures.

    The latent variable is a per-joint `wear` level in [0, 1]. Every sensor is
    generated as a monotone function of wear plus noise, which is exactly the
    physical story we want the model to learn. An arm is labelled WORN (1) if
    ANY of its six joints has crossed the wear cutoff.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        global_wear = rng.uniform(-0.10, 0.95)
        wear = {j: float(np.clip(global_wear + rng.normal(0, 0.13), 0.0, 1.0))
                for j in JOINTS}
        row = _make_row(rng, wear, f"R{i:04d}", f"ARM-{(i % 8) + 1:02d}")
        label = int(max(wear.values()) >= WEAR_LABEL_CUTOFF)
        if rng.random() < 0.04:          # 4 % real-world label noise
            label = 1 - label
        row["wear_label"] = label
        row["failure_mode"] = str(rng.choice(FAILURE_MODES)) if label else "none"
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def synth_reading(wear_by_joint: dict, seed: int = 0,
                  noise_scale: float = 1.0) -> dict:
    """Build one live sensor reading from specified per-joint wear levels."""
    rng = np.random.default_rng(seed)
    wear = {j: float(wear_by_joint.get(j, 0.0)) for j in JOINTS}
    row = _make_row(rng, wear, "live", "live", noise_scale=noise_scale)
    return {k: row[k] for k in READING_COLUMNS}