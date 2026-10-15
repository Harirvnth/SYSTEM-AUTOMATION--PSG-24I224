"""LAYER 1 -- spoofed / untrusted sensor detection.

Four independent checks, each of which can stand alone:

  A. plausibility_bounds    the value is outside the physical envelope
  B. rate_of_change         torque teleported between two cycles
  C. frozen_signal          torque is pinned while other sensors still move
                            (classic replay / stuck-sensor attack)
  D. cross_signal_masking   torque reads LOW while temperature and vibration
                            read HIGH -- the signature of a masking attack

Check D uses a *static physics prior* fitted offline (torque ~ temp + vib).
The prior is not the ML model; if it is missing, A/B/C still run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C


@dataclass
class SpoofReport:
    spoofed: bool = False
    confidence: float = 0.0
    reasons: list = field(default_factory=list)
    checks_run: list = field(default_factory=list)

    @property
    def summary(self) -> str:
        return " | ".join(self.reasons) if self.reasons else "no anomaly"


def detect_spoof(reading: dict, history: list | None = None,
                 torque_prior: dict | None = None) -> SpoofReport:
    history = list(history or [])
    rep = SpoofReport()
    score = 0.0

    # --- Check A: absolute physical envelope ------------------------------
    rep.checks_run.append("plausibility_bounds")
    for j in C.JOINTS:
        for s in C.SENSORS:
            v = reading.get(C.sensor_key(j, s))
            if v is None:
                continue
            lo, hi = C.PLAUSIBLE_RANGE[s]
            if not (lo <= float(v) <= hi):
                rep.reasons.append(
                    f"{j} {s}={v}{C.UNITS[s]} outside physical envelope [{lo}, {hi}]")
                score += 1.0

    # --- Check B: rate of change ------------------------------------------
    rep.checks_run.append("rate_of_change")
    if history:
        prev = history[-1]
        for j in C.JOINTS:
            k = C.sensor_key(j, "torque")
            a, b = reading.get(k), prev.get(k)
            if a is None or b is None:
                continue
            if abs(float(a) - float(b)) > C.MAX_TORQUE_STEP:
                rep.reasons.append(
                    f"{j} torque jumped {abs(float(a) - float(b)):.1f} Nm in one cycle "
                    f"(physical max {C.MAX_TORQUE_STEP})")
                score += 0.8

    # --- Check C: frozen / replayed torque --------------------------------
    rep.checks_run.append("frozen_signal")
    if len(history) >= C.FROZEN_WINDOW - 1:
        window = history[-(C.FROZEN_WINDOW - 1):] + [reading]
        for j in C.JOINTS:
            tk = C.sensor_key(j, "torque")
            vals = [r.get(tk) for r in window]
            if any(v is None for v in vals):
                continue
            if float(np.std(vals)) < C.FROZEN_STD_EPS:
                others = []
                for s in ("temperature", "vibration"):
                    key = C.sensor_key(j, s)
                    others += [float(r[key]) for r in window if r.get(key) is not None]
                if others and float(np.std(others)) > C.FROZEN_OTHER_STD_MIN:
                    rep.reasons.append(
                        f"{j} torque frozen at {float(vals[0]):.2f} Nm for "
                        f"{len(window)} cycles while temperature/vibration keep "
                        f"moving -> stuck or replayed sensor")
                    score += 0.9

    # --- Check D: cross-signal masking ------------------------------------
    rep.checks_run.append("cross_signal_masking")
    if torque_prior:
        for j in C.JOINTS:
            m = torque_prior.get(j)
            if not m:
                continue
            t = reading.get(C.sensor_key(j, "temperature"))
            v = reading.get(C.sensor_key(j, "vibration"))
            y = reading.get(C.sensor_key(j, "torque"))
            if t is None or v is None or y is None:
                continue
            a_t, a_v, b = m["coef"]
            expected = a_t * float(t) + a_v * float(v) + b
            resid = float(y) - expected
            limit = max(C.MASK_RESIDUAL_SIGMA * m["sigma"], C.MIN_MASK_DELTA)
            corroborated = (float(t) > m["temp_ref"] + C.MASK_TEMP_MARGIN or
                            float(v) > m["vib_ref"] + C.MASK_VIB_MARGIN)
            if resid < -limit and corroborated:
                rep.reasons.append(
                    f"{j} torque reports {float(y):.1f} Nm but temperature "
                    f"{float(t):.1f}C / vibration {float(v):.1f}mm/s imply "
                    f"~{expected:.1f} Nm (residual {resid:+.1f}) -> possible "
                    f"masking attack")
                score += 1.0

    rep.spoofed = score >= C.SPOOF_SCORE_THRESHOLD
    rep.confidence = float(min(1.0, score / 2.0))
    return rep