"""LAYER 3 -- fuse every signal into exactly ONE decision per arm."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config as C
from .log_nlp import LogSignal
from .safety_check import SafetyViolation
from .spoof_detect import SpoofReport


@dataclass
class Decision:
    arm_id: str
    decision: str
    risk: float
    ml_prob: float
    log_score: float
    reasons: list = field(default_factory=list)
    violations: list = field(default_factory=list)
    spoof: SpoofReport | None = None


def decide(arm_id: str,
           violations: list[SafetyViolation],
           spoof: SpoofReport | None,
           ml_prob: float,
           log_signal: LogSignal) -> Decision:
    """Priority order matters: safety beats security beats prediction."""

    # ---- LAYER 0: deterministic hard limit (model-free) -------------------
    if violations:
        return Decision(
            arm_id, "EMERGENCY_STOP", 1.0, ml_prob, log_signal.score,
            reasons=[f"HARD LIMIT: {v.describe()}" for v in violations],
            violations=violations, spoof=spoof)

    # ---- LAYER 1: untrusted data -----------------------------------------
    # If torque is being spoofed, the Layer 0 check is blind, so we must
    # fail safe rather than trust a reading we know is fabricated.
    if spoof is not None and spoof.spoofed:
        return Decision(
            arm_id, "EMERGENCY_STOP", 1.0, ml_prob, log_signal.score,
            reasons=["UNTRUSTED SENSOR: " + r for r in spoof.reasons],
            spoof=spoof)

    # ---- LAYER 2: risk fusion --------------------------------------------
    risk = C.RISK_WEIGHT_MODEL * ml_prob + C.RISK_WEIGHT_LOG * log_signal.score
    reasons = [
        f"model wear probability = {ml_prob:.2f} (weight {C.RISK_WEIGHT_MODEL})",
        f"maintenance-note signal = {log_signal.score:.2f} (weight {C.RISK_WEIGHT_LOG})",
    ]

    if ml_prob >= C.ML_CONFIDENT_PROB:
        risk = max(risk, C.RISK_FLOOR_ML_CONFIDENT)
        reasons.append(f"model highly confident (>= {C.ML_CONFIDENT_PROB}) -> risk floor "
                       f"{C.RISK_FLOOR_ML_CONFIDENT}")

    if log_signal.critical and ml_prob >= C.LOG_CRITICAL_MIN_MODEL_PROB:
        risk = max(risk, C.RISK_FLOOR_LOG_CRITICAL)
        reasons.append("technician note is critical AND model agrees -> risk floor "
                       f"{C.RISK_FLOOR_LOG_CRITICAL}")

    if risk >= C.RISK_SCHEDULE:
        label = "SCHEDULE_MAINTENANCE"
    elif risk >= C.RISK_MONITOR:
        label = "MONITOR"
    else:
        label = "SAFE"

    return Decision(arm_id, label, round(risk, 4), ml_prob,
                    log_signal.score, reasons=reasons, spoof=spoof)