"""Glue: one object that evaluates a live reading end to end."""
from __future__ import annotations

from .decision import Decision, decide
from .log_nlp import LogEntry, analyse
from .safety_check import check_hard_limits
from .spoof_detect import detect_spoof
from .wear_model import WearModel


class ArmHealthPipeline:
    def __init__(self, wear_model: WearModel, torque_prior: dict,
                 log_entries: list[LogEntry]):
        self.wear_model = wear_model
        self.torque_prior = torque_prior
        self.log_entries = log_entries

    def evaluate(self, arm_id: str, reading: dict,
                 history: list | None = None) -> Decision:
        violations = check_hard_limits(reading)                 # Layer 0
        spoof = detect_spoof(reading, history, self.torque_prior)  # Layer 1
        ml_prob = self.wear_model.predict_proba(reading)        # Layer 2a
        log_signal = analyse(self.log_entries, arm_id)          # Layer 2b
        return decide(arm_id, violations, spoof, ml_prob, log_signal)  # Layer 3