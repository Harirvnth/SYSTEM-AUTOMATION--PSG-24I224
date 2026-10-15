"""LAYER 2b -- free-text technician notes as an independent signal.

A weighted keyword scorer with a three-token negation window. It never looks
at sensor numbers, so it is genuinely independent evidence: a technician can
write "wrist feels loose" before any sensor moves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Vocabulary. Weight 0.9+ marks a "critical" term.
# --------------------------------------------------------------------------
RELEVANT_KEYWORDS: dict[str, float] = {
    # thermal
    "overheat": 0.9, "overheating": 0.9, "overheated": 0.9, "hot": 0.5,
    "burning": 1.0, "burnt": 1.0, "smoke": 1.0, "smoking": 1.0, "scorch": 1.0,
    # vibration / acoustic
    "vibration": 0.9, "vibrating": 0.9, "vibrates": 0.9, "shaking": 0.7,
    "grinding": 1.0, "growling": 0.8, "knocking": 0.9, "rattling": 0.8,
    "squealing": 0.8, "squeal": 0.8, "noise": 0.6, "noisy": 0.6,
    # mechanical
    "worn": 0.9, "wear": 0.9, "crack": 1.0, "cracked": 1.0, "loose": 0.8,
    "backlash": 1.0, "play": 0.6, "misaligned": 0.9, "misalignment": 0.9,
    "corrosion": 0.9, "corroded": 0.9, "rust": 0.7, "rusted": 0.7,
    # lubrication
    "leak": 0.9, "leaking": 0.9, "leaked": 0.9, "grease": 0.3, "oil": 0.3,
    # control / behaviour
    "erratic": 0.9, "drift": 0.8, "drifting": 0.8, "spike": 0.8, "spikes": 0.8,
    "spiking": 0.8, "stuck": 0.8, "jammed": 1.0, "jam": 0.9, "fault": 0.8,
    "alarm": 0.7, "intermittent": 0.6, "worse": 0.5, "degrading": 0.8,
}

NEGATION_TOKENS = {
    "no", "not", "never", "without", "none", "free", "n't", "isn't", "wasn't",
    "aren't", "doesn't", "didn't", "don't", "cannot", "can't", "hardly", "rarely",
}

SATURATION = 1.5          # raw weight that maps to a score of 1.0
CRITICAL_WEIGHT = 0.9

TOKEN_RE = re.compile(r"[a-z']+")


@dataclass
class LogEntry:
    timestamp: str
    arm_id: str
    text: str


@dataclass
class LogSignal:
    arm_id: str
    score: float = 0.0
    critical: bool = False
    matched: list = field(default_factory=list)
    entries: list = field(default_factory=list)


def parse_log(path: str = "data/maintenance_log.txt") -> list[LogEntry]:
    """Parse 'YYYY-MM-DD HH:MM | ARM-XX | free text' lines."""
    entries: list[LogEntry] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        entries.append(LogEntry(parts[0], parts[1], " | ".join(parts[2:])))
    return entries


def _score_text(text: str) -> tuple[float, list[str]]:
    tokens = TOKEN_RE.findall(text.lower())
    raw, matched = 0.0, []
    for i, tok in enumerate(tokens):
        weight = RELEVANT_KEYWORDS.get(tok)
        if weight is None:
            continue
        window = tokens[max(0, i - 3):i]
        if any(t in NEGATION_TOKENS or t.endswith("n't") for t in window):
            continue                      # "no abnormal noise" -> ignore
        raw += weight
        matched.append(tok)
    return raw, matched


def analyse(entries: list[LogEntry], arm_id: str) -> LogSignal:
    sig = LogSignal(arm_id=arm_id)
    raw_total = 0.0
    critical = False
    for e in entries:
        if e.arm_id != arm_id:
            continue
        sig.entries.append(f"{e.timestamp}: {e.text}")
        raw, matched = _score_text(e.text)
        raw_total += raw
        sig.matched.extend(matched)
        if any(RELEVANT_KEYWORDS[t] >= CRITICAL_WEIGHT for t in matched):
            critical = True
    sig.score = float(min(1.0, raw_total / SATURATION))
    sig.critical = critical
    return sig