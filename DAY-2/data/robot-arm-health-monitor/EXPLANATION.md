# Design rationale

## 1. Why these specific safety thresholds

The immediate-stop check must be deterministic, so the thresholds are fixed
numbers tied to physical limits rather than learned statistics.

**Torque = 120 Nm.** The arm's continuous rating is 100 Nm, so 120 Nm is 120 %
of rated load — the region where gear teeth and harmonic-drive flexsplines
start to yield. Above this, damage is happening *now*, not later.

**Temperature = 85 °C.** This is the class-F winding insulation limit and the
temperature at which lithium-complex bearing grease loses its base oil fast.
Below it, heat is a wear accelerant; above it, insulation breakdown is minutes
away.

**Vibration = 12 mm/s RMS.** ISO 10816 zone D — "vibration of this magnitude
will cause damage". Zones A/B (normal) top out around 4.5 mm/s, so 12 mm/s is
unambiguously abnormal and cannot be produced by a healthy gearbox.

The three thresholds sit far outside the normal operating distribution
(torque ~40 Nm, temp ~48 °C, vibration ~3 mm/s), so they never false-fire on a
healthy arm, and they are checked before any model runs. `safety_check.py`
imports nothing but `config`, so it keeps working with no model file, no
scikit-learn, and no network.

## 2. How the model and the log combine

Two independent sources of evidence, combined with an explicit weighted sum:

```
risk = 0.7 · P(model says worn)  +  0.3 · (normalised log signal)
```

The model gets the larger weight because it is quantitative and continuous.
The log gets a smaller weight because it is sparse and subjective — but it is
*independent*: a technician can write "wrist feels loose" weeks before any
sensor channel moves, and the model can never see that.

Two floors then apply:
- If `P(model) ≥ 0.85`, risk is raised to at least 0.70 — a confident model
  should not need log corroboration.
- If the log contains a critical term (weight ≥ 0.9) **and** `P(model) ≥ 0.45`,
  risk is raised to at least 0.70 — two independent signals agreeing is strong
  evidence even when each is individually weak.

Finally: `risk ≥ 0.65 → SCHEDULE_MAINTENANCE`, `≥ 0.30 → MONITOR`, else `SAFE`.

A worked example: the model says 0.60, the log says 0.90 (a technician wrote
"loud grinding noise"). The critical term fires and the model clears 0.45, so
risk is floored at 0.70 → `SCHEDULE_MAINTENANCE`. If the model had said 0.10,
the log alone would not have floored the risk — a noisy note should not by
itself pull an arm off the line.

## 3. How the spoofed-sensor defence works

Four independent checks, each of which can fire on its own:

1. **Plausibility bounds.** Torque outside [0, 160] Nm, temperature outside
   [5, 120] °C, vibration outside [0, 25] mm/s cannot come from a real sensor.
2. **Rate of change.** Torque cannot move more than 45 Nm in one control cycle.
3. **Frozen signal.** If a joint's torque has a standard deviation below
   0.05 Nm over five consecutive cycles *while its temperature and vibration
   are still varying*, the torque channel has been pinned or replayed.
4. **Cross-signal masking.** Offline, we fit a static physics prior
   `torque ≈ a·temperature + b·vibration + c` per joint on the historical
   dataset. At runtime, if the reported torque is more than
   `max(2.5σ, 12 Nm)` *below* what the joint's heat and vibration imply — and
   those two signals are themselves elevated — we flag a masking attack.

Check 4 is the core defence, because the obvious attack is exactly this:
leave the temperature and vibration sensors alone and rewrite torque to a
comfortable value so the Layer 0 check never trips. The physics prior makes
that self-contradictory.

Any flag escalates straight to `EMERGENCY_STOP`. The justification is
fail-safe: the Layer 0 safety check reads torque, so if torque is not
trustworthy the safety check is blind, and we must stop rather than pretend
we know the arm is safe.

**What it catches:**
- Replay attacks that pin a stale torque value (check 3).
- Single-channel overwrites with a low constant, the classic masking attack (4).
- Implausible or out-of-range injections, including negative torque (1).
- Sudden step injections (2).

**What it does not catch:**
- A coordinated attacker who spoofs torque, temperature *and* vibration
  consistently. All four checks are cross-signal checks, so a consistent lie
  looks exactly like the truth.
- Slow drift attacks that stay inside the plausibility envelope and below the
  residual limit — the attacker trades a large lie for many small ones.
- Attacks that only lower a *non-torque* channel. The prior only guards torque,
  because torque is what the safety layer depends on.
- A compromised sensor that is physically consistent with a *different* real
  arm (a swapped-identity attack) — we have no per-arm identity binding.

Closing these gaps would require cryptographic sensor attestation or hardware
redundancy, which is out of scope for a software-only pipeline.