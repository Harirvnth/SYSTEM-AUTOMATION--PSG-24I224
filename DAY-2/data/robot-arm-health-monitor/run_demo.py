"""
End-to-end demo.

Builds the synthetic dataset, trains the wear model, fits the physics prior,
parses the maintenance log, then evaluates five live arms so that all four
possible decisions are produced at least once.
"""
from __future__ import annotations

from src import config as C
from src.dataset_gen import generate_dataset, synth_reading
from src.log_nlp import analyse, parse_log
from src.pipeline import ArmHealthPipeline
from src.wear_model import WearModel, fit_torque_prior, save_prior

RULE = "=" * 74


def make_history(wear_by_joint, n=4, seed=100):
    return [synth_reading(wear_by_joint, seed=seed + i) for i in range(n)]


def build_scenarios():
    """Five hand-crafted arms covering every decision."""
    scenarios = {}

    # ---- ARM-01: healthy -> SAFE ----------------------------------------
    nominal = {j: 0.08 for j in C.JOINTS}
    scenarios["ARM-01"] = dict(
        reading=synth_reading(nominal, seed=11, noise_scale=0.3),
        history=make_history(nominal, seed=101),
        note="nominal arm, clean log")

    # ---- ARM-02: borderline wear -> MONITOR ------------------------------
    mild = {"J1": 0.25, "J2": 0.30, "J3": 0.35,
            "J4": 0.40, "J5": 0.40, "J6": 0.45}
    scenarios["ARM-02"] = dict(
        reading=synth_reading(mild, seed=22, noise_scale=0.3),
        history=make_history(mild, seed=102),
        note="incipient wear, only a minor log note")

    # ---- ARM-03: heavy wear + critical note -> SCHEDULE_MAINTENANCE ------
    heavy = {"J1": 0.62, "J2": 0.70, "J3": 0.78,
             "J4": 0.74, "J5": 0.80, "J6": 0.66}
    scenarios["ARM-03"] = dict(
        reading=synth_reading(heavy, seed=33, noise_scale=0.3),
        history=make_history(heavy, seed=103),
        note="advanced wear, technician flagged vibration")

    # ---- ARM-04: hard limit breach -> EMERGENCY_STOP (Layer 0) -----------
    arm4 = synth_reading(heavy, seed=44, noise_scale=0.3)
    arm4_history = make_history(heavy, seed=104)
    for i, r in enumerate(arm4_history):                 # smooth ramp-up
        r[C.sensor_key("J3", "torque")] = 100.0 + 6.0 * i
    arm4[C.sensor_key("J3", "torque")] = 137.4           # breach!
    scenarios["ARM-04"] = dict(
        reading=arm4, history=arm4_history,
        note="elbow torque crosses the hard limit")

    # ---- ARM-05: spoofed torque -> EMERGENCY_STOP (Layer 1) --------------
    spoof_wear = {"J1": 0.40, "J2": 0.45, "J3": 0.50,
                  "J4": 0.72, "J5": 0.60, "J6": 0.50}
    arm5 = synth_reading(spoof_wear, seed=55, noise_scale=0.3)
    arm5_history = make_history(spoof_wear, seed=105)
    for r in arm5_history + [arm5]:                      # attacker pins torque
        for j in C.JOINTS:
            r[C.sensor_key(j, "torque")] = 11.0
    scenarios["ARM-05"] = dict(
        reading=arm5, history=arm5_history,
        note="wrist is hot and vibrating, but torque reads 11 Nm everywhere")

    return scenarios


def main() -> None:
    print(RULE)
    print(" ROBOTIC ARM HEALTH MONITORING -- END-TO-END DEMO")
    print(RULE)

    print("\n[1/4] Generating synthetic historical dataset ...")
    df = generate_dataset(n_rows=240, seed=7)
    worn = int(df["wear_label"].sum())
    print(f"      {len(df)} labelled readings -> data/historical_readings.csv")
    print(f"      class balance: {worn} WORN / {len(df) - worn} HEALTHY")

    print("\n[2/4] Training the predictive wear model ...")
    model = WearModel().fit(df)
    if model.metrics_:
        print(f"      hold-out ROC-AUC = {model.metrics_['roc_auc']:.3f}")
        for line in model.metrics_["report"].rstrip().splitlines():
            print("      " + line)

    print("\n[3/4] Fitting the torque physics prior (spoof defence) ...")
    prior = fit_torque_prior(df)
    save_prior(prior)
    print("      saved -> data/torque_prior.json")

    print("\n[4/4] Parsing maintenance log ...")
    entries = parse_log("data/maintenance_log.txt")
    print(f"      {len(entries)} entries parsed from data/maintenance_log.txt")

    pipeline = ArmHealthPipeline(model, prior, entries)
    scenarios = build_scenarios()

    seen = set()
    print("\n" + RULE)
    print(" LIVE ARM EVALUATION")
    print(RULE)
    for arm_id, sc in scenarios.items():
        decision = pipeline.evaluate(arm_id, sc["reading"], sc["history"])
        seen.add(decision.decision)
        print(f"\n--- {arm_id}  ({sc['note']}) ---")
        print(f"    ML wear probability : {decision.ml_prob:.3f}")
        print(f"    Log signal          : {decision.log_score:.2f}"
              f"   critical={analyse(entries, arm_id).critical}")
        print(f"    Fused risk          : {decision.risk:.3f}")
        print(f"    >>> DECISION        : {decision.decision}")
        for r in decision.reasons:
            print(f"        - {r}")

    print("\n" + RULE)
    print(f" Outcomes produced: {sorted(seen)}")
    missing = set(C.DECISIONS) - seen
    print(" All four decisions reachable: " + ("YES" if not missing
                                               else f"NO (missing {missing})"))
    print(RULE)


if __name__ == "__main__":
    main()