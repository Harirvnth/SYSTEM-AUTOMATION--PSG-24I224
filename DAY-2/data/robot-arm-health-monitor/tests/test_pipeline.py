"""Smoke + behaviour tests for the full pipeline."""
from __future__ import annotations

import pytest

from src import config as C
from src.dataset_gen import generate_dataset, synth_reading
from src.decision import decide
from src.log_nlp import LogSignal, analyse, parse_log
from src.safety_check import check_hard_limits
from src.spoof_detect import detect_spoof
from src.wear_model import WearModel, fit_torque_prior

NOMINAL = {j: 0.08 for j in C.JOINTS}
HEAVY = {j: 0.80 for j in C.JOINTS}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    df = generate_dataset(n_rows=200, seed=3, out_csv="data/historical_readings.csv")
    return df, WearModel().fit(df, evaluate=False), fit_torque_prior(df)


# ---- Layer 0 -------------------------------------------------------------
def test_hard_limit_fires_on_high_torque():
    r = synth_reading(NOMINAL, seed=1, noise_scale=0.1)
    r["J3_torque"] = 137.0
    v = check_hard_limits(r)
    assert len(v) == 1 and v[0].joint == "J3"


def test_hard_limit_is_model_free():
    """No ML object is passed in -- the check must still work."""
    r = synth_reading(NOMINAL, seed=2, noise_scale=0.1)
    assert check_hard_limits(r) == []
    r["J5_temperature"] = 99.0
    assert len(check_hard_limits(r)) == 1


def test_nominal_reading_has_no_violation():
    assert check_hard_limits(synth_reading(NOMINAL, seed=4, noise_scale=0.1)) == []


# ---- Layer 1 -------------------------------------------------------------
def test_detects_frozen_masked_torque(trained):
    _, _, prior = trained
    wear = {"J4": 0.72, **{j: 0.4 for j in C.JOINTS if j != "J4"}}
    hist = [synth_reading(wear, seed=200 + i) for i in range(4)]
    live = synth_reading(wear, seed=99, noise_scale=0.3)
    for r in hist + [live]:
        for j in C.JOINTS:
            r[C.sensor_key(j, "torque")] = 11.0
    rep = detect_spoof(live, hist, prior)
    assert rep.spoofed, rep.reasons
    assert rep.confidence > 0.0


def test_no_false_positive_on_healthy_arm(trained):
    _, _, prior = trained
    hist = [synth_reading(NOMINAL, seed=300 + i) for i in range(4)]
    live = synth_reading(NOMINAL, seed=77, noise_scale=0.3)
    assert not detect_spoof(live, hist, prior).spoofed


# ---- Layer 2b ------------------------------------------------------------
def test_negation_is_respected(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(
        "2024-01-01 08:00 | ARM-01 | No overheating observed, temps look normal.\n"
        "2024-01-01 09:00 | ARM-02 | Loud grinding noise from the shoulder.\n",
        encoding="utf-8")
    entries = parse_log(str(log))
    assert analyse(entries, "ARM-01").score == 0.0
    sig2 = analyse(entries, "ARM-02")
    assert sig2.score > 0.5 and sig2.critical


# ---- Layer 3: all four decisions reachable -------------------------------
def test_all_four_decisions_reachable(trained):
    _, model, prior = trained
    none_log = LogSignal("X", 0.0, False)
    hot_log = LogSignal("X", 0.95, True)

    # SAFE
    safe = synth_reading(NOMINAL, seed=5, noise_scale=0.3)
    d_safe = decide("A", [], detect_spoof(safe, None, prior),
                    model.predict_proba(safe), none_log)

    # MONITOR
   
    # MONITOR
    d_mon = decide(
        "B",
        [],
        None,
        0.50,
        LogSignal("X", 0.0, False)
    )

    # SCHEDULE_MAINTENANCE
    worn = synth_reading(HEAVY, seed=7, noise_scale=0.3)
    d_sch = decide("C", [], detect_spoof(worn, None, prior),
                   model.predict_proba(worn), hot_log)

    # EMERGENCY_STOP
    bad = synth_reading(HEAVY, seed=8, noise_scale=0.3)
    bad["J2_torque"] = 150.0
    d_emg = decide("D", check_hard_limits(bad), None,
                   model.predict_proba(bad), none_log)

    assert {d_safe.decision, d_mon.decision,
            d_sch.decision, d_emg.decision} == set(C.DECISIONS)