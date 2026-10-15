# Robotic Arm Health Monitoring

An end-to-end decision pipeline for a fleet of 6-axis robotic arms. It turns
raw joint telemetry plus free-text technician notes into exactly one action
per arm: `SAFE`, `MONITOR`, `SCHEDULE_MAINTENANCE` or `EMERGENCY_STOP`.

## Quick start

```bash
git clone https://github.com/<your-username>/robot-arm-health-monitor.git
cd robot-arm-health-monitor
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_demo.py
pytest -q
```

## How the five requirements are met

| # | Requirement | Where | How |
|---|-------------|-------|-----|
| 1 | Immediate safety stop | `src/safety_check.py` | Pure function, hard thresholds, zero ML imports |
| 2 | Predictive wear detection | `src/wear_model.py`, `src/dataset_gen.py` | RandomForest on a self-generated 240-row labelled dataset |
| 3 | Maintenance notes as 2nd signal | `src/log_nlp.py` | Weighted keyword scorer with a 3-token negation window |
| 4 | Spoofed-sensor defence | `src/spoof_detect.py` | 4 independent checks incl. cross-signal masking |
| 5 | One final decision per arm | `src/decision.py` | 3-layer priority fusion |

See [EXPLANATION.md](EXPLANATION.md) for the design rationale.

## Repo layout

```
src/        pipeline modules
data/       generated dataset + hand-written maintenance log
tests/      pytest suite
run_demo.py demonstrates all four decisions
```