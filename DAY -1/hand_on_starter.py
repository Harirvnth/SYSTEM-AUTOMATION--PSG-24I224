
"""
DAY 1: System Automation Rule Engine
Generates 500 sensor samples for Day 2 ML training.
"""

import csv
import random
import time
from pathlib import Path

# Shared project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = PROJECT_DIR / "sensor_data.csv"
LOG_FILE = Path(__file__).resolve().parent / "day1_log.txt"

NUM_SAMPLES = 500

# Priority: lower number means checked first.
RULES = [
    {
        "name": "EMERGENCY_TEMPERATURE",
        "priority": 1,
        "condition": lambda r: r["temperature"] > 90,
        "action": "EMERGENCY_SHUTDOWN",
        "state": "OVERHEATING",
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 2,
        "condition": lambda r: r["vibration"] > 10,
        "action": "STOP_MACHINE",
        "state": "HIGH_VIBRATION",
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 3,
        "condition": lambda r: r["pressure"] > 8,
        "action": "REDUCE_PRESSURE",
        "state": "HIGH_PRESSURE",
    },
    {
        "name": "HIGH_CURRENT",
        "priority": 4,
        "condition": lambda r: r["current"] > 25,
        "action": "CHECK_MOTOR",
        "state": "MOTOR_OVERLOAD",
    },
    {
        "name": "HIGH_TEMPERATURE",
        "priority": 5,
        "condition": lambda r: r["temperature"] > 80,
        "action": "COOL_MACHINE",
        "state": "OVERHEATING",
    },
    {
        "name": "LOW_PRESSURE",
        "priority": 6,
        "condition": lambda r: r["pressure"] < 3,
        "action": "CHECK_PRESSURE_SYSTEM",
        "state": "LOW_PRESSURE",
    },
]


def generate_sensor_data():
    """Generate one simulated sensor reading."""
    return {
        "temperature": round(random.uniform(50, 95), 2),
        "pressure": round(random.uniform(2, 10), 2),
        "vibration": round(random.uniform(0, 12), 2),
        "current": round(random.uniform(5, 30), 2),
        "rpm": random.randint(1400, 2200),
    }


def evaluate_rules(reading):
    """Return the first matching rule in priority order."""
    for rule in sorted(RULES, key=lambda r: r["priority"]):
        if rule["condition"](reading):
            return rule["name"], rule["action"], rule["state"]

    return "NO_RULE_FIRED", "CONTINUE_NORMAL", "NORMAL"


def take_action(action):
    """Describe the simulated automation action."""
    actions = {
        "EMERGENCY_SHUTDOWN":
            "Simulated emergency shutdown",
        "STOP_MACHINE":
            "Simulated machine stop for inspection",
        "REDUCE_PRESSURE":
            "Simulated pressure reduction",
        "CHECK_MOTOR":
            "Inspect motor load",
        "COOL_MACHINE":
            "Activate simulated cooling",
        "CHECK_PRESSURE_SYSTEM":
            "Inspect pressure system",
        "CONTINUE_NORMAL":
            "Continue normal operation",
    }
    return actions.get(action, "Unknown action")


def main():
    print("=== DAY 1: SENSOR DATA GENERATION ===")

    # Use a fixed seed for reproducible data.
    random.seed(42)

    fieldnames = [
        "temperature",
        "pressure",
        "vibration",
        "current",
        "rpm",
        "state",
    ]

    state_counts = {}

    with (
        open(CSV_FILE, "w", newline="", encoding="utf-8") as csv_file,
        open(LOG_FILE, "w", encoding="utf-8") as log_file,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for tick in range(1, NUM_SAMPLES + 1):
            reading = generate_sensor_data()

            rule_name, action, state = evaluate_rules(reading)
            outcome = take_action(action)

            # Save features and the rule-derived target label.
            writer.writerow({**reading, "state": state})

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(
                f"{timestamp} | Sample: {tick} | "
                f"Reading: {reading} | State: {state} | "
                f"Rule: {rule_name} | Action: {action}\n"
            )

            state_counts[state] = state_counts.get(state, 0) + 1

            if tick <= 10 or tick % 100 == 0:
                print(
                    f"[{tick:03d}/{NUM_SAMPLES}] "
                    f"State={state} | Rule={rule_name} | {outcome}"
                )

    print("\n=== DAY 1 COMPLETED ===")
    print(f"Dataset saved: {CSV_FILE}")
    print(f"Log saved: {LOG_FILE}")
    print(f"Total samples: {NUM_SAMPLES}")
    print("\nGenerated class distribution:")

    for state, count in sorted(state_counts.items()):
        print(f"{state}: {count}")


if __name__ == "__main__":
    main()
