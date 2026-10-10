
"""
DAY 1 · HANDS-ON — Build Your Rule Engine
23IF02 System Automation

Steps:
1. Generate simulated sensor data
2. Define automation rules
3. Evaluate rules by priority
4. Trigger actions
5. Log every decision
6. Add a second sensor (current)
7. Add a custom rule
8. Add a priority rule
"""

import random
import time

LOG_FILE = "day1_log.txt"


# ---------------------------------------------------------------------------
# STEP 1 & 6 — Generate simulated sensor data
# ---------------------------------------------------------------------------
def generate_sensor_data():
    """Return simulated sensor readings."""
    return {
        "temperature": round(random.uniform(50, 95), 2),  # Degree Celsius
        "pressure": round(random.uniform(2, 10), 2),      # Bar
        "vibration": round(random.uniform(0, 12), 2),     # mm/s
        "current": round(random.uniform(5, 30), 2),      # Amperes
    }


# ---------------------------------------------------------------------------
# STEP 2, 7 & 8 — Define automation rules
# Lower priority number = checked first
# ---------------------------------------------------------------------------
RULES = [
    {
        "name": "EMERGENCY_TEMPERATURE",
        "priority": 1,
        "condition": lambda r: r["temperature"] > 90,
        "action": "EMERGENCY_SHUTDOWN",
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 2,
        "condition": lambda r: r["vibration"] > 10,
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_PRESSURE",
        "priority": 3,
        "condition": lambda r: r["pressure"] > 8,
        "action": "REDUCE_PRESSURE",
    },
    {
        "name": "HIGH_CURRENT",
        "priority": 4,
        "condition": lambda r: r["current"] > 25,
        "action": "CHECK_MOTOR",
    },
    {
        "name": "HIGH_TEMPERATURE",
        "priority": 5,
        "condition": lambda r: r["temperature"] > 80,
        "action": "COOL_MACHINE",
    },
    # STEP 7: Custom rule
    {
        "name": "LOW_PRESSURE",
        "priority": 6,
        "condition": lambda r: r["pressure"] < 3,
        "action": "CHECK_PRESSURE_SYSTEM",
    },
]


# ---------------------------------------------------------------------------
# STEP 3 — Evaluate rules in priority order
# ---------------------------------------------------------------------------
def evaluate_rules(reading, rules):
    """
    Return (rule_name, action) for the first matching rule.
    Lowest priority number is checked first.
    """
    sorted_rules = sorted(rules, key=lambda rule: rule["priority"])

    for rule in sorted_rules:
        try:
            if rule["condition"](reading):
                return rule["name"], rule["action"]
        except (KeyError, TypeError):
            # Ignore rules whose required sensor data is missing or invalid.
            continue

    return "NO_RULE_FIRED", "CONTINUE_NORMAL"

def check_sensor_status(temperature):
    if temperature > 80:
        return "WARNING: High temperature"
    return "Temperature normal"
# ---------------------------------------------------------------------------
# STEP 4 — Trigger an action
# ---------------------------------------------------------------------------
def take_action(action):
    """Return a human-readable response for an action."""
    actions = {
        "EMERGENCY_SHUTDOWN": "EMERGENCY! Shut down the machine immediately.",
        "STOP_MACHINE": "Stop the machine for safety inspection.",
        "REDUCE_PRESSURE": "Activate pressure reduction procedure.",
        "CHECK_MOTOR": "Inspect motor current and electrical load.",
        "COOL_MACHINE": "Activate the cooling system.",
        "CHECK_PRESSURE_SYSTEM": "Inspect the pressure system for leaks.",
        "CONTINUE_NORMAL": "All conditions normal. Continue operation.",
    }

    return actions.get(action, f"No handler available for action: {action}")


# ---------------------------------------------------------------------------
# STEP 5 — Log every decision
# ---------------------------------------------------------------------------
def log_decision(reading, rule_name, action):
    """Append each sensor reading and decision to the log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(
            f"{timestamp} | "
            f"Reading: {reading} | "
            f"Rule: {rule_name} | "
            f"Action: {action}\n"
        )


# ---------------------------------------------------------------------------
# MAIN — Run the automation engine
# ---------------------------------------------------------------------------
def main():
    print("=== SYSTEM AUTOMATION: RULE ENGINE ===\n")

    for tick in range(1, 11):
        reading = generate_sensor_data()

        rule_name, action = evaluate_rules(reading, RULES)
        outcome = take_action(action)

        print(f"[Tick {tick:02d}] SENSE -> {reading}")
        print(f"           THINK -> {rule_name}")
        print(f"           ACT   -> {outcome}\n")

        log_decision(reading, rule_name, action)

        time.sleep(0.2)

    print(f"Execution completed. Decisions saved in '{LOG_FILE}'.")


if __name__ == "__main__":
    main()
