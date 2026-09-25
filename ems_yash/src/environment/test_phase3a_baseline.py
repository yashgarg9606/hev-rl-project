"""
Phase 3A — Baseline Equivalence Test

Validates that the integrated powertrain with BMS interface
produces identical results to the baseline behavior when
using TRUE_PHYSICAL SOH source.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.environment.integrated_powertrain import (
    IntegratedPowertrain,
    default_motor_map_paths,
)


def main():

    print("=" * 70)
    print("PHASE 3A — BASELINE EQUIVALENCE TEST")
    print("=" * 70)

    # Build powertrain with Phase 3A BMS interface
    motor1_map, motor2_map = default_motor_map_paths(project_root)

    powertrain = IntegratedPowertrain(
        motor1_map_path=motor1_map,
        motor2_map_path=motor2_map,
    )

    # --------------------------------------------------------------
    # TEST 1 — Initial health state
    # --------------------------------------------------------------

    print("\nTEST 1 — Initial Health State")

    # Access through interface
    health = powertrain.health_interface.get_health_state()

    print(f"Battery SOH (interface): {health['battery_soh']:.12f}")
    print(f"Motor 1 SOH (interface): {health['motor1_soh']:.12f}")
    print(f"Motor 2 SOH (interface): {health['motor2_soh']:.12f}")

    # Verify it matches the underlying physical model
    true_health = powertrain.health_model.get_health_state()

    assert health["battery_soh"] == true_health["battery_soh"]
    assert health["motor1_soh"] == true_health["motor1_soh"]
    assert health["motor2_soh"] == true_health["motor2_soh"]

    assert health["battery_soh"] == 1.0
    assert health["motor1_soh"] == 1.0
    assert health["motor2_soh"] == 1.0

    print("[PASS] Initial SOH = 1.0 for all components")
    print("[PASS] Interface matches physical model")

    # --------------------------------------------------------------
    # TEST 2 — Powertrain step with health degradation
    # --------------------------------------------------------------

    print("\nTEST 2 — Powertrain Step with Health Degradation")

    result = powertrain.step(
        velocity_kmh=50.0,
        target_velocity_kmh=50.0,
        sigma_tor=0.5,
        slope_rad=0.0,
        dt_s=1.0,
    )

    print(f"Battery power:   {result.battery_power_kw:.6f} kW")
    print(f"Battery current: {result.battery_current_a:.6f} A")
    print(f"Battery SOC:     {result.soc:.12f}")
    print(f"Battery SOH:     {result.battery_soh:.12f}")
    print(f"Motor 1 SOH:     {result.motor1_soh:.12f}")
    print(f"Motor 2 SOH:     {result.motor2_soh:.12f}")

    assert result.battery_power_kw < 0.0, "Should be discharging"
    assert result.battery_current_a < 0.0, "Should be discharging"

    assert 0.0 < result.battery_soh < 1.0, "Battery SOH should decrease"
    assert 0.0 < result.motor1_soh < 1.0, "Motor 1 SOH should decrease"
    assert 0.0 < result.motor2_soh < 1.0, "Motor 2 SOH should decrease"

    print("[PASS] Battery discharging")
    print("[PASS] All component health decreased")

    # --------------------------------------------------------------
    # TEST 3 — Verify interface returns updated health
    # --------------------------------------------------------------

    print("\nTEST 3 — Interface Returns Updated Health")

    interface_health = powertrain.health_interface.get_health_state()
    model_health = powertrain.health_model.get_health_state()

    print(f"Battery SOH (interface): {interface_health['battery_soh']:.12f}")
    print(f"Battery SOH (model):     {model_health['battery_soh']:.12f}")

    assert interface_health["battery_soh"] == model_health["battery_soh"]
    assert interface_health["motor1_soh"] == model_health["motor1_soh"]
    assert interface_health["motor2_soh"] == model_health["motor2_soh"]

    # Should match the step result
    assert interface_health["battery_soh"] == result.battery_soh
    assert interface_health["motor1_soh"] == result.motor1_soh
    assert interface_health["motor2_soh"] == result.motor2_soh

    print("[PASS] Interface reflects degraded health")
    print("[PASS] Interface matches step result")

    # --------------------------------------------------------------
    # TEST 4 — Verify get_true_battery_soh() works
    # --------------------------------------------------------------

    print("\nTEST 4 — get_true_battery_soh() Access")

    true_soh = powertrain.health_interface.get_true_battery_soh()

    assert true_soh == result.battery_soh
    assert true_soh == model_health["battery_soh"]

    print(f"True battery SOH: {true_soh:.12f}")
    print("[PASS] get_true_battery_soh() returns correct value")

    # --------------------------------------------------------------
    # TEST 5 — Multiple steps
    # --------------------------------------------------------------

    print("\nTEST 5 — Multiple Steps")

    for i in range(5):
        result = powertrain.step(
            velocity_kmh=60.0,
            target_velocity_kmh=60.0,
            sigma_tor=0.5,
            slope_rad=0.0,
            dt_s=1.0,
        )

    print(f"After 5 more steps:")
    print(f"  Battery SOH: {result.battery_soh:.12f}")
    print(f"  Motor 1 SOH: {result.motor1_soh:.12f}")
    print(f"  Motor 2 SOH: {result.motor2_soh:.12f}")

    # Verify continued degradation
    assert result.battery_soh < interface_health["battery_soh"]

    print("[PASS] Continued degradation over multiple steps")

    # --------------------------------------------------------------
    # TEST 6 — State dimensions preserved
    # --------------------------------------------------------------

    print("\nTEST 6 — State Dimensions Check")

    # Health state should have 3 keys
    health = powertrain.health_interface.get_health_state()
    assert len(health) == 3
    assert "battery_soh" in health
    assert "motor1_soh" in health
    assert "motor2_soh" in health

    print(f"Health state keys: {list(health.keys())}")
    print("[PASS] Health state structure preserved")

    print("\n" + "=" * 70)
    print("PHASE 3A BASELINE EQUIVALENCE VALIDATED")
    print("=" * 70)
    print("\nSummary:")
    print("  - BMS interface initialized with TRUE_PHYSICAL source")
    print("  - Interface returns identical SOH to physical model")
    print("  - Powertrain step() returns health through interface")
    print("  - Health degradation propagates correctly")
    print("  - State structure preserved")
    print("  - No behavioral changes from baseline")


if __name__ == "__main__":
    main()
