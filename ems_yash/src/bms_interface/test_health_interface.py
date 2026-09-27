"""
Phase 3A — BMS-EMS Interface Validation Test

Validates that the BMSHealthInterface correctly:
1. Returns true physical SOH when source is TRUE_PHYSICAL
2. Passes through motor SOH values unchanged
3. Provides access to true battery SOH via get_true_battery_soh()
4. Requires a trace adapter for BMS_ESTIMATED
5. Does not create a second physical battery SOH state
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.bms_interface import BMSHealthInterface, SOHSource
from src.environment.health_model import (
    HealthDegradationModel,
    BatteryHealthParameters,
    MotorHealthParameters,
)


def test_interface_initialization():
    """Test that the interface initializes correctly."""

    print("TEST 1 — Interface Initialization")

    health_model = HealthDegradationModel(
        battery_parameters=BatteryHealthParameters(),
        motor1_parameters=MotorHealthParameters(
            rated_power_kw=100.0,
            rated_efficiency=0.95,
        ),
        motor2_parameters=MotorHealthParameters(
            rated_power_kw=100.0,
            rated_efficiency=0.95,
        ),
    )

    interface = BMSHealthInterface(
        health_model=health_model,
        soh_source=SOHSource.TRUE_PHYSICAL,
    )

    assert interface.health_model is health_model
    assert interface.soh_source == SOHSource.TRUE_PHYSICAL

    print("[PASS] Interface initialized with TRUE_PHYSICAL source")


def test_true_physical_source():
    """Test that TRUE_PHYSICAL returns physical SOH values."""

    print("\nTEST 2 — TRUE_PHYSICAL Source")

    health_model = HealthDegradationModel(
        battery_parameters=BatteryHealthParameters(),
        motor1_parameters=MotorHealthParameters(
            rated_power_kw=100.0,
            rated_efficiency=0.95,
        ),
        motor2_parameters=MotorHealthParameters(
            rated_power_kw=100.0,
            rated_efficiency=0.95,
        ),
    )

    interface = BMSHealthInterface(
        health_model=health_model,
        soh_source=SOHSource.TRUE_PHYSICAL,
    )

    # Get health state from both interface and model
    interface_state = interface.get_health_state()
    model_state = health_model.get_health_state()

    # They should be identical
    assert interface_state["battery_soh"] == model_state["battery_soh"]
    assert interface_state["motor1_soh"] == model_state["motor1_soh"]
    assert interface_state["motor2_soh"] == model_state["motor2_soh"]

    print(f"Interface battery SOH: {interface_state['battery_soh']:.12f}")
    print(f"Model battery SOH:     {model_state['battery_soh']:.12f}")
    print("[PASS] Interface returns identical SOH to physical model")


def test_get_true_battery_soh():
    """Test that get_true_battery_soh() always returns physical SOH."""

    print("\nTEST 3 — get_true_battery_soh() Method")

    health_model = HealthDegradationModel(
        battery_parameters=BatteryHealthParameters(),
        motor1_parameters=MotorHealthParameters(
            rated_power_kw=100.0,
            rated_efficiency=0.95,
        ),
        motor2_parameters=MotorHealthParameters(
            rated_power_kw=100.0,
            rated_efficiency=0.95,
        ),
    )

    interface = BMSHealthInterface(
        health_model=health_model,
        soh_source=SOHSource.TRUE_PHYSICAL,
    )

    true_soh = interface.get_true_battery_soh()
    model_soh = health_model.get_health_state()["battery_soh"]

    assert true_soh == model_soh

    print(f"True battery SOH: {true_soh:.12f}")
    print("[PASS] get_true_battery_soh() returns physical model SOH")


def test_get_true_motor_soh():
    """Test that get_true_motor_soh() returns physical motor SOH."""

    print("\nTEST 4 — get_true_motor_soh() Method")

    health_model = HealthDegradationModel(
        battery_parameters=BatteryHealthParameters(),
        motor1_parameters=MotorHealthParameters(
            rated_power_kw=100.0,
            rated_efficiency=0.95,
        ),
        motor2_parameters=MotorHealthParameters(
            rated_power_kw=100.0,
            rated_efficiency=0.95,
        ),
    )

    interface = BMSHealthInterface(
        health_model=health_model,
        soh_source=SOHSource.TRUE_PHYSICAL,
    )

    motor1_soh, motor2_soh = interface.get_true_motor_soh()
    model_state = health_model.get_health_state()

    assert motor1_soh == model_state["motor1_soh"]
    assert motor2_soh == model_state["motor2_soh"]

    print(f"Motor 1 SOH: {motor1_soh:.12f}")
    print(f"Motor 2 SOH: {motor2_soh:.12f}")
    print("[PASS] get_true_motor_soh() returns physical model motor SOH")


def test_bms_estimated_requires_adapter():
    """Test that BMS_ESTIMATED rejects a missing trace adapter."""

    print("\nTEST 5 — BMS_ESTIMATED Requires Adapter")

    health_model = HealthDegradationModel(
        battery_parameters=BatteryHealthParameters(),
        motor1_parameters=MotorHealthParameters(
            rated_power_kw=100.0,
            rated_efficiency=0.95,
        ),
        motor2_parameters=MotorHealthParameters(
            rated_power_kw=100.0,
            rated_efficiency=0.95,
        ),
    )

    try:
        BMSHealthInterface(
            health_model=health_model,
            soh_source=SOHSource.BMS_ESTIMATED,
        )
    except ValueError as e:
        assert "requires soh_trace_adapter" in str(e)
        print(f"Raised: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("[PASS] BMS_ESTIMATED correctly rejects a missing adapter")
    else:
        raise AssertionError("Should have raised ValueError for a missing adapter")


def test_no_duplicate_battery_state():
    """Test that interface does not create duplicate battery SOH state."""

    print("\nTEST 6 — No Duplicate Battery State")

    health_model = HealthDegradationModel(
        battery_parameters=BatteryHealthParameters(),
        motor1_parameters=MotorHealthParameters(
            rated_power_kw=100.0,
            rated_efficiency=0.95,
        ),
        motor2_parameters=MotorHealthParameters(
            rated_power_kw=100.0,
            rated_efficiency=0.95,
        ),
    )

    interface = BMSHealthInterface(
        health_model=health_model,
        soh_source=SOHSource.TRUE_PHYSICAL,
    )

    # Interface should not have its own battery_state attribute
    assert not hasattr(interface, "battery_state")
    assert not hasattr(interface, "battery_soh")

    # Interface only has reference to health_model
    assert hasattr(interface, "health_model")
    assert interface.health_model is health_model

    print("[PASS] Interface does not create duplicate battery state")
    print("[PASS] Interface only references the physical health_model")


def test_soh_after_degradation():
    """Test that interface reflects degradation in the physical model."""

    print("\nTEST 7 — SOH After Degradation")

    health_model = HealthDegradationModel(
        battery_parameters=BatteryHealthParameters(
            nominal_capacity_ah=72.0,
        ),
        motor1_parameters=MotorHealthParameters(
            rated_power_kw=100.0,
            rated_efficiency=0.95,
        ),
        motor2_parameters=MotorHealthParameters(
            rated_power_kw=100.0,
            rated_efficiency=0.95,
        ),
    )

    interface = BMSHealthInterface(
        health_model=health_model,
        soh_source=SOHSource.TRUE_PHYSICAL,
    )

    initial_soh = interface.get_health_state()["battery_soh"]
    print(f"Initial battery SOH: {initial_soh:.12f}")

    # Simulate battery degradation through discharge
    health_model.update_battery(
        current_a=-50.0,  # 50A discharge
        temperature_c=25.0,
        dt_s=3600.0,  # 1 hour
    )

    degraded_soh = interface.get_health_state()["battery_soh"]
    print(f"After 1h discharge:  {degraded_soh:.12f}")

    assert degraded_soh < initial_soh, "SOH should decrease after discharge"

    # Verify it matches the physical model
    model_soh = health_model.get_health_state()["battery_soh"]
    assert degraded_soh == model_soh

    print("[PASS] Interface correctly reflects physical battery degradation")


def main():

    print("=" * 70)
    print("PHASE 3A — BMS-EMS INTERFACE VALIDATION")
    print("=" * 70)

    test_interface_initialization()
    test_true_physical_source()
    test_get_true_battery_soh()
    test_get_true_motor_soh()
    test_bms_estimated_requires_adapter()
    test_no_duplicate_battery_state()
    test_soh_after_degradation()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
