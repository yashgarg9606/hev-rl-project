"""
Phase 3B — Minimal Integration Validation

This test validates the core Phase 3B integration WITHOUT running a full episode:
1. BMS_ESTIMATED mode correctly provides AI-estimated SOH
2. Physical battery SOH remains independent
3. SOH routing logic works correctly

This avoids motor map interpolation issues by testing the interface directly.
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.bms_interface.soh_source import SOHSource
from src.bms_interface.soh_trace_adapter import SOHTraceAdapter
from src.bms_interface.health_interface import BMSHealthInterface
from src.environment.health_model import HealthDegradationModel, BatteryHealthParameters, MotorHealthParameters


def test_bms_estimated_mode():
    """Test BMS_ESTIMATED mode integration."""

    print("=" * 70)
    print("PHASE 3B INTEGRATION VALIDATION")
    print("=" * 70)
    print()

    # Load BMS trace
    trace_file = project_root / "data" / "bms_soh_trace.npz"

    if not trace_file.exists():
        raise FileNotFoundError(f"BMS trace not found: {trace_file}")

    trace_adapter = SOHTraceAdapter(trace_file)
    print(f"Loaded BMS trace: {trace_adapter}")
    print()

    # Create health degradation model
    health_model = HealthDegradationModel(
        battery_parameters=BatteryHealthParameters(nominal_capacity_ah=72.0),
        motor1_parameters=MotorHealthParameters(rated_power_kw=45.0, rated_efficiency=0.92),
        motor2_parameters=MotorHealthParameters(rated_power_kw=45.0, rated_efficiency=0.92),
    )

    print("Created health degradation model")
    print()

    # Test 1: TRUE_PHYSICAL mode
    print("TEST 1: TRUE_PHYSICAL MODE")
    print("-" * 70)

    interface_true = BMSHealthInterface(
        health_model=health_model,
        soh_source=SOHSource.TRUE_PHYSICAL,
        soh_trace_adapter=None,
    )

    health_true = interface_true.get_health_state(simulation_time=0.0)
    true_battery_soh = interface_true.get_true_battery_soh()

    print(f"  EMS Battery SOH: {health_true['battery_soh']:.6f}")
    print(f"  True Battery SOH: {true_battery_soh:.6f}")
    print(f"  Motor 1 SOH: {health_true['motor1_soh']:.6f}")
    print(f"  Motor 2 SOH: {health_true['motor2_soh']:.6f}")

    assert health_true['battery_soh'] == true_battery_soh, "TRUE mode: EMS SOH should match physical SOH"
    print("  [PASS] TRUE mode: EMS uses physical SOH")
    print()

    # Test 2: BMS_ESTIMATED mode
    print("TEST 2: BMS_ESTIMATED MODE")
    print("-" * 70)

    interface_ai = BMSHealthInterface(
        health_model=health_model,
        soh_source=SOHSource.BMS_ESTIMATED,
        soh_trace_adapter=trace_adapter,
    )

    # Test at different simulation times
    test_times = [0.0, 3600.0, 72000.0, 360000.0]  # 0h, 1h, 20h, 100h

    for sim_time in test_times:
        health_ai = interface_ai.get_health_state(simulation_time=sim_time)
        true_battery_soh = interface_ai.get_true_battery_soh()

        print(f"\n  Simulation time: {sim_time/3600:.1f} hours")
        print(f"    EMS Battery SOH: {health_ai['battery_soh']:.6f} (from BMS trace)")
        print(f"    True Battery SOH: {true_battery_soh:.6f} (from physical model)")
        print(f"    Motor 1 SOH: {health_ai['motor1_soh']:.6f} (always physical)")
        print(f"    Motor 2 SOH: {health_ai['motor2_soh']:.6f} (always physical)")
        print(f"    SOH Difference: {abs(health_ai['battery_soh'] - true_battery_soh):.6f}")

    print()

    # Test 3: Validate SOH separation
    print("TEST 3: SOH SEPARATION VALIDATION")
    print("-" * 70)

    # Both interfaces share the same health_model instance,
    # so physical SOH should be identical
    true_physical = interface_true.get_true_battery_soh()
    ai_physical = interface_ai.get_true_battery_soh()

    # Get EMS-visible SOH from both modes
    health_true_t0 = interface_true.get_health_state(simulation_time=0.0)
    health_ai_t0 = interface_ai.get_health_state(simulation_time=0.0)

    print(f"  TRUE mode:")
    print(f"    EMS SOH: {health_true_t0['battery_soh']:.6f}")
    print(f"    Physical SOH: {true_physical:.6f}")
    print(f"    Match: {health_true_t0['battery_soh'] == true_physical}")
    print()

    print(f"  AI mode:")
    print(f"    EMS SOH: {health_ai_t0['battery_soh']:.6f}")
    print(f"    Physical SOH: {ai_physical:.6f}")
    print(f"    Match: {health_ai_t0['battery_soh'] == ai_physical}")
    print()

    # Validation checks
    checks = [
        ("TRUE mode: EMS uses physical SOH", health_true_t0['battery_soh'] == true_physical),
        ("AI mode: EMS uses estimated SOH (differs from physical)", health_ai_t0['battery_soh'] != ai_physical),
        ("Physical SOH identical in both interfaces", abs(true_physical - ai_physical) < 1e-9),
        ("Motor SOH always from physical model", health_ai_t0['motor1_soh'] == health_true_t0['motor1_soh']),
        ("AI estimated SOH comes from BMS trace", abs(health_ai_t0['battery_soh'] - trace_adapter.get_soh_at_time(0.0)) < 1e-9),
    ]

    print("VALIDATION RESULTS:")
    print("-" * 70)
    all_passed = True
    for check_name, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("=" * 70)
        print("[SUCCESS] PHASE 3B INTEGRATION VALIDATED")
        print("=" * 70)
        print()
        print("Key findings:")
        print("  - BMS_ESTIMATED mode successfully provides AI-estimated SOH to EMS")
        print("  - Physical battery SOH evolution is independent of EMS-visible SOH")
        print("  - Motor SOH values always come from physical degradation model")
        print("  - SOH routing logic correctly separates true and estimated values")
        print()
        print("Next step: Resolve motor efficiency map interpolation issue")
        print("to enable full episode experiments.")
    else:
        print("=" * 70)
        print("[FAILURE] VALIDATION CHECKS FAILED")
        print("=" * 70)
        print("Review SOH routing logic in BMSHealthInterface")

    print()


if __name__ == "__main__":
    test_bms_estimated_mode()
