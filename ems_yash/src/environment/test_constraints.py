"""
Phase 3D.6 — Physical constraint validation.
"""

from __future__ import annotations

from src.environment.constraints import (
    ConstraintParameters,
    PhysicalConstraintChecker,
)


def main() -> None:

    print("=" * 70)
    print("PHASE 3D.6 — PHYSICAL CONSTRAINT VALIDATION")
    print("=" * 70)

    checker = PhysicalConstraintChecker()

    p = checker.parameters

    # ---------------------------------------------------------------
    # TEST 1 — Paper constraint parameters
    # ---------------------------------------------------------------

    print("\nTEST 1 — Constraint parameters")

    print(f"SOC:       {p.soc_min} - {p.soc_max}")
    print(
        f"Battery SOH: "
        f"{p.battery_soh_min} - {p.battery_soh_max}"
    )
    print(
        f"Motor 1 SOH: "
        f"{p.motor1_soh_min} - {p.motor1_soh_max}"
    )
    print(
        f"Motor 2 SOH: "
        f"{p.motor2_soh_min} - {p.motor2_soh_max}"
    )
    print(
        f"Motor 1 speed: "
        f"{p.motor1_max_speed_rpm} rpm"
    )
    print(
        f"Motor 2 speed: "
        f"{p.motor2_max_speed_rpm} rpm"
    )
    print(
        f"Motor 1 torque: "
        f"{p.motor1_max_torque_nm} Nm"
    )
    print(
        f"Motor 2 torque: "
        f"{p.motor2_max_torque_nm} Nm"
    )

    assert p.soc_min == 0.2
    assert p.soc_max == 0.9

    assert p.battery_soh_min == 0.8
    assert p.battery_soh_max == 1.0

    assert p.motor1_soh_min == 0.8
    assert p.motor1_soh_max == 1.0

    assert p.motor2_soh_min == 0.8
    assert p.motor2_soh_max == 1.0

    assert p.motor1_max_speed_rpm == 12000.0
    assert p.motor2_max_speed_rpm == 12000.0

    assert p.motor1_max_torque_nm == 180.0
    assert p.motor2_max_torque_nm == 155.0

    print("✓ Constraint parameters match the paper")

    # ---------------------------------------------------------------
    # TEST 2 — Valid operating point
    # ---------------------------------------------------------------

    print("\nTEST 2 — Valid operating point")

    result = checker.check(
        soc=0.60,
        battery_soh=1.0,
        motor1_soh=1.0,
        motor2_soh=1.0,
        motor1_speed_rpm=5000.0,
        motor2_speed_rpm=3000.0,
        motor1_torque_nm=100.0,
        motor2_torque_nm=100.0,
    )

    assert result.overall_valid
    assert result.violated_constraints == ()

    print("✓ Valid operating point accepted")

    # ---------------------------------------------------------------
    # TEST 3 — Boundary values
    # ---------------------------------------------------------------

    print("\nTEST 3 — Exact boundary values")

    result = checker.check(
        soc=0.2,
        battery_soh=0.8,
        motor1_soh=0.8,
        motor2_soh=0.8,
        motor1_speed_rpm=12000.0,
        motor2_speed_rpm=-12000.0,
        motor1_torque_nm=-180.0,
        motor2_torque_nm=155.0,
    )

    assert result.overall_valid
    assert result.violated_constraints == ()

    print("✓ Exact constraint boundaries accepted")

    # ---------------------------------------------------------------
    # TEST 4 — SOC violations
    # ---------------------------------------------------------------

    print("\nTEST 4 — SOC violations")

    result = checker.check(
        soc=0.19,
        battery_soh=1.0,
        motor1_soh=1.0,
        motor2_soh=1.0,
        motor1_speed_rpm=5000.0,
        motor2_speed_rpm=3000.0,
        motor1_torque_nm=100.0,
        motor2_torque_nm=100.0,
    )

    assert not result.overall_valid
    assert "soc" in result.violated_constraints

    result = checker.check(
        soc=0.91,
        battery_soh=1.0,
        motor1_soh=1.0,
        motor2_soh=1.0,
        motor1_speed_rpm=5000.0,
        motor2_speed_rpm=3000.0,
        motor1_torque_nm=100.0,
        motor2_torque_nm=100.0,
    )

    assert not result.overall_valid
    assert "soc" in result.violated_constraints

    print("✓ SOC lower/upper violations detected")

    # ---------------------------------------------------------------
    # TEST 5 — SOH violations
    # ---------------------------------------------------------------

    print("\nTEST 5 — SOH violations")

    result = checker.check(
        soc=0.60,
        battery_soh=0.79,
        motor1_soh=1.0,
        motor2_soh=1.0,
        motor1_speed_rpm=5000.0,
        motor2_speed_rpm=3000.0,
        motor1_torque_nm=100.0,
        motor2_torque_nm=100.0,
    )

    assert "battery_soh" in result.violated_constraints

    result = checker.check(
        soc=0.60,
        battery_soh=1.0,
        motor1_soh=0.79,
        motor2_soh=0.79,
        motor1_speed_rpm=5000.0,
        motor2_speed_rpm=3000.0,
        motor1_torque_nm=100.0,
        motor2_torque_nm=100.0,
    )

    assert "motor1_soh" in result.violated_constraints
    assert "motor2_soh" in result.violated_constraints

    print("✓ Battery and motor SOH violations detected")

    # ---------------------------------------------------------------
    # TEST 6 — Motor speed violations
    # ---------------------------------------------------------------

    print("\nTEST 6 — Motor speed violations")

    result = checker.check(
        soc=0.60,
        battery_soh=1.0,
        motor1_soh=1.0,
        motor2_soh=1.0,
        motor1_speed_rpm=12000.1,
        motor2_speed_rpm=3000.0,
        motor1_torque_nm=100.0,
        motor2_torque_nm=100.0,
    )

    assert "motor1_speed" in result.violated_constraints

    result = checker.check(
        soc=0.60,
        battery_soh=1.0,
        motor1_soh=1.0,
        motor2_soh=1.0,
        motor1_speed_rpm=5000.0,
        motor2_speed_rpm=-12000.1,
        motor1_torque_nm=100.0,
        motor2_torque_nm=100.0,
    )

    assert "motor2_speed" in result.violated_constraints

    print("✓ Motor speed violations detected")

    # ---------------------------------------------------------------
    # TEST 7 — Motor torque violations
    # ---------------------------------------------------------------

    print("\nTEST 7 — Motor torque violations")

    result = checker.check(
        soc=0.60,
        battery_soh=1.0,
        motor1_soh=1.0,
        motor2_soh=1.0,
        motor1_speed_rpm=5000.0,
        motor2_speed_rpm=3000.0,
        motor1_torque_nm=180.1,
        motor2_torque_nm=100.0,
    )

    assert "motor1_torque" in result.violated_constraints

    result = checker.check(
        soc=0.60,
        battery_soh=1.0,
        motor1_soh=1.0,
        motor2_soh=1.0,
        motor1_speed_rpm=5000.0,
        motor2_speed_rpm=3000.0,
        motor1_torque_nm=100.0,
        motor2_torque_nm=-155.1,
    )

    assert "motor2_torque" in result.violated_constraints

    print("✓ Motor torque violations detected")

    # ---------------------------------------------------------------
    # TEST 8 — Multiple simultaneous violations
    # ---------------------------------------------------------------

    print("\nTEST 8 — Multiple violations")

    result = checker.check(
        soc=0.1,
        battery_soh=0.7,
        motor1_soh=0.7,
        motor2_soh=0.7,
        motor1_speed_rpm=13000.0,
        motor2_speed_rpm=13000.0,
        motor1_torque_nm=200.0,
        motor2_torque_nm=170.0,
    )

    assert not result.overall_valid

    expected_violations = {
        "soc",
        "battery_soh",
        "motor1_soh",
        "motor2_soh",
        "motor1_speed",
        "motor2_speed",
        "motor1_torque",
        "motor2_torque",
    }

    assert set(result.violated_constraints) == expected_violations

    print(
        "Violations detected:",
        result.violated_constraints,
    )

    print("✓ Multiple simultaneous violations detected")

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("✓ PHASE 3D.6 PHYSICAL CONSTRAINT VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()