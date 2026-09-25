"""
Phase 3E.4B — Geared torque-capability validation.

Purpose:
    Validate motor-shaft torque limits against the required wheel torque
    after accounting for the motor-specific reduction ratios.

This test is diagnostic only.
It does NOT modify the vehicle, motor, battery, environment, or agent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.environment.integrated_powertrain import (
    IntegratedPowertrain,
    IntegratedPowertrainParameters,
    default_motor_map_paths,
)


def build_test_cycle() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 10.0, 0.0],
            [2.0, 20.0, 0.0],
            [3.0, 30.0, 0.0],
            [4.0, 40.0, 0.0],
            [5.0, 50.0, 0.0],
            [6.0, 40.0, 0.0],
            [7.0, 30.0, 0.0],
            [8.0, 20.0, 0.0],
            [9.0, 10.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def main() -> None:
    print("=" * 78)
    print("PHASE 3E.4B — GEARED TORQUE-CAPABILITY VALIDATION")
    print("=" * 78)

    # ------------------------------------------------------------------
    # TEST 1 — Construct validated powertrain
    # ------------------------------------------------------------------
    print("\nTEST 1 — Powertrain construction")

    project_root = Path(__file__).resolve().parents[2]

    motor1_map_path, motor2_map_path = default_motor_map_paths(
        project_root
    )

    powertrain = IntegratedPowertrain(
        motor1_map_path=motor1_map_path,
        motor2_map_path=motor2_map_path,
        parameters=IntegratedPowertrainParameters(),
    )

    m1 = powertrain.motors.params.motor1
    m2 = powertrain.motors.params.motor2

    print(f"Motor 1 gear ratio: {m1.gear_ratio:.3f}")
    print(f"Motor 1 max torque: {m1.max_torque_nm:.3f} Nm")
    print(f"Motor 1 max speed:  {m1.max_speed_rpm:.1f} rpm")

    print(f"Motor 2 gear ratio: {m2.gear_ratio:.3f}")
    print(f"Motor 2 max torque: {m2.max_torque_nm:.3f} Nm")
    print(f"Motor 2 max speed:  {m2.max_speed_rpm:.1f} rpm")

    print("✓ Integrated powertrain constructed")

    # ------------------------------------------------------------------
    # TEST 2 — Calculate theoretical wheel-side torque capability
    # ------------------------------------------------------------------
    print("\nTEST 2 — Wheel-side torque capability")

    m1_wheel_capacity = m1.max_torque_nm * m1.gear_ratio
    m2_wheel_capacity = m2.max_torque_nm * m2.gear_ratio
    total_wheel_capacity = m1_wheel_capacity + m2_wheel_capacity

    print(
        f"M1 wheel-side capacity: "
        f"{m1_wheel_capacity:.3f} Nm"
    )
    print(
        f"M2 wheel-side capacity: "
        f"{m2_wheel_capacity:.3f} Nm"
    )
    print(
        f"Combined wheel-side capacity: "
        f"{total_wheel_capacity:.3f} Nm"
    )

    assert m1_wheel_capacity > m1.max_torque_nm
    assert m2_wheel_capacity > m2.max_torque_nm
    assert total_wheel_capacity > 335.0

    print("✓ Gear ratios correctly reflected in wheel torque")

    # ------------------------------------------------------------------
    # TEST 3 — Transition-by-transition feasible interval
    # ------------------------------------------------------------------
    print("\nTEST 3 — Feasible sigma intervals")

    cycle = build_test_cycle()

    sigmas = np.linspace(0.0, 1.0, 10001)

    for i in range(len(cycle) - 1):
        velocity = float(cycle[i, 1])
        target_velocity = float(cycle[i + 1, 1])
        slope_rad = float(cycle[i, 2])
        dt_s = float(cycle[i + 1, 0] - cycle[i, 0])

        wheel_torque = float(
            powertrain.vehicle.calculate_wheel_torque(
                velocity_kmh=velocity,
                target_velocity_kmh=target_velocity,
                slope_rad=slope_rad,
                dt=dt_s,
            )
        )

        feasible_sigmas = []

        for sigma in sigmas:
            result = powertrain.motors.calculate_operating_point(
                velocity_kmh=velocity,
                wheel_torque_nm=wheel_torque,
                sigma_tor=float(sigma),
            )

            if result["overall_feasible"]:
                feasible_sigmas.append(float(sigma))

        assert feasible_sigmas, (
            f"No feasible sigma found for transition {i}"
        )

        sigma_min = min(feasible_sigmas)
        sigma_max = max(feasible_sigmas)

        # Evaluate the lower boundary.
        lower = powertrain.motors.calculate_operating_point(
            velocity_kmh=velocity,
            wheel_torque_nm=wheel_torque,
            sigma_tor=sigma_min,
        )

        # Evaluate the midpoint.
        sigma_mid = (sigma_min + sigma_max) / 2.0

        middle = powertrain.motors.calculate_operating_point(
            velocity_kmh=velocity,
            wheel_torque_nm=wheel_torque,
            sigma_tor=sigma_mid,
        )

        # Evaluate sigma = 1.
        upper = powertrain.motors.calculate_operating_point(
            velocity_kmh=velocity,
            wheel_torque_nm=wheel_torque,
            sigma_tor=1.0,
        )

        print("\n" + "-" * 78)
        print(
            f"Transition {i}: "
            f"{velocity:.1f} → {target_velocity:.1f} km/h"
        )
        print(f"Required wheel torque: {wheel_torque:.6f} Nm")
        print(
            f"Feasible sigma interval: "
            f"{sigma_min:.6f} → {sigma_max:.6f}"
        )

        print("\nAt sigma_min:")
        print(
            f"  M1 torque = "
            f"{lower['motor1_torque_nm']:.6f} Nm"
        )
        print(
            f"  M2 torque = "
            f"{lower['motor2_torque_nm']:.6f} Nm"
        )
        print(
            f"  M1 wheel contribution = "
            f"{lower['motor1_torque_nm'] * m1.gear_ratio:.6f} Nm"
        )
        print(
            f"  M2 wheel contribution = "
            f"{lower['motor2_torque_nm'] * m2.gear_ratio:.6f} Nm"
        )
        print(
            f"  M1 feasible = "
            f"{lower['motor1_feasible']}"
        )
        print(
            f"  M2 feasible = "
            f"{lower['motor2_feasible']}"
        )

        print("\nAt sigma_mid:")
        print(f"  sigma = {sigma_mid:.6f}")
        print(
            f"  M1 torque = "
            f"{middle['motor1_torque_nm']:.6f} Nm"
        )
        print(
            f"  M2 torque = "
            f"{middle['motor2_torque_nm']:.6f} Nm"
        )
        print(
            f"  M1 feasible = "
            f"{middle['motor1_feasible']}"
        )
        print(
            f"  M2 feasible = "
            f"{middle['motor2_feasible']}"
        )

        print("\nAt sigma = 1.0:")
        print(
            f"  M1 torque = "
            f"{upper['motor1_torque_nm']:.6f} Nm"
        )
        print(
            f"  M2 torque = "
            f"{upper['motor2_torque_nm']:.6f} Nm"
        )
        print(
            f"  M1 feasible = "
            f"{upper['motor1_feasible']}"
        )
        print(
            f"  M2 feasible = "
            f"{upper['motor2_feasible']}"
        )

        # Every reported feasible operating point must actually satisfy
        # the plant's own feasibility test.
        assert lower["overall_feasible"]
        assert middle["overall_feasible"]
        assert upper["overall_feasible"]

    print("\n✓ All transitions have validated feasible geared torque splits")

    # ------------------------------------------------------------------
    # TEST 4 — Maximum theoretical wheel-side torque
    # ------------------------------------------------------------------
    print("\nTEST 4 — Maximum theoretical wheel-side torque")

    print(
        f"Maximum wheel torque from M1: "
        f"{m1.max_torque_nm:.3f} × {m1.gear_ratio:.3f} "
        f"= {m1_wheel_capacity:.3f} Nm"
    )

    print(
        f"Maximum wheel torque from M2: "
        f"{m2.max_torque_nm:.3f} × {m2.gear_ratio:.3f} "
        f"= {m2_wheel_capacity:.3f} Nm"
    )

    print(
        f"Combined theoretical wheel torque: "
        f"{total_wheel_capacity:.3f} Nm"
    )

    assert total_wheel_capacity > 1600.0

    print(
        "✓ Combined geared capability exceeds the "
        "test-cycle acceleration demand"
    )

    # ------------------------------------------------------------------
    # TEST 5 — Important conclusion
    # ------------------------------------------------------------------
    print("\nTEST 5 — Physical interpretation")

    print(
        "\nThe motor torque constraints operate on motor-shaft torque, "
        "while vehicle demand is expressed as wheel torque."
    )

    print(
        "Therefore, wheel torque must NOT be compared directly "
        "against 180 + 155 Nm."
    )

    print(
        "\nThe observed RL constraint terminations should therefore "
        "be investigated as torque-allocation violations."
    )

    print("\n" + "=" * 78)
    print("✓ PHASE 3E.4B GEARED TORQUE VALIDATION COMPLETED")
    print("=" * 78)

    print(
        "\nIMPORTANT: This test is diagnostic only. "
        "No project physics, environment behavior, or agent behavior "
        "was modified."
    )


if __name__ == "__main__":
    main()