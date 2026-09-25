"""
Phase 3E.4A — Motor operating-point feasibility audit.

Purpose:
    Determine which sigma_tor values are physically feasible for
    every driving-cycle transition.

This test does NOT modify the environment, motor model, or agent.

For each transition:

    wheel torque demand
            ↓
    sigma_tor sweep [0, 1]
            ↓
    motor 1 torque
    motor 2 torque
            ↓
    feasible sigma interval

The audit is used to distinguish:
    1. inherently infeasible wheel-torque demands, from
    2. feasible demands for which the Actor selects an infeasible
       torque split.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.environment.integrated_powertrain import (
    IntegratedPowertrain,
    IntegratedPowertrainParameters,
    default_motor_map_paths,
)


def build_test_cycle() -> np.ndarray:
    """
    Same deterministic validation cycle used in Phase 3E.

    Columns:
        [time_s, velocity_kmh, slope_rad]
    """

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


def calculate_wheel_torque(
    powertrain: IntegratedPowertrain,
    current_velocity_kmh: float,
    target_velocity_kmh: float,
    slope_rad: float,
    dt_s: float,
) -> float:
    """Calculate the wheel torque required for one cycle transition."""

    return float(
        powertrain.vehicle.calculate_wheel_torque(
            velocity_kmh=current_velocity_kmh,
            target_velocity_kmh=target_velocity_kmh,
            slope_rad=slope_rad,
            dt=dt_s,
        )
    )


def find_feasible_sigma_interval(
    powertrain: IntegratedPowertrain,
    velocity_kmh: float,
    wheel_torque_nm: float,
    sigma_points: int = 10001,
) -> tuple[float | None, float | None, int]:
    """
    Sweep sigma_tor over [0, 1] and identify feasible points.

    Returns:
        minimum feasible sigma,
        maximum feasible sigma,
        number of feasible grid points
    """

    sigmas = np.linspace(
        0.0,
        1.0,
        sigma_points,
    )

    feasible = []

    for sigma in sigmas:

        result = powertrain.motors.calculate_operating_point(
            velocity_kmh=velocity_kmh,
            wheel_torque_nm=wheel_torque_nm,
            sigma_tor=float(sigma),
        )

        if result["overall_feasible"]:
            feasible.append(
                float(sigma)
            )

    if not feasible:
        return None, None, 0

    return (
        min(feasible),
        max(feasible),
        len(feasible),
    )


def main() -> None:

    print("=" * 70)
    print("PHASE 3E.4A — MOTOR OPERATING-POINT FEASIBILITY AUDIT")
    print("=" * 70)

    # ---------------------------------------------------------------
    # TEST 1 — Construct powertrain
    # ---------------------------------------------------------------

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

    print("Project root:", project_root)
    print("Motor 1 map:", motor1_map_path)
    print("Motor 2 map:", motor2_map_path)
    print("✓ Integrated powertrain constructed")

    # ---------------------------------------------------------------
    # TEST 2 — Test-cycle definition
    # ---------------------------------------------------------------

    print("\nTEST 2 — Test cycle")

    cycle = build_test_cycle()

    assert cycle.ndim == 2
    assert cycle.shape[1] == 3
    assert len(cycle) >= 2

    print(
        "Transitions:",
        len(cycle) - 1,
    )

    print("✓ Test cycle valid")

    # ---------------------------------------------------------------
    # TEST 3 — Feasibility audit
    # ---------------------------------------------------------------

    print("\nTEST 3 — Sigma feasibility sweep")

    inherently_infeasible = 0
    feasible_transitions = 0

    feasibility_results = []

    for i in range(len(cycle) - 1):

        current_velocity = float(
            cycle[i, 1]
        )

        target_velocity = float(
            cycle[i + 1, 1]
        )

        slope_rad = float(
            cycle[i, 2]
        )

        dt_s = float(
            cycle[i + 1, 0] - cycle[i, 0]
        )

        wheel_torque = calculate_wheel_torque(
            powertrain=powertrain,
            current_velocity_kmh=current_velocity,
            target_velocity_kmh=target_velocity,
            slope_rad=slope_rad,
            dt_s=dt_s,
        )

        sigma_min, sigma_max, feasible_count = (
            find_feasible_sigma_interval(
                powertrain=powertrain,
                velocity_kmh=current_velocity,
                wheel_torque_nm=wheel_torque,
            )
        )

        if feasible_count == 0:
            status = "INFEASIBLE"
            inherently_infeasible += 1
        else:
            status = "FEASIBLE"
            feasible_transitions += 1

        feasibility_results.append(
            (
                i,
                current_velocity,
                target_velocity,
                wheel_torque,
                sigma_min,
                sigma_max,
                feasible_count,
                status,
            )
        )

        if sigma_min is None:
            interval_text = "none"
        else:
            interval_text = (
                f"{sigma_min:.4f} - {sigma_max:.4f}"
            )

        print(
            f"Transition {i:2d} | "
            f"{current_velocity:6.1f} → "
            f"{target_velocity:6.1f} km/h | "
            f"Tw={wheel_torque:10.3f} Nm | "
            f"feasible sigma={interval_text:15s} | "
            f"{status}"
        )

    # ---------------------------------------------------------------
    # TEST 4 — Feasibility accounting
    # ---------------------------------------------------------------

    print("\nTEST 4 — Feasibility accounting")

    total_transitions = len(cycle) - 1

    assert (
        feasible_transitions
        + inherently_infeasible
        == total_transitions
    )

    print(
        "Total transitions:",
        total_transitions,
    )

    print(
        "Feasible transitions:",
        feasible_transitions,
    )

    print(
        "Inherently infeasible transitions:",
        inherently_infeasible,
    )

    print("✓ Feasibility accounting is consistent")

    # ---------------------------------------------------------------
    # TEST 5 — Compare theoretical torque-capability bound
    # ---------------------------------------------------------------

    print("\nTEST 5 — Combined motor torque capability")

    max_total_torque = (
        180.0 + 155.0
    )

    print(
        f"Combined motor torque capability: "
        f"{max_total_torque:.3f} Nm"
    )

    for result in feasibility_results:

        (
            index,
            current_velocity,
            target_velocity,
            wheel_torque,
            sigma_min,
            sigma_max,
            feasible_count,
            status,
        ) = result

        if abs(wheel_torque) > max_total_torque:

            print(
                f"Transition {index:2d}: "
                f"|Tw|={abs(wheel_torque):.3f} Nm "
                f"> {max_total_torque:.3f} Nm "
                f"→ inherently infeasible"
            )

    # ---------------------------------------------------------------
    # TEST 6 — Explicit first-transition diagnosis
    # ---------------------------------------------------------------

    print("\nTEST 6 — First-transition diagnosis")

    first = feasibility_results[0]

    (
        index,
        current_velocity,
        target_velocity,
        wheel_torque,
        sigma_min,
        sigma_max,
        feasible_count,
        status,
    ) = first

    print(
        f"First transition: "
        f"{current_velocity:.1f} → "
        f"{target_velocity:.1f} km/h"
    )

    print(
        f"Wheel torque demand: "
        f"{wheel_torque:.6f} Nm"
    )

    print(
        f"Combined torque capability: "
        f"{max_total_torque:.6f} Nm"
    )

    if sigma_min is None:
        print(
            "Result: no sigma_tor in [0,1] satisfies "
            "both motor torque limits."
        )
    else:
        print(
            f"Feasible sigma interval: "
            f"{sigma_min:.6f} - {sigma_max:.6f}"
        )

    assert feasible_count >= 0

    print("✓ First-transition diagnosis completed")

    # ---------------------------------------------------------------
    # TEST 7 — Feasible operating point sanity check
    # ---------------------------------------------------------------

    print("\nTEST 7 — Feasible operating-point sanity check")

    # Find the first transition for which at least one feasible
    # sigma exists.
    feasible_example = None

    for result in feasibility_results:

        if result[6] > 0:
            feasible_example = result
            break

    if feasible_example is not None:

        (
            index,
            current_velocity,
            target_velocity,
            wheel_torque,
            sigma_min,
            sigma_max,
            feasible_count,
            status,
        ) = feasible_example

        midpoint_sigma = (
            sigma_min + sigma_max
        ) / 2.0

        operating_point = (
            powertrain.motors.calculate_operating_point(
                velocity_kmh=current_velocity,
                wheel_torque_nm=wheel_torque,
                sigma_tor=midpoint_sigma,
            )
        )

        assert operating_point["overall_feasible"]

        print(
            f"Example feasible transition: {index}"
        )

        print(
            f"Example sigma: "
            f"{midpoint_sigma:.6f}"
        )

        print(
            f"M1 torque: "
            f"{operating_point['motor1_torque_nm']:.6f} Nm"
        )

        print(
            f"M2 torque: "
            f"{operating_point['motor2_torque_nm']:.6f} Nm"
        )

        print("✓ Feasible operating point verified")

    else:

        print(
            "No feasible transition exists in the "
            "test cycle."
        )

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("✓ PHASE 3E.4A FEASIBILITY AUDIT COMPLETED")
    print("=" * 70)

    print(
        "\nIMPORTANT: This audit is diagnostic only. "
        "No project physics or environment behavior was modified."
    )


if __name__ == "__main__":
    main()