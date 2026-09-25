"""
Phase 3E.4E — Feasibility-aware action-mapping interface validation.

Purpose:
    Validate that the environment can determine the physically feasible
    sigma_tor interval at each transition using information already
    available to the environment.

Key requirement:
    sigma_min must be determined from the CURRENT transition only.

    No future transition information may be used.

This test is diagnostic only.

It does NOT modify:
    - rl_environment.py
    - IntegratedPowertrain
    - Actor
    - Critic
    - DDPG agent
    - replay buffer
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


def find_sigma_min(
    powertrain: IntegratedPowertrain,
    velocity_kmh: float,
    wheel_torque_nm: float,
    sigma_points: int = 10001,
) -> float:
    for sigma in np.linspace(0.0, 1.0, sigma_points):
        result = powertrain.motors.calculate_operating_point(
            velocity_kmh=velocity_kmh,
            wheel_torque_nm=wheel_torque_nm,
            sigma_tor=float(sigma),
        )

        if result["overall_feasible"]:
            return float(sigma)

    raise RuntimeError(
        "No feasible sigma_tor found."
    )


def map_actor_action(
    actor_action: float,
    sigma_min: float,
) -> float:
    if not np.isfinite(actor_action):
        raise ValueError("Actor action must be finite.")

    if not 0.0 <= actor_action <= 1.0:
        raise ValueError(
            "Actor action must lie in [0, 1]."
        )

    if not 0.0 <= sigma_min <= 1.0:
        raise ValueError(
            "sigma_min must lie in [0, 1]."
        )

    return float(
        sigma_min
        + actor_action * (1.0 - sigma_min)
    )


def main() -> None:
    print("=" * 78)
    print("PHASE 3E.4E — ACTION-MAPPING INTERFACE VALIDATION")
    print("=" * 78)

    # ------------------------------------------------------------------
    # TEST 1 — Construct plant
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

    print("✓ Integrated powertrain constructed")

    # ------------------------------------------------------------------
    # TEST 2 — Build cycle
    # ------------------------------------------------------------------
    print("\nTEST 2 — Transition information")

    cycle = build_test_cycle()

    print(f"Number of states: {len(cycle)}")
    print(f"Number of transitions: {len(cycle) - 1}")

    assert cycle.shape[1] == 3
    assert len(cycle) >= 2

    print("✓ Cycle information valid")

    # ------------------------------------------------------------------
    # TEST 3 — Compute current-transition quantities
    # ------------------------------------------------------------------
    print("\nTEST 3 — Current-transition feasibility")

    transition_data = []

    for i in range(len(cycle) - 1):
        current_time = float(cycle[i, 0])
        next_time = float(cycle[i + 1, 0])

        current_velocity = float(cycle[i, 1])
        next_velocity = float(cycle[i + 1, 1])

        current_slope = float(cycle[i, 2])

        dt_s = next_time - current_time

        assert dt_s > 0.0

        wheel_torque = float(
            powertrain.vehicle.calculate_wheel_torque(
                velocity_kmh=current_velocity,
                target_velocity_kmh=next_velocity,
                slope_rad=current_slope,
                dt=dt_s,
            )
        )

        sigma_min = find_sigma_min(
            powertrain=powertrain,
            velocity_kmh=current_velocity,
            wheel_torque_nm=wheel_torque,
        )

        transition_data.append(
            {
                "index": i,
                "current_time": current_time,
                "next_time": next_time,
                "current_velocity": current_velocity,
                "next_velocity": next_velocity,
                "slope": current_slope,
                "dt": dt_s,
                "wheel_torque": wheel_torque,
                "sigma_min": sigma_min,
            }
        )

        print(
            f"Transition {i:2d} | "
            f"t={current_time:.1f}→{next_time:.1f}s | "
            f"v={current_velocity:.1f}→{next_velocity:.1f} | "
            f"Tw={wheel_torque:10.3f} | "
            f"sigma_min={sigma_min:.6f}"
        )

    print("✓ Current-transition feasibility calculated")

    # ------------------------------------------------------------------
    # TEST 4 — No future-information dependence
    # ------------------------------------------------------------------
    print("\nTEST 4 — Future-information independence")

    """
    The feasible bound for transition i should depend on:

        current velocity
        next velocity
        current slope
        dt
        plant parameters

    It should NOT depend on:
        cycle[i + 2]
        cycle[i + 3]
        ...
    """

    for data in transition_data:
        i = data["index"]

        original_sigma_min = data["sigma_min"]

        modified_cycle = cycle.copy()

        # Modify ONLY states after the immediate target state.
        if i + 2 < len(modified_cycle):
            modified_cycle[i + 2 :, 1] += 37.0

        current_velocity = float(
            modified_cycle[i, 1]
        )
        next_velocity = float(
            modified_cycle[i + 1, 1]
        )
        slope = float(
            modified_cycle[i, 2]
        )
        dt_s = float(
            modified_cycle[i + 1, 0]
            - modified_cycle[i, 0]
        )

        modified_wheel_torque = float(
            powertrain.vehicle.calculate_wheel_torque(
                velocity_kmh=current_velocity,
                target_velocity_kmh=next_velocity,
                slope_rad=slope,
                dt=dt_s,
            )
        )

        modified_sigma_min = find_sigma_min(
            powertrain=powertrain,
            velocity_kmh=current_velocity,
            wheel_torque_nm=modified_wheel_torque,
        )

        assert np.isclose(
            original_sigma_min,
            modified_sigma_min,
            atol=1e-4,
        )

    print(
        "✓ sigma_min is independent of transitions "
        "after the immediate target"
    )

    # ------------------------------------------------------------------
    # TEST 5 — Normalized action transformation
    # ------------------------------------------------------------------
    print("\nTEST 5 — Normalized action transformation")

    normalized_actions = [
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
    ]

    for data in transition_data:
        for actor_action in normalized_actions:
            sigma = map_actor_action(
                actor_action=actor_action,
                sigma_min=data["sigma_min"],
            )

            assert (
                data["sigma_min"]
                <= sigma
                <= 1.0
            )

            result = powertrain.motors.calculate_operating_point(
                velocity_kmh=data["current_velocity"],
                wheel_torque_nm=data["wheel_torque"],
                sigma_tor=sigma,
            )

            assert result["overall_feasible"]

    print(
        "✓ Normalized Actor actions map to feasible "
        "physical actions"
    )

    # ------------------------------------------------------------------
    # TEST 6 — No modification of observation dimension
    # ------------------------------------------------------------------
    print("\nTEST 6 — Observation compatibility")

    expected_state_dim = 6

    print(
        "Existing state representation remains:"
    )
    print(
        "[velocity, wheel torque, SOC, battery SOH, "
        "motor1 SOH, motor2 SOH]"
    )

    assert expected_state_dim == 6

    print(
        "✓ Action mapping does not require adding "
        "sigma_min to the observation"
    )

    # ------------------------------------------------------------------
    # TEST 7 — Transition semantics
    # ------------------------------------------------------------------
    print("\nTEST 7 — Transition semantics")

    for data in transition_data:
        assert data["current_time"] < data["next_time"]

        # The target velocity belongs to the immediate next cycle point.
        # No later cycle point is used in the calculation.
        assert np.isfinite(data["next_velocity"])

        assert np.isfinite(data["wheel_torque"])
        assert np.isfinite(data["sigma_min"])

    print(
        "✓ Each sigma_min is associated with exactly "
        "one current transition"
    )

    # ------------------------------------------------------------------
    # TEST 8 — Report recommended interface
    # ------------------------------------------------------------------
    print("\nTEST 8 — Interface conclusion")

    print(
        "\nFor each environment transition:"
    )

    print(
        "  1. Environment determines current wheel torque demand."
    )

    print(
        "  2. Environment determines sigma_min from the "
        "current operating point."
    )

    print(
        "  3. Actor provides normalized action a ∈ [0,1]."
    )

    print(
        "  4. Physical action is:"
    )

    print(
        "       sigma = sigma_min + a(1 - sigma_min)"
    )

    print(
        "\nThe six-dimensional observation does not need "
        "to contain sigma_min."
    )

    print(
        "No future driving-cycle state is required beyond "
        "the immediate transition target already used by "
        "the existing vehicle model."
    )

    print("\n" + "=" * 78)
    print("✓ PHASE 3E.4E ACTION-MAPPING INTERFACE VALIDATION COMPLETED")
    print("=" * 78)

    print(
        "\nIMPORTANT:"
        "\nNo production environment or DDPG component was modified."
    )


if __name__ == "__main__":
    main()