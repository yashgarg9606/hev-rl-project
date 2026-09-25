"""
Phase 3E.4D — Feasibility-aware action mapping validation.

Purpose:
    Validate the transformation:

        actor_action in [0, 1]
                    ↓
        sigma_min + actor_action * (1 - sigma_min)
                    ↓
        physically feasible sigma_tor

    This preserves a continuous normalized action space for the Actor
    while mapping it into the physically feasible torque-allocation region.

This test is diagnostic only.

It does NOT modify:
    - the RL environment
    - the vehicle model
    - the motor model
    - the Actor
    - the Critic
    - the replay buffer
    - the DDPG agent
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


def calculate_wheel_torque(
    powertrain: IntegratedPowertrain,
    velocity_kmh: float,
    target_velocity_kmh: float,
    slope_rad: float,
    dt_s: float,
) -> float:
    return float(
        powertrain.vehicle.calculate_wheel_torque(
            velocity_kmh=velocity_kmh,
            target_velocity_kmh=target_velocity_kmh,
            slope_rad=slope_rad,
            dt=dt_s,
        )
    )


def find_sigma_min(
    powertrain: IntegratedPowertrain,
    velocity_kmh: float,
    wheel_torque_nm: float,
    sigma_points: int = 10001,
) -> float:
    """
    Determine the smallest feasible sigma_tor.

    The search uses the validated motor operating-point feasibility
    calculation rather than independently reproducing motor physics.
    """

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
    """
    Map normalized Actor action [0, 1] to feasible sigma_tor.

        sigma = sigma_min + a * (1 - sigma_min)

    This is an affine transformation.
    """

    if not np.isfinite(actor_action):
        raise ValueError("Actor action must be finite.")

    if not 0.0 <= actor_action <= 1.0:
        raise ValueError(
            "Actor action must lie within [0, 1]."
        )

    if not 0.0 <= sigma_min <= 1.0:
        raise ValueError(
            "sigma_min must lie within [0, 1]."
        )

    return float(
        sigma_min
        + actor_action * (1.0 - sigma_min)
    )


def main() -> None:
    print("=" * 78)
    print("PHASE 3E.4D — FEASIBILITY-AWARE ACTION MAPPING VALIDATION")
    print("=" * 78)

    # ------------------------------------------------------------------
    # TEST 1 — Construct powertrain
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
    # TEST 2 — Calculate feasible regions
    # ------------------------------------------------------------------
    print("\nTEST 2 — Calculate feasible action regions")

    cycle = build_test_cycle()

    transition_data = []

    for i in range(len(cycle) - 1):
        velocity = float(cycle[i, 1])
        target_velocity = float(cycle[i + 1, 1])
        slope_rad = float(cycle[i, 2])
        dt_s = float(cycle[i + 1, 0] - cycle[i, 0])

        wheel_torque = calculate_wheel_torque(
            powertrain=powertrain,
            velocity_kmh=velocity,
            target_velocity_kmh=target_velocity,
            slope_rad=slope_rad,
            dt_s=dt_s,
        )

        sigma_min = find_sigma_min(
            powertrain=powertrain,
            velocity_kmh=velocity,
            wheel_torque_nm=wheel_torque,
        )

        transition_data.append(
            {
                "index": i,
                "velocity": velocity,
                "target_velocity": target_velocity,
                "wheel_torque": wheel_torque,
                "sigma_min": sigma_min,
            }
        )

        print(
            f"Transition {i:2d} | "
            f"Tw={wheel_torque:10.3f} Nm | "
            f"sigma_min={sigma_min:.6f}"
        )

    print("✓ Feasible regions calculated")

    # ------------------------------------------------------------------
    # TEST 3 — Affine mapping boundary validation
    # ------------------------------------------------------------------
    print("\nTEST 3 — Mapping boundary validation")

    for data in transition_data:
        sigma_min = data["sigma_min"]

        mapped_zero = map_actor_action(
            actor_action=0.0,
            sigma_min=sigma_min,
        )

        mapped_one = map_actor_action(
            actor_action=1.0,
            sigma_min=sigma_min,
        )

        assert np.isclose(
            mapped_zero,
            sigma_min,
            atol=1e-12,
        )

        assert np.isclose(
            mapped_one,
            1.0,
            atol=1e-12,
        )

    print(
        "Actor action 0 → sigma_min"
    )
    print(
        "Actor action 1 → sigma = 1"
    )
    print("✓ Mapping boundaries are correct")

    # ------------------------------------------------------------------
    # TEST 4 — Monotonicity and continuity
    # ------------------------------------------------------------------
    print("\nTEST 4 — Mapping monotonicity")

    actor_actions = np.linspace(0.0, 1.0, 101)

    for data in transition_data:
        sigma_min = data["sigma_min"]

        mapped_actions = np.array(
            [
                map_actor_action(
                    actor_action=float(action),
                    sigma_min=sigma_min,
                )
                for action in actor_actions
            ]
        )

        differences = np.diff(mapped_actions)

        assert np.all(differences >= 0.0)

        assert np.all(mapped_actions >= sigma_min)
        assert np.all(mapped_actions <= 1.0)

    print(
        "✓ Mapping is monotonic and remains within "
        "the feasible interval"
    )

    # ------------------------------------------------------------------
    # TEST 5 — Physical feasibility of mapped actions
    # ------------------------------------------------------------------
    print("\nTEST 5 — Physical feasibility")

    rng = np.random.default_rng(20260913)

    total_trials = 0
    violations = 0

    for data in transition_data:
        sigma_min = data["sigma_min"]

        for _ in range(1000):
            actor_action = float(
                rng.uniform(0.0, 1.0)
            )

            sigma = map_actor_action(
                actor_action=actor_action,
                sigma_min=sigma_min,
            )

            result = powertrain.motors.calculate_operating_point(
                velocity_kmh=data["velocity"],
                wheel_torque_nm=data["wheel_torque"],
                sigma_tor=sigma,
            )

            total_trials += 1

            if not result["overall_feasible"]:
                violations += 1

    print(f"Trials: {total_trials}")
    print(f"Violations: {violations}")

    assert violations == 0

    print(
        "✓ Every mapped action produced a "
        "physically feasible motor operating point"
    )

    # ------------------------------------------------------------------
    # TEST 6 — Example mapping
    # ------------------------------------------------------------------
    print("\nTEST 6 — Example action transformation")

    example = transition_data[0]
    sigma_min = example["sigma_min"]

    print(
        f"Transition 0 sigma_min = "
        f"{sigma_min:.6f}"
    )

    for actor_action in [0.0, 0.25, 0.5, 0.75, 1.0]:
        sigma = map_actor_action(
            actor_action=actor_action,
            sigma_min=sigma_min,
        )

        result = powertrain.motors.calculate_operating_point(
            velocity_kmh=example["velocity"],
            wheel_torque_nm=example["wheel_torque"],
            sigma_tor=sigma,
        )

        print(
            f"Actor action={actor_action:.2f} "
            f"→ sigma={sigma:.6f} "
            f"→ feasible={result['overall_feasible']}"
        )

        assert result["overall_feasible"]

    print("✓ Example mapping verified")

    # ------------------------------------------------------------------
    # TEST 7 — Differentiability
    # ------------------------------------------------------------------
    print("\nTEST 7 — Mapping derivative")

    for data in transition_data:
        sigma_min = data["sigma_min"]

        numerical_delta = 1e-6

        action_a = 0.4
        action_b = action_a + numerical_delta

        sigma_a = map_actor_action(
            actor_action=action_a,
            sigma_min=sigma_min,
        )

        sigma_b = map_actor_action(
            actor_action=action_b,
            sigma_min=sigma_min,
        )

        numerical_derivative = (
            sigma_b - sigma_a
        ) / numerical_delta

        expected_derivative = (
            1.0 - sigma_min
        )

        assert np.isclose(
            numerical_derivative,
            expected_derivative,
            rtol=1e-5,
            atol=1e-8,
        )

    print(
        "✓ Mapping is continuous and has a "
        "constant positive derivative"
    )

    # ------------------------------------------------------------------
    # TEST 8 — No direct clipping
    # ------------------------------------------------------------------
    print("\nTEST 8 — Clipping distinction")

    sigma_min = transition_data[0]["sigma_min"]

    actor_values = np.array(
        [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    )

    mapped_values = np.array(
        [
            map_actor_action(
                actor_action=float(a),
                sigma_min=sigma_min,
            )
            for a in actor_values
        ]
    )

    clipped_values = np.clip(
        actor_values,
        sigma_min,
        1.0,
    )

    print("Actor actions:")
    print(actor_values)

    print("Feasibility mapping:")
    print(mapped_values)

    print("Direct clipping:")
    print(clipped_values)

    # The two mechanisms should differ for at least some actions
    # whenever sigma_min > 0.
    assert not np.allclose(
        mapped_values,
        clipped_values,
    )

    print(
        "✓ Feasibility mapping is distinct from "
        "post-action clipping"
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("✓ PHASE 3E.4D ACTION MAPPING VALIDATION COMPLETED")
    print("=" * 78)

    print(
        "\nIMPORTANT:"
        "\nThe mapping has only been validated diagnostically."
        "\nNo production RL component has been modified."
    )


if __name__ == "__main__":
    main()