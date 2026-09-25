"""
Phase 3E.4C — Action-feasibility strategy experiment.

Purpose:
    Compare three torque-allocation strategies on the same driving cycle:

    A. Unconstrained random actions:
           sigma ~ Uniform(0, 1)

    B. Feasibility-aware random actions:
           sigma ~ Uniform(sigma_min, 1)

    C. Feasibility-aware midpoint:
           sigma = (sigma_min + 1) / 2

    The experiment is diagnostic only.

    It does NOT modify:
        - the RL environment
        - the vehicle model
        - the motor model
        - the battery model
        - the Actor
        - the replay buffer
        - the training algorithm
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
    Find the smallest feasible sigma in [0, 1].

    Uses the validated motor operating-point feasibility function.
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
        "No feasible sigma_tor found for a transition that was "
        "previously shown to be feasible."
    )


def evaluate_action(
    powertrain: IntegratedPowertrain,
    velocity_kmh: float,
    wheel_torque_nm: float,
    sigma: float,
) -> dict:
    return powertrain.motors.calculate_operating_point(
        velocity_kmh=velocity_kmh,
        wheel_torque_nm=wheel_torque_nm,
        sigma_tor=float(sigma),
    )


def main() -> None:
    print("=" * 78)
    print("PHASE 3E.4C — ACTION-FEASIBILITY STRATEGY EXPERIMENT")
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
    # TEST 2 — Prepare cycle
    # ------------------------------------------------------------------
    print("\nTEST 2 — Test cycle")

    cycle = build_test_cycle()

    print(f"Transitions: {len(cycle) - 1}")
    print("✓ Test cycle valid")

    # ------------------------------------------------------------------
    # TEST 3 — Determine feasible intervals
    # ------------------------------------------------------------------
    print("\nTEST 3 — Determine feasible sigma intervals")

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
            f"{velocity:5.1f} → {target_velocity:5.1f} km/h | "
            f"Tw={wheel_torque:10.3f} Nm | "
            f"sigma_min={sigma_min:.6f}"
        )

    print("✓ Feasible intervals determined")

    # ------------------------------------------------------------------
    # TEST 4 — Strategy A: unconstrained random
    # ------------------------------------------------------------------
    print("\nTEST 4 — Strategy A: unconstrained random actions")

    rng = np.random.default_rng(20260913)

    trials_per_transition = 1000

    strategy_a_violations = 0
    strategy_a_total = 0

    for data in transition_data:
        for _ in range(trials_per_transition):
            sigma = float(rng.uniform(0.0, 1.0))

            result = evaluate_action(
                powertrain=powertrain,
                velocity_kmh=data["velocity"],
                wheel_torque_nm=data["wheel_torque"],
                sigma=sigma,
            )

            strategy_a_total += 1

            if not result["overall_feasible"]:
                strategy_a_violations += 1

    strategy_a_rate = (
        strategy_a_violations / strategy_a_total
    )

    print(f"Trials: {strategy_a_total}")
    print(f"Violations: {strategy_a_violations}")
    print(f"Violation rate: {strategy_a_rate:.4%}")

    # ------------------------------------------------------------------
    # TEST 5 — Strategy B: feasibility-aware random
    # ------------------------------------------------------------------
    print("\nTEST 5 — Strategy B: feasibility-aware random actions")

    strategy_b_violations = 0
    strategy_b_total = 0

    for data in transition_data:
        sigma_min = data["sigma_min"]

        for _ in range(trials_per_transition):
            sigma = float(
                rng.uniform(
                    sigma_min,
                    1.0,
                )
            )

            result = evaluate_action(
                powertrain=powertrain,
                velocity_kmh=data["velocity"],
                wheel_torque_nm=data["wheel_torque"],
                sigma=sigma,
            )

            strategy_b_total += 1

            if not result["overall_feasible"]:
                strategy_b_violations += 1

    strategy_b_rate = (
        strategy_b_violations / strategy_b_total
    )

    print(f"Trials: {strategy_b_total}")
    print(f"Violations: {strategy_b_violations}")
    print(f"Violation rate: {strategy_b_rate:.4%}")

    # ------------------------------------------------------------------
    # TEST 6 — Strategy C: feasibility-aware midpoint
    # ------------------------------------------------------------------
    print("\nTEST 6 — Strategy C: feasibility-aware midpoint")

    strategy_c_violations = 0

    for data in transition_data:
        sigma_min = data["sigma_min"]

        sigma = (sigma_min + 1.0) / 2.0

        result = evaluate_action(
            powertrain=powertrain,
            velocity_kmh=data["velocity"],
            wheel_torque_nm=data["wheel_torque"],
            sigma=sigma,
        )

        print(
            f"Transition {data['index']:2d} | "
            f"sigma={sigma:.6f} | "
            f"feasible={result['overall_feasible']} | "
            f"M1={result['motor1_torque_nm']:.3f} Nm | "
            f"M2={result['motor2_torque_nm']:.3f} Nm"
        )

        if not result["overall_feasible"]:
            strategy_c_violations += 1

    print(f"Midpoint violations: {strategy_c_violations}")

    # ------------------------------------------------------------------
    # TEST 7 — Compare strategies
    # ------------------------------------------------------------------
    print("\nTEST 7 — Strategy comparison")

    print(
        f"Strategy A — unconstrained random: "
        f"{strategy_a_rate:.4%} violation rate"
    )

    print(
        f"Strategy B — feasible random:       "
        f"{strategy_b_rate:.4%} violation rate"
    )

    print(
        f"Strategy C — feasible midpoint:     "
        f"{strategy_c_violations}/"
        f"{len(transition_data)} violations"
    )

    # The feasibility-aware strategies should not produce violations
    # because their actions are deliberately sampled from the region
    # established by the plant's own feasibility function.
    assert strategy_b_violations == 0
    assert strategy_c_violations == 0

    assert strategy_a_rate >= 0.0
    assert strategy_a_rate <= 1.0

    print("✓ Strategy comparison completed")

    # ------------------------------------------------------------------
    # TEST 8 — Research interpretation
    # ------------------------------------------------------------------
    print("\nTEST 8 — Research interpretation")

    print(
        "\nThe experiment distinguishes:"
    )

    print(
        "  A. unconstrained policy exploration"
        "\n     → may select physically infeasible torque splits"
    )

    print(
        "\n  B. feasibility-aware exploration"
        "\n     → samples only from the physically feasible region"
    )

    print(
        "\n  C. feasibility-aware deterministic allocation"
        "\n     → selects the midpoint of the feasible region"
    )

    print(
        "\nNo strategy has been installed into the RL environment."
    )

    print("\n" + "=" * 78)
    print("✓ PHASE 3E.4C ACTION-FEASIBILITY EXPERIMENT COMPLETED")
    print("=" * 78)


if __name__ == "__main__":
    main()
    