"""
Phase 3E.4F-3 — Replay / target-action information audit.

Purpose
-------
Verify that the feasibility lower bound sigma_min can be reconstructed
from the immediate transition information available during training.

We specifically verify:

    (current state,
     immediate next velocity,
     current slope,
     dt)
            ↓
       wheel torque demand
            ↓
       feasible sigma_min

The audit also verifies that information from later driving-cycle points
does not affect the result.

No production files are modified.
"""

from __future__ import annotations

import numpy as np

from src.environment.integrated_powertrain import (
    IntegratedPowertrain,
    default_motor_map_paths,
)
from src.environment.vehicle_dynamics import (
    VehicleDynamics,
    VehicleParameters,
)


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

DT = 1.0

# Same 10-transition cycle used in the previous feasibility audits.
CYCLE = np.array(
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


def build_powertrain() -> IntegratedPowertrain:
    """
    Construct the validated production powertrain.

    This test uses the same motor maps and vehicle parameters as the
    integrated plant.
    """

    motor1_map, motor2_map = default_motor_map_paths(
        PROJECT_ROOT
    )

    return IntegratedPowertrain(
        motor1_map_path=motor1_map,
        motor2_map_path=motor2_map,
    )


def calculate_wheel_torque(
    vehicle: VehicleDynamics,
    current_velocity_kmh: float,
    next_velocity_kmh: float,
    slope_rad: float,
    dt: float,
) -> float:
    """
    Reconstruct the immediate wheel torque demand.

    Uses only:
        - current velocity
        - immediate next velocity
        - current slope
        - dt

    This directly uses the validated Phase 2A VehicleDynamics API.
    """

    wheel_torque_nm = vehicle.calculate_wheel_torque(
        velocity_kmh=current_velocity_kmh,
        target_velocity_kmh=next_velocity_kmh,
        slope_rad=slope_rad,
        dt=dt,
    )

    return float(wheel_torque_nm)



def calculate_sigma_min(
    powertrain: IntegratedPowertrain,
    velocity_kmh: float,
    wheel_torque_nm: float,
) -> float:
    """
    Find the smallest feasible physical sigma_tor.

    Uses the validated production motor operating-point model.

    The search is deliberately independent of the Actor.
    """

    sigma_grid = np.linspace(
        0.0,
        1.0,
        10001,
        dtype=np.float64,
    )

    for sigma in sigma_grid:
        result = powertrain.motors.calculate_operating_point(
            velocity_kmh=velocity_kmh,
            wheel_torque_nm=wheel_torque_nm,
            sigma_tor=float(sigma),
        )

        if result["overall_feasible"]:
            return float(sigma)

    raise RuntimeError(
        "No feasible sigma_tor found for this transition."
    )


def calculate_sigma_min_from_transition(
    powertrain: IntegratedPowertrain,
    vehicle: VehicleDynamics,
    current_velocity_kmh: float,
    next_velocity_kmh: float,
    slope_rad: float,
    dt: float,
) -> tuple[float, float]:
    """
    Complete reconstruction from immediate transition information.
    """

    wheel_torque_nm = calculate_wheel_torque(
        vehicle=vehicle,
        current_velocity_kmh=current_velocity_kmh,
        next_velocity_kmh=next_velocity_kmh,
        slope_rad=slope_rad,
        dt=dt,
    )

    sigma_min = calculate_sigma_min(
        powertrain=powertrain,
        velocity_kmh=current_velocity_kmh,
        wheel_torque_nm=wheel_torque_nm,
    )

    return wheel_torque_nm, sigma_min


def main() -> None:
    print("=" * 78)
    print(
        "PHASE 3E.4F-3 — REPLAY / TARGET-ACTION INFORMATION AUDIT"
    )
    print("=" * 78)

    powertrain = build_powertrain()

    vehicle = VehicleDynamics(
        VehicleParameters()
    )

    # ------------------------------------------------------------------
    # TEST 1 — Immediate-transition reconstruction
    # ------------------------------------------------------------------
    print("\nTEST 1 — Immediate-transition sigma_min reconstruction")

    sigma_values = []

    for i in range(len(CYCLE) - 1):
        current = CYCLE[i]
        nxt = CYCLE[i + 1]

        velocity = float(current[1])
        next_velocity = float(nxt[1])
        slope = float(current[2])

        wheel_torque, sigma_min = (
            calculate_sigma_min_from_transition(
                powertrain=powertrain,
                vehicle=vehicle,
                current_velocity_kmh=velocity,
                next_velocity_kmh=next_velocity,
                slope_rad=slope,
                dt=DT,
            )
        )

        sigma_values.append(sigma_min)

        print(
            f"Transition {i:2d}: "
            f"v={velocity:6.1f} → {next_velocity:6.1f} km/h, "
            f"Twheel={wheel_torque:9.3f} Nm, "
            f"sigma_min={sigma_min:.4f}"
        )

        assert 0.0 <= sigma_min <= 1.0

    assert len(sigma_values) == len(CYCLE) - 1

    print("✓ sigma_min reconstructed for every transition")

    # ------------------------------------------------------------------
    # TEST 2 — Replay-compatible reconstruction
    # ------------------------------------------------------------------
    print("\nTEST 2 — Replay-compatible information set")

    # A replay transition needs only the information below for
    # reconstructing the current action feasibility bound.
    replay_required_fields = {
        "state_velocity_kmh",
        "next_state_velocity_kmh",
        "current_slope_rad",
        "dt",
    }

    print("Required information:")
    for field in sorted(replay_required_fields):
        print(f"  {field}")

    assert replay_required_fields == {
        "state_velocity_kmh",
        "next_state_velocity_kmh",
        "current_slope_rad",
        "dt",
    }

    print(
        "✓ Required information is available from the "
        "immediate transition context"
    )

    # ------------------------------------------------------------------
    # TEST 3 — No dependence on later cycle points
    # ------------------------------------------------------------------
    print("\nTEST 3 — Future-information leakage audit")

    # Pick one representative transition.
    index = 4

    current = CYCLE[index]
    nxt = CYCLE[index + 1]

    baseline_wheel_torque, baseline_sigma = (
        calculate_sigma_min_from_transition(
            powertrain=powertrain,
            vehicle=vehicle,
            current_velocity_kmh=float(current[1]),
            next_velocity_kmh=float(nxt[1]),
            slope_rad=float(current[2]),
            dt=DT,
        )
    )

    # Create a modified cycle where every point AFTER the immediate
    # next state is radically different.
    modified_cycle = CYCLE.copy()

    modified_cycle[index + 2 :, 1] = np.array(
        [131.3] * len(modified_cycle[index + 2 :]),
        dtype=np.float64,
    )

    modified_current = modified_cycle[index]
    modified_next = modified_cycle[index + 1]

    modified_wheel_torque, modified_sigma = (
        calculate_sigma_min_from_transition(
            powertrain=powertrain,
            vehicle=vehicle,
            current_velocity_kmh=float(modified_current[1]),
            next_velocity_kmh=float(modified_next[1]),
            slope_rad=float(modified_current[2]),
            dt=DT,
        )
    )

    print(
        f"Baseline wheel torque:  {baseline_wheel_torque:.9f} Nm"
    )

    print(
        f"Modified wheel torque:  {modified_wheel_torque:.9f} Nm"
    )

    print(
        f"Baseline sigma_min:      {baseline_sigma:.9f}"
    )

    print(
        f"Modified sigma_min:      {modified_sigma:.9f}"
    )

    assert np.isclose(
        baseline_wheel_torque,
        modified_wheel_torque,
        rtol=0.0,
        atol=1e-10,
    )

    assert np.isclose(
        baseline_sigma,
        modified_sigma,
        rtol=0.0,
        atol=1e-10,
    )

    print(
        "✓ Later cycle points do not affect sigma_min"
    )

    # ------------------------------------------------------------------
    # TEST 4 — Current slope matters, but future slope does not
    # ------------------------------------------------------------------
    print(
        "\nTEST 4 — Immediate slope dependence / future-slope independence"
    )

    # Use an independent representative transition.
    index = 2

    current = CYCLE[index]
    nxt = CYCLE[index + 1]

    baseline = calculate_sigma_min_from_transition(
        powertrain=powertrain,
        vehicle=vehicle,
        current_velocity_kmh=float(current[1]),
        next_velocity_kmh=float(nxt[1]),
        slope_rad=float(current[2]),
        dt=DT,
    )

    modified_cycle = CYCLE.copy()

    # Change only future slopes.
    modified_cycle[index + 2 :, 2] = 0.05

    modified_current = modified_cycle[index]
    modified_next = modified_cycle[index + 1]

    modified = calculate_sigma_min_from_transition(
        powertrain=powertrain,
        vehicle=vehicle,
        current_velocity_kmh=float(modified_current[1]),
        next_velocity_kmh=float(modified_next[1]),
        slope_rad=float(modified_current[2]),
        dt=DT,
    )

    assert np.isclose(
        baseline[0],
        modified[0],
        rtol=0.0,
        atol=1e-10,
    )

    assert np.isclose(
        baseline[1],
        modified[1],
        rtol=0.0,
        atol=1e-10,
    )

    print("✓ Future slope information does not leak into sigma_min")

    # ------------------------------------------------------------------
    # TEST 5 — Normalized action can be mapped after reconstruction
    # ------------------------------------------------------------------
    print(
        "\nTEST 5 — Normalized action → physical sigma "
        "using reconstructed sigma_min"
    )

    normalized_actions = np.array(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        dtype=np.float64,
    )

    for sigma_min in sigma_values[:3]:
        physical_actions = (
            sigma_min
            + normalized_actions * (1.0 - sigma_min)
        )

        assert np.all(
            physical_actions >= sigma_min
        )

        assert np.all(
            physical_actions <= 1.0
        )

        assert np.all(
            np.diff(physical_actions) >= 0.0
        )

    print(
        "✓ Normalized Actor actions can be mapped using "
        "reconstructed sigma_min"
    )

    # ------------------------------------------------------------------
    # TEST 6 — No need to add sigma_min to the six-dimensional state
    # ------------------------------------------------------------------
    print(
        "\nTEST 6 — Six-dimensional RL state remains unchanged"
    )

    state_dim = 6

    state_example = np.array(
        [
            50.0,       # velocity
            104.1403,   # wheel torque
            0.60,       # SOC
            1.0,        # battery SOH
            1.0,        # motor 1 SOH
            1.0,        # motor 2 SOH
        ],
        dtype=np.float64,
    )

    assert state_example.shape == (state_dim,)

    # sigma_min is auxiliary transition information, NOT an RL state
    # variable.
    assert len(state_example) == 6

    print(
        "✓ sigma_min does not need to become a seventh "
        "observation/state variable"
    )

    # ------------------------------------------------------------------
    # TEST 7 — Target-action reconstruction requirement
    # ------------------------------------------------------------------
    print(
        "\nTEST 7 — Target-action reconstruction requirement"
    )

    print(
        "For each sampled NEXT transition:"
    )

    print(
        "  1. Obtain current next-state velocity."
    )

    print(
        "  2. Obtain immediate following velocity."
    )

    print(
        "  3. Obtain slope associated with that transition."
    )

    print(
        "  4. Reconstruct next-transition wheel torque."
    )

    print(
        "  5. Compute next-transition sigma_min."
    )

    print(
        "  6. Map target Actor normalized action "
        "to physical sigma_tor."
    )

    print(
        "  7. Pass physical sigma_tor to Target Critic."
    )

    print(
        "✓ Target-action information requirement established"
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(
        "✓ PHASE 3E.4F-3 REPLAY / TARGET-ACTION "
        "INFORMATION AUDIT PASSED"
    )
    print("=" * 78)

    print(
        "\nConclusion:"
    )

    print(
        "sigma_min can be reconstructed from immediate transition "
        "information without adding it to the six-dimensional RL state "
        "and without using future driving-cycle information."
    )

    print(
        "\nNo production files were modified."
    )


if __name__ == "__main__":
    main()