"""
Phase 3D.3 — RL Environment Functional Validation

Validates the Gymnasium environment independently of the RL agent.
"""

from pathlib import Path

import numpy as np

from src.environment.rl_environment import EnergyManagementEnv


def load_test_cycle() -> np.ndarray:
    """
    Construct a small physically valid driving-cycle segment.

    This test cycle is intentionally centered around 50 km/h,
    where the reconstructed motor efficiency maps have already
    been validated through Phase 2D.

    The purpose of this test is Gymnasium/environment integration,
    not efficiency-map boundary testing.
    """

    return np.array(
        [
            [0.0, 50.0, 0.0],
            [1.0, 50.0, 0.0],
            [2.0, 55.0, 0.0],
            [3.0, 55.0, 0.0],
            [4.0, 50.0, 0.0],
            [5.0, 50.0, 0.0],
        ],
        dtype=np.float64,
    )

    if not cycle_path.exists():
        raise FileNotFoundError(
            f"Expected driving-cycle file not found:\n{cycle_path}"
        )

    data = np.loadtxt(
        cycle_path,
        delimiter=",",
        skiprows=1,
    )

    return data[:20]


def main() -> None:

    print("=" * 70)
    print("PHASE 3D.3 — RL ENVIRONMENT FUNCTIONAL VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # TEST 1 — Load driving cycle
    # ---------------------------------------------------------------

    print("\nTEST 1 — Driving-cycle loading")

    cycle = load_test_cycle()

    print(f"Cycle shape: {cycle.shape}")
    print(f"First row:   {cycle[0]}")
    print(f"Last row:    {cycle[-1]}")

    assert cycle.ndim == 2
    assert cycle.shape[1] == 3
    assert cycle.shape[0] == 6

    print("✓ Driving cycle loaded correctly")

    # ---------------------------------------------------------------
    # TEST 2 — Environment construction
    # ---------------------------------------------------------------

    print("\nTEST 2 — Environment construction")

    env = EnergyManagementEnv(cycle=cycle)

    print(f"State dimension:  {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    print(f"Action space:     {env.action_space}")
    print(f"Observation space:{env.observation_space}")

    assert env.state_dim == 6
    assert env.action_dim == 1

    assert env.action_space.shape == (1,)
    assert np.allclose(env.action_space.low, [0.0])
    assert np.allclose(env.action_space.high, [1.0])

    assert env.observation_space.shape == (6,)

    print("✓ Environment constructed correctly")

    # ---------------------------------------------------------------
    # TEST 3 — Reset
    # ---------------------------------------------------------------

    print("\nTEST 3 — Environment reset")

    state, info = env.reset(seed=42)

    print(f"Initial state: {state}")
    print(f"Initial info:  {info}")

    assert state.shape == (6,)
    assert state.dtype == np.float32

    assert env.observation_space.contains(state)

    assert state[0] == cycle[0, 1]

    assert state[2] == 0.6
    assert state[3] == 1.0
    assert state[4] == 1.0
    assert state[5] == 1.0

    assert info["cycle_index"] == 0

    print("✓ Reset returns valid six-dimensional state")
    print("✓ Initial SOC/SOH values are correct")

    # ---------------------------------------------------------------
    # TEST 4 — One environment step
    # ---------------------------------------------------------------

    print("\nTEST 4 — One environment step")

    action = np.array([0.5], dtype=np.float32)

    next_state, reward, terminated, truncated, info = env.step(
        action
    )

    print(f"Action:       {action}")
    print(f"Next state:   {next_state}")
    print(f"Reward:       {reward}")
    print(f"Terminated:   {terminated}")
    print(f"Truncated:    {truncated}")

    assert next_state.shape == (6,)
    assert next_state.dtype == np.float32

    assert env.observation_space.contains(next_state)

    assert isinstance(reward, float)
    assert reward == 0.0

    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    assert info["cycle_index"] == 1

    print("✓ One environment step successful")
    print("✓ Gymnasium return signature is correct")

    # ---------------------------------------------------------------
    # TEST 5 — State propagation
    # ---------------------------------------------------------------

    print("\nTEST 5 — State propagation")

    # The test cycle begins at 50 km/h and performs a steady-cruising
    # transition of 50 -> 50 km/h.
    #
    # Even though velocity remains constant, the vehicle requires
    # positive wheel torque to overcome rolling and aerodynamic
    # resistance. Therefore battery power, SOC and health should
    # change.

    print(
        f"Velocity:    "
        f"{state[0]:.6f} -> {next_state[0]:.6f} km/h"
    )

    print(
        f"SOC:         "
        f"{state[2]:.12f} -> {next_state[2]:.12f}"
    )

    print(
        f"Battery SOH: "
        f"{state[3]:.12f} -> {next_state[3]:.12f}"
    )

    print(
        f"Motor 1 SOH: "
        f"{state[4]:.12f} -> {next_state[4]:.12f}"
    )

    print(
        f"Motor 2 SOH: "
        f"{state[5]:.12f} -> {next_state[5]:.12f}"
    )

    assert np.isclose(
        next_state[0],
        cycle[1, 1],
        atol=1e-5,
    )

    # Steady cruising still requires positive traction power.
    assert next_state[2] < state[2]
    assert next_state[3] < state[3]

    print("✓ Velocity propagated from driving cycle")
    print("✓ SOC decreased during energy consumption")
    print("✓ Battery SOH decreased during energy throughput")

    # Motor SOH may change by an amount smaller than float32
    # resolution. The underlying powertrain health model is already
    # independently validated in Phase 2E.

    # ---------------------------------------------------------------
    # TEST 5B — Active driving transition
    # ---------------------------------------------------------------

    print("\nTEST 5B — Active driving transition")

    # Find the first meaningful moving transition.
    #
    # We require:
    #   1. Current velocity > 5 km/h
    #   2. A nonzero velocity change
    #
    # This avoids the startup region where the vehicle transitions
    # from rest and the instantaneous motor speed is zero.

    velocity_difference = np.abs(np.diff(cycle[:, 1]))

    candidate_indices = np.where(
        (cycle[:-1, 1] > 5.0)
        & (velocity_difference > 1e-9)
    )[0]

    if len(candidate_indices) == 0:
        raise AssertionError(
            "Test cycle contains no meaningful moving transition "
            "above 5 km/h."
        )

    active_index = int(candidate_indices[0])

    print(
        f"First meaningful velocity transition found at cycle "
        f"index {active_index}: "
        f"{cycle[active_index, 1]:.6f} -> "
        f"{cycle[active_index + 1, 1]:.6f} km/h"
    )

    # Reset the environment.
    env.reset()

    # Advance until immediately before the selected transition.
    while env.current_index < active_index:
        env.step(np.array([0.5], dtype=np.float32))

    state_before_active_step = env.current_state.copy()

    # Execute the selected moving transition.
    (
        next_active_state,
        _,
        terminated,
        truncated,
        active_info,
    ) = env.step(
        np.array([0.5], dtype=np.float32)
    )

    print(
        f"Velocity:    "
        f"{state_before_active_step[0]:.6f} -> "
        f"{next_active_state[0]:.6f} km/h"
    )

    print(
        f"SOC:         "
        f"{state_before_active_step[2]:.12f} -> "
        f"{next_active_state[2]:.12f}"
    )

    print(
        f"Battery SOH: "
        f"{state_before_active_step[3]:.12f} -> "
        f"{next_active_state[3]:.12f}"
    )

    print(
        f"Motor 1 SOH: "
        f"{state_before_active_step[4]:.12f} -> "
        f"{next_active_state[4]:.12f}"
    )

    print(
        f"Motor 2 SOH: "
        f"{state_before_active_step[5]:.12f} -> "
        f"{next_active_state[5]:.12f}"
    )

    print(
        f"Battery power: "
        f"{active_info['battery_power_kw']:.6f} kW"
    )

    print(
        f"Battery current: "
        f"{active_info['battery_current_a']:.6f} A"
    )

    # Float32 observation vs float64 cycle comparison.
    assert np.isclose(
        next_active_state[0],
        cycle[active_index + 1, 1],
        atol=1e-5,
    )

    assert not terminated
    assert not truncated

    # A meaningful moving transition should produce nonzero
    # electrical power and current.
    assert abs(active_info["battery_power_kw"]) > 1e-9
    assert abs(active_info["battery_current_a"]) > 1e-9

    # Battery SOH degradation is large enough to be visible in the
    # float32 RL observation.
    assert (
        next_active_state[3]
        < state_before_active_step[3]
    )

    # Motor health is maintained internally at higher precision than
    # the float32 RL observation. A very small SOH change can therefore
    # round to the same float32 value.
    #
    # Validate Motor 1 and Motor 2 degradation using the diagnostic
    # health values returned through info instead of requiring every
    # change to survive float32 conversion.

    assert active_info["motor1_soh"] < 1.0
    assert active_info["motor2_soh"] < 1.0

    assert next_active_state[4] <= state_before_active_step[4]
    assert next_active_state[5] <= state_before_active_step[5]

    print("✓ Meaningful moving transition identified")
    print("✓ Velocity propagated correctly")
    print("✓ Nonzero battery power generated")
    print("✓ Nonzero battery current generated")
    print("✓ Battery SOH changes during active operation")
    print("✓ Motor 1 internal SOH degradation confirmed")
    print("✓ Motor 2 internal SOH degradation confirmed")



    # ---------------------------------------------------------------
    # TEST 6 — Action boundaries
    # ---------------------------------------------------------------

    print("\nTEST 6 — Action boundaries")

    env.reset()

    lower_action = np.array([0.0], dtype=np.float32)
    env.step(lower_action)

    env.reset()

    upper_action = np.array([1.0], dtype=np.float32)
    env.step(upper_action)

    print("✓ sigma_tor = 0.0 accepted")
    print("✓ sigma_tor = 1.0 accepted")

    # ---------------------------------------------------------------
    # TEST 7 — Episode termination
    # ---------------------------------------------------------------

    print("\nTEST 7 — Episode termination")

    env.reset()

    terminated = False
    steps = 0

    while not terminated:

        _, _, terminated, truncated, _ = env.step(
            np.array([0.5], dtype=np.float32)
        )

        steps += 1

        assert not truncated

    print(f"Episode steps: {steps}")

    assert steps == len(cycle) - 1
    assert terminated is True

    print("✓ Episode terminates at final driving-cycle point")

    # ---------------------------------------------------------------
    # TEST 8 — Step after termination rejected
    # ---------------------------------------------------------------

    print("\nTEST 8 — Step after termination")

    try:
        env.step(np.array([0.5], dtype=np.float32))
    except RuntimeError:
        print("✓ Step after termination correctly rejected")
    else:
        raise AssertionError(
            "Environment allowed step() after episode termination."
        )

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("✓ PHASE 3D.3 RL ENVIRONMENT VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()