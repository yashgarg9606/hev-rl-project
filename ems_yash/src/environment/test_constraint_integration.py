"""
Phase 3D.7 — Physical constraint integration validation.

Validates that the physical constraints are correctly integrated
into the Gymnasium RL environment without modifying the validated
powertrain physics or reward model.
"""

from __future__ import annotations

import numpy as np

from src.environment.rl_environment import EnergyManagementEnv


def build_test_cycle() -> np.ndarray:
    """
    Small deterministic driving cycle for integration testing.

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
        ],
        dtype=float,
    )


def make_environment() -> EnergyManagementEnv:
    """Create a fresh test environment."""

    return EnergyManagementEnv(
        cycle=build_test_cycle()
    )


def main() -> None:

    print("=" * 70)
    print("PHASE 3D.7 — ENVIRONMENT CONSTRAINT INTEGRATION VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # TEST 1 — Environment construction
    # ---------------------------------------------------------------

    print("\nTEST 1 — Environment construction")

    env = make_environment()

    assert env.constraint_checker is not None
    assert env.reward_model is not None
    assert env.powertrain is not None

    print("✓ Constraint checker attached to environment")
    print("✓ Reward model attached to environment")
    print("✓ Integrated powertrain attached to environment")

    # ---------------------------------------------------------------
    # TEST 2 — Reset and initial state
    # ---------------------------------------------------------------

    print("\nTEST 2 — Environment reset")

    state, info = env.reset(seed=42)

    assert state.shape == (6,)
    assert state.dtype == np.float32

    assert np.isfinite(state).all()

    assert 0.2 <= float(state[2]) <= 0.9
    assert 0.8 <= float(state[3]) <= 1.0
    assert 0.8 <= float(state[4]) <= 1.0
    assert 0.8 <= float(state[5]) <= 1.0

    print("Initial state:", state)
    print("✓ Initial state satisfies physical health/SOC constraints")

    # ---------------------------------------------------------------
    # TEST 3 — Normal feasible action
    # ---------------------------------------------------------------

    print("\nTEST 3 — Normal feasible action")

    state_before = state.copy()

    next_state, reward, terminated, truncated, info = env.step(
        np.array([0.5], dtype=np.float32)
    )

    assert next_state.shape == (6,)
    assert next_state.dtype == np.float32
    assert np.isfinite(next_state).all()

    assert np.isfinite(reward)

    assert "constraint_violation" in info
    assert "violated_constraints" in info
    assert "constraint_stage" in info

    assert info["constraint_violation"] is False
    assert info["violated_constraints"] == ()
    assert info["constraint_stage"] is None

    assert terminated is False
    assert truncated is False

    print(f"Reward: {reward:.12f}")
    print("Constraint violation:", info["constraint_violation"])
    print("✓ Feasible transition proceeds normally")

    # ---------------------------------------------------------------
    # TEST 4 — Constraint diagnostics on normal transition
    # ---------------------------------------------------------------

    print("\nTEST 4 — Constraint diagnostics")

    assert "soc" in info
    assert "battery_soh" in info
    assert "motor1_soh" in info
    assert "motor2_soh" in info

    assert 0.2 <= float(info["soc"]) <= 0.9
    assert 0.8 <= float(info["battery_soh"]) <= 1.0
    assert 0.8 <= float(info["motor1_soh"]) <= 1.0
    assert 0.8 <= float(info["motor2_soh"]) <= 1.0

    print(f"SOC:       {info['soc']:.12f}")
    print(f"Battery SOH: {info['battery_soh']:.12f}")
    print(f"Motor 1 SOH: {info['motor1_soh']:.12f}")
    print(f"Motor 2 SOH: {info['motor2_soh']:.12f}")

    print("✓ Post-step constraint state is physically valid")

    # ---------------------------------------------------------------
    # TEST 5 — Force SOC violation before a step
    # ---------------------------------------------------------------

    print("\nTEST 5 — Pre-step SOC violation")

    # Directly modify the validated battery state only for testing
    # environment constraint handling.
    env.powertrain.battery.state.soc = 0.1

    current_index_before = env.current_index

    next_state, reward, terminated, truncated, info = env.step(
        np.array([0.5], dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False

    assert reward == 0.0

    assert info["constraint_violation"] is True
    assert info["constraint_stage"] == "pre_step"
    assert "soc" in info["violated_constraints"]

    # The physical plant must not be advanced when the pre-step
    # constraint check rejects the state.
    assert env.current_index == current_index_before + 1

    print("Violation:", info["violated_constraints"])
    print("Constraint stage:", info["constraint_stage"])
    print("Reward:", reward)

    print("✓ Pre-step constraint violation handled")

    # ---------------------------------------------------------------
    # TEST 6 — Environment remains resettable after violation
    # ---------------------------------------------------------------

    print("\nTEST 6 — Reset after constraint termination")

    state, info = env.reset(seed=123)

    assert state.shape == (6,)
    assert np.isfinite(state).all()

    assert 0.2 <= float(state[2]) <= 0.9
    assert 0.8 <= float(state[3]) <= 1.0
    assert 0.8 <= float(state[4]) <= 1.0
    assert 0.8 <= float(state[5]) <= 1.0

    print("✓ Environment resets correctly after constraint termination")

    # ---------------------------------------------------------------
    # TEST 7 — Force motor torque violation before step
    # ---------------------------------------------------------------

    print("\nTEST 7 — Motor torque constraint path")

    original_checker = env.constraint_checker

    class ForcedTorqueViolationChecker:
        """
        Test-only wrapper that forces a motor torque violation.

        The underlying checker remains unchanged.
        """

        def __init__(self, wrapped_checker):
            self.wrapped_checker = wrapped_checker

        def check(self, **kwargs):
            result = self.wrapped_checker.check(**kwargs)

            if result.overall_valid:
                from src.environment.constraints import ConstraintResult

                return ConstraintResult(
                    soc_valid=result.soc_valid,
                    battery_soh_valid=result.battery_soh_valid,
                    motor1_soh_valid=result.motor1_soh_valid,
                    motor2_soh_valid=result.motor2_soh_valid,
                    motor1_speed_valid=result.motor1_speed_valid,
                    motor2_speed_valid=result.motor2_speed_valid,
                    motor1_torque_valid=False,
                    motor2_torque_valid=result.motor2_torque_valid,
                    overall_valid=False,
                    violated_constraints=(
                        *result.violated_constraints,
                        "motor1_torque",
                    ),
                )

            return result

    env.constraint_checker = ForcedTorqueViolationChecker(
        original_checker
    )

    next_state, reward, terminated, truncated, info = env.step(
        np.array([0.5], dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert reward == 0.0

    assert info["constraint_violation"] is True
    assert info["constraint_stage"] == "pre_step"
    assert "motor1_torque" in info["violated_constraints"]

    print("Violation:", info["violated_constraints"])
    print("Constraint stage:", info["constraint_stage"])

    print("✓ Motor torque constraint path handled")

    # ---------------------------------------------------------------
    # TEST 8 — Restore checker and verify reset
    # ---------------------------------------------------------------

    print("\nTEST 8 — Restore normal environment")

    env.constraint_checker = original_checker

    state, info = env.reset(seed=456)

    assert state.shape == (6,)
    assert np.isfinite(state).all()

    print("✓ Normal constraint checker restored")
    print("✓ Environment remains reusable")

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    env.close()

    print("\n" + "=" * 70)
    print("PHASE 3D.7 ENVIRONMENT CONSTRAINT INTEGRATION VALIDATION")
    print("=" * 70)


if __name__ == "__main__":
    main()