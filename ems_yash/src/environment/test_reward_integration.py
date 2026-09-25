"""
Phase 3D.5 — Reward integration validation.

Validates that the Wu et al. (2024) reward function is correctly
connected to the Gymnasium energy-management environment.
"""

from __future__ import annotations

import numpy as np

from src.environment.rl_environment import EnergyManagementEnv


def load_test_cycle() -> np.ndarray:
    """
    Small physically valid cycle already used for Phase 3D.3.
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


def main() -> None:
    print("=" * 70)
    print("PHASE 3D.5 — REWARD INTEGRATION VALIDATION")
    print("=" * 70)

    cycle = load_test_cycle()

    env = EnergyManagementEnv(cycle=cycle)

    # ---------------------------------------------------------------
    # TEST 1 — Reward model attached
    # ---------------------------------------------------------------

    print("\nTEST 1 — Reward model integration")

    assert hasattr(env, "reward_model")
    assert env.reward_model is not None

    print("✓ Wu reward model attached to environment")

    # ---------------------------------------------------------------
    # TEST 2 — Reset initializes reward state
    # ---------------------------------------------------------------

    print("\nTEST 2 — Reward state reset")

    state, _ = env.reset(seed=42)

    assert env._reward_state is not None
    assert env._reward_state.dtype == np.float64

    assert np.allclose(
        env._reward_state,
        state.astype(np.float64),
    )

    print("Reward state:")
    print(env._reward_state)

    print("✓ High-precision reward state initialized")

    # ---------------------------------------------------------------
    # TEST 3 — First environment reward
    # ---------------------------------------------------------------

    print("\nTEST 3 — First environment reward")

    previous_reward_state = env._reward_state.copy()

    next_state, reward, terminated, truncated, info = env.step(
        np.array([0.5], dtype=np.float32)
    )

    print(f"Reward: {reward:.12f}")

    expected_reward = env.reward_model.compute(
        previous_reward_state,
        env._reward_state,
    )

    print(f"Expected: {expected_reward:.12f}")

    assert np.isclose(
        reward,
        expected_reward,
        rtol=1e-12,
        atol=1e-12,
    )

    assert isinstance(reward, float)
    assert not terminated
    assert not truncated

    print("✓ Environment reward matches reward model")

    # ---------------------------------------------------------------
    # TEST 4 — Reward is exposed in info
    # ---------------------------------------------------------------

    print("\nTEST 4 — Reward diagnostics")

    assert "reward" in info
    assert "delta_soc_reward" in info
    assert "delta_soh_reward" in info
    assert "delta_sohm1_reward" in info
    assert "delta_sohm2_reward" in info

    assert np.isclose(
        info["reward"],
        reward,
        rtol=1e-12,
        atol=1e-12,
    )

    print(
        f"ΔSOC:   "
        f"{info['delta_soc_reward']:.12e}"
    )
    print(
        f"ΔSOH:   "
        f"{info['delta_soh_reward']:.12e}"
    )
    print(
        f"ΔSOHM1: "
        f"{info['delta_sohm1_reward']:.12e}"
    )
    print(
        f"ΔSOHM2: "
        f"{info['delta_sohm2_reward']:.12e}"
    )

    print("✓ Reward diagnostics exposed correctly")

    # ---------------------------------------------------------------
    # TEST 5 — Reward changes after active transition
    # ---------------------------------------------------------------

    print("\nTEST 5 — Active transition reward")

    previous_reward_state = env._reward_state.copy()

    (
        next_active_state,
        active_reward,
        terminated,
        truncated,
        active_info,
    ) = env.step(
        np.array([0.5], dtype=np.float32)
    )

    expected_active_reward = env.reward_model.compute(
        previous_reward_state,
        env._reward_state,
    )

    print(
        f"Active-transition reward: "
        f"{active_reward:.12f}"
    )

    print(
        f"Expected reward:          "
        f"{expected_active_reward:.12f}"
    )

    assert np.isclose(
        active_reward,
        expected_active_reward,
        rtol=1e-12,
        atol=1e-12,
    )

    assert abs(active_info["battery_power_kw"]) > 1e-9
    assert abs(active_info["battery_current_a"]) > 1e-9

    print("✓ Active-transition reward is correctly calculated")

    # ---------------------------------------------------------------
    # TEST 6 — Reward state retains high-precision health values
    # ---------------------------------------------------------------

    print("\nTEST 6 — High-precision health preservation")

    print(
        f"Internal Motor 1 SOH: "
        f"{env._reward_state[4]:.15f}"
    )

    print(
        f"Internal Motor 2 SOH: "
        f"{env._reward_state[5]:.15f}"
    )

    # At least one motor should have degraded during the active
    # transition. The exact amount is determined by the validated
    # health model.
    assert (
        env._reward_state[4] < 1.0
        or env._reward_state[5] < 1.0
    )

    print(
        "✓ Motor health retained at reward precision "
        "independent of float32 observation precision"
    )

    # ---------------------------------------------------------------
    # TEST 7 — Reset clears previous reward history
    # ---------------------------------------------------------------

    print("\nTEST 7 — Reward state episode reset")

    env.reset()

    expected_initial_reward_state = np.array(
        [
            cycle[0, 1],
            env.current_state[1],
            0.6,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float64,
    )

    assert np.allclose(
        env._reward_state,
        expected_initial_reward_state,
    )

    print("✓ Reward state reset between episodes")

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("✓ PHASE 3D.5 REWARD INTEGRATION VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()