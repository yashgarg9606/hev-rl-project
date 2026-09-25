"""
Phase 3E.2 — Full episode rollout validation.

Validates that the DDPG-GRU-SA Actor can interact with the
EnergyManagementEnv for an entire driving-cycle episode.

This is a validation rollout only.
No learning/update is performed.
"""

from __future__ import annotations

import numpy as np
import torch

from src.environment.rl_environment import EnergyManagementEnv
from src.agent.ddpg_agent import DDPGAgent


def build_test_cycle() -> np.ndarray:
    """
    Deterministic driving cycle.

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


def build_initial_history() -> np.ndarray:
    """Return the zero-padded initial history."""

    return np.zeros(
        (2, 7),
        dtype=np.float32,
    )


def main() -> None:

    print("=" * 70)
    print("PHASE 3E.2 — FULL EPISODE ROLLOUT VALIDATION")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    # ---------------------------------------------------------------
    # TEST 1 — Construct environment and agent
    # ---------------------------------------------------------------

    print("\nTEST 1 — Environment and agent construction")

    cycle = build_test_cycle()

    env = EnergyManagementEnv(
        cycle=cycle
    )

    agent = DDPGAgent(
        state_dim=6,
        action_dim=1,
        history_length=2,
        device="cpu",
    )

    print("✓ Environment constructed")
    print("✓ DDPG-GRU-SA agent constructed")

    # ---------------------------------------------------------------
    # TEST 2 — Reset
    # ---------------------------------------------------------------

    print("\nTEST 2 — Episode reset")

    state, info = env.reset(seed=42)

    history = build_initial_history()

    assert state.shape == (6,)
    assert state.dtype == np.float32
    assert np.isfinite(state).all()

    assert history.shape == (2, 7)
    assert history.dtype == np.float32
    assert np.all(history == 0.0)

    print("Initial state:", state)
    print("Initial history shape:", history.shape)

    print("✓ Initial state valid")
    print("✓ Initial history valid")

    # ---------------------------------------------------------------
    # TEST 3 — Full deterministic rollout
    # ---------------------------------------------------------------

    print("\nTEST 3 — Full Actor/environment rollout")

    transition_count = 0
    cumulative_reward = 0.0

    visited_indices = []

    initial_soc = float(state[2])
    initial_battery_soh = float(state[3])
    initial_motor1_soh = float(state[4])
    initial_motor2_soh = float(state[5])

    terminated = False
    truncated = False

    while not (terminated or truncated):

        # -----------------------------------------------------------
        # Actor inference
        # -----------------------------------------------------------

        history_tensor = torch.as_tensor(
            history,
            dtype=torch.float32,
        ).unsqueeze(0)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
        ).unsqueeze(0)

        action_tensor = agent.select_action(
            history=history_tensor,
            state=state_tensor,
            explore=False,
        )

        assert action_tensor.shape == (1, 1)
        assert torch.isfinite(action_tensor).all()

        action_value = float(
            action_tensor[0, 0].item()
        )

        assert 0.0 <= action_value <= 1.0

        action = np.array(
            [action_value],
            dtype=np.float32,
        )

        # -----------------------------------------------------------
        # Environment transition
        # -----------------------------------------------------------

        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        transition_count += 1
        cumulative_reward += float(reward)

        visited_indices.append(
            int(info["cycle_index"])
        )

        # -----------------------------------------------------------
        # State validation
        # -----------------------------------------------------------

        assert next_state.shape == (6,)
        assert next_state.dtype == np.float32
        assert np.isfinite(next_state).all()

        assert np.isfinite(reward)

        # -----------------------------------------------------------
        # Physical constraint validation
        # -----------------------------------------------------------

        assert info["constraint_violation"] is False
        assert info["violated_constraints"] == ()

        # -----------------------------------------------------------
        # State bounds
        # -----------------------------------------------------------

        soc = float(next_state[2])
        battery_soh = float(next_state[3])
        motor1_soh = float(next_state[4])
        motor2_soh = float(next_state[5])

        assert 0.2 <= soc <= 0.9
        assert 0.8 <= battery_soh <= 1.0
        assert 0.8 <= motor1_soh <= 1.0
        assert 0.8 <= motor2_soh <= 1.0

        # -----------------------------------------------------------
        # Update history
        # -----------------------------------------------------------

        next_history = np.roll(
            history,
            shift=-1,
            axis=0,
        )

        next_history[-1, :6] = state
        next_history[-1, 6] = action_value

        assert next_history.shape == (2, 7)
        assert np.isfinite(next_history).all()

        state = next_state
        history = next_history

        print(
            f"Step {transition_count:2d} | "
            f"cycle index={info['cycle_index']:2d} | "
            f"sigma={action_value:.6f} | "
            f"reward={float(reward):.6f} | "
            f"SOC={soc:.9f}"
        )

    # ---------------------------------------------------------------
    # TEST 4 — Episode termination
    # ---------------------------------------------------------------

    print("\nTEST 4 — Episode termination")

    assert terminated is True
    assert truncated is False

    expected_transitions = len(cycle) - 1

    assert transition_count == expected_transitions

    assert visited_indices == list(
        range(1, len(cycle))
    )

    print("Transitions:", transition_count)
    print("Expected:", expected_transitions)
    print("Visited cycle indices:", visited_indices)

    print("✓ Episode terminated at the correct cycle boundary")
    print("✓ No premature termination occurred")

    # ---------------------------------------------------------------
    # TEST 5 — Cumulative reward
    # ---------------------------------------------------------------

    print("\nTEST 5 — Cumulative reward")

    assert np.isfinite(cumulative_reward)

    print(
        f"Cumulative reward: "
        f"{cumulative_reward:.12f}"
    )

    print("✓ Cumulative reward is finite")

    # ---------------------------------------------------------------
    # TEST 6 — Final physical state
    # ---------------------------------------------------------------

    print("\nTEST 6 — Final physical state")

    final_soc = float(state[2])
    final_battery_soh = float(state[3])
    final_motor1_soh = float(state[4])
    final_motor2_soh = float(state[5])

    assert 0.2 <= final_soc <= 0.9
    assert 0.8 <= final_battery_soh <= 1.0
    assert 0.8 <= final_motor1_soh <= 1.0
    assert 0.8 <= final_motor2_soh <= 1.0

    print(f"Initial SOC:      {initial_soc:.12f}")
    print(f"Final SOC:        {final_soc:.12f}")

    print(
        f"Initial battery SOH: "
        f"{initial_battery_soh:.12f}"
    )
    print(
        f"Final battery SOH:   "
        f"{final_battery_soh:.12f}"
    )

    print(
        f"Initial motor 1 SOH: "
        f"{initial_motor1_soh:.12f}"
    )
    print(
        f"Final motor 1 SOH:   "
        f"{final_motor1_soh:.12f}"
    )

    print(
        f"Initial motor 2 SOH: "
        f"{initial_motor2_soh:.12f}"
    )
    print(
        f"Final motor 2 SOH:   "
        f"{final_motor2_soh:.12f}"
    )

    print("✓ Final physical state remains within constraints")

    # ---------------------------------------------------------------
    # TEST 7 — History after complete rollout
    # ---------------------------------------------------------------

    print("\nTEST 7 — Historical sequence continuity")

    assert history.shape == (2, 7)
    assert np.isfinite(history).all()

    # Last action stored in the final history entry must remain
    # inside the action range.
    final_history_action = float(
        history[-1, 6]
    )

    assert 0.0 <= final_history_action <= 1.0

    print("Final history shape:", history.shape)
    print(
        "Final historical action:",
        final_history_action,
    )

    print("✓ Historical sequence remains valid")

    # ---------------------------------------------------------------
    # TEST 8 — Reset after complete episode
    # ---------------------------------------------------------------

    print("\nTEST 8 — Reset after completed episode")

    reset_state, reset_info = env.reset(seed=123)

    assert reset_state.shape == (6,)
    assert reset_state.dtype == np.float32
    assert np.isfinite(reset_state).all()

    assert 0.2 <= float(reset_state[2]) <= 0.9
    assert 0.8 <= float(reset_state[3]) <= 1.0
    assert 0.8 <= float(reset_state[4]) <= 1.0
    assert 0.8 <= float(reset_state[5]) <= 1.0

    print("Reset state:", reset_state)

    print("✓ Environment successfully resets after full episode")

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    env.close()

    print("\n" + "=" * 70)
    print("✓ PHASE 3E.2 FULL EPISODE ROLLOUT PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()