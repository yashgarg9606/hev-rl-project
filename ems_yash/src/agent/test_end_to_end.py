"""
Phase 3E.1 — RL environment / agent end-to-end smoke test.

Validates the complete pipeline:

    Environment
        ↓
    State
        ↓
    Historical state-action sequence
        ↓
    DDPG-GRU-SA Actor
        ↓
    Action
        ↓
    Environment step
        ↓
    Reward
        ↓
    Historical replay buffer
        ↓
    DDPG update

This is a smoke test only.
It is NOT a training run.
"""

from __future__ import annotations

import numpy as np
import torch

from src.environment.rl_environment import EnergyManagementEnv
from src.agent.ddpg_agent import DDPGAgent
from src.agent.replay_buffer import HistoricalReplayBuffer


def build_test_cycle() -> np.ndarray:
    """
    Small deterministic driving cycle.

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
        dtype=np.float64,
    )


def build_history(
    history_length: int,
    state_dim: int,
    action_dim: int,
) -> np.ndarray:
    """Create zero-padded initial history."""

    return np.zeros(
        (
            history_length,
            state_dim + action_dim,
        ),
        dtype=np.float32,
    )


def main() -> None:

    print("=" * 70)
    print("PHASE 3E.1 — RL END-TO-END SMOKE TEST")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    state_dim = 6
    action_dim = 1
    history_length = 2
    batch_size = 50

    # ---------------------------------------------------------------
    # TEST 1 — Create environment and agent
    # ---------------------------------------------------------------

    print("\nTEST 1 — Environment and DDPG agent construction")

    cycle = build_test_cycle()

    env = EnergyManagementEnv(
        cycle=cycle
    )

    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        history_length=history_length,
        device="cpu",
    )

    replay_buffer = HistoricalReplayBuffer(
        capacity=100,
        state_dim=state_dim,
        action_dim=action_dim,
        history_length=history_length,
    )

    assert env.observation_space.shape == (state_dim,)
    assert env.action_space.shape == (action_dim,)

    assert agent.state_dim == state_dim
    assert agent.action_dim == action_dim
    assert agent.history_length == history_length

    print("✓ Environment constructed")
    print("✓ DDPG agent constructed")
    print("✓ Historical replay buffer constructed")

    # ---------------------------------------------------------------
    # TEST 2 — Environment reset
    # ---------------------------------------------------------------

    print("\nTEST 2 — Environment reset")

    state, info = env.reset(seed=42)

    assert state.shape == (state_dim,)
    assert state.dtype == np.float32
    assert np.isfinite(state).all()

    print("State shape:", state.shape)
    print("State:", state)

    print("✓ Environment reset successful")

    # ---------------------------------------------------------------
    # TEST 3 — Initial history construction
    # ---------------------------------------------------------------

    print("\nTEST 3 — Historical state-action sequence")

    history = build_history(
        history_length=history_length,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    assert history.shape == (
        history_length,
        state_dim + action_dim,
    )

    assert history.dtype == np.float32
    assert np.all(history == 0.0)

    print("History shape:", history.shape)

    print("✓ Initial zero-padded history is valid")

    # ---------------------------------------------------------------
    # TEST 4 — Actor action generation
    # ---------------------------------------------------------------

    print("\nTEST 4 — DDPG-GRU-SA action generation")

    history_tensor = torch.as_tensor(
        history,
        dtype=torch.float32,
    ).unsqueeze(0)

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
    ).unsqueeze(0)

    assert history_tensor.shape == (
        1,
        history_length,
        state_dim + action_dim,
    )

    assert state_tensor.shape == (
        1,
        state_dim,
    )

    action_tensor = agent.select_action(
        history=history_tensor,
        state=state_tensor,
        explore=False,
    )

    assert action_tensor.shape == (
        1,
        action_dim,
    )

    assert torch.isfinite(action_tensor).all()

    action_value = float(action_tensor[0, 0].item())

    assert 0.0 <= action_value <= 1.0

    print("History tensor:", tuple(history_tensor.shape))
    print("State tensor:", tuple(state_tensor.shape))
    print(f"Actor action: {action_value:.12f}")

    print("✓ Actor produces valid deterministic action")

    # ---------------------------------------------------------------
    # TEST 5 — Environment / Actor interaction
    # ---------------------------------------------------------------

    print("\nTEST 5 — Agent → environment transition")

    next_state, reward, terminated, truncated, info = env.step(
        np.array([action_value], dtype=np.float32)
    )

    assert next_state.shape == (state_dim,)
    assert next_state.dtype == np.float32
    assert np.isfinite(next_state).all()

    assert np.isfinite(reward)

    assert terminated is False
    assert truncated is False

    assert info["constraint_violation"] is False
    assert info["violated_constraints"] == ()

    print("Action:", action_value)
    print("Reward:", reward)
    print("Next state:", next_state)

    print("✓ Agent action successfully passed through environment")
    print("✓ Reward returned successfully")
    print("✓ No physical constraint violation")

    # ---------------------------------------------------------------
    # TEST 6 — Historical transition construction
    # ---------------------------------------------------------------

    print("\nTEST 6 — Historical replay transition")

    action = np.array(
        [action_value],
        dtype=np.float32,
    )

    next_history = np.zeros_like(history)

    # Previous state-action pair enters the most recent history slot.
    next_history[-1, :state_dim] = state
    next_history[-1, state_dim:] = action

    replay_buffer.add(
        history=history,
        state=state,
        action=action,
        reward=float(reward),
        next_state=next_state,
        next_history=next_history,
        done=bool(terminated or truncated),
    )

    assert len(replay_buffer) == 1

    print("Stored transitions:", len(replay_buffer))
    print("✓ Environment transition stored in historical replay")

    # ---------------------------------------------------------------
    # TEST 7 — Build enough transitions for DDPG update
    # ---------------------------------------------------------------

    print("\nTEST 7 — Populate replay buffer")

    # We already have one transition.
    #
    # The environment has only four transitions per episode.
    # Therefore we repeatedly reset and collect deterministic
    # transitions until the replay buffer contains >= batch_size.

    transitions_required = batch_size - len(replay_buffer)

    for transition_number in range(transitions_required):

        state, _ = env.reset(
            seed=1000 + transition_number
        )

        history = build_history(
            history_length=history_length,
            state_dim=state_dim,
            action_dim=action_dim,
        )

        episode_done = False

        while not episode_done:

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

            action_value = float(
                action_tensor[0, 0].item()
            )

            action = np.array(
                [action_value],
                dtype=np.float32,
            )

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            # A constraint violation should not occur during this
            # normal smoke-test trajectory.
            assert info["constraint_violation"] is False

            next_history = np.roll(
                history,
                shift=-1,
                axis=0,
            )

            next_history[-1, :state_dim] = state
            next_history[-1, state_dim:] = action

            done = bool(
                terminated or truncated
            )

            replay_buffer.add(
                history=history,
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                next_history=next_history,
                done=done,
            )

            state = next_state
            history = next_history
            episode_done = done

            if len(replay_buffer) >= batch_size:
                break

    assert len(replay_buffer) >= batch_size

    print(
        "Replay buffer size:",
        len(replay_buffer),
    )

    print("✓ Sufficient transitions collected")

    # ---------------------------------------------------------------
    # TEST 8 — Sample DDPG training batch
    # ---------------------------------------------------------------

    print("\nTEST 8 — Historical replay batch")

    batch = replay_buffer.sample(
        batch_size=batch_size,
        device=agent.device,
    )

    assert batch.history.shape == (
        batch_size,
        history_length,
        state_dim + action_dim,
    )

    assert batch.state.shape == (
        batch_size,
        state_dim,
    )

    assert batch.action.shape == (
        batch_size,
        action_dim,
    )

    assert batch.reward.shape == (
        batch_size,
        1,
    )

    assert batch.next_state.shape == (
        batch_size,
        state_dim,
    )

    assert batch.next_history.shape == (
        batch_size,
        history_length,
        state_dim + action_dim,
    )

    assert batch.done.shape == (
        batch_size,
        1,
    )

    print("History:", tuple(batch.history.shape))
    print("State:", tuple(batch.state.shape))
    print("Action:", tuple(batch.action.shape))
    print("Reward:", tuple(batch.reward.shape))
    print("Next state:", tuple(batch.next_state.shape))
    print("Next history:", tuple(batch.next_history.shape))
    print("Done:", tuple(batch.done.shape))

    print("✓ Replay batch has correct DDPG shapes")

    # ---------------------------------------------------------------
    # TEST 9 — Complete DDPG update
    # ---------------------------------------------------------------

    print("\nTEST 9 — Complete DDPG update")

    actor_loss, critic_loss = agent.update(
        batch
    )

    assert np.isfinite(actor_loss)
    assert np.isfinite(critic_loss)

    print(f"Actor loss:  {actor_loss:.12f}")
    print(f"Critic loss: {critic_loss:.12f}")

    print("✓ Complete DDPG update successful")

    # ---------------------------------------------------------------
    # TEST 10 — Post-update action remains valid
    # ---------------------------------------------------------------

    print("\nTEST 10 — Post-update action validation")

    state, _ = env.reset(seed=999)

    history = build_history(
        history_length=history_length,
        state_dim=state_dim,
        action_dim=action_dim,
    )

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

    assert action_tensor.shape == (
        1,
        action_dim,
    )

    assert torch.isfinite(action_tensor).all()

    action_value = float(
        action_tensor[0, 0].item()
    )

    assert 0.0 <= action_value <= 1.0

    print(
        f"Post-update action: "
        f"{action_value:.12f}"
    )

    print("✓ Post-update Actor action remains bounded")

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    env.close()

    print("\n" + "=" * 70)
    print("✓ PHASE 3E.1 RL END-TO-END SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()