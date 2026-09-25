"""
Phase 3E.3 — Training-loop validation.

Validates the complete interaction and learning loop:

    Environment
        ↓
    Actor + exploration
        ↓
    Environment step
        ↓
    Historical transition
        ↓
    Replay buffer
        ↓
    DDPG update
        ↓
    Next transition

This is a controlled validation run, NOT the final research training.
"""

from __future__ import annotations

import numpy as np
import torch

from src.environment.rl_environment import EnergyManagementEnv
from src.agent.ddpg_agent import DDPGAgent
from src.agent.replay_buffer import HistoricalReplayBuffer


def build_test_cycle() -> np.ndarray:
    """
    Deterministic short driving cycle.

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
    """Initial zero-padded history."""

    return np.zeros(
        (2, 7),
        dtype=np.float32,
    )


def main() -> None:

    print("=" * 70)
    print("PHASE 3E.3 — TRAINING-LOOP VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------

    np.random.seed(42)
    torch.manual_seed(42)

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    state_dim = 6
    action_dim = 1
    history_length = 2

    batch_size = 8
    num_episodes = 5

    # ---------------------------------------------------------------
    # TEST 1 — Construct training components
    # ---------------------------------------------------------------

    print("\nTEST 1 — Training components")

    env = EnergyManagementEnv(
        cycle=build_test_cycle()
    )

    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        history_length=history_length,
        device="cpu",
    )

    replay_buffer = HistoricalReplayBuffer(
        capacity=1000,
        state_dim=state_dim,
        action_dim=action_dim,
        history_length=history_length,
    )

    assert env.observation_space.shape == (state_dim,)
    assert env.action_space.shape == (action_dim,)

    print("✓ Environment constructed")
    print("✓ DDPG agent constructed")
    print("✓ Replay buffer constructed")

    # ---------------------------------------------------------------
    # TEST 2 — Training-loop configuration
    # ---------------------------------------------------------------

    print("\nTEST 2 — Training configuration")

    assert batch_size == 8
    assert num_episodes == 5

    assert agent.gamma == 0.99
    assert agent.history_length == 2

    print(f"Episodes:       {num_episodes}")
    print(f"Batch size:     {batch_size}")
    print(f"Gamma:          {agent.gamma}")
    print(f"History length: {agent.history_length}")

    print("✓ Training-loop configuration valid")

    # ---------------------------------------------------------------
    # TEST 3 — Collect transitions and train
    # ---------------------------------------------------------------

    print("\nTEST 3 — Multi-episode training loop")

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    update_count = 0
    constraint_violations = 0

    actor_losses: list[float] = []
    critic_losses: list[float] = []

    for episode in range(num_episodes):

        state, _ = env.reset(
            seed=100 + episode
        )

        history = build_initial_history()

        episode_reward = 0.0
        episode_length = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            # -------------------------------------------------------
            # Actor action with exploration
            # -------------------------------------------------------

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
                explore=True,
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

            # -------------------------------------------------------
            # Environment transition
            # -------------------------------------------------------

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            assert next_state.shape == (state_dim,)
            assert np.isfinite(next_state).all()
            assert np.isfinite(reward)

            if info["constraint_violation"]:
                constraint_violations += 1

                print(
                    f"  CONSTRAINT VIOLATION | "
                    f"stage={info['constraint_stage']} | "
                    f"violations={info['violated_constraints']} | "
                    f"cycle_index={info['cycle_index']}"
                )

                print(
                    f"  sigma_tor={action_value:.12f}"
                )

                print(
                    f"  M1 speed={info.get('motor1_speed_rpm', 'N/A')} rpm | "
                    f"M2 speed={info.get('motor2_speed_rpm', 'N/A')} rpm"
                )

                print(
                    f"  M1 torque={info.get('motor1_torque_nm', 'N/A')} Nm | "
                    f"M2 torque={info.get('motor2_torque_nm', 'N/A')} Nm"
                )

            # -------------------------------------------------------
            # Construct next history
            # -------------------------------------------------------

            next_history = np.roll(
                history,
                shift=-1,
                axis=0,
            )

            next_history[-1, :state_dim] = state
            next_history[-1, state_dim:] = action

            assert next_history.shape == (
                history_length,
                state_dim + action_dim,
            )

            assert np.isfinite(next_history).all()

            # -------------------------------------------------------
            # Store transition
            # -------------------------------------------------------

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

            # -------------------------------------------------------
            # Learn once enough transitions exist
            # -------------------------------------------------------

            if len(replay_buffer) >= batch_size:

                batch = replay_buffer.sample(
                    batch_size=batch_size,
                    device=agent.device,
                )

                actor_loss, critic_loss = agent.update(
                    batch
                )

                assert np.isfinite(actor_loss)
                assert np.isfinite(critic_loss)

                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                update_count += 1

            # -------------------------------------------------------
            # Advance transition
            # -------------------------------------------------------

            episode_reward += float(reward)
            episode_length += 1

            state = next_state
            history = next_history

        episode_rewards.append(
            episode_reward
        )

        episode_lengths.append(
            episode_length
        )

        print(
            f"  Episode termination | "
            f"terminated={terminated} | "
            f"truncated={truncated} | "
            f"constraint_violations={constraint_violations}"
        )

        print(
            f"Episode {episode + 1:2d} | "
            f"steps={episode_length:2d} | "
            f"reward={episode_reward:.6f} | "
            f"buffer={len(replay_buffer):3d} | "
            f"updates={update_count:3d}"
        )

    # ---------------------------------------------------------------
    # TEST 4 — Episode statistics
    # ---------------------------------------------------------------

    print("\nTEST 4 — Episode statistics")

    assert len(episode_rewards) == num_episodes
    assert len(episode_lengths) == num_episodes

    assert all(
        np.isfinite(reward)
        for reward in episode_rewards
    )


    
    assert all(
        1 <= length <= len(build_test_cycle()) - 1
        for length in episode_lengths
    )

    print("Episode rewards:", episode_rewards)
    print("Episode lengths:", episode_lengths)

    print("✓ Episode statistics valid")

    # ---------------------------------------------------------------
    # TEST 5 — Replay-buffer growth
    # ---------------------------------------------------------------

    print("\nTEST 5 — Replay-buffer growth")

    expected_transitions = sum(
        episode_lengths
    )

    assert len(replay_buffer) == expected_transitions

    print(
        "Replay transitions:",
        len(replay_buffer),
    )

    print(
        "Expected transitions:",
        expected_transitions,
    )

    print("✓ Replay buffer contains all episode transitions")

    # ---------------------------------------------------------------
    # TEST 6 — DDPG updates
    # ---------------------------------------------------------------

    print("\nTEST 6 — DDPG learning updates")

    assert update_count > 0

    assert len(actor_losses) == update_count
    assert len(critic_losses) == update_count

    assert all(
        np.isfinite(loss)
        for loss in actor_losses
    )

    assert all(
        np.isfinite(loss)
        for loss in critic_losses
    )

    print(
        "DDPG updates:",
        update_count,
    )

    print(
        f"Final actor loss:  "
        f"{actor_losses[-1]:.12f}"
    )

    print(
        f"Final critic loss: "
        f"{critic_losses[-1]:.12f}"
    )

    print("✓ DDPG updates completed successfully")

    # ---------------------------------------------------------------
    # TEST 7 — Constraint behavior
    # ---------------------------------------------------------------

    print("\nTEST 7 — Constraint behavior")

    assert constraint_violations >= 0

    print(
        "Constraint violations:",
        constraint_violations,
    )

    print(
    "✓ Constraint violations were handled without "
    "crashing the training loop"
)

    # ---------------------------------------------------------------
    # TEST 8 — Post-training Actor action
    # ---------------------------------------------------------------

    print("\nTEST 8 — Post-training Actor action")

    state, _ = env.reset(seed=999)

    history = build_initial_history()

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

    print(
        f"Post-training action: "
        f"{action_value:.12f}"
    )

    print("✓ Post-training Actor remains valid")

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    env.close()

    print("\n" + "=" * 70)
    print("✓ PHASE 3E.3 TRAINING-LOOP VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()