"""
Phase 3E.4 — Multi-episode training stability validation.

Purpose:
    Validate that the DDPG-GRU-SA training loop remains operational
    over repeated episodes and that training statistics can be
    collected reliably.

This is a controlled validation experiment.
It is NOT the final 500-episode research training.
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
    """Initial zero-padded historical state-action sequence."""

    return np.zeros(
        (2, 7),
        dtype=np.float32,
    )


def main() -> None:

    print("=" * 70)
    print("PHASE 3E.4 — MULTI-EPISODE TRAINING STABILITY VALIDATION")
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

    # Longer than 3E.3, but still deliberately small.
    num_episodes = 20

    # ---------------------------------------------------------------
    # TEST 1 — Construct components
    # ---------------------------------------------------------------

    print("\nTEST 1 — Training components")

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
    # TEST 2 — Record training configuration
    # ---------------------------------------------------------------

    print("\nTEST 2 — Training configuration")

    print(f"Episodes:       {num_episodes}")
    print(f"Batch size:     {batch_size}")
    print(f"Gamma:          {agent.gamma}")
    print(f"History length: {agent.history_length}")
    print(f"Device:         {agent.device}")

    assert agent.gamma == 0.99
    assert agent.history_length == 2

    print("✓ Training configuration valid")

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    final_socs: list[float] = []
    final_battery_sohs: list[float] = []
    final_motor1_sohs: list[float] = []
    final_motor2_sohs: list[float] = []

    actor_losses: list[float] = []
    critic_losses: list[float] = []

    cycle_completions = 0
    constraint_terminations = 0
    truncated_episodes = 0

    total_updates = 0

    # ---------------------------------------------------------------
    # TEST 3 — Multi-episode training
    # ---------------------------------------------------------------

    print("\nTEST 3 — Multi-episode training stability")

    for episode in range(num_episodes):

        state, _ = env.reset(
            seed=1000 + episode
        )

        history = build_initial_history()

        episode_reward = 0.0
        episode_length = 0

        terminated = False
        truncated = False

        final_info = None

        while not (terminated or truncated):

            # -------------------------------------------------------
            # Actor with exploration
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

            final_info = info

            assert next_state.shape == (state_dim,)
            assert next_state.dtype == np.float32
            assert np.isfinite(next_state).all()
            assert np.isfinite(reward)

            # -------------------------------------------------------
            # Next historical sequence
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
            # Replay transition
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
            # DDPG update
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

                actor_losses.append(
                    actor_loss
                )

                critic_losses.append(
                    critic_loss
                )

                total_updates += 1

            # -------------------------------------------------------
            # Advance
            # -------------------------------------------------------

            episode_reward += float(reward)
            episode_length += 1

            state = next_state
            history = next_history

        assert final_info is not None

        # -----------------------------------------------------------
        # Determine termination type
        # -----------------------------------------------------------

        if final_info["constraint_violation"]:
            constraint_terminations += 1
            termination_type = "constraint"

        elif terminated:
            cycle_completions += 1
            termination_type = "cycle_end"

        else:
            truncated_episodes += 1
            termination_type = "truncated"

        # -----------------------------------------------------------
        # Record episode statistics
        # -----------------------------------------------------------

        episode_rewards.append(
            episode_reward
        )

        episode_lengths.append(
            episode_length
        )

        final_socs.append(
            float(state[2])
        )

        final_battery_sohs.append(
            float(state[3])
        )

        final_motor1_sohs.append(
            float(state[4])
        )

        final_motor2_sohs.append(
            float(state[5])
        )

        print(
            f"Episode {episode + 1:2d} | "
            f"steps={episode_length:2d} | "
            f"reward={episode_reward:12.6f} | "
            f"termination={termination_type:9s} | "
            f"buffer={len(replay_buffer):3d} | "
            f"updates={total_updates:4d}"
        )

    # ---------------------------------------------------------------
    # TEST 4 — Episode statistics
    # ---------------------------------------------------------------

    print("\nTEST 4 — Episode statistics")

    assert len(episode_rewards) == num_episodes
    assert len(episode_lengths) == num_episodes

    assert len(final_socs) == num_episodes
    assert len(final_battery_sohs) == num_episodes
    assert len(final_motor1_sohs) == num_episodes
    assert len(final_motor2_sohs) == num_episodes

    assert all(
        np.isfinite(x)
        for x in episode_rewards
    )

    assert all(
        1 <= length <= len(cycle) - 1
        for length in episode_lengths
    )

    print(
        "Reward range:",
        min(episode_rewards),
        "to",
        max(episode_rewards),
    )

    print(
        "Episode length range:",
        min(episode_lengths),
        "to",
        max(episode_lengths),
    )

    print(
        "Cycle completions:",
        cycle_completions,
    )

    print(
        "Constraint terminations:",
        constraint_terminations,
    )

    print(
        "Truncated episodes:",
        truncated_episodes,
    )

    print("✓ Episode statistics remain finite and valid")

    # ---------------------------------------------------------------
    # TEST 5 — Constraint termination accounting
    # ---------------------------------------------------------------

    print("\nTEST 5 — Termination accounting")

    assert (
        cycle_completions
        + constraint_terminations
        + truncated_episodes
        == num_episodes
    )

    print(
        "Termination total:",
        (
            cycle_completions
            + constraint_terminations
            + truncated_episodes
        ),
    )

    print(
        "Expected:",
        num_episodes,
    )

    print("✓ Every episode has exactly one termination classification")

    # ---------------------------------------------------------------
    # TEST 6 — Physical state stability
    # ---------------------------------------------------------------

    print("\nTEST 6 — Physical state stability")

    assert all(
        0.2 <= soc <= 0.9
        for soc in final_socs
    )

    assert all(
        0.8 <= soh <= 1.0
        for soh in final_battery_sohs
    )

    assert all(
        0.8 <= soh <= 1.0
        for soh in final_motor1_sohs
    )

    assert all(
        0.8 <= soh <= 1.0
        for soh in final_motor2_sohs
    )

    print(
        f"Final SOC range: "
        f"{min(final_socs):.12f} - "
        f"{max(final_socs):.12f}"
    )

    print(
        f"Final battery SOH range: "
        f"{min(final_battery_sohs):.12f} - "
        f"{max(final_battery_sohs):.12f}"
    )

    print("✓ Physical state remains within allowed bounds")

    # ---------------------------------------------------------------
    # TEST 7 — DDPG update stability
    # ---------------------------------------------------------------

    print("\nTEST 7 — DDPG update stability")

    assert total_updates > 0
    assert len(actor_losses) == total_updates
    assert len(critic_losses) == total_updates

    assert all(
        np.isfinite(loss)
        for loss in actor_losses
    )

    assert all(
        np.isfinite(loss)
        for loss in critic_losses
    )

    print(
        "Total DDPG updates:",
        total_updates,
    )

    print(
        f"Actor loss range: "
        f"{min(actor_losses):.6f} - "
        f"{max(actor_losses):.6f}"
    )

    print(
        f"Critic loss range: "
        f"{min(critic_losses):.6f} - "
        f"{max(critic_losses):.6f}"
    )

    print("✓ All DDPG updates remained finite")

    # ---------------------------------------------------------------
    # TEST 8 — Replay-buffer consistency
    # ---------------------------------------------------------------

    print("\nTEST 8 — Replay-buffer consistency")

    assert len(replay_buffer) == sum(
        episode_lengths
    )

    print(
        "Replay-buffer transitions:",
        len(replay_buffer),
    )

    print(
        "Expected:",
        sum(episode_lengths),
    )

    print("✓ Replay-buffer count matches actual transitions")

    # ---------------------------------------------------------------
    # TEST 9 — Action validity after repeated training
    # ---------------------------------------------------------------

    print("\nTEST 9 — Post-training action validity")

    state, _ = env.reset(seed=9999)

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

    final_action = float(
        action_tensor[0, 0].item()
    )

    assert 0.0 <= final_action <= 1.0

    print(
        f"Post-training deterministic action: "
        f"{final_action:.12f}"
    )

    print("✓ Actor remains numerically valid")

    # ---------------------------------------------------------------
    # TEST 10 — Training summary
    # ---------------------------------------------------------------

    print("\nTEST 10 — Stability summary")

    completion_rate = (
        cycle_completions / num_episodes
    )

    constraint_rate = (
        constraint_terminations / num_episodes
    )

    truncation_rate = (
        truncated_episodes / num_episodes
    )

    print(
        f"Cycle completion rate: "
        f"{completion_rate:.3f}"
    )

    print(
        f"Constraint termination rate: "
        f"{constraint_rate:.3f}"
    )

    print(
        f"Truncation rate: "
        f"{truncation_rate:.3f}"
    )

    print(
        f"Mean episode reward: "
        f"{np.mean(episode_rewards):.6f}"
    )

    print(
        f"Mean episode length: "
        f"{np.mean(episode_lengths):.3f}"
    )

    assert (
        abs(
            completion_rate
            + constraint_rate
            + truncation_rate
            - 1.0
        )
        < 1e-12
    )

    print("✓ Stability statistics are internally consistent")

    # ---------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------

    env.close()

    print("\n" + "=" * 70)
    print("✓ PHASE 3E.4 MULTI-EPISODE TRAINING STABILITY PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()