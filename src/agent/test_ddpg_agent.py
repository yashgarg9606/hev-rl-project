"""
Phase 3C — DDPG Agent Validation
"""

import numpy as np
import torch

from ddpg_agent import DDPGAgent
from replay_buffer import HistoricalReplayBuffer


STATE_DIM = 6
ACTION_DIM = 1
HISTORY_LENGTH = 2
BATCH_SIZE = 50


def networks_identical(
    network_a,
    network_b,
):
    for parameter_a, parameter_b in zip(
        network_a.parameters(),
        network_b.parameters(),
    ):
        if not torch.allclose(
            parameter_a,
            parameter_b,
        ):
            return False

    return True


def networks_different(
    network_a,
    network_b,
):
    for parameter_a, parameter_b in zip(
        network_a.parameters(),
        network_b.parameters(),
    ):
        if not torch.allclose(
            parameter_a,
            parameter_b,
        ):
            return True

    return False


def make_batch():

    buffer = HistoricalReplayBuffer(
        capacity=1000,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        history_length=HISTORY_LENGTH,
    )

    for i in range(100):

        history = np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(
                HISTORY_LENGTH,
                STATE_DIM + ACTION_DIM,
            ),
        ).astype(np.float32)

        state = np.random.uniform(
            low=-1.0,
            high=1.0,
            size=STATE_DIM,
        ).astype(np.float32)

        action = np.random.uniform(
            low=0.0,
            high=1.0,
            size=ACTION_DIM,
        ).astype(np.float32)

        reward = float(
            np.random.uniform(-1.0, 1.0)
        )

        next_state = np.random.uniform(
            low=-1.0,
            high=1.0,
            size=STATE_DIM,
        ).astype(np.float32)

        next_history = np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(
                HISTORY_LENGTH,
                STATE_DIM + ACTION_DIM,
            ),
        ).astype(np.float32)

        done = bool(i % 10 == 0)

        buffer.add(
            history=history,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_history=next_history,
            done=done,
        )

    return buffer.sample(
        batch_size=BATCH_SIZE,
        device="cpu",
    )


def main():

    print("=" * 75)
    print("PHASE 3C — DDPG AGENT VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------------
    # 1. Create Agent
    # --------------------------------------------------------------

    print("\nAGENT")
    print("-" * 75)

    agent = DDPGAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        history_length=HISTORY_LENGTH,
        hidden_dim=50,
        gamma=0.99,
        actor_lr=1e-4,
        critic_lr=1e-5,
        noise_std=0.35,
        tau=0.005,
        device="cpu",
    )

    print(f"Gamma:                 {agent.gamma}")
    print(f"Actor learning rate:   1e-4")
    print(f"Critic learning rate:  1e-5")
    print(f"Exploration noise:     {agent.noise_std}")
    print(f"Target update tau:     {agent.tau}")

    # --------------------------------------------------------------
    # 2. Target initialization
    # --------------------------------------------------------------

    print("\nTARGET NETWORK INITIALIZATION")
    print("-" * 75)

    assert networks_identical(
        agent.actor,
        agent.target_actor,
    )

    assert networks_identical(
        agent.critic,
        agent.target_critic,
    )

    print("✓ Target Actor initially matches Actor")
    print("✓ Target Critic initially matches Critic")

    # --------------------------------------------------------------
    # 3. Target gradients disabled
    # --------------------------------------------------------------

    print("\nTARGET NETWORK GRADIENTS")
    print("-" * 75)

    assert all(
        not parameter.requires_grad
        for parameter in agent.target_actor.parameters()
    )

    assert all(
        not parameter.requires_grad
        for parameter in agent.target_critic.parameters()
    )

    print("✓ Target Actor gradients disabled")
    print("✓ Target Critic gradients disabled")

    # --------------------------------------------------------------
    # 4. Batch
    # --------------------------------------------------------------

    print("\nTRAINING BATCH")
    print("-" * 75)

    batch = make_batch()

    print(f"History:               {tuple(batch.history.shape)}")
    print(f"State:                 {tuple(batch.state.shape)}")
    print(f"Action:                {tuple(batch.action.shape)}")
    print(f"Reward:                {tuple(batch.reward.shape)}")
    print(f"Next state:            {tuple(batch.next_state.shape)}")
    print(f"Next history:          {tuple(batch.next_history.shape)}")
    print(f"Done:                  {tuple(batch.done.shape)}")

    assert batch.history.shape == (
        BATCH_SIZE,
        HISTORY_LENGTH,
        STATE_DIM + ACTION_DIM,
    )

    assert batch.state.shape == (
        BATCH_SIZE,
        STATE_DIM,
    )

    assert batch.action.shape == (
        BATCH_SIZE,
        ACTION_DIM,
    )

    assert batch.reward.shape == (
        BATCH_SIZE,
        1,
    )

    assert batch.next_state.shape == (
        BATCH_SIZE,
        STATE_DIM,
    )

    assert batch.next_history.shape == (
        BATCH_SIZE,
        HISTORY_LENGTH,
        STATE_DIM + ACTION_DIM,
    )

    assert batch.done.shape == (
        BATCH_SIZE,
        1,
    )

    print("✓ Batch dimensions correct")

    # --------------------------------------------------------------
    # 5. Action selection
    # --------------------------------------------------------------

    print("\nACTION SELECTION")
    print("-" * 75)

    deterministic_action = agent.select_action(
        batch.history,
        batch.state,
        explore=False,
    )

    exploratory_action = agent.select_action(
        batch.history,
        batch.state,
        explore=True,
    )

    print(
        f"Deterministic action range: "
        f"{deterministic_action.min().item():.6f} "
        f"to "
        f"{deterministic_action.max().item():.6f}"
    )

    print(
        f"Exploratory action range:   "
        f"{exploratory_action.min().item():.6f} "
        f"to "
        f"{exploratory_action.max().item():.6f}"
    )

    assert deterministic_action.shape == (
        BATCH_SIZE,
        ACTION_DIM,
    )

    assert exploratory_action.shape == (
        BATCH_SIZE,
        ACTION_DIM,
    )

    assert torch.all(
        deterministic_action >= 0.0
    )

    assert torch.all(
        deterministic_action <= 1.0
    )

    assert torch.all(
        exploratory_action >= 0.0
    )

    assert torch.all(
        exploratory_action <= 1.0
    )

    print("✓ Deterministic actions bounded [0,1]")
    print("✓ Exploratory actions bounded [0,1]")

    # --------------------------------------------------------------
    # 6. Critic update
    # --------------------------------------------------------------

    print("\nCRITIC UPDATE")
    print("-" * 75)

    critic_loss = agent.update_critic(batch)

    print(
        f"Critic loss:            {critic_loss:.8f}"
    )

    assert np.isfinite(critic_loss)
    assert critic_loss >= 0.0

    print("✓ Critic update produced finite loss")

    # --------------------------------------------------------------
    # 7. Actor update
    # --------------------------------------------------------------

    print("\nACTOR UPDATE")
    print("-" * 75)

    actor_loss = agent.update_actor(batch)

    print(
        f"Actor loss:             {actor_loss:.8f}"
    )

    assert np.isfinite(actor_loss)

    print("✓ Actor update produced finite loss")

    # --------------------------------------------------------------
    # 8. Target update
    # --------------------------------------------------------------

    print("\nTARGET SOFT UPDATE")
    print("-" * 75)

    actor_target_before = [
        parameter.detach().clone()
        for parameter in agent.target_actor.parameters()
    ]

    critic_target_before = [
        parameter.detach().clone()
        for parameter in agent.target_critic.parameters()
    ]

    agent.update_targets()

    actor_target_changed = any(
        not torch.allclose(
            before,
            after,
        )
        for before, after in zip(
            actor_target_before,
            agent.target_actor.parameters(),
        )
    )

    critic_target_changed = any(
        not torch.allclose(
            before,
            after,
        )
        for before, after in zip(
            critic_target_before,
            agent.target_critic.parameters(),
        )
    )

    assert actor_target_changed
    assert critic_target_changed

    print("✓ Target Actor changed after soft update")
    print("✓ Target Critic changed after soft update")

    # --------------------------------------------------------------
    # 9. Target independence
    # --------------------------------------------------------------

    print("\nTARGET INDEPENDENCE")
    print("-" * 75)

    assert networks_different(
        agent.actor,
        agent.target_actor,
    )

    assert networks_different(
        agent.critic,
        agent.target_critic,
    )

    print("✓ Target Actor remains independent")
    print("✓ Target Critic remains independent")

    # --------------------------------------------------------------
    # 10. Complete update
    # --------------------------------------------------------------

    print("\nCOMPLETE DDPG UPDATE")
    print("-" * 75)

    actor_loss, critic_loss = agent.update(batch)

    print(
        f"Actor loss:             {actor_loss:.8f}"
    )

    print(
        f"Critic loss:            {critic_loss:.8f}"
    )

    assert np.isfinite(actor_loss)
    assert np.isfinite(critic_loss)

    print("✓ Complete DDPG update successful")

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    print("\n" + "=" * 75)
    print("✓ PHASE 3C DDPG AGENT VALIDATION PASSED")
    print("=" * 75)


if __name__ == "__main__":
    main()