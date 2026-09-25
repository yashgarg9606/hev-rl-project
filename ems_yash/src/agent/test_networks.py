"""
Phase 3A — DDPG-GRU-SA network validation.

Validates:

1. GRU input/output dimensions.
2. Sequence length L=2.
3. Self-attention output dimensions.
4. Actor output dimension.
5. Actor action bounds.
6. Critic scalar output.
7. Actor gradients.
8. Critic gradients.
9. Different actions can be produced for different states.
"""

import torch

from recurrent_encoder import RecurrentEncoder
from self_attention import SelfAttention
from history_encoder import HistoryEncoder
from actor import Actor
from critic import Critic


STATE_DIM = 6
ACTION_DIM = 1
SEQUENCE_LENGTH = 2
GRU_HIDDEN = 50
FC_HIDDEN = 64
BATCH_SIZE = 8


def main():

    torch.manual_seed(42)

    print("=" * 75)
    print("PHASE 3A — DDPG-GRU-SA NETWORK VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------------
    # 1. GRU
    # --------------------------------------------------------------

    history = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        STATE_DIM + ACTION_DIM,
    )

    encoder = RecurrentEncoder(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=GRU_HIDDEN,
    )

    gru_output = encoder(history)

    print("\nGRU")
    print("-" * 75)
    print(f"Input shape:              {tuple(history.shape)}")
    print(f"Output shape:             {tuple(gru_output.shape)}")

    assert gru_output.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        GRU_HIDDEN,
    )

    print("✓ GRU dimensions correct")

    # --------------------------------------------------------------
    # 2. Self-attention
    # --------------------------------------------------------------

    attention = SelfAttention(
        hidden_dim=GRU_HIDDEN
    )

    attention_output = attention(
        gru_output
    )

    print("\nSELF-ATTENTION")
    print("-" * 75)
    print(
        f"Input shape:              "
        f"{tuple(gru_output.shape)}"
    )
    print(
        f"Output shape:             "
        f"{tuple(attention_output.shape)}"
    )

    assert attention_output.shape == (
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        GRU_HIDDEN,
    )

    print("✓ Self-attention dimensions correct")

    # --------------------------------------------------------------
    # 3. History encoder
    # --------------------------------------------------------------

    history_encoder = HistoryEncoder(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=GRU_HIDDEN,
    )

    history_feature = history_encoder(
        history
    )

    print("\nHISTORY ENCODER")
    print("-" * 75)
    print(
        f"Output shape:             "
        f"{tuple(history_feature.shape)}"
    )

    assert history_feature.shape == (
        BATCH_SIZE,
        GRU_HIDDEN,
    )

    print("✓ History feature dimensions correct")

    # --------------------------------------------------------------
    # 4. Actor
    # --------------------------------------------------------------

    actor = Actor(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        gru_hidden_dim=GRU_HIDDEN,
        hidden_dim=FC_HIDDEN,
    )

    state = torch.randn(
        BATCH_SIZE,
        STATE_DIM,
    )

    action = actor(
        history,
        state,
    )

    print("\nACTOR")
    print("-" * 75)
    print(
        f"Input history:            "
        f"{tuple(history.shape)}"
    )
    print(
        f"Input state:              "
        f"{tuple(state.shape)}"
    )
    print(
        f"Output action:            "
        f"{tuple(action.shape)}"
    )
    print(
        f"Action minimum:           "
        f"{action.min().item():.6f}"
    )
    print(
        f"Action maximum:           "
        f"{action.max().item():.6f}"
    )

    assert action.shape == (
        BATCH_SIZE,
        ACTION_DIM,
    )

    assert torch.all(
        action >= 0.0
    )

    assert torch.all(
        action <= 1.0
    )

    print("✓ Actor output dimension correct")
    print("✓ Actor action bounds correct")

    # --------------------------------------------------------------
    # 5. Critic
    # --------------------------------------------------------------

    critic = Critic(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        gru_hidden_dim=GRU_HIDDEN,
        hidden_dim=FC_HIDDEN,
    )

    q_value = critic(
        history,
        state,
        action,
    )

    print("\nCRITIC")
    print("-" * 75)
    print(
        f"Output shape:             "
        f"{tuple(q_value.shape)}"
    )
    print(
        f"Q-value mean:             "
        f"{q_value.mean().item():.6f}"
    )

    assert q_value.shape == (
        BATCH_SIZE,
        1,
    )

    print("✓ Critic scalar output correct")

       # --------------------------------------------------------------
    # 6. Gradient test
    # --------------------------------------------------------------

    # Actor gradient test
    actor_loss = -action.mean()

    actor.zero_grad()
    actor_loss.backward()

    actor_gradients = [
        parameter.grad
        for parameter in actor.parameters()
        if parameter.grad is not None
    ]

    assert len(actor_gradients) > 0

    assert all(
        torch.isfinite(gradient).all()
        for gradient in actor_gradients
    )

    print("✓ Actor gradients valid")

    # Critic gradient test
    # Recompute Q-value to create a fresh computation graph.
    critic.zero_grad()

    q_value_for_gradient_test = critic(
        history,
        state,
        action.detach(),
    )
    critic_loss = q_value_for_gradient_test.mean()

    critic_loss.backward()

    critic_gradients = [
        parameter.grad
        for parameter in critic.parameters()
        if parameter.grad is not None
    ]

    assert len(critic_gradients) > 0

    assert all(
        torch.isfinite(gradient).all()
        for gradient in critic_gradients
    )

    print("✓ Critic gradients valid")
    # --------------------------------------------------------------
    # 7. Parameter count
    # --------------------------------------------------------------

    actor_parameters = sum(
        parameter.numel()
        for parameter in actor.parameters()
    )

    critic_parameters = sum(
        parameter.numel()
        for parameter in critic.parameters()
    )

    print("\nPARAMETERS")
    print("-" * 75)
    print(
        f"Actor parameters:        "
        f"{actor_parameters:,}"
    )
    print(
        f"Critic parameters:       "
        f"{critic_parameters:,}"
    )

    print("\n" + "=" * 75)
    print("✓ PHASE 3A NETWORK VALIDATION PASSED")
    print("=" * 75)


if __name__ == "__main__":
    main()
