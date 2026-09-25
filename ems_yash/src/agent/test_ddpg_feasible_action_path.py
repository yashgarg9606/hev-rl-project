"""
Phase 3E.4F-2 — DDPG feasible-action learning-path validation.

Purpose
-------
Validate the complete learning path when the Actor produces a normalized
action a ∈ [0, 1] and the physical motor action is:

    sigma_tor = sigma_min + a * (1 - sigma_min)

The Critic continues to evaluate the PHYSICAL action sigma_tor.

This test validates:

    Actor
      ↓
    normalized action a
      ↓
    feasibility mapping
      ↓
    physical sigma_tor
      ↓
    Critic
      ↓
    Actor gradient

and the corresponding target-network path.

No production RL component is modified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.agent.actor import Actor
from src.agent.critic import Critic
from src.agent.feasible_action_mapper import map_action


STATE_DIM = 6
ACTION_DIM = 1
HISTORY_LENGTH = 2
HISTORY_DIM = STATE_DIM + ACTION_DIM

GRU_HIDDEN = 50
FC_HIDDEN = 64

BATCH_SIZE = 50

SEED = 20260913


def build_inputs() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Build deterministic synthetic DDPG inputs.

    Returns
    -------
    history:
        [B, L, 7]

    state:
        [B, 6]

    sigma_min:
        [B, 1]
    """

    torch.manual_seed(SEED)

    history = torch.randn(
        BATCH_SIZE,
        HISTORY_LENGTH,
        HISTORY_DIM,
        dtype=torch.float32,
    )

    state = torch.randn(
        BATCH_SIZE,
        STATE_DIM,
        dtype=torch.float32,
    )

    # Use the physically observed range from Phase 3E.4A/4B.
    sigma_min = torch.linspace(
        0.35,
        0.43,
        BATCH_SIZE,
        dtype=torch.float32,
    ).unsqueeze(1)

    return history, state, sigma_min


def main() -> None:
    print("=" * 78)
    print("PHASE 3E.4F-2 — DDPG FEASIBLE-ACTION LEARNING-PATH VALIDATION")
    print("=" * 78)

    torch.manual_seed(SEED)

    # ------------------------------------------------------------------
    # TEST 1 — Network construction
    # ------------------------------------------------------------------
    print("\nTEST 1 — Actor/Critic construction")

    actor = Actor(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        gru_hidden_dim=GRU_HIDDEN,
        hidden_dim=FC_HIDDEN,
    )

    critic = Critic(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        gru_hidden_dim=GRU_HIDDEN,
        hidden_dim=FC_HIDDEN,
    )

    target_actor = Actor(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        gru_hidden_dim=GRU_HIDDEN,
        hidden_dim=FC_HIDDEN,
    )

    target_critic = Critic(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        gru_hidden_dim=GRU_HIDDEN,
        hidden_dim=FC_HIDDEN,
    )

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    print("✓ Networks constructed")

    # ------------------------------------------------------------------
    # TEST 2 — Build learning inputs
    # ------------------------------------------------------------------
    print("\nTEST 2 — Learning-path inputs")

    history, state, sigma_min = build_inputs()

    print(f"History shape:   {tuple(history.shape)}")
    print(f"State shape:     {tuple(state.shape)}")
    print(f"sigma_min shape: {tuple(sigma_min.shape)}")

    assert history.shape == (
        BATCH_SIZE,
        HISTORY_LENGTH,
        HISTORY_DIM,
    )

    assert state.shape == (
        BATCH_SIZE,
        STATE_DIM,
    )

    assert sigma_min.shape == (
        BATCH_SIZE,
        ACTION_DIM,
    )

    assert torch.all(sigma_min >= 0.0)
    assert torch.all(sigma_min <= 1.0)

    print("✓ Input shapes valid")

    # ------------------------------------------------------------------
    # TEST 3 — Actor normalized action
    # ------------------------------------------------------------------
    print("\nTEST 3 — Actor normalized action")

    normalized_action = actor(
        history,
        state,
    )

    print(
        f"Normalized action shape: "
        f"{tuple(normalized_action.shape)}"
    )

    print(
        f"Normalized action range: "
        f"{normalized_action.min().item():.6f} → "
        f"{normalized_action.max().item():.6f}"
    )

    assert normalized_action.shape == (
        BATCH_SIZE,
        ACTION_DIM,
    )

    assert torch.all(normalized_action >= 0.0)
    assert torch.all(normalized_action <= 1.0)

    print("✓ Actor action is normalized and bounded")

    # ------------------------------------------------------------------
    # TEST 4 — Map to physical action
    # ------------------------------------------------------------------
    print("\nTEST 4 — Feasibility mapping")

    physical_action = map_action(
        normalized_action,
        sigma_min,
    )

    print(
        f"Physical sigma shape: "
        f"{tuple(physical_action.shape)}"
    )

    print(
        f"Physical sigma range: "
        f"{physical_action.min().item():.6f} → "
        f"{physical_action.max().item():.6f}"
    )

    assert physical_action.shape == (
        BATCH_SIZE,
        ACTION_DIM,
    )

    assert torch.all(
        physical_action >= sigma_min
    )

    assert torch.all(
        physical_action <= 1.0
    )

    print("✓ Physical action is feasible")

    # ------------------------------------------------------------------
    # TEST 5 — Critic evaluates physical action
    # ------------------------------------------------------------------
    print("\nTEST 5 — Critic physical-action interface")

    critic_value = critic(
        history,
        state,
        physical_action,
    )

    print(
        f"Critic output shape: "
        f"{tuple(critic_value.shape)}"
    )

    print(
        f"Critic value range: "
        f"{critic_value.min().item():.6f} → "
        f"{critic_value.max().item():.6f}"
    )

    assert critic_value.shape == (
        BATCH_SIZE,
        1,
    )

    assert torch.isfinite(critic_value).all()

    print(
        "✓ Critic successfully evaluates "
        "physical sigma_tor"
    )

    # ------------------------------------------------------------------
    # TEST 6 — Actor gradient through mapper and Critic
    # ------------------------------------------------------------------
    print("\nTEST 6 — Actor gradient through physical action")

    actor.zero_grad()
    critic.zero_grad()

    normalized_action = actor(
        history,
        state,
    )

    physical_action = map_action(
        normalized_action,
        sigma_min,
    )

    actor_q = critic(
        history,
        state,
        physical_action,
    )

    actor_loss = -actor_q.mean()

    actor_loss.backward()

    actor_gradient_count = 0

    for parameter in actor.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(
                parameter.grad
            ).all()

            actor_gradient_count += 1

    assert actor_gradient_count > 0

    print(
        f"Actor loss: {actor_loss.item():.8f}"
    )

    print(
        f"Actor parameters receiving gradients: "
        f"{actor_gradient_count}"
    )

    print(
        "✓ Actor gradient successfully propagates "
        "through feasibility mapping and Critic"
    )

    # ------------------------------------------------------------------
    # TEST 7 — Critic gradient
    # ------------------------------------------------------------------
    print("\nTEST 7 — Critic gradient")

    critic.zero_grad()

    normalized_action = actor(
        history,
        state,
    ).detach()

    physical_action = map_action(
        normalized_action,
        sigma_min,
    )

    critic_value = critic(
        history,
        state,
        physical_action,
    )

    critic_loss = critic_value.pow(2).mean()

    critic_loss.backward()

    critic_gradient_count = 0

    for parameter in critic.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(
                parameter.grad
            ).all()

            critic_gradient_count += 1

    assert critic_gradient_count > 0

    print(
        f"Critic loss: {critic_loss.item():.8f}"
    )

    print(
        f"Critic parameters receiving gradients: "
        f"{critic_gradient_count}"
    )

    print("✓ Critic gradients valid")

    # ------------------------------------------------------------------
    # TEST 8 — Target Actor → mapper → target Critic
    # ------------------------------------------------------------------
    print("\nTEST 8 — Target-network physical-action path")

    with torch.no_grad():
        target_normalized_action = target_actor(
            history,
            state,
        )

        target_physical_action = map_action(
            target_normalized_action,
            sigma_min,
        )

        target_q = target_critic(
            history,
            state,
            target_physical_action,
        )

    print(
        f"Target normalized action range: "
        f"{target_normalized_action.min().item():.6f} → "
        f"{target_normalized_action.max().item():.6f}"
    )

    print(
        f"Target physical sigma range: "
        f"{target_physical_action.min().item():.6f} → "
        f"{target_physical_action.max().item():.6f}"
    )

    print(
        f"Target Critic output shape: "
        f"{tuple(target_q.shape)}"
    )

    assert torch.all(
        target_physical_action >= sigma_min
    )

    assert torch.all(
        target_physical_action <= 1.0
    )

    assert torch.isfinite(target_q).all()

    print(
        "✓ Target Actor → mapper → Target Critic "
        "path is valid"
    )

    # ------------------------------------------------------------------
    # TEST 9 — Target networks remain gradient-free
    # ------------------------------------------------------------------
    print("\nTEST 9 — Target-network gradient isolation")

    target_actor_gradient_count = sum(
        parameter.grad is not None
        for parameter in target_actor.parameters()
    )

    target_critic_gradient_count = sum(
        parameter.grad is not None
        for parameter in target_critic.parameters()
    )

    assert target_actor_gradient_count == 0
    assert target_critic_gradient_count == 0

    print(
        "✓ Target networks remain gradient-free"
    )

    # ------------------------------------------------------------------
    # TEST 10 — Compare physical-action Critic with normalized-action
    #           Critic
    # ------------------------------------------------------------------
    print("\nTEST 10 — Action representation distinction")

    normalized_for_comparison = torch.tensor(
        [[0.2], [0.4], [0.6], [0.8]],
        dtype=torch.float32,
    )

    sigma_min_for_comparison = torch.full(
        (4, 1),
        0.4254,
        dtype=torch.float32,
    )

    physical_for_comparison = map_action(
        normalized_for_comparison,
        sigma_min_for_comparison,
    )

    print("Normalized actions:")
    print(normalized_for_comparison.flatten())

    print("Physical sigma actions:")
    print(physical_for_comparison.flatten())

    assert not torch.allclose(
        normalized_for_comparison,
        physical_for_comparison,
    )

    print(
        "✓ Normalized Actor action and physical "
        "sigma_tor are demonstrably distinct"
    )

    # ------------------------------------------------------------------
    # TEST 11 — Replay representation recommendation
    # ------------------------------------------------------------------
    print("\nTEST 11 — Replay representation")

    print(
        "\nRecommended replay action representation:"
    )

    print(
        "  Store the PHYSICAL sigma_tor actually executed "
        "by the environment."
    )

    print(
        "\nReason:"
    )

    print(
        "  The existing Critic is defined over the physical "
        "motor-allocation action."
    )

    print(
        "\nThe normalized Actor action can be regenerated "
        "during policy evaluation, while replay retains the "
        "actual physical action associated with each transition."
    )

    print(
        "\nFor target actions, the target Actor produces "
        "normalized a', which must be mapped to physical "
        "sigma_tor' before being passed to the target Critic."
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("✓ PHASE 3E.4F-2 DDPG FEASIBLE-ACTION PATH VALIDATION PASSED")
    print("=" * 78)

    print(
        "\nNo production DDPG/environment files were modified."
    )


if __name__ == "__main__":
    main()