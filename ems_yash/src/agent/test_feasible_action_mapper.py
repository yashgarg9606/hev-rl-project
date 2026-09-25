"""
Phase 3E.4F-1 — Feasibility-aware action mapper validation.

Validates the standalone mapper using:
    1. scalar NumPy actions
    2. batched NumPy actions
    3. PyTorch tensors
    4. Actor-generated actions
    5. gradient propagation

No environment or DDPG production code is modified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.agent.actor import Actor
from src.agent.feasible_action_mapper import map_action


def main() -> None:
    print("=" * 78)
    print("PHASE 3E.4F-1 — FEASIBLE ACTION MAPPER VALIDATION")
    print("=" * 78)

    # ------------------------------------------------------------------
    # TEST 1 — Scalar NumPy mapping
    # ------------------------------------------------------------------
    print("\nTEST 1 — Scalar mapping")

    sigma_min = 0.4254

    assert np.isclose(
        map_action(0.0, sigma_min),
        sigma_min,
    )

    assert np.isclose(
        map_action(0.5, sigma_min),
        0.7127,
    )

    assert np.isclose(
        map_action(1.0, sigma_min),
        1.0,
    )

    print(
        f"a=0.00 → sigma={map_action(0.0, sigma_min):.6f}"
    )
    print(
        f"a=0.50 → sigma={map_action(0.5, sigma_min):.6f}"
    )
    print(
        f"a=1.00 → sigma={map_action(1.0, sigma_min):.6f}"
    )

    print("✓ Scalar mapping valid")

    # ------------------------------------------------------------------
    # TEST 2 — Batched NumPy mapping
    # ------------------------------------------------------------------
    print("\nTEST 2 — Batched NumPy mapping")

    actions = np.array(
        [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        dtype=np.float64,
    )

    sigma_mins = np.array(
        [0.4254, 0.4257, 0.4266, 0.4281, 0.4301, 0.3501],
        dtype=np.float64,
    )

    mapped = map_action(
        actions,
        sigma_mins,
    )

    print("Actor actions:")
    print(actions)

    print("sigma_min:")
    print(sigma_mins)

    print("Mapped physical sigma:")
    print(mapped)

    assert np.all(mapped >= sigma_mins)
    assert np.all(mapped <= 1.0)

    print("✓ Batched NumPy mapping valid")

    # ------------------------------------------------------------------
    # TEST 3 — PyTorch mapping
    # ------------------------------------------------------------------
    print("\nTEST 3 — PyTorch mapping")

    torch_actions = torch.tensor(
        [[0.0], [0.25], [0.5], [0.75], [1.0]],
        dtype=torch.float32,
    )

    torch_sigma_min = torch.full(
        (5, 1),
        0.4254,
        dtype=torch.float32,
    )

    torch_mapped = map_action(
        torch_actions,
        torch_sigma_min,
    )

    print("Mapped tensor:")
    print(torch_mapped)

    assert torch.all(
        torch_mapped >= torch_sigma_min
    )

    assert torch.all(
        torch_mapped <= 1.0
    )

    print("✓ PyTorch mapping valid")

    # ------------------------------------------------------------------
    # TEST 4 — Actor integration
    # ------------------------------------------------------------------
    print("\nTEST 4 — Actor output integration")

    torch.manual_seed(20260913)

    actor = Actor(
        state_dim=6,
        action_dim=1,
        gru_hidden_dim=50,
        hidden_dim=64,
    )

    actor.eval()

    batch_size = 16

    state = torch.randn(
        batch_size,
        6,
        dtype=torch.float32,
    )

    history = torch.randn(
        batch_size,
        2,
        7,
        dtype=torch.float32,
    )

    with torch.no_grad():
        actor_actions = actor(
            history,
            state,
        )

    assert actor_actions.shape == (16, 1)

    assert torch.all(actor_actions >= 0.0)
    assert torch.all(actor_actions <= 1.0)

    sigma_min_batch = torch.linspace(
        0.35,
        0.43,
        batch_size,
        dtype=torch.float32,
    ).unsqueeze(1)

    physical_actions = map_action(
        actor_actions,
        sigma_min_batch,
    )

    print(
        f"Actor action range: "
        f"{actor_actions.min().item():.6f} → "
        f"{actor_actions.max().item():.6f}"
    )

    print(
        f"Physical sigma range: "
        f"{physical_actions.min().item():.6f} → "
        f"{physical_actions.max().item():.6f}"
    )

    assert torch.all(
        physical_actions >= sigma_min_batch
    )

    assert torch.all(
        physical_actions <= 1.0
    )

    print("✓ Actor → mapper interface valid")

    # ------------------------------------------------------------------
    # TEST 5 — Gradient propagation
    # ------------------------------------------------------------------
    print("\nTEST 5 — Gradient propagation")

    actor.train()

    state = torch.randn(
        batch_size,
        6,
        dtype=torch.float32,
    )

    history = torch.randn(
        batch_size,
        2,
        7,
        dtype=torch.float32,
    )

    actor_actions = actor(
        history,
        state,
    )

    sigma_min_batch = torch.full(
        (batch_size, 1),
        0.4254,
        dtype=torch.float32,
    )

    physical_actions = map_action(
        actor_actions,
        sigma_min_batch,
    )

    loss = physical_actions.mean()

    actor.zero_grad()
    loss.backward()

    gradient_count = 0

    for parameter in actor.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(
                parameter.grad
            ).all()

            gradient_count += 1

    assert gradient_count > 0

    print(
        f"Parameters receiving gradients: "
        f"{gradient_count}"
    )

    print(
        "✓ Feasibility mapping preserves "
        "Actor gradient flow"
    )

    # ------------------------------------------------------------------
    # TEST 6 — Mapping is not clipping
    # ------------------------------------------------------------------
    print("\nTEST 6 — Mapping versus clipping")

    test_actions = torch.tensor(
        [[0.0], [0.1], [0.25], [0.5], [0.75], [1.0]],
        dtype=torch.float32,
    )

    lower = torch.full_like(
        test_actions,
        0.4254,
    )

    mapped = map_action(
        test_actions,
        lower,
    )

    clipped = torch.clamp(
        test_actions,
        min=0.4254,
        max=1.0,
    )

    print("Mapped:")
    print(mapped.flatten())

    print("Clipped:")
    print(clipped.flatten())

    assert not torch.allclose(
        mapped,
        clipped,
    )

    print(
        "✓ Mapper retains continuous action information"
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("✓ PHASE 3E.4F-1 FEASIBLE ACTION MAPPER VALIDATION PASSED")
    print("=" * 78)

    print(
        "\nNo production environment or DDPG code was modified."
    )


if __name__ == "__main__":
    main()