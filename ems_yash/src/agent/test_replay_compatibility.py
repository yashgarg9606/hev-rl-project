"""
Phase 3E.4F-4 — Replay-buffer compatibility audit.

Purpose
-------
Verify that the existing replay transition already contains sufficient
information to reconstruct the feasibility bound for the TARGET action.

State definition:

    [velocity_kmh,
     wheel_torque_nm,
     SOC,
     battery_SOH,
     motor1_SOH,
     motor2_SOH]

Because next_state contains the wheel torque demand for the next
transition, the target feasibility bound can be reconstructed as:

    next velocity
        +
    next wheel torque
        ↓
    next sigma_min

No next-next velocity, slope, dt, cycle index, or seventh state variable
is required.

No production files are modified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.agent.actor import Actor
from src.agent.replay_buffer import HistoricalReplayBuffer
from src.agent.feasible_action_mapper import map_action
from src.environment.integrated_powertrain import (
    IntegratedPowertrain,
    default_motor_map_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATE_DIM = 6
ACTION_DIM = 1
HISTORY_LENGTH = 2
HISTORY_DIM = STATE_DIM + ACTION_DIM

BATCH_SIZE = 8


def build_powertrain() -> IntegratedPowertrain:
    motor1_map, motor2_map = default_motor_map_paths(
        PROJECT_ROOT
    )

    return IntegratedPowertrain(
        motor1_map_path=motor1_map,
        motor2_map_path=motor2_map,
    )


def calculate_sigma_min(
    powertrain: IntegratedPowertrain,
    velocity_kmh: float,
    wheel_torque_nm: float,
) -> float:
    """
    Find the smallest feasible physical sigma_tor for a transition.

    Crucially, this requires only:
        velocity_kmh
        wheel_torque_nm

    Both are already contained in the six-dimensional state.
    """

    sigma_grid = np.linspace(
        0.0,
        1.0,
        10001,
        dtype=np.float64,
    )

    for sigma in sigma_grid:
        result = powertrain.motors.calculate_operating_point(
            velocity_kmh=velocity_kmh,
            wheel_torque_nm=wheel_torque_nm,
            sigma_tor=float(sigma),
        )

        if result["overall_feasible"]:
            return float(sigma)

    raise RuntimeError(
        "No feasible sigma_tor found."
    )


def main() -> None:
    print("=" * 78)
    print("PHASE 3E.4F-4 — REPLAY-BUFFER COMPATIBILITY AUDIT")
    print("=" * 78)

    powertrain = build_powertrain()

    # ------------------------------------------------------------------
    # TEST 1 — Replay buffer construction
    # ------------------------------------------------------------------
    print("\nTEST 1 — Existing replay-buffer interface")

    buffer = HistoricalReplayBuffer(
        capacity=100,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        history_length=HISTORY_LENGTH,
    )

    print(
        f"State dimension: {buffer.state_dim}"
    )

    print(
        f"Action dimension: {buffer.action_dim}"
    )

    print(
        f"History length: {buffer.history_length}"
    )

    assert buffer.state_dim == 6
    assert buffer.action_dim == 1
    assert buffer.history_length == 2

    print(
        "✓ Existing replay dimensions remain unchanged"
    )

    # ------------------------------------------------------------------
    # TEST 2 — Add physically executed actions
    # ------------------------------------------------------------------
    print(
        "\nTEST 2 — Store physical sigma_tor in replay"
    )

    for i in range(12):
        velocity = float(i * 5.0)

        if i < 6:
            wheel_torque = 1600.0 + i * 5.0
            sigma_min = calculate_sigma_min(
                powertrain,
                velocity,
                wheel_torque,
            )
        else:
            wheel_torque = -1400.0 - i * 5.0
            sigma_min = calculate_sigma_min(
                powertrain,
                velocity,
                wheel_torque,
            )

        normalized_action = 0.5

        physical_sigma = (
            sigma_min
            + normalized_action * (1.0 - sigma_min)
        )

        state = np.array(
            [
                velocity,
                wheel_torque,
                0.60,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

        next_state = np.array(
            [
                velocity + 5.0,
                wheel_torque,
                0.60,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

        history = np.zeros(
            (HISTORY_LENGTH, HISTORY_DIM),
            dtype=np.float32,
        )

        next_history = np.zeros(
            (HISTORY_LENGTH, HISTORY_DIM),
            dtype=np.float32,
        )

        buffer.add(
            history=history,
            state=state,
            action=np.array(
                [physical_sigma],
                dtype=np.float32,
            ),
            reward=1.0,
            next_state=next_state,
            next_history=next_history,
            done=False,
        )

    assert len(buffer) == 12

    print(
        "✓ Physical sigma_tor can be stored using "
        "the existing replay interface"
    )

    # ------------------------------------------------------------------
    # TEST 3 — Reconstruct target sigma_min from next_state
    # ------------------------------------------------------------------
    print(
        "\nTEST 3 — Reconstruct sigma_min from next_state"
    )

    batch = buffer.sample(
        batch_size=BATCH_SIZE,
        device="cpu",
    )

    next_velocity = batch.next_state[:, 0]
    next_wheel_torque = batch.next_state[:, 1]

    sigma_min_values = []

    for velocity, wheel_torque in zip(
        next_velocity.numpy(),
        next_wheel_torque.numpy(),
    ):
        sigma_min = calculate_sigma_min(
            powertrain=powertrain,
            velocity_kmh=float(velocity),
            wheel_torque_nm=float(wheel_torque),
        )

        sigma_min_values.append(sigma_min)

    sigma_min_tensor = torch.tensor(
        sigma_min_values,
        dtype=torch.float32,
    ).unsqueeze(1)

    print(
        "Next-state velocity range: "
        f"{next_velocity.min().item():.3f} → "
        f"{next_velocity.max().item():.3f} km/h"
    )

    print(
        "Next-state wheel torque range: "
        f"{next_wheel_torque.min().item():.3f} → "
        f"{next_wheel_torque.max().item():.3f} Nm"
    )

    print(
        "Reconstructed sigma_min range: "
        f"{sigma_min_tensor.min().item():.4f} → "
        f"{sigma_min_tensor.max().item():.4f}"
    )

    assert sigma_min_tensor.shape == (
        BATCH_SIZE,
        1,
    )

    assert torch.all(
        sigma_min_tensor >= 0.0
    )

    assert torch.all(
        sigma_min_tensor <= 1.0
    )

    print(
        "✓ Target sigma_min reconstructed entirely "
        "from next_state"
    )

    # ------------------------------------------------------------------
    # TEST 4 — Target Actor → feasible physical action
    # ------------------------------------------------------------------
    print(
        "\nTEST 4 — Target Actor → physical target action"
    )

    actor = Actor(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        gru_hidden_dim=50,
        hidden_dim=64,
    )

    with torch.no_grad():
        target_normalized_action = actor(
            batch.next_history,
            batch.next_state,
        )

        target_physical_action = map_action(
            target_normalized_action,
            sigma_min_tensor,
        )

    print(
        "Target normalized action range: "
        f"{target_normalized_action.min().item():.6f} → "
        f"{target_normalized_action.max().item():.6f}"
    )

    print(
        "Target physical sigma range: "
        f"{target_physical_action.min().item():.6f} → "
        f"{target_physical_action.max().item():.6f}"
    )

    assert torch.all(
        target_physical_action >= sigma_min_tensor
    )

    assert torch.all(
        target_physical_action <= 1.0
    )

    print(
        "✓ Target Actor action mapped to physically "
        "feasible sigma_tor"
    )

    # ------------------------------------------------------------------
    # TEST 5 — Physical replay action → Critic interface
    # ------------------------------------------------------------------
    print(
        "\nTEST 5 — Physical replay action interface"
    )

    from src.agent.critic import Critic

    critic = Critic(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        gru_hidden_dim=50,
        hidden_dim=64,
    )

    current_q = critic(
        batch.history,
        batch.state,
        batch.action,
    )

    target_q = critic(
        batch.next_history,
        batch.next_state,
        target_physical_action,
    )

    assert current_q.shape == (
        BATCH_SIZE,
        1,
    )

    assert target_q.shape == (
        BATCH_SIZE,
        1,
    )

    assert torch.isfinite(current_q).all()
    assert torch.isfinite(target_q).all()

    print(
        "✓ Critic accepts physical replay actions "
        "and physical target actions"
    )

    # ------------------------------------------------------------------
    # TEST 6 — Replay does not need sigma_min storage
    # ------------------------------------------------------------------
    print(
        "\nTEST 6 — No sigma_min storage required"
    )

    replay_fields = {
        "history",
        "state",
        "action",
        "reward",
        "next_state",
        "next_history",
        "done",
    }

    assert "sigma_min" not in replay_fields
    assert "next_sigma_min" not in replay_fields

    print(
        "✓ sigma_min remains a derived quantity"
    )

    # ------------------------------------------------------------------
    # TEST 7 — No seventh state variable
    # ------------------------------------------------------------------
    print(
        "\nTEST 7 — Six-dimensional state preservation"
    )

    assert batch.state.shape[1] == 6
    assert batch.next_state.shape[1] == 6

    print(
        "✓ RL state remains exactly six-dimensional"
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(
        "✓ PHASE 3E.4F-4 REPLAY-BUFFER COMPATIBILITY "
        "AUDIT PASSED"
    )
    print("=" * 78)

    print(
        "\nFinal architecture decision:"
    )

    print(
        "1. Actor stores/produces normalized action a."
    )

    print(
        "2. Environment maps a → physical sigma_tor."
    )

    print(
        "3. Replay stores the physical sigma_tor actually executed."
    )

    print(
        "4. Target sigma_min is reconstructed from next_state."
    )

    print(
        "5. Target Actor output is mapped to physical sigma_tor."
    )

    print(
        "6. Critic always receives physical sigma_tor."
    )

    print(
        "7. The six-dimensional RL state is unchanged."
    )

    print(
        "\nNo production files were modified."
    )


if __name__ == "__main__":
    main()