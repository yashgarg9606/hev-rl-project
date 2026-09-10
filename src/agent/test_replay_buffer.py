"""
Phase 3B — Historical Replay Buffer Validation
"""

import numpy as np
import torch

from replay_buffer import HistoricalReplayBuffer


STATE_DIM = 6
ACTION_DIM = 1
HISTORY_LENGTH = 2


def make_history(values):
    """
    Convert two scalar identifiers into a [2, 7] history.

    Each row represents:
        [state(6), action(1)]
    """

    history = np.zeros(
        (HISTORY_LENGTH, STATE_DIM + ACTION_DIM),
        dtype=np.float32,
    )

    history[0, :] = values[0]
    history[1, :] = values[1]

    return history


def main():

    print("=" * 75)
    print("PHASE 3B — HISTORICAL REPLAY BUFFER VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------------
    # 1. Create buffer
    # --------------------------------------------------------------

    print("\nBUFFER")
    print("-" * 75)

    buffer = HistoricalReplayBuffer(
        capacity=100,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        history_length=HISTORY_LENGTH,
    )

    print(f"Capacity:              {buffer.capacity}")
    print(f"State dimension:       {buffer.state_dim}")
    print(f"Action dimension:      {buffer.action_dim}")
    print(f"History length:        {buffer.history_length}")
    print(f"History dimension:     {buffer.history_dim}")

    assert len(buffer) == 0

    print("✓ Empty buffer initialized correctly")

    # --------------------------------------------------------------
    # 2. Zero-padded initial history
    # --------------------------------------------------------------

    print("\nZERO-PADDED HISTORY")
    print("-" * 75)

    zero_history = np.zeros(
        (HISTORY_LENGTH, STATE_DIM + ACTION_DIM),
        dtype=np.float32,
    )

    state = np.array(
        [50.0, 0.0, 0.6, 1.0, 1.0, 1.0],
        dtype=np.float32,
    )

    action = np.array(
        [0.5],
        dtype=np.float32,
    )

    next_state = np.array(
        [51.0, 0.0, 0.5999, 1.0, 1.0, 1.0],
        dtype=np.float32,
    )

    next_history = np.zeros_like(zero_history)

    next_history[0] = zero_history[1]
    next_history[1] = np.concatenate(
        [state, action]
    )

    buffer.add(
        history=zero_history,
        state=state,
        action=action,
        reward=1.0,
        next_state=next_state,
        next_history=next_history,
        done=False,
    )

    assert len(buffer) == 1

    stored = buffer.histories[0]

    assert np.allclose(
        stored,
        zero_history,
    )

    print("✓ Initial history correctly zero-padded")

    # --------------------------------------------------------------
    # 3. History ordering
    # --------------------------------------------------------------

    print("\nHISTORY ORDERING")
    print("-" * 75)

    history = make_history(
        [
            10.0,
            20.0,
        ]
    )

    next_history = make_history(
        [
            20.0,
            30.0,
        ]
    )

    state = np.full(
        STATE_DIM,
        30.0,
        dtype=np.float32,
    )

    action = np.array(
        [0.3],
        dtype=np.float32,
    )

    next_state = np.full(
        STATE_DIM,
        40.0,
        dtype=np.float32,
    )

    buffer.add(
        history=history,
        state=state,
        action=action,
        reward=2.0,
        next_state=next_state,
        next_history=next_history,
        done=False,
    )

    assert np.allclose(
        buffer.histories[1],
        history,
    )

    assert np.allclose(
        buffer.next_histories[1],
        next_history,
    )

    print("✓ Historical sequence ordering preserved")

    # --------------------------------------------------------------
    # 4. Episode boundary
    # --------------------------------------------------------------

    print("\nEPISODE BOUNDARY")
    print("-" * 75)

    terminal_history = make_history(
        [
            100.0,
            200.0,
        ]
    )

    terminal_next_history = np.zeros(
        (HISTORY_LENGTH, STATE_DIM + ACTION_DIM),
        dtype=np.float32,
    )

    terminal_state = np.full(
        STATE_DIM,
        300.0,
        dtype=np.float32,
    )

    terminal_next_state = np.zeros(
        STATE_DIM,
        dtype=np.float32,
    )

    buffer.add(
        history=terminal_history,
        state=terminal_state,
        action=np.array([0.8], dtype=np.float32),
        reward=-5.0,
        next_state=terminal_next_state,
        next_history=terminal_next_history,
        done=True,
    )

    assert buffer.dones[2] == 1.0

    assert np.allclose(
        buffer.next_histories[2],
        0.0,
    )

    print("✓ Terminal transition stored correctly")
    print("✓ Terminal next-history contains no leaked history")

    # --------------------------------------------------------------
    # 5. Fill buffer for sampling
    # --------------------------------------------------------------

    print("\nSAMPLING")
    print("-" * 75)

    for i in range(20):

        history = np.full(
            (HISTORY_LENGTH, STATE_DIM + ACTION_DIM),
            float(i),
            dtype=np.float32,
        )

        next_history = np.full(
            (HISTORY_LENGTH, STATE_DIM + ACTION_DIM),
            float(i + 1),
            dtype=np.float32,
        )

        state = np.full(
            STATE_DIM,
            float(i),
            dtype=np.float32,
        )

        action = np.array(
            [i / 20.0],
            dtype=np.float32,
        )

        next_state = np.full(
            STATE_DIM,
            float(i + 1),
            dtype=np.float32,
        )

        buffer.add(
            history=history,
            state=state,
            action=action,
            reward=float(i),
            next_state=next_state,
            next_history=next_history,
            done=False,
        )

    batch_size = 8

    batch = buffer.sample(
        batch_size=batch_size,
        device="cpu",
    )

    print(f"Buffer size:            {len(buffer)}")
    print(f"Batch size:             {batch_size}")
    print(f"History shape:          {tuple(batch.history.shape)}")
    print(f"State shape:            {tuple(batch.state.shape)}")
    print(f"Action shape:           {tuple(batch.action.shape)}")
    print(f"Reward shape:           {tuple(batch.reward.shape)}")
    print(f"Next-state shape:       {tuple(batch.next_state.shape)}")
    print(f"Next-history shape:     {tuple(batch.next_history.shape)}")
    print(f"Done shape:             {tuple(batch.done.shape)}")

    assert batch.history.shape == (
        batch_size,
        HISTORY_LENGTH,
        STATE_DIM + ACTION_DIM,
    )

    assert batch.state.shape == (
        batch_size,
        STATE_DIM,
    )

    assert batch.action.shape == (
        batch_size,
        ACTION_DIM,
    )

    assert batch.reward.shape == (
        batch_size,
        1,
    )

    assert batch.next_state.shape == (
        batch_size,
        STATE_DIM,
    )

    assert batch.next_history.shape == (
        batch_size,
        HISTORY_LENGTH,
        STATE_DIM + ACTION_DIM,
    )

    assert batch.done.shape == (
        batch_size,
        1,
    )

    print("✓ Mini-batch dimensions correct")

    # --------------------------------------------------------------
    # 6. Tensor validation
    # --------------------------------------------------------------

    print("\nTENSOR VALIDATION")
    print("-" * 75)

    assert isinstance(batch.history, torch.Tensor)
    assert isinstance(batch.state, torch.Tensor)
    assert isinstance(batch.action, torch.Tensor)
    assert isinstance(batch.reward, torch.Tensor)
    assert isinstance(batch.next_state, torch.Tensor)
    assert isinstance(batch.next_history, torch.Tensor)
    assert isinstance(batch.done, torch.Tensor)

    assert batch.history.dtype == torch.float32
    assert batch.state.dtype == torch.float32
    assert batch.action.dtype == torch.float32

    print("✓ PyTorch tensor types correct")
    print("✓ Tensor dtype correct")

    # --------------------------------------------------------------
    # 7. Insufficient sample protection
    # --------------------------------------------------------------

    print("\nERROR HANDLING")
    print("-" * 75)

    small_buffer = HistoricalReplayBuffer(
        capacity=10,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        history_length=HISTORY_LENGTH,
    )

    try:
        small_buffer.sample(1)
        raise AssertionError(
            "Sampling from an empty buffer should fail"
        )
    except ValueError:
        print("✓ Empty-buffer sampling correctly rejected")

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    print("\n" + "=" * 75)
    print("✓ PHASE 3B HISTORICAL REPLAY BUFFER VALIDATION PASSED")
    print("=" * 75)


if __name__ == "__main__":
    main()