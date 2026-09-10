"""
Phase 3B — Historical Replay Buffer

Implements sequence-aware replay for DDPG-GRU-SA.

The controller uses:
    state_dim  = 6
    action_dim = 1
    history length L = 2

Each sampled transition contains:
    history
    current_state
    current_action
    reward
    next_state
    next_history
    done

History consists of previous state-action pairs:

    [(s_{t-L}, a_{t-L}),
     ...
     (s_{t-1}, a_{t-1})]

At the beginning of an episode, missing history is
zero-padded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReplayBatch:
    """
    Mini-batch returned by the historical replay buffer.
    """

    history: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    next_history: torch.Tensor
    done: torch.Tensor


class HistoricalReplayBuffer:
    """
    Sequence-aware replay buffer for DDPG-GRU-SA.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    state_dim : int
        Dimension of the environment state.
    action_dim : int
        Dimension of the action.
    history_length : int
        Number of previous state-action pairs used by GRU.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        state_dim: int = 6,
        action_dim: int = 1,
        history_length: int = 2,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        if state_dim <= 0:
            raise ValueError("state_dim must be positive")

        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        if history_length <= 0:
            raise ValueError("history_length must be positive")

        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_length = history_length

        self.history_dim = state_dim + action_dim

        # Main transition storage.
        self.histories = np.zeros(
            (capacity, history_length, self.history_dim),
            dtype=np.float32,
        )

        self.states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (capacity, action_dim),
            dtype=np.float32,
        )

        self.rewards = np.zeros(
            capacity,
            dtype=np.float32,
        )

        self.next_states = np.zeros(
            (capacity, state_dim),
            dtype=np.float32,
        )

        self.next_histories = np.zeros(
            (capacity, history_length, self.history_dim),
            dtype=np.float32,
        )

        self.dones = np.zeros(
            capacity,
            dtype=np.float32,
        )

        self.position = 0
        self.size = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------------
    # Add transition
    # ------------------------------------------------------------------

    def add(
        self,
        history: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_history: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store one transition.
        """

        history = np.asarray(history, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        next_history = np.asarray(
            next_history,
            dtype=np.float32,
        )

        expected_history_shape = (
            self.history_length,
            self.history_dim,
        )

        if history.shape != expected_history_shape:
            raise ValueError(
                f"history shape must be {expected_history_shape}, "
                f"got {history.shape}"
            )

        if state.shape != (self.state_dim,):
            raise ValueError(
                f"state shape must be {(self.state_dim,)}, "
                f"got {state.shape}"
            )

        if action.shape != (self.action_dim,):
            raise ValueError(
                f"action shape must be {(self.action_dim,)}, "
                f"got {action.shape}"
            )

        if next_state.shape != (self.state_dim,):
            raise ValueError(
                f"next_state shape must be {(self.state_dim,)}, "
                f"got {next_state.shape}"
            )

        if next_history.shape != expected_history_shape:
            raise ValueError(
                f"next_history shape must be {expected_history_shape}, "
                f"got {next_history.shape}"
            )

        index = self.position

        self.histories[index] = history
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = float(reward)
        self.next_states[index] = next_state
        self.next_histories[index] = next_history
        self.dones[index] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> ReplayBatch:
        """
        Randomly sample a mini-batch.
        """

        if self.size < batch_size:
            raise ValueError(
                f"Not enough samples: buffer contains {self.size}, "
                f"requested {batch_size}"
            )

        indices = np.random.randint(
            0,
            self.size,
            size=batch_size,
        )

        return ReplayBatch(
            history=torch.as_tensor(
                self.histories[indices],
                dtype=torch.float32,
                device=device,
            ),
            state=torch.as_tensor(
                self.states[indices],
                dtype=torch.float32,
                device=device,
            ),
            action=torch.as_tensor(
                self.actions[indices],
                dtype=torch.float32,
                device=device,
            ),
            reward=torch.as_tensor(
                self.rewards[indices],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(-1),
            next_state=torch.as_tensor(
                self.next_states[indices],
                dtype=torch.float32,
                device=device,
            ),
            next_history=torch.as_tensor(
                self.next_histories[indices],
                dtype=torch.float32,
                device=device,
            ),
            done=torch.as_tensor(
                self.dones[indices],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(-1),
        )