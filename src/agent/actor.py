"""
Actor network for DDPG-GRU-SA.

Architecture:

    historical state-action sequence
        ↓
    GRU(50)
        ↓
    Self-Attention
        ↓
    Mean Pool
        ↓
    concatenate current state
        ↓
    FC 64
        ↓
    FC 64
        ↓
    FC 64
        ↓
    Sigmoid
        ↓
    sigma_tor ∈ [0, 1]
"""

import torch
import torch.nn as nn

from history_encoder import HistoryEncoder


class Actor(nn.Module):
    """DDPG-GRU-SA Actor."""

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 1,
        gru_hidden_dim: int = 50,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.history_encoder = HistoryEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=gru_hidden_dim,
        )

        self.fc1 = nn.Linear(
            gru_hidden_dim + state_dim,
            hidden_dim,
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.fc3 = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.output = nn.Linear(
            hidden_dim,
            action_dim,
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        history: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        history:
            [batch, L, state_dim + action_dim]

        state:
            [batch, state_dim]

        Returns
        -------
        action:
            [batch, action_dim], bounded to [0, 1]
        """

        if state.ndim != 2:
            raise ValueError(
                "state must have shape [batch, state_dim]."
            )

        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, "
                f"received {state.shape[-1]}."
            )

        history_feature = self.history_encoder(
            history
        )

        x = torch.cat(
            [
                history_feature,
                state,
            ],
            dim=-1,
        )

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        action = self.sigmoid(
            self.output(x)
        )

        return action
