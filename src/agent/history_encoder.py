"""
GRU + self-attention history encoder.

Pipeline:

    history
       ↓
    GRU(50)
       ↓
    Self-Attention
       ↓
    mean pooling
       ↓
    50-dimensional history feature
"""

import torch
import torch.nn as nn

from recurrent_encoder import RecurrentEncoder
from self_attention import SelfAttention


class HistoryEncoder(nn.Module):
    """Encode L-step state-action history."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 50,
    ):
        super().__init__()

        self.recurrent = RecurrentEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        self.attention = SelfAttention(
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """
        history:
            [batch, L, state_dim + action_dim]

        returns:
            [batch, hidden_dim]
        """

        gru_output = self.recurrent(history)

        attention_output = self.attention(
            gru_output
        )

        # Preserve information from both historical timesteps.
        history_feature = attention_output.mean(
            dim=1
        )

        return history_feature
