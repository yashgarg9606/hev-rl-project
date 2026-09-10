"""
Self-attention block for DDPG-GRU-SA.

The attention mechanism follows:

    Hs = softmax(Q K^T / sqrt(d_Q)) V

where Q, K and V are learned linear projections of the GRU output.
"""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Single self-attention block."""

    def __init__(self, hidden_dim: int = 50):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.query = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.key = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.value = nn.Linear(
            hidden_dim,
            hidden_dim,
        )

        self.scale = hidden_dim ** 0.5

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            [batch, sequence_length, hidden_dim]

        Returns
        -------
        Tensor:
            [batch, sequence_length, hidden_dim]
        """

        if x.ndim != 3:
            raise ValueError(
                "Self-attention input must be 3-dimensional."
            )

        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dimension {self.hidden_dim}, "
                f"received {x.shape[-1]}."
            )

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / self.scale

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attended = torch.matmul(
            attention_weights,
            v,
        )

        return attended
