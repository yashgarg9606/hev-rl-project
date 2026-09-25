"""
Recurrent history encoder for DDPG-GRU-SA.

Wu et al. (2024):
    Historical sequence length L = 2
    GRU hidden units = 50

Each historical element is:

    [state_t, action_t]

For this project:

    state_dim  = 6
    action_dim = 1

Therefore:

    GRU input dimension = 7
    sequence shape      = [batch, 2, 7]
    GRU output shape    = [batch, 2, 50]
"""

import torch
import torch.nn as nn


class RecurrentEncoder(nn.Module):
    """Encode historical state-action sequences using a GRU."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 50,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.input_dim = state_dim + action_dim

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(
        self,
        history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        history:
            Tensor with shape [batch, sequence_length, state_dim + action_dim].

        Returns
        -------
        Tensor:
            GRU sequence output with shape
            [batch, sequence_length, hidden_dim].
        """

        if history.ndim != 3:
            raise ValueError(
                "history must have shape "
                "[batch, sequence_length, state_dim + action_dim]."
            )

        if history.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected history feature dimension {self.input_dim}, "
                f"received {history.shape[-1]}."
            )

        output, _ = self.gru(history)

        return output
