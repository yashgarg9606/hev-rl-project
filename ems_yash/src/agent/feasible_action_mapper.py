"""
Phase 3E.4F-1 — Feasibility-aware action mapper.

Maps the Actor's normalized action:

    a ∈ [0, 1]

to the physically feasible motor torque split:

    sigma_tor = sigma_min + a * (1 - sigma_min)

The mapper is deliberately kept separate from the environment and
DDPG agent during the first integration stage.

This preserves:
    - the Actor action space [0, 1]
    - differentiability
    - monotonicity
    - explicit physical-action representation
"""

from __future__ import annotations

import numpy as np
import torch


def map_action(
    actor_action: float | np.ndarray | torch.Tensor,
    sigma_min: float | np.ndarray | torch.Tensor,
):
    """
    Map normalized Actor action to feasible physical sigma_tor.

    Formula:

        sigma_tor = sigma_min + actor_action * (1 - sigma_min)

    Parameters
    ----------
    actor_action:
        Normalized Actor output in [0, 1].

    sigma_min:
        Minimum physically feasible sigma_tor.

    Returns
    -------
    Same general numerical type as the input action.
    """

    if isinstance(actor_action, torch.Tensor):
        return _map_torch(actor_action, sigma_min)

    return _map_numpy(actor_action, sigma_min)


def _map_numpy(
    actor_action,
    sigma_min,
):
    action = np.asarray(actor_action, dtype=np.float64)
    lower = np.asarray(sigma_min, dtype=np.float64)

    if not np.all(np.isfinite(action)):
        raise ValueError("Actor action contains non-finite values.")

    if not np.all(np.isfinite(lower)):
        raise ValueError("sigma_min contains non-finite values.")

    if np.any(action < 0.0) or np.any(action > 1.0):
        raise ValueError(
            "Actor action must lie within [0, 1]."
        )

    if np.any(lower < 0.0) or np.any(lower > 1.0):
        raise ValueError(
            "sigma_min must lie within [0, 1]."
        )

    sigma = lower + action * (1.0 - lower)

    if np.any(sigma < lower - 1e-12):
        raise RuntimeError(
            "Mapped action fell below sigma_min."
        )

    if np.any(sigma > 1.0 + 1e-12):
        raise RuntimeError(
            "Mapped action exceeded sigma=1."
        )

    return sigma


def _map_torch(
    actor_action: torch.Tensor,
    sigma_min,
):
    if not torch.isfinite(actor_action).all():
        raise ValueError(
            "Actor action contains non-finite values."
        )

    if isinstance(sigma_min, torch.Tensor):
        lower = sigma_min.to(
            device=actor_action.device,
            dtype=actor_action.dtype,
        )
    else:
        lower = torch.as_tensor(
            sigma_min,
            device=actor_action.device,
            dtype=actor_action.dtype,
        )

    if not torch.isfinite(lower).all():
        raise ValueError(
            "sigma_min contains non-finite values."
        )

    if torch.any(actor_action < 0.0) or torch.any(
        actor_action > 1.0
    ):
        raise ValueError(
            "Actor action must lie within [0, 1]."
        )

    if torch.any(lower < 0.0) or torch.any(
        lower > 1.0
    ):
        raise ValueError(
            "sigma_min must lie within [0, 1]."
        )

    sigma = lower + actor_action * (1.0 - lower)

    if torch.any(sigma < lower - 1e-6):
        raise RuntimeError(
            "Mapped action fell below sigma_min."
        )

    if torch.any(sigma > 1.0 + 1e-6):
        raise RuntimeError(
            "Mapped action exceeded sigma=1."
        )

    return sigma