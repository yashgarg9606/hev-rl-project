"""
Phase 3C — DDPG Agent

DDPG implementation for the DDPG-GRU-SA controller.

Architecture:
    Online Actor  -> action
    Target Actor  -> target action

    Online Critic -> Q(s, a)
    Target Critic -> target Q

The networks themselves are defined in:
    actor.py
    critic.py

Historical sequences are supplied by:
    replay_buffer.py
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from actor import Actor
from critic import Critic
from replay_buffer import ReplayBatch


class DDPGAgent:
    """
    DDPG agent for the GRU + Self-Attention architecture.

    Paper-aligned hyperparameters:
        gamma       = 0.99
        actor_lr    = 1e-4
        critic_lr   = 1e-5
        noise_std   = 0.35
        tau         = 0.005

    Notes
    -----
    The paper specifies the first four hyperparameters explicitly.
    tau is required for the standard DDPG soft target update; since
    the paper's table does not explicitly specify tau, 0.005 is an
    implementation choice and is documented separately.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 1,
        history_length: int = 2,
        hidden_dim: int = 50,
        gamma: float = 0.99,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-5,
        noise_std: float = 0.35,
        tau: float = 0.005,
        device: torch.device | str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.hidden_dim = hidden_dim

        self.gamma = gamma
        self.noise_std = noise_std
        self.tau = tau

        self.device = torch.device(device)

        # ----------------------------------------------------------
        # Online networks
        # ----------------------------------------------------------

        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.critic = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # ----------------------------------------------------------
        # Target networks
        # ----------------------------------------------------------

        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        # Target networks are not directly optimized.
        for parameter in self.target_actor.parameters():
            parameter.requires_grad = False

        for parameter in self.target_critic.parameters():
            parameter.requires_grad = False

        # ----------------------------------------------------------
        # Optimizers
        # ----------------------------------------------------------

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

        self.mse_loss = nn.MSELoss()

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(
        self,
        history: torch.Tensor,
        state: torch.Tensor,
        explore: bool = True,
    ) -> torch.Tensor:
        """
        Select an action from the Actor.

        Actor output is already constrained to [0, 1] through sigmoid.

        Exploration noise is added after the Actor and then clipped
        back to the valid action range.
        """

        history = history.to(self.device)
        state = state.to(self.device)

        action = self.actor(
            history,
            state,
        )

        if explore and self.noise_std > 0.0:
            noise = torch.randn_like(action) * self.noise_std
            action = action + noise

        return torch.clamp(
            action,
            min=0.0,
            max=1.0,
        )

    # ------------------------------------------------------------------
    # Critic update
    # ------------------------------------------------------------------

    def update_critic(
        self,
        batch: ReplayBatch,
    ) -> float:
        """
        Perform one Critic update.

        Bellman target:

            y = r + gamma * (1-done) * Q_target(s', a')

        where:

            a' = Actor_target(s')
        """

        self.critic_optimizer.zero_grad()

        with torch.no_grad():

            next_action = self.target_actor(
                batch.next_history,
                batch.next_state,
            )

            target_q = self.target_critic(
                batch.next_history,
                batch.next_state,
                next_action,
            )

            y = batch.reward + (
                self.gamma
                * (1.0 - batch.done)
                * target_q
            )

        current_q = self.critic(
            batch.history,
            batch.state,
            batch.action,
        )

        critic_loss = self.mse_loss(
            current_q,
            y,
        )

        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=1.0,
        )

        self.critic_optimizer.step()

        return float(critic_loss.item())

    # ------------------------------------------------------------------
    # Actor update
    # ------------------------------------------------------------------

    def update_actor(
        self,
        batch: ReplayBatch,
    ) -> float:
        """
        Perform one Actor update.

        DDPG objective:

            maximize Q(s, Actor(s))

        Therefore the loss minimized by gradient descent is:

            L_actor = -mean(Q(s, Actor(s)))
        """

        self.actor_optimizer.zero_grad()

        action = self.actor(
            batch.history,
            batch.state,
        )

        actor_q = self.critic(
            batch.history,
            batch.state,
            action,
        )

        actor_loss = -actor_q.mean()

        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=1.0,
        )

        self.actor_optimizer.step()

        return float(actor_loss.item())

    # ------------------------------------------------------------------
    # Target-network soft update
    # ------------------------------------------------------------------

    def soft_update(
        self,
        source: nn.Module,
        target: nn.Module,
    ) -> None:
        """
        Soft-update target parameters:

            theta_target =
                tau * theta_source
                + (1-tau) * theta_target
        """

        with torch.no_grad():

            for target_parameter, source_parameter in zip(
                target.parameters(),
                source.parameters(),
            ):
                target_parameter.data.mul_(
                    1.0 - self.tau
                )

                target_parameter.data.add_(
                    self.tau * source_parameter.data
                )

    def update_targets(self) -> None:
        """
        Soft-update both target networks.
        """

        self.soft_update(
            self.actor,
            self.target_actor,
        )

        self.soft_update(
            self.critic,
            self.target_critic,
        )

    # ------------------------------------------------------------------
    # Complete training update
    # ------------------------------------------------------------------

    def update(
        self,
        batch: ReplayBatch,
    ) -> tuple[float, float]:
        """
        Perform one complete DDPG update.
        """

        critic_loss = self.update_critic(batch)

        actor_loss = self.update_actor(batch)

        self.update_targets()

        return actor_loss, critic_loss