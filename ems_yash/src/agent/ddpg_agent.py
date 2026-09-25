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

from .actor import Actor
from .critic import Critic
from .feasible_action_mapper import map_action
from .replay_buffer import ReplayBatch


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
    # Feasibility-aware action mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_sigma_min(
        velocity_kmh: torch.Tensor,
        wheel_torque_nm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the minimum feasible physical torque split.

        Parameters
        ----------
        velocity_kmh:
            [batch, 1] vehicle velocity.

        wheel_torque_nm:
            [batch, 1] wheel torque demand.

        Returns
        -------
        sigma_min:
            [batch, 1] minimum feasible physical sigma_tor.

        Notes
        -----
        The exact feasibility boundary is determined by the dual-motor
        physical constraints. This helper is intentionally kept separate
        from the Actor so the six-dimensional RL state remains unchanged.

        This method will be replaced by the validated vectorized
        feasibility calculation once the production environment interface
        is integrated.
        """

        velocity = velocity_kmh.detach().cpu().numpy()
        wheel_torque = wheel_torque_nm.detach().cpu().numpy()

        sigma_min_values = []

        # Import locally to avoid changing the module-level dependency
        # structure.
        from pathlib import Path
        from src.environment.integrated_powertrain import (
            IntegratedPowertrain,
            default_motor_map_paths,
        )

        project_root = Path(__file__).resolve().parents[2]

        motor1_map, motor2_map = default_motor_map_paths(
            project_root
        )

        powertrain = IntegratedPowertrain(
            motor1_map_path=motor1_map,
            motor2_map_path=motor2_map,
        )

        for v, torque in zip(
            velocity.reshape(-1),
            wheel_torque.reshape(-1),
        ):
            sigma_grid = np.linspace(
                0.0,
                1.0,
                10001,
                dtype=np.float64,
            )

            sigma_found = None

            for sigma in sigma_grid:
                result = powertrain.motors.calculate_operating_point(
                    velocity_kmh=float(v),
                    wheel_torque_nm=float(torque),
                    sigma_tor=float(sigma),
                )

                if result["overall_feasible"]:
                    sigma_found = float(sigma)
                    break

            if sigma_found is None:
                raise ValueError(
                    "No feasible sigma_tor exists for the sampled "
                    f"transition: velocity={float(v):.6f} km/h, "
                    f"wheel_torque={float(torque):.6f} Nm"
                )

            sigma_min_values.append(sigma_found)

        return torch.tensor(
            sigma_min_values,
            dtype=velocity_kmh.dtype,
            device=velocity_kmh.device,
        ).reshape(-1, 1)

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

            next_normalized_action = self.target_actor(
                batch.next_history,
                batch.next_state,
            )

            next_velocity = batch.next_state[:, 0:1]
            next_wheel_torque = batch.next_state[:, 1:2]

            next_sigma_min = self._calculate_sigma_min(
                velocity_kmh=next_velocity,
                wheel_torque_nm=next_wheel_torque,
            )

            next_physical_action = map_action(
                next_normalized_action,
                next_sigma_min,
            )

            target_q = self.target_critic(
                batch.next_history,
                batch.next_state,
                next_physical_action,
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

        normalized_action = self.actor(
            batch.history,
            batch.state,
        )

        current_velocity = batch.state[:, 0:1]
        current_wheel_torque = batch.state[:, 1:2]

        sigma_min = self._calculate_sigma_min(
            velocity_kmh=current_velocity,
            wheel_torque_nm=current_wheel_torque,
        )

        physical_action = map_action(
            normalized_action,
            sigma_min,
        )

        actor_q = self.critic(
            batch.history,
            batch.state,
            physical_action,
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