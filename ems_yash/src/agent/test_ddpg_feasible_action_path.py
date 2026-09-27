"""Verify the normalized action contract using actual DDPG update methods."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.ddpg_agent import DDPGAgent
from src.agent.replay_buffer import ReplayBatch


class NormalizedDDPGTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.agent = DDPGAgent(device="cpu")
        # Deterministic, differentiable networks: actor a=.5 and critic Q=a.
        with torch.no_grad():
            for parameter in self.agent.actor.parameters():
                parameter.zero_()
            for parameter in self.agent.critic.parameters():
                parameter.zero_()
            self.agent.critic.fc1.weight[0, -1] = 1.0
            self.agent.critic.fc2.weight[0, 0] = 1.0
            self.agent.critic.fc3.weight[0, 0] = 1.0
            self.agent.critic.output.weight[0, 0] = 1.0
        self.agent.target_actor.load_state_dict(self.agent.actor.state_dict())
        self.agent.target_critic.load_state_dict(self.agent.critic.state_dict())
        state = torch.tensor([[50., 2500., .6, 1., 1., 1.]]).repeat(2, 1)
        next_state = state.clone()
        next_state[1, :2] = torch.tensor([140., 100000.])  # Terminal, physically infeasible.
        self.batch = ReplayBatch(
            history=torch.zeros(2, 2, 7), state=state,
            action=torch.tensor([[.2], [.8]]), reward=torch.tensor([[1.], [2.]]),
            next_state=next_state, next_history=torch.zeros(2, 2, 7),
            done=torch.tensor([[0.], [1.]]),
        )

    def record_inputs(self, module):
        calls = []
        handle = module.register_forward_pre_hook(
            lambda _, args: calls.append(tuple(value.detach().clone() for value in args))
        )
        self.addCleanup(handle.remove)
        return calls

    def test_current_and_target_critics_receive_normalized_actions(self):
        current_calls = self.record_inputs(self.agent.critic)
        target_calls = self.record_inputs(self.agent.target_critic)
        loss_calls = self.record_inputs(self.agent.mse_loss)
        loss = self.agent.update_critic(self.batch)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        torch.testing.assert_close(current_calls[0][2], self.batch.action)
        # The physical split for a=.5 at Td=2500 is .710, not .5.
        torch.testing.assert_close(target_calls[0][2], torch.tensor([[.5]]))
        self.assertEqual(target_calls[0][1].shape[0], 1)
        torch.testing.assert_close(loss_calls[0][1], torch.tensor([[1.495], [2.]]))

    def test_actor_gradient_uses_normalized_action(self):
        calls = self.record_inputs(self.agent.critic)
        before = self.agent.select_action(self.batch.history, self.batch.state, explore=False)
        self.assertEqual(self.agent.update_actor(self.batch), -.5)
        torch.testing.assert_close(calls[0][2], before)
        after = self.agent.select_action(self.batch.history, self.batch.state, explore=False)
        self.assertTrue(torch.all(after > before))
        self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0) for p in self.agent.actor.parameters()))

    def test_all_terminal_targets_equal_rewards_without_network_evaluation(self):
        self.batch.done.fill_(1.)
        self.batch.next_state.fill_(float("nan"))
        loss_calls = self.record_inputs(self.agent.mse_loss)
        with patch.object(self.agent.target_actor, "forward", side_effect=AssertionError("terminal actor called")), \
             patch.object(self.agent.target_critic, "forward", side_effect=AssertionError("terminal critic called")):
            self.agent.update_critic(self.batch)
        torch.testing.assert_close(loss_calls[0][1], self.batch.reward, rtol=0, atol=0)

    def test_learning_never_constructs_physics_or_maps_actions(self):
        with patch("src.environment.integrated_powertrain.IntegratedPowertrain.__init__", side_effect=AssertionError("plant constructed")), \
             patch("src.environment.motor_model.DualMotorModel.calculate_feasible_sigma_bounds", side_effect=AssertionError("physics queried")), \
             patch("src.agent.feasible_action_mapper.map_action", side_effect=AssertionError("action mapped")):
            self.agent.update_critic(self.batch)
            self.agent.update_actor(self.batch)

    def test_network_checkpoint_shapes_remain_loadable(self):
        from src.agent.actor import Actor
        from src.agent.critic import Critic
        # These constructors are unchanged; no feasibility features were added.
        actor = Actor(hidden_dim=50)
        critic = Critic(hidden_dim=50)
        actor.load_state_dict(self.agent.actor.state_dict(), strict=True)
        critic.load_state_dict(self.agent.critic.state_dict(), strict=True)
        torch.testing.assert_close(actor(self.batch.history, self.batch.state), self.agent.actor(self.batch.history, self.batch.state))


def main():
    unittest.main()


if __name__ == "__main__":
    main()
