"""Replay retains normalized actions while the environment executes sigma_tor."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.ddpg_agent import DDPGAgent
from src.agent.replay_buffer import HistoricalReplayBuffer
from src.environment.rl_environment import EnergyManagementEnv
from src.environment.vehicle_dynamics import VehicleDynamics


class NormalizedReplayTests(unittest.TestCase):
    def test_executed_physical_split_does_not_replace_replay_or_history_action(self):
        vehicle = VehicleDynamics()
        base = vehicle.calculate_wheel_torque(50., 50., 0., 1.)
        gain = vehicle.calculate_wheel_torque(50., 51., 0., 1.) - base
        target = 50. + (2500. - base) / gain
        cycle = np.array([[0., 50., 0.], [1., target, 0.], [2., target, 0.]])
        env = EnergyManagementEnv(cycle)
        self.addCleanup(env.close)
        state, _ = env.reset()
        history = np.zeros((2, 7), dtype=np.float32)
        action = np.array([.5], dtype=np.float32)
        next_state, reward, terminated, truncated, info = env.step(action)
        self.assertAlmostEqual(info["sigma_min"], .628)
        self.assertAlmostEqual(info["sigma_max"], .792)
        self.assertAlmostEqual(info["sigma_tor"], .710)
        self.assertEqual(info["normalized_action"], .5)
        self.assertFalse(info["constraint_violation"])
        next_history = np.vstack([history[1:], np.concatenate([state, action])])
        replay = HistoricalReplayBuffer(capacity=2)
        replay.add(history, state, action, reward, next_state, next_history, terminated or truncated)
        self.assertEqual(replay.actions[0, 0], .5)
        self.assertEqual(replay.next_histories[0, -1, -1], .5)
        self.assertNotAlmostEqual(float(replay.actions[0, 0]), info["sigma_tor"])
        batch = replay.sample(1)
        torch.testing.assert_close(batch.action, torch.tensor([[.5]]))
        torch.testing.assert_close(batch.next_history[0, -1, -1], torch.tensor(.5))
        agent = DDPGAgent()
        self.assertTrue(np.isfinite(agent.update_critic(batch)))
        self.assertTrue(np.isfinite(agent.update_actor(batch)))


def main():
    unittest.main()


if __name__ == "__main__":
    main()
