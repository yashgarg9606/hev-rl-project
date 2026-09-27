"""Regression tests for elapsed-time BMS sampling through the real powertrain."""

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.bms_interface import SOHSource
from src.bms_interface.soh_trace_adapter import SOHTraceAdapter
from .integrated_powertrain import (
    IntegratedPowertrain,
    IntegratedPowertrainParameters,
    default_motor_map_paths,
)
from .rl_environment import EnergyManagementEnv


class SOHClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.trace_path = Path(cls.directory.name) / "trace.npz"
        np.savez(cls.trace_path, time_seconds=[0.0, 1.0, 3.0],
                 soh_predicted=[0.95, 0.93, 0.91], soh_true=[0.96, 0.94, 0.92])
        cls.project_root = Path(__file__).resolve().parents[2]
        cls.maps = default_motor_map_paths(cls.project_root)
        # The cycle starts at 50 seconds; BMS time is elapsed episode time.
        cls.cycle = np.array([[50.0, 20.0, 0.0], [51.0, 20.0, 0.0],
                              [53.0, 20.0, 0.0], [54.0, 20.0, 0.0]])

    def parameters(self, estimated=True, adapter=None, **overrides):
        return IntegratedPowertrainParameters(
            soh_source=SOHSource.BMS_ESTIMATED if estimated else SOHSource.TRUE_PHYSICAL,
            soh_trace_adapter=adapter if adapter is not None else SOHTraceAdapter(self.trace_path),
            **overrides,
        )

    def plant(self, **kwargs):
        return IntegratedPowertrain(*self.maps, parameters=self.parameters(**kwargs))

    def env(self, **kwargs):
        env = EnergyManagementEnv(self.cycle, project_root=self.project_root,
                                  powertrain_parameters=self.parameters(**kwargs))
        self.addCleanup(env.close)
        return env

    def test_direct_plant_samples_end_of_variable_length_steps(self):
        plant = self.plant()
        self.assertEqual(plant.simulation_time_s, 0.0)
        for dt, expected_time, expected_soh in [(0.5, 0.5, 0.95), (0.5, 1.0, 0.93),
                                               (2.0, 3.0, 0.91), (1.0, 4.0, 0.91)]:
            with self.subTest(time=expected_time):
                # Existing positional call signature remains supported.
                result = plant.step(20.0, 20.0, 0.5, 0.0, dt)
                self.assertEqual(plant.simulation_time_s, expected_time)
                self.assertEqual(result.battery_soh, expected_soh)
                self.assertEqual(result.motor1_soh, plant.health_model.get_health_state()["motor1_soh"])

    def test_rejected_plant_steps_do_not_advance_time_or_physics(self):
        plant = self.plant()
        before_battery = asdict(plant.battery.state)
        before_health = plant.health_model.get_health_state()
        for dt in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                plant.step(20.0, 20.0, 0.5, dt_s=dt)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            plant.step(20.0, 100.0, 0.5)
        self.assertEqual(plant.simulation_time_s, 0.0)
        self.assertEqual(asdict(plant.battery.state), before_battery)
        self.assertEqual(plant.health_model.get_health_state(), before_health)
        self.assertEqual(plant.step(20.0, 20.0, 0.5).battery_soh, 0.93)
        self.assertEqual(plant.simulation_time_s, 1.0)

    def test_source_selection_preserves_same_action_physical_trajectory(self):
        true_plant = self.plant(estimated=False)
        estimated_plant = self.plant()
        initial_health = true_plant.health_model.get_health_state()["battery_soh"]
        for velocity, target, sigma, dt in [(20.0, 21.0, 0.2, 1.0),
                                             (21.0, 20.0, 0.8, 2.0),
                                             (20.0, 20.0, 0.5, 1.0)]:
            true_result = true_plant.step(velocity, target, sigma, dt_s=dt)
            estimated_result = estimated_plant.step(velocity, target, sigma, dt_s=dt)
            true_values, estimated_values = asdict(true_result), asdict(estimated_result)
            true_values.pop("battery_soh")
            estimated_values.pop("battery_soh")
            self.assertEqual(true_values, estimated_values)
            self.assertEqual(asdict(true_plant.battery.state), asdict(estimated_plant.battery.state))
            self.assertEqual(true_plant.health_model.get_health_state(), estimated_plant.health_model.get_health_state())
            self.assertEqual(true_result.battery_soh, true_plant.health_interface.get_true_battery_soh())
            self.assertEqual(estimated_plant.health_interface.get_true_battery_soh(), true_result.battery_soh)
            self.assertNotEqual(estimated_result.battery_soh, true_result.battery_soh)
        self.assertLess(true_plant.health_interface.get_true_battery_soh(), initial_health)

    def test_environment_end_step_observation_info_reward_state_and_reset(self):
        env = self.env()
        for episode in range(2):
            observation, info = env.reset()
            self.assertEqual(env.simulation_time_s, 0.0)
            self.assertEqual(env.powertrain.simulation_time_s, 0.0)
            self.assertEqual(info["time_s"], 50.0)
            self.assertEqual(info["battery_soh"], 0.95)
            self.assertEqual(observation[3], np.float32(0.95))
            for action, time, expected_soh in [(0.2, 1.0, 0.93), (0.5, 3.0, 0.91), (0.8, 4.0, 0.91)]:
                with self.subTest(episode=episode, time=time):
                    observation, _, terminated, truncated, info = env.step([action])
                    self.assertEqual(env.simulation_time_s, time)
                    self.assertEqual(env.powertrain.simulation_time_s, time)
                    self.assertEqual(info["battery_soh"], expected_soh)
                    self.assertEqual(env._reward_state[3], expected_soh)
                    self.assertEqual(observation[3], np.float32(expected_soh))
                    self.assertEqual(terminated, time == 4.0)
                    self.assertFalse(truncated)
                    self.assertFalse(info["constraint_violation"])
            with self.assertRaises(RuntimeError):
                env.step([0.5])
            self.assertEqual(env.simulation_time_s, 4.0)

    def test_shared_adapter_supports_interleaved_environments(self):
        adapter = SOHTraceAdapter(self.trace_path)
        first, second = self.env(adapter=adapter), self.env(adapter=adapter)
        first.reset()
        first.step([0.5])
        self.assertEqual(first.step([0.5])[4]["battery_soh"], 0.91)
        self.assertEqual(second.reset()[1]["battery_soh"], 0.95)
        self.assertEqual(second.step([0.5])[4]["battery_soh"], 0.93)
        self.assertEqual(first.step([0.5])[4]["battery_soh"], 0.91)

    def test_minimal_custom_trace_does_not_require_reset_method(self):
        class Trace:
            def get_soh_at_time(self, time):
                return 0.95 if time < 1.0 else 0.93

        env = self.env(adapter=Trace())
        for _ in range(2):
            self.assertEqual(env.reset()[1]["battery_soh"], 0.95)
            self.assertEqual(env.step([0.5])[4]["battery_soh"], 0.93)

    def test_pre_step_rejection_does_not_advance_either_clock(self):
        env = self.env(initial_soc=0.1)
        env.reset()
        before_health = env.powertrain.health_model.get_health_state()
        _, _, terminated, _, info = env.step([0.5])
        self.assertTrue(terminated)
        self.assertEqual(info["constraint_stage"], "pre_step")
        self.assertEqual(env.simulation_time_s, 0.0)
        self.assertEqual(env.powertrain.simulation_time_s, 0.0)
        self.assertEqual(env.powertrain.health_model.get_health_state(), before_health)

    def test_post_step_constraint_uses_new_trace_sample(self):
        class Trace:
            def get_soh_at_time(self, time):
                return 0.95 if time < 1.0 else 0.79

        env = self.env(adapter=Trace())
        env.reset()
        observation, _, terminated, _, info = env.step([0.5])
        self.assertTrue(terminated)
        self.assertTrue(info["constraint_violation"])
        self.assertIn("battery_soh", info["violated_constraints"])
        self.assertEqual(info["battery_soh"], 0.79)
        self.assertEqual(observation[3], np.float32(0.79))
        self.assertEqual(env.simulation_time_s, 1.0)
        self.assertEqual(env.powertrain.simulation_time_s, 1.0)
        with self.assertRaises(RuntimeError):
            env.step([0.5])
        self.assertEqual(env.powertrain.simulation_time_s, 1.0)


if __name__ == "__main__":
    unittest.main()
