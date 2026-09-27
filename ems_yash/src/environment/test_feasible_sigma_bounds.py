"""Analytic motor bounds and environment-only action mapping regressions."""

from dataclasses import asdict
import unittest

import numpy as np
import torch

from src.agent.feasible_action_mapper import map_action
from .constraints import PhysicalConstraintChecker
from .motor_model import DualMotorModel, DualMotorParameters, MotorParameters
from .rl_environment import EnergyManagementEnv
from .vehicle_dynamics import VehicleDynamics


def cycle_for_demand(torque, velocity=50.):
    vehicle = VehicleDynamics()
    base = vehicle.calculate_wheel_torque(velocity, velocity, 0., 1.)
    gain = vehicle.calculate_wheel_torque(velocity, velocity + 1., 0., 1.) - base
    target = velocity + (torque - base) / gain
    return np.array([[0., velocity, 0.], [1., target, 0.], [2., target, 0.]])


class FeasibleSigmaTests(unittest.TestCase):
    def test_full_interval_both_signs_zero_singleton_and_empty(self):
        model = DualMotorModel()
        for sign in (-1, 1):
            lower, upper = model.calculate_feasible_sigma_bounds(50., sign * 2500.)
            self.assertAlmostEqual(lower, .628)
            self.assertAlmostEqual(upper, .792)
            self.assertAlmostEqual(float(map_action(.5, lower, upper)), .710)
            lower, upper = model.calculate_feasible_sigma_bounds(50., sign * 2910.)
            self.assertAlmostEqual(lower, upper)
            self.assertTrue(model.calculate_operating_point(50., sign * 2910., lower)["overall_feasible"])
            self.assertIsNone(model.calculate_feasible_sigma_bounds(50., sign * 2910.001))
        self.assertEqual(model.calculate_feasible_sigma_bounds(50., 0.), (0., 1.))
        self.assertIsNone(model.calculate_feasible_sigma_bounds(140., 0.))
        self.assertIsNone(model.calculate_feasible_sigma_bounds(-140., 2500.))

    def test_numerical_endpoints_agree_with_motor_and_constraint_checks(self):
        model = DualMotorModel()
        checker = PhysicalConstraintChecker()
        demands = list(np.linspace(1., 2910., 31)) + [1046.9223922392239]
        for torque in demands:
            for sign in (-1, 1):
                for sigma in model.calculate_feasible_sigma_bounds(50., sign * torque):
                    with self.subTest(torque=sign * torque, sigma=sigma):
                        op = model.calculate_operating_point(50., sign * torque, sigma)
                        self.assertTrue(op["overall_feasible"])
                        result = checker.check(soc=.6, battery_soh=1., motor1_soh=1., motor2_soh=1.,
                            **{name: op[name] for name in ("motor1_speed_rpm", "motor2_speed_rpm", "motor1_torque_nm", "motor2_torque_nm")})
                        self.assertTrue(result.overall_valid)
        # Tolerance must not turn a material excess into a feasible point.
        self.assertFalse(model.calculate_operating_point(50., 11. * (180. + 1e-6), 1.)["overall_feasible"])
        self.assertFalse(checker.check(soc=.6, battery_soh=1., motor1_soh=1., motor2_soh=1.,
            motor1_speed_rpm=100., motor2_speed_rpm=100., motor1_torque_nm=180. + 1e-6,
            motor2_torque_nm=0.).overall_valid)

    def test_numpy_torch_mapping_upper_bounds_and_gradients(self):
        np.testing.assert_allclose(map_action(np.array([0., .5, 1.]), .628, .792), [.628, .710, .792])
        action = torch.tensor([0., .5, 1.], dtype=torch.float64, requires_grad=True)
        mapped = map_action(action, .628, .792)
        mapped.sum().backward()
        torch.testing.assert_close(action.grad, torch.full_like(action, .164))
        self.assertEqual(float(map_action(.5, .7, .7)), .7)
        self.assertEqual(float(map_action(1., .4)), 1.)  # Legacy two-argument API.
        for lower, upper in ((.8, .7), (-.1, .8), (.2, 1.1), (.2, np.nan)):
            for action in (.5, torch.tensor(.5)):
                with self.subTest(lower=lower, upper=upper), self.assertRaises(ValueError):
                    map_action(action, lower, upper)

    def test_custom_motor_limits_gearing_and_radius(self):
        parameters = DualMotorParameters(
            MotorParameters(300., 9000., 28., .98, 8.),
            MotorParameters(220., 7000., 24., .98, 5.),
        )
        model = DualMotorModel(parameters, wheel_radius_m=.4)
        lower, upper = model.calculate_feasible_sigma_bounds(20., 2500.)
        self.assertAlmostEqual(lower, .56)
        self.assertAlmostEqual(upper, .96)
        env = EnergyManagementEnv(cycle_for_demand(2500., 20.), motor_parameters=parameters)
        self.addCleanup(env.close)
        env.reset()
        _, _, _, _, info = env.step([1.])
        self.assertAlmostEqual(info["sigma_min"], lower)
        self.assertAlmostEqual(info["sigma_max"], upper)
        self.assertFalse(info["constraint_violation"])
        self.assertAlmostEqual(env.powertrain.motors.params.motor1.max_torque_nm, 300.)
        self.assertAlmostEqual(env.constraint_checker.parameters.motor1_max_torque_nm, 300.)
        self.assertEqual(env.constraint_checker.parameters.motor1_max_speed_rpm, 9000.)

    def test_environment_executes_both_full_interval_endpoints(self):
        for torque in (-2500., 2500., 2910.):
            for action in (0., .5, 1.):
                with self.subTest(torque=torque, action=action):
                    env = EnergyManagementEnv(cycle_for_demand(torque))
                    self.addCleanup(env.close)
                    env.reset()
                    _, _, _, _, info = env.step([action])
                    self.assertFalse(info["constraint_violation"])
                    self.assertTrue(info["overall_feasible"])
                    self.assertAlmostEqual(info["sigma_tor"], info["sigma_min"] + action * (info["sigma_max"] - info["sigma_min"]))

    def test_empty_interval_terminates_without_state_or_clock_mutation(self):
        for cycle in (cycle_for_demand(4000.), np.array([[0., 140., 0.], [1., 140., 0.]])):
            env = EnergyManagementEnv(cycle)
            self.addCleanup(env.close)
            before, _ = env.reset()
            battery = asdict(env.powertrain.battery.state)
            health = env.powertrain.health_model.get_health_state()
            observation, reward, terminated, truncated, info = env.step([.5])
            np.testing.assert_array_equal(observation, before)
            self.assertEqual(reward, 0.)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info["constraint_stage"], "pre_step")
            self.assertIn("motor_feasibility", info["violated_constraints"])
            self.assertIsNone(info["sigma_tor"])
            self.assertEqual(env.current_index, 0)
            self.assertEqual(env.simulation_time_s, 0.)
            self.assertEqual(env.powertrain.simulation_time_s, 0.)
            self.assertEqual(asdict(env.powertrain.battery.state), battery)
            self.assertEqual(env.powertrain.health_model.get_health_state(), health)
            with self.assertRaises(RuntimeError):
                env.step([.5])
            env.reset()  # Clears the terminal latch for another episode.
            self.assertTrue(env.step([.5])[2])


if __name__ == "__main__":
    unittest.main()
