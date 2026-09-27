"""Failed transitions must not leave partially advanced physical or RL state."""

from dataclasses import asdict, replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from .health_model import HealthDegradationModel, MotorHealthParameters
from .integrated_powertrain import IntegratedPowertrain, IntegratedPowertrainParameters, default_motor_map_paths
from .motor_model import DualMotorModel, DualMotorParameters
from .rl_environment import EnergyManagementEnv
from ..bms_interface import SOHSource


def plant_state(plant):
    return (asdict(plant.battery.state), asdict(plant.health_model.battery_state),
            asdict(plant.health_model.motor1_state), asdict(plant.health_model.motor2_state),
            plant.simulation_time_s)


class AtomicStepTests(unittest.TestCase):
    def plant(self, **kwargs):
        return IntegratedPowertrain(*default_motor_map_paths(Path(__file__).resolve().parents[2]), **kwargs)

    def test_unsupported_future_observation_rejected_before_episode(self):
        with self.assertRaisesRegex(ValueError, "transition 1.*observation space"):
            EnergyManagementEnv(np.array([[0., 0., 0.], [1., 10., 0.], [2., 50., 0.]]))

    def test_late_health_update_failure_restores_original_objects_and_retry(self):
        for method in ("update_battery", "update_motor1", "update_motor2"):
            with self.subTest(method=method):
                plant = self.plant()
                before = plant_state(plant)
                references = (plant.battery.state, plant.health_model.battery_state,
                              plant.health_model.motor1_state, plant.health_model.motor2_state)
                original = getattr(plant.health_model, method)
                def fail(*args, **kwargs):
                    original(*args, **kwargs)
                    raise RuntimeError("injected after mutation")
                with patch.object(plant.health_model, method, side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, "injected"):
                        plant.step(20., 20., .5)
                self.assertEqual(plant_state(plant), before)
                restored = (plant.battery.state, plant.health_model.battery_state,
                            plant.health_model.motor1_state, plant.health_model.motor2_state)
                self.assertTrue(all(left is right for left, right in zip(references, restored)))
                self.assertEqual(asdict(plant.step(20., 20., .5)), asdict(self.plant().step(20., 20., .5)))

    def test_external_lookup_failure_restores_owned_state_and_can_retry(self):
        class Trace:
            fail = True
            def get_soh_at_time(self, time):
                if time > 0 and self.fail:
                    raise RuntimeError("trace unavailable")
                return .95
        trace = Trace()
        params = IntegratedPowertrainParameters(soh_source=SOHSource.BMS_ESTIMATED, soh_trace_adapter=trace)
        plant = self.plant(parameters=params)
        before = plant_state(plant)
        with self.assertRaisesRegex(RuntimeError, "trace unavailable"):
            plant.step(20., 20., .5)
        self.assertEqual(plant_state(plant), before)
        trace.fail = False
        self.assertEqual(asdict(plant.step(20., 20., .5)), asdict(self.plant(parameters=params).step(20., 20., .5)))

    def test_environment_downstream_failures_roll_back_and_retry(self):
        cycle = np.array([[50., 20., 0.], [51., 21., 0.], [53., 23., 0.]])
        reference = EnergyManagementEnv(cycle)
        reference.reset()
        expected = reference.step([.5])
        for component, method in (("reward_model", "compute"), (None, "_build_state"),
                                  ("constraint_checker", "check")):
            env = EnergyManagementEnv(cycle)
            state, _ = env.reset()
            physical, reward = plant_state(env.powertrain), env._reward_state.copy()
            owner = getattr(env, component) if component else env
            original = getattr(owner, method)
            count = 0
            def fail(*args, **kwargs):
                nonlocal count
                count += 1
                if component == "constraint_checker" and count == 1:
                    return original(*args, **kwargs)
                raise RuntimeError("downstream failure")
            with patch.object(owner, method, side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "downstream"):
                    env.step([.5])
            self.assertEqual(plant_state(env.powertrain), physical)
            self.assertEqual((env.current_index, env.current_velocity_kmh, env.simulation_time_s, env._terminated), (0, 20., 0., False))
            np.testing.assert_array_equal(env.current_state, state)
            np.testing.assert_array_equal(env._reward_state, reward)
            actual = env.step([.5])
            np.testing.assert_array_equal(actual[0], expected[0])
            self.assertEqual(actual[1:], expected[1:])

    def test_singular_or_nonfinite_motor_lifetime_configuration_rejected(self):
        for changes in ({"rated_efficiency": 1.}, {"rated_efficiency": np.nan},
                        {"rated_power_kw": np.inf}, {"life_hours": np.nan},
                        {"rated_power_kw": 1e308, "life_hours": 1e308},
                        {"rated_power_kw": 1e-300, "life_hours": 1e-300}):
            parameters = replace(MotorHealthParameters(28., .98), **changes)
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    HealthDegradationModel(motor1_parameters=parameters)
                with self.assertRaises(ValueError):
                    HealthDegradationModel.motor_lifetime_energy_loss_kwh(parameters)

    def test_zero_and_near_zero_motor_capacity_bounds_execute(self):
        for tiny in (0., 1e-15, 1e-10):
            for reverse in (False, True):
                parameters = DualMotorParameters.from_wu_paper()
                if reverse:
                    parameters.motor2.max_torque_nm = tiny
                else:
                    parameters.motor1.max_torque_nm = tiny
                model = DualMotorModel(parameters)
                total = sum(m.max_torque_nm * m.gear_ratio for m in (parameters.motor1, parameters.motor2))
                for demand in (total, np.nextafter(total, 0.), np.nextafter(total, np.inf), total * (1. + 5e-13)):
                    for sign in (-1., 1.):
                        bounds = model.calculate_feasible_sigma_bounds(20., sign * demand)
                        if demand <= total:
                            self.assertIsNotNone(bounds)
                            if tiny == 0.:
                                expected = 1. if reverse else 0.
                                self.assertEqual(bounds, (expected, expected))
                        if bounds is not None:
                            for sigma in (*bounds, sum(bounds) / 2.):
                                self.assertTrue(model.calculate_operating_point(20., sign * demand, sigma)["overall_feasible"])
                self.assertIsNone(model.calculate_feasible_sigma_bounds(20., total * (1. + 1e-7)))


if __name__ == "__main__":
    unittest.main()
