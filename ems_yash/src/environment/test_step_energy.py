"""Physical work and endpoint-feasibility tests for a finite driving step."""

from dataclasses import asdict
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from .integrated_powertrain import (
    IntegratedPowertrain,
    IntegratedPowertrainParameters,
    default_motor_map_paths,
)
from .rl_environment import EnergyManagementEnv
from .vehicle_dynamics import VehicleParameters


def mechanical_work(result, dt):
    return dt * 2. * math.pi / 60. * (
        result.motor1_torque_nm * result.motor1_speed_rpm
        + result.motor2_torque_nm * result.motor2_speed_rpm
    )


def physical_state(plant):
    return asdict(plant.battery.state), plant.health_model.get_health_state(), plant.simulation_time_s


class StepEnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maps = default_motor_map_paths(Path(__file__).resolve().parents[2])

    def plant(self, **kwargs):
        return IntegratedPowertrain(*self.maps, **kwargs)

    def test_roadload_free_work_equals_kinetic_energy_change(self):
        for radius in (.325, .4):
            vehicle = VehicleParameters(wheel_radius_m=radius, rolling_resistance=0., drag_coefficient=0.)
            plant = self.plant(vehicle_parameters=vehicle,
                               parameters=IntegratedPowertrainParameters(wheel_radius_m=radius))
            for v0, v1, dt in ((0., 10., 1.), (10., 0., 1.), (30., 37., 2.5), (37., 30., 2.5)):
                # At radius .4 the launch needs sigma>=.507; all splits below
                # are feasible for every case and exercise split invariance.
                for sigma in (.55, .7, .85):
                    with self.subTest(radius=radius, v0=v0, v1=v1, dt=dt, sigma=sigma):
                        result = plant.step(v0, v1, sigma, dt_s=dt)
                        expected = .5 * vehicle.mass_kg * vehicle.rotational_mass_factor * ((v1 / 3.6)**2 - (v0 / 3.6)**2)
                        self.assertAlmostEqual(mechanical_work(result, dt), expected, delta=1e-8)
                        self.assertEqual(result.velocity_kmh, v0)
                        self.assertEqual(result.target_velocity_kmh, v1)
                        expected_rpm = plant.motors.calculate_motor_speed((v0 + v1) / 2., plant.motors.params.motor1.gear_ratio)
                        self.assertEqual(result.motor1_speed_rpm, expected_rpm)

    def test_existing_roadload_and_torque_are_preserved(self):
        plant = self.plant()
        v0, v1, dt, slope = 30., 37., 2.5, .01
        forces = plant.vehicle.calculate_required_force(v0, v1, slope, dt)
        expected_torque = plant.vehicle.calculate_wheel_torque(v0, v1, slope, dt)
        result = plant.step(v0, v1, .5, slope, dt)
        self.assertEqual(result.wheel_torque_nm, expected_torque)
        self.assertEqual(result.acceleration_mps2, forces["acceleration_mps2"])
        parameters = plant.vehicle.params
        kinetic_change = .5 * parameters.mass_kg * parameters.rotational_mass_factor * ((v1 / 3.6)**2 - (v0 / 3.6)**2)
        distance = (v0 + v1) / 2. / 3.6 * dt
        roadload_work = (forces["force_rolling_n"] + forces["force_aero_n"] + forces["force_grade_n"]) * distance
        self.assertAlmostEqual(mechanical_work(result, dt), kinetic_change + roadload_work, delta=1e-8)

    def test_actual_map_start_stop_consumes_net_energy(self):
        plant = self.plant()
        initial_soc = plant.battery.state.soc
        start = plant.step(0., 10., .5)
        stop = plant.step(10., 0., .5)
        self.assertGreater(start.motor1_speed_rpm, 0.)
        self.assertLess(start.battery_power_kw, 0.)
        self.assertLess(start.soc, initial_soc)
        self.assertGreater(stop.battery_power_kw, 0.)
        self.assertGreater(stop.soc, start.soc)
        self.assertLess(stop.soc, initial_soc)
        net_discharge_j = -(start.battery_power_kw + stop.battery_power_kw) * 1000.
        self.assertGreater(net_discharge_j, 0.)
        for result in (start, stop):
            self.assertAlmostEqual(result.battery_power_kw + result.motor_total_electrical_power_kw, 0.)

    def test_steady_cruise_matches_recorded_prechange_values(self):
        result = self.plant().step(50., 50., .5)
        # Captured before changing the integration rule, with shipped maps.
        expected = {
            "motor1_speed_rpm": 4488.985574386792,
            "motor2_speed_rpm": 2448.5375860291592,
            "battery_power_kw": -5.842281911661818,
            "battery_current_a": -22.87242140669737,
            "soc": .5999117576334618,
            "battery_soh": .9999998209052489,
        }
        for name, value in expected.items():
            with self.subTest(field=name):
                self.assertAlmostEqual(getattr(result, name), value, delta=abs(value) * 1e-12)

    def test_endpoint_overspeed_rejected_even_when_midpoint_is_feasible(self):
        plant = self.plant()
        motor = plant.motors.params.motor1
        speed_limit = motor.max_speed_rpm * 2. * math.pi / 60. * plant.parameters.wheel_radius_m * 3.6 / motor.gear_ratio
        for v0, v1 in ((speed_limit - 1., speed_limit + .5), (speed_limit + .5, speed_limit - 1.)):
            torque = plant.vehicle.calculate_wheel_torque(v0, v1, 0., 1.)
            self.assertTrue(plant.motors.calculate_operating_point((v0 + v1) / 2., torque, .5)["overall_feasible"])
            before = physical_state(plant)
            with self.assertRaisesRegex(ValueError, "endpoint.*infeasible"):
                plant.step(v0, v1, .5)
            self.assertEqual(physical_state(plant), before)

            cycle = np.array([[0., v0, 0.], [1., v1, 0.], [2., v1, 0.]])
            env = EnergyManagementEnv(cycle)
            self.addCleanup(env.close)
            initial_observation, _ = env.reset()
            before = physical_state(env.powertrain)
            reward_state = env._reward_state.copy()
            observation, reward, terminated, _, info = env.step([.5])
            self.assertTrue(terminated)
            self.assertEqual(reward, 0.)
            self.assertEqual(info["constraint_stage"], "pre_step")
            self.assertEqual(env.current_index, 0)
            self.assertEqual(env.current_velocity_kmh, v0)
            self.assertEqual(env.simulation_time_s, 0.)
            self.assertEqual(physical_state(env.powertrain), before)
            np.testing.assert_array_equal(observation, initial_observation)
            np.testing.assert_array_equal(env._reward_state, reward_state)
            with self.assertRaises(RuntimeError):
                env.step([.5])
            self.assertEqual(physical_state(env.powertrain), before)

    def test_configured_radius_is_shared_and_explicit_conflicts_fail_early(self):
        parameters = IntegratedPowertrainParameters(wheel_radius_m=.4)
        plant = self.plant(parameters=parameters)
        self.assertEqual(plant.vehicle.params.wheel_radius_m, .4)
        self.assertEqual(plant.motors.wheel_radius_m, .4)
        vehicle = VehicleParameters(wheel_radius_m=.4, rolling_resistance=.02)
        before = asdict(vehicle)
        plant = self.plant(parameters=parameters, vehicle_parameters=vehicle)
        self.assertEqual(asdict(vehicle), before)
        self.assertIsNot(plant.vehicle.params, vehicle)
        self.assertEqual(plant.vehicle.params.rolling_resistance, .02)
        with patch("src.environment.integrated_powertrain.MotorEfficiencyMap", side_effect=AssertionError("map load reached")):
            with self.assertRaisesRegex(ValueError, "wheel radius must match"):
                self.plant(parameters=parameters, vehicle_parameters=VehicleParameters())
            for radius in (0., -1., np.nan, np.inf):
                with self.subTest(radius=radius), self.assertRaisesRegex(ValueError, "radius"):
                    self.plant(parameters=IntegratedPowertrainParameters(wheel_radius_m=radius))

    def test_invalid_endpoint_velocities_reject_before_mutation(self):
        plant = self.plant()
        before = physical_state(plant)
        for speed in (-1., np.nan, np.inf):
            for v0, v1 in ((speed, 10.), (10., speed)):
                with self.subTest(v0=v0, v1=v1), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    plant.step(v0, v1, .5)
                self.assertEqual(physical_state(plant), before)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            EnergyManagementEnv(np.array([[0., 0., 0.], [1., -1., 0.]]))


if __name__ == "__main__":
    unittest.main()
