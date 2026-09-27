"""Legacy entrypoints must obey the current action, timing and logging contracts."""

import contextlib
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from ems_yash import baseline_compare_soh, baseline_true_soh, run_phase3b_experiment
from src.agent.health_aware_policy import HealthAwareDeterministicPolicy
from src.bms_interface import SOHSource
from src.data.velocity_cycle_input import export_from_cli, load_velocity_cycle
from src.environment.integrated_powertrain import IntegratedPowertrainParameters
from src.environment.rl_environment import EnergyManagementEnv


class FixedPolicy:
    def select_action(self, state):
        return np.array([.5], dtype=np.float32)


class LegacyContractTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()

    def test_comparison_logs_executed_sigma_and_valid_completion(self):
        probe = EnergyManagementEnv(np.array([[0., 50., 0.], [1., 50., 0.]]))
        vehicle = probe.powertrain.vehicle
        steady = vehicle.calculate_wheel_torque(50., 50., 0., 1.)
        params = vehicle.params
        target = 50. + (2500. - steady) * 3.6 / (params.mass_kg * params.rotational_mass_factor * params.wheel_radius_m)
        env = EnergyManagementEnv(np.array([[0., 50., 0.], [1., target, 0.]]))
        with contextlib.redirect_stdout(self.output):
            result = baseline_compare_soh.run_episode(env, FixedPolicy(), "TRUE")
        self.assertFalse(result['constraint_violation'])
        self.assertTrue(result['completed_cycle'])
        self.assertEqual(result['soh_history']['normalized_action'][1], .5)
        self.assertAlmostEqual(result['soh_history']['sigma_tor'][1], .710, places=12)

    def test_comparison_preserves_actual_rejection_flag(self):
        env = EnergyManagementEnv(np.array([[0., 20., 0.], [1., 20., 0.]]),
            powertrain_parameters=IntegratedPowertrainParameters(initial_soc=.19))
        with contextlib.redirect_stdout(self.output):
            result = baseline_compare_soh.run_episode(env, FixedPolicy(), "TRUE")
        self.assertTrue(result['constraint_violation'])
        self.assertFalse(result['completed_cycle'])
        self.assertEqual(env.simulation_time_s, 0.)
        self.assertTrue(np.isnan(result['soh_history']['sigma_tor'][1]))

    def test_baseline_final_state_and_variable_dt_use_completed_interval(self):
        cycle = np.array([[50., 20., 0.], [52., 20., 0.]])
        expected_env = EnergyManagementEnv(cycle)
        expected_env.reset()
        _, _, _, _, info = expected_env.step([.5])
        with patch.object(baseline_true_soh, 'build_wltp_class3_excerpt', return_value=cycle), \
             contextlib.redirect_stdout(self.output):
            frame, metrics = baseline_true_soh.run_baseline_experiment(use_random_actions=False)
        self.assertEqual(metrics['final_soc'], expected_env.powertrain.battery.state.soc)
        self.assertEqual(metrics['final_battery_soh'], expected_env.powertrain.health_model.battery_state.soh)
        self.assertAlmostEqual(metrics['total_battery_energy_kwh'], -info['battery_power_kw'] * 2. / 3600.)
        self.assertEqual(frame['completed_dt_s'].iloc[0], 2.)
        self.assertEqual(frame['sigma_tor'].iloc[0], info['sigma_tor'])
        self.assertTrue(metrics['completed_cycle'])

    def test_phase3b_uses_environment_transitions_and_paired_trace_targets(self):
        class Trace:
            def get_soh_at_time(self, time):
                return .95 if time < 2. else .94
            def get_true_soh_at_time(self, time):
                return .96 if time < 2. else .93
        trace = Trace()
        cycle = np.array([[50., 20., 0.], [51., 21., 0.], [53., 23., 0.]])
        with contextlib.redirect_stdout(self.output):
            actual = run_phase3b_experiment.run_experiment(SOHSource.BMS_ESTIMATED, trace, cycle)
        env = EnergyManagementEnv(cycle, powertrain_parameters=IntegratedPowertrainParameters(
            soh_source=SOHSource.BMS_ESTIMATED, soh_trace_adapter=trace))
        state, _ = env.reset()
        policy = HealthAwareDeterministicPolicy()
        for index in (1, 2):
            action = policy.select_action(state)
            state, reward, _, _, info = env.step(action)
            self.assertEqual(actual['normalized_action'][index], float(action[0]))
            self.assertEqual(actual['sigma_tor'][index], info['sigma_tor'])
            self.assertEqual(actual['reward'][index], reward)
            self.assertEqual(actual['soc'][index], env.powertrain.battery.state.soc)
        np.testing.assert_array_equal(actual['time_s'], [0., 1., 3.])
        np.testing.assert_array_equal(actual['trace_true_soh'], [.96, .96, .93])
        self.assertTrue(actual['completed_cycle'])
        self.assertEqual(actual['completed_transitions'], 2)

    def test_policy_contract_and_nonfinite_sensitivity(self):
        policy = HealthAwareDeterministicPolicy()
        value = policy.select_action(np.array([50., 2500., .6, .8, 1., 1.]))
        self.assertAlmostEqual(float(value[0]), .56, places=6)
        for invalid in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                HealthAwareDeterministicPolicy(soh_sensitivity=invalid)

    def test_external_cycle_input_is_explicit_validated_and_never_overwritten(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            source, output = directory / 'source.txt', directory / 'new.csv'
            with self.assertRaisesRegex(FileNotFoundError, '--source'):
                load_velocity_cycle(source, 3, 120.)
            source.write_text('0 10 0')
            frame = load_velocity_cycle(source, 3, 120.)
            np.testing.assert_array_equal(frame['time_s'], [0., 1., 2.])
            with patch.object(sys, 'argv', ['extract', '--source', str(source), '--output', str(output)]), \
                 contextlib.redirect_stdout(self.output):
                export_from_cli('fixture', 'source.txt', 3, 120.)
                before = output.read_bytes()
                with contextlib.redirect_stderr(self.output), self.assertRaises(SystemExit):
                    export_from_cli('fixture', 'source.txt', 3, 120.)
                self.assertEqual(output.read_bytes(), before)
            for content in ('0 nan 0', '0 -1 0', '0 121 0', '0 0'):
                source.write_text(content)
                with self.assertRaises(ValueError):
                    load_velocity_cycle(source, 3, 120.)

    def test_baseline_entrypoints_refuse_existing_output_before_running(self):
        with TemporaryDirectory() as directory:
            for module in (baseline_true_soh, baseline_compare_soh):
                with self.subTest(module=module.__name__), \
                     patch.object(sys, 'argv', ['baseline', '--output-dir', directory]), \
                     contextlib.redirect_stderr(self.output):
                    with self.assertRaises(SystemExit):
                        module.main()


if __name__ == '__main__':
    unittest.main()
