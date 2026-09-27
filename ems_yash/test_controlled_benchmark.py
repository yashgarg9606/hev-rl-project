"""Controlled benchmark contracts tested on short synthetic driving cycles."""

from dataclasses import asdict
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from ems_yash import run_controlled_benchmark as benchmark
from src.environment.rl_environment import EnergyManagementEnv


def env_snapshot(env):
    return dict(index=env.current_index, env_time=env.simulation_time_s,
                plant_time=env.powertrain.simulation_time_s,
                battery=asdict(env.powertrain.battery.state),
                health=env.powertrain.health_model.get_health_state(),
                state=env.current_state.tolist(), reward_state=env._reward_state.tolist())


class ControlledBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.cycle = np.array([[50., 20., 0.], [51., 20., 0.], [53., 20., 0.]])
        self.fixed = benchmark.CONTROLLERS[0]

    def env(self, cycle):
        env = EnergyManagementEnv(cycle)
        self.addCleanup(env.close)
        env.reset()
        return env

    def test_grid_objective_and_execution_match_without_state_mutation(self):
        for v0, v1 in ((0., 10.), (20., 30.), (30., 20.)):
            with self.subTest(v0=v0, v1=v1):
                env = self.env(np.array([[0., v0, 0.], [1., v1, 0.]]))
                before = env_snapshot(env)
                action, predicted = benchmark.grid_action(env)
                self.assertEqual(action.dtype, np.float32)
                self.assertIn(action[0], benchmark.GRID_ACTIONS)
                for endpoint in (0., .5, 1.):
                    self.assertLessEqual(predicted, benchmark.candidate_motor_power(env, endpoint) + 1e-12)
                self.assertEqual(env_snapshot(env), before)
                _, _, _, _, info = env.step(action)
                self.assertAlmostEqual(predicted, -info["battery_power_kw"], delta=1e-12)
                self.assertFalse(info["constraint_violation"])

    def test_grid_uses_first_argmin_for_zero_power_ties(self):
        env = self.env(np.array([[0., 0., 0.], [1., 0., 0.]]))
        action, predicted = benchmark.grid_action(env)
        self.assertEqual(action[0], 0.)
        self.assertEqual(predicted, 0.)

    def test_bias_changes_only_copied_observation(self):
        env = self.env(self.cycle)
        state = env.current_state.copy()
        before = env_snapshot(env)
        action, observed, _ = benchmark.select_action(env, state, benchmark.CONTROLLERS[3])
        np.testing.assert_array_equal(state, env.current_state)
        np.testing.assert_array_equal(observed[[0, 1, 2, 4, 5]], state[[0, 1, 2, 4, 5]])
        self.assertEqual(observed[3], np.float32(.95))
        self.assertEqual(action[0], np.float32(.5 + .3 * (1. - float(np.float32(.95)))))
        self.assertEqual(env_snapshot(env), before)
        for bias in (-1.1, .1, np.nan):
            with self.subTest(bias=bias), self.assertRaises(ValueError):
                benchmark.observed_state(state, bias)

    def test_grid_prediction_mismatch_fails_instead_of_recording_success(self):
        with patch.object(benchmark, "grid_action", return_value=(np.array([.5], dtype=np.float32), 1000.)):
            with self.assertRaisesRegex(RuntimeError, "does not match executed"):
                benchmark.run_episode(self.cycle, benchmark.CONTROLLERS[-1])

    def test_zero_bias_equivalence_and_same_action_physical_invariance(self):
        oracle, _ = benchmark.run_episode(self.cycle, benchmark.CONTROLLERS[1])
        zero, _ = benchmark.run_episode(self.cycle, benchmark.Controller("zero_bias", "health_rule", 0.))
        self.assertEqual(oracle, zero)
        true_rows, true_summary = benchmark.run_episode(self.cycle, self.fixed)
        biased_rows, biased_summary = benchmark.run_episode(self.cycle, benchmark.Controller("fixed_biased", "fixed", -.5))
        # The artificial observation is below the health constraint, yet fixed
        # actions leave physical evolution, constraints, and rewards identical.
        self.assertTrue(true_summary["completed_cycle"])
        self.assertTrue(biased_summary["completed_cycle"])
        for true_row, biased_row in zip(true_rows, biased_rows):
            self.assertLess(biased_row["observed_battery_soh"], .8)
            true_row, biased_row = dict(true_row), dict(biased_row)
            true_row.pop("observed_battery_soh")
            biased_row.pop("observed_battery_soh")
            self.assertEqual(true_row, biased_row)

    def test_variable_timestep_energy_distance_and_current_aggregation(self):
        rows = [
            dict(completed_step=True, dt_completed_s=2., battery_power_kw=-4., battery_current_a=-2.,
                 velocity_before_kmh=10., target_velocity_kmh=20.),
            dict(completed_step=True, dt_completed_s=3., battery_power_kw=1., battery_current_a=4.,
                 velocity_before_kmh=20., target_velocity_kmh=0.),
            dict(completed_step=False, dt_completed_s=0., battery_power_kw=None, battery_current_a=None,
                 velocity_before_kmh=0., target_velocity_kmh=100.),
        ]
        result = benchmark.aggregate_steps(rows)
        self.assertEqual(result["attempted_transitions"], 3)
        self.assertEqual(result["completed_transitions"], 2)
        self.assertEqual(result["elapsed_s"], 5.)
        self.assertAlmostEqual(result["distance_km"], 60. / 3600.)
        self.assertAlmostEqual(result["discharge_kwh"], 8. / 3600.)
        self.assertAlmostEqual(result["recovered_kwh"], 3. / 3600.)
        self.assertAlmostEqual(result["net_kwh"], 5. / 3600.)
        self.assertEqual(result["peak_abs_current_a"], 4.)
        self.assertAlmostEqual(result["rms_current_a"], np.sqrt(56. / 5.))

    def test_pre_and_post_constraint_rejections_have_distinct_accounting(self):
        cycle = np.array([[0., 50., 0.], [1., 50., 0.], [2., 50., 0.]])
        before_rows, before = benchmark.run_episode(cycle, self.fixed, initial_soc=.1)
        self.assertEqual(before["attempted_transitions"], 1)
        self.assertEqual(before["completed_transitions"], 0)
        self.assertEqual(before["elapsed_s"], 0.)
        self.assertEqual(before["net_kwh"], 0.)
        self.assertIsNone(before["rms_current_a"])
        self.assertIsNone(before_rows[0]["battery_power_kw"])
        self.assertIsNone(before_rows[0]["battery_current_a"])
        self.assertEqual(before_rows[0]["cycle_index_before"], before_rows[0]["cycle_index_after"])
        self.assertEqual(before["constraint_stage"], "pre_step")
        after_rows, after = benchmark.run_episode(cycle, self.fixed, initial_soc=.200001)
        self.assertEqual(after["attempted_transitions"], 1)
        self.assertEqual(after["completed_transitions"], 1)
        self.assertEqual(after["elapsed_s"], 1.)
        self.assertGreater(after["net_kwh"], 0.)
        self.assertEqual(after["constraint_stage"], "post_step")
        self.assertTrue(after_rows[0]["completed_step"])
        for summary in (before, after):
            self.assertFalse(summary["completed_cycle"])
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(summary["completion_reason"], "constraint_violation")

    def test_infeasible_grid_demand_remains_a_recorded_rejection(self):
        rows, summary = benchmark.run_episode(np.array([[0., 140., 0.], [1., 140., 0.]]), benchmark.CONTROLLERS[-1])
        self.assertEqual(summary["attempted_transitions"], 1)
        self.assertEqual(summary["completed_transitions"], 0)
        self.assertFalse(summary["completed_cycle"])
        self.assertIn("motor_feasibility", rows[0]["violated_constraints"])
        self.assertIsNone(rows[0]["predicted_motor_power_kw"])
        self.assertIsNone(rows[0]["sigma_tor"])
        self.assertEqual(summary["battery_soh_loss"], 0.)

    def test_final_true_health_and_completed_variable_duration_are_reported(self):
        rows, summary = benchmark.run_episode(self.cycle, self.fixed)
        self.assertEqual([row["dt_completed_s"] for row in rows], [1., 2.])
        self.assertEqual(summary["elapsed_s"], 3.)
        self.assertEqual(rows[0]["original_cycle_time_before_s"], 50.)
        self.assertEqual(rows[-1]["original_cycle_time_after_s"], 53.)
        self.assertEqual(rows[0]["time_before_s"], 0.)
        self.assertEqual(rows[-1]["time_after_s"], 3.)
        self.assertEqual(rows[0]["slope_rad"], 0.)
        self.assertTrue(summary["completed_cycle"])
        self.assertEqual(summary["final_soc"], rows[-1]["soc_after"])
        for component in ("battery", "motor1", "motor2"):
            self.assertEqual(summary[f"final_{component}_soh"], rows[-1][f"{component}_soh_after"])
            self.assertEqual(summary[f"{component}_soh_loss"], summary[f"initial_{component}_soh"] - summary[f"final_{component}_soh"])
            self.assertGreater(summary[f"{component}_soh_loss"], 0.)

    def test_fresh_output_provenance_and_paired_step_records(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cycle_path = root / "cycle.csv"
            np.savetxt(cycle_path, self.cycle, delimiter=",", header="time_s,velocity_kmh,slope_rad", comments="")
            source_hash = benchmark.file_sha256(cycle_path)
            output = root / "run"
            summaries = benchmark.run_benchmark(output, cycles={"synthetic": cycle_path},
                controllers=(self.fixed, benchmark.CONTROLLERS[-1]))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["cycle_sha256"]["synthetic"]["sha256"], source_hash)
            self.assertEqual(benchmark.file_sha256(cycle_path), source_hash)
            self.assertEqual(manifest["config"]["soh_source"], "true_physical")
            self.assertFalse(manifest["config"]["incomplete_runs_comparable"])
            self.assertEqual(len(manifest["config"]["grid_actions"]), 101)
            self.assertEqual(len(manifest["map_sha256"]), 2)
            self.assertIn("run_controlled_benchmark.py", manifest["source_sha256"])
            self.assertIn("torch", manifest["versions"])
            self.assertIn("pandas", manifest["versions"])
            physical = manifest["physical_configuration"]
            self.assertEqual(physical["battery_electrical"]["nominal_capacity_ah"], 72.)
            self.assertEqual(physical["battery_electrical"]["nominal_ocv_v"], 255.5)
            self.assertEqual(physical["battery_initial_state"]["polarization_voltage_1_v"], 0.)
            self.assertEqual(physical["vehicle"]["mass_kg"], 1635.)
            self.assertEqual(physical["powertrain"]["soh_source"], "true_physical")
            self.assertGreater(physical["initial_rc_parameters"]["discharge"]["r1_ohm"], 0.)
            self.assertEqual(json.loads((output / "summary.json").read_text()), summaries)
            with (output / "summary.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            for summary in summaries:
                with (output / "steps" / f"synthetic__{summary['controller']}.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), summary["attempted_transitions"])
                self.assertEqual(float(rows[-1]["soc_after"]), summary["final_soc"])
            with self.assertRaises(FileExistsError):
                benchmark.run_benchmark(output, cycles={"synthetic": cycle_path})
            self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "completed")

    def test_failed_arm_is_explicit_in_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cycle_path = root / "cycle.csv"
            np.savetxt(cycle_path, self.cycle, delimiter=",", header="time_s,velocity_kmh,slope_rad", comments="")
            output = root / "failed"
            with patch.object(benchmark, "run_episode", side_effect=RuntimeError("fixture failure")):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    benchmark.run_benchmark(output, cycles={"synthetic": cycle_path}, controllers=(self.fixed,))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["episodes"]["synthetic__fixed_0p5"]["status"], "failed")
            self.assertIn("fixture failure", manifest["error"])


if __name__ == "__main__":
    unittest.main()
