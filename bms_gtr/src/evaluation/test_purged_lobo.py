"""Protocol regressions use temporary fixtures; historical artifacts are read-only."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from . import purged_lobo as protocol


FIXTURE_BATTERIES = ("A", "B", "C")


def save_sequence_archive(directory, length=2):
    X, y, batteries = [], [], []
    for battery in FIXTURE_BATTERIES:
        with np.load(directory / f"{battery}.npz", allow_pickle=False) as data:
            features, targets = data["features"], data["soh"]
        n = len(features) - length
        X.extend(features[i:i + length] for i in range(n))
        y.extend(targets[length:])
        batteries.extend([battery] * n)
    np.savez(directory / "sequences.npz", X=np.asarray(X, dtype=np.float32),
             y=np.asarray(y, dtype=np.float32), batteries=np.asarray(batteries))


def make_fixture(directory):
    directory.mkdir()
    generator = np.random.default_rng(42)
    for index, battery in enumerate(FIXTURE_BATTERIES):
        features = generator.normal(size=(18, 3, 8)) + .05 * np.arange(18)[:, None, None] + .1 * index
        targets = .99 - .01 * np.arange(18) - .005 * index
        np.savez(directory / f"{battery}.npz", features=features, soh=targets,
                 cycle_indices=1 + index + 3 * np.arange(18))
    save_sequence_archive(directory)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data_dir = self.root / "processed"
        make_fixture(self.data_dir)
        self.config = protocol.ProtocolConfig(sequence_length=2, max_epochs=2, patience=1, batch_size=4)

    def data(self):
        return protocol.reconstruct_sequences(self.data_dir, 2, FIXTURE_BATTERIES)

    def run_baselines(self, output, **kwargs):
        return protocol.run_protocol(self.data_dir, output, self.config, models=("mean", "ridge"),
            battery_names=FIXTURE_BATTERIES, held_out_batteries=("A",), **kwargs)

    def test_reconstruction_alignment_original_ids_and_exact_archive_guard(self):
        data = self.data()
        self.assertEqual(data.X.shape, (48, 2, 3, 8))
        self.assertEqual(data.target_retained_indices[0], 2)
        self.assertEqual(data.target_cycle_ids[0], 7)
        np.testing.assert_array_equal(data.support_cycle_ids[0], [1, 4, 7])
        self.assertEqual(data.local_indices[16], 0)
        self.assertEqual(data.batteries[16], "B")
        with np.load(self.data_dir / "sequences.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["X"][0, 0, 0, 0] += .1
        np.savez(self.data_dir / "sequences.npz", **arrays)
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            self.data()

    def test_purged_supports_and_outer_batteries_are_disjoint(self):
        data = self.data()
        split = protocol.make_fold(data, "A")
        self.assertEqual({name: len(rows) for name, rows in split.items()},
                         dict(train=20, purged=4, validation=8, test=16))
        np.testing.assert_array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(48))
        self.assertTrue(np.all(data.batteries[split["test"]] == "A"))
        self.assertFalse(np.any(data.batteries[split["train"]] == "A"))
        for battery in ("B", "C"):
            train = split["train"][data.batteries[split["train"]] == battery]
            validation = split["validation"][data.batteries[split["validation"]] == battery]
            self.assertLess(data.local_indices[train[-1]] + 2, data.local_indices[validation[0]])
            self.assertEqual(np.intersect1d(data.support_cycle_ids[train], data.support_cycle_ids[validation]).size, 0)

    def test_shipped_data_reconstruction_and_predeclared_fold_counts(self):
        directory = Path(__file__).resolve().parents[2] / "data" / "processed"
        data = protocol.reconstruct_sequences(directory)
        self.assertEqual(data.X.shape, (556, 20, 3, 300))
        for battery in protocol.BATTERIES:
            split = protocol.make_fold(data, battery)
            expected = (294, 90, 112) if battery == "B0018" else (265, 83, 148)
            self.assertEqual(tuple(len(split[name]) for name in ("train", "validation", "test")), expected)
            self.assertEqual(len(split["purged"]), 60)

    def test_scaler_uses_training_only_and_constant_channels(self):
        data = self.data()
        split = protocol.make_fold(data, "A")
        scaler = protocol.ChannelScaler.fit(data.X[split["train"]])
        expected_mean = data.X[split["train"]].astype(np.float64).mean(axis=(0, 1, 3), keepdims=True)
        np.testing.assert_array_equal(scaler.mean, expected_mean)
        data.X[split["validation"]] += 1e6
        data.X[split["test"]] -= 1e6
        after = protocol.ChannelScaler.fit(data.X[split["train"]])
        np.testing.assert_array_equal(after.mean, scaler.mean)
        np.testing.assert_array_equal(after.std, scaler.std)
        constant = np.ones((4, 2, 3, 8))
        fitted = protocol.ChannelScaler.fit(constant)
        np.testing.assert_array_equal(fitted.std, np.ones((1, 1, 3, 1)))
        np.testing.assert_array_equal(fitted.transform(constant), np.zeros_like(constant))

    def test_frozen_ridge_grid_ties_and_dual_solution(self):
        X = np.zeros((4, 2, 3, 8))
        model, selection = protocol.select_ridge(X, np.full(4, .8), X[:2], np.full(2, .8))
        self.assertEqual(tuple(row["alpha"] for row in selection), protocol.RIDGE_ALPHAS)
        self.assertEqual(model.alpha, .001)
        np.testing.assert_array_equal(model.predict(X), np.full(4, .8))
        generator = np.random.default_rng(9)
        X = generator.normal(size=(8, 2, 3, 2))
        y = generator.normal(size=8)
        model, _ = protocol.select_ridge(X, y, X[:2], y[:2])
        U = (X.reshape(8, -1) - model.center) / np.sqrt(12)
        primal = np.linalg.solve(U.T @ U + model.alpha * np.eye(12), U.T @ (y - y.mean()))
        np.testing.assert_allclose(model.beta, primal, rtol=1e-11, atol=1e-11)

    def test_saved_baselines_reload_with_paired_ids_and_provenance(self):
        before = {path.name: protocol.sha256(path) for path in self.data_dir.glob("*.npz")}
        output = self.root / "run"
        summary = self.run_baselines(output)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["data_sha256"], before)
        self.assertIsNone(manifest["scheduler"])
        self.assertFalse(manifest["refit_after_selection"])
        self.assertIn("src/models/hybrid_model.py", manifest["source_sha256"])
        self.assertNotIn("torch", manifest["versions"])
        self.assertEqual(before, {path.name: protocol.sha256(path) for path in self.data_dir.glob("*.npz")})
        data = self.data()
        with (output / "predictions.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual((output / "predictions.csv").read_text(), (output / "folds/A/predictions.csv").read_text())
        indices = np.asarray([int(row["combined_row"]) for row in rows])
        for row, index in zip(rows, indices):
            self.assertEqual(row["battery"], str(data.batteries[index]))
            self.assertEqual(int(row["local_sequence_index"]), int(data.local_indices[index]))
            self.assertEqual(int(row["target_cycle_id"]), int(data.target_cycle_ids[index]))
            self.assertEqual(int(row["target_retained_index"]), int(data.target_retained_indices[index]))
            self.assertEqual(float(row["soh_true"]), float(data.y[index]))
        for name in ("mean", "ridge"):
            predictions = protocol.load_frozen_predictor(output / "folds/A", name)(data.X[indices])
            np.testing.assert_allclose(predictions, [float(row[name]) for row in rows], rtol=1e-12, atol=1e-12)
            self.assertEqual(summary["pooled"][name], protocol.metrics(data.y[indices], predictions))
        with self.assertRaises(FileExistsError):
            self.run_baselines(output)
        self.assertEqual(json.loads((output / "manifest.json").read_text())["status"], "completed")

    def test_holdout_perturbations_cannot_change_scaler_or_selected_model(self):
        first, second = self.root / "first", self.root / "second"
        self.run_baselines(first)
        path = self.data_dir / "A.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["features"] += 1000.
        arrays["soh"] -= .1
        np.savez(path, **arrays)
        save_sequence_archive(self.data_dir)
        self.run_baselines(second)
        for name in ("scaler.npz", "ridge_model.npz", "mean_model.npz"):
            with np.load(first / "folds/A" / name, allow_pickle=False) as before, \
                 np.load(second / "folds/A" / name, allow_pickle=False) as after:
                for key in before.files:
                    np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual((first / "folds/A/ridge_selection.json").read_text(), (second / "folds/A/ridge_selection.json").read_text())

    def test_failures_are_recorded_without_overwriting_existing_outputs(self):
        output = self.root / "failed"
        with patch.object(protocol, "reconstruct_sequences", side_effect=ValueError("fixture failure")):
            with self.assertRaisesRegex(ValueError, "fixture failure"):
                self.run_baselines(output)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("fixture failure", manifest["error"])
        self.assertFalse((output / "summary.json").exists())

    def test_metrics_units_negative_r2_and_constant_target_convention(self):
        self.assertAlmostEqual(protocol.metrics([.8, .9], [.7, .8])["mae"], .1)
        self.assertLess(protocol.metrics([.8, .9], [0., 0.])["r2"], 0.)
        self.assertIsNone(protocol.metrics([.8, .8], [.7, .9])["r2"])
        with self.assertRaises(ValueError):
            protocol.metrics([.8], [np.nan])

    def test_protocol_import_does_not_load_torch_or_sklearn(self):
        command = "import sys; import bms_gtr.src.evaluation.purged_lobo; assert 'torch' not in sys.modules; assert 'sklearn' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", command],
            cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is needed only for hybrid tests")
class HybridProtocolTests(unittest.TestCase):
    # Reuse only fixture setup; protocol tests above remain their own suite.
    setUp = ProtocolTests.setUp
    data = ProtocolTests.data

    def test_eval_inference_preserves_batchnorm_and_disables_gradients(self):
        import torch
        from ..models.hybrid_model import CNNTCNLSTMAttention
        torch.set_num_threads(2)
        model = CNNTCNLSTMAttention()
        model.train()
        before = {name: value.clone() for name, value in model.named_buffers()}
        grad_enabled = []
        handle = model.register_forward_pre_hook(lambda *_: grad_enabled.append(torch.is_grad_enabled()))
        self.addCleanup(handle.remove)
        X = self.data().X[:4]
        first = protocol.predict_hybrid(model, X, 2)
        second = protocol.predict_hybrid(model, X, 2)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(any(grad_enabled))
        self.assertFalse(model.training)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_best_checkpoint_is_restored_after_worse_validation_epoch(self):
        import torch
        X, y = self.data().X[:8], self.data().y[:8]
        checkpoint = self.root / "best.pt"
        with patch.object(protocol, "predict_hybrid", side_effect=[y[4:].astype(float), y[4:].astype(float) + .1]):
            model, history = protocol.train_hybrid(X[:4], y[:4], X[4:], y[4:], self.config, checkpoint)
        self.assertEqual(history["best_epoch"], 1)
        self.assertEqual(len(history["epochs"]), 2)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_saved_hybrid_predictor_reproduces_paired_predictions(self):
        output = self.root / "hybrid_run"
        protocol.run_protocol(self.data_dir, output, self.config, models=("hybrid",),
            battery_names=FIXTURE_BATTERIES, held_out_batteries=("A",))
        with (output / "predictions.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        data = self.data()
        indices = [int(row["combined_row"]) for row in rows]
        predictor = protocol.load_frozen_predictor(output / "folds/A", "hybrid")
        np.testing.assert_allclose(predictor(data.X[indices]), [float(row["hybrid"]) for row in rows], rtol=1e-6, atol=1e-7)
        with self.assertRaisesRegex(ValueError, "input shape"):
            predictor(data.X[indices, :1])
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertIn("torch", manifest["versions"])
        self.assertEqual(manifest["config"]["learning_rate"], .001)
        self.assertIsNone(manifest["scheduler"])


if __name__ == "__main__":
    unittest.main()
