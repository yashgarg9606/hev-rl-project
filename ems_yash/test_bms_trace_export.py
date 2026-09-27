"""Model isolation, explicit time alignment, and preservation contracts."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from ems_yash import generate_bms_trace as bridge


class TraceExportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def cell(self, name='test_cell', channels=3, **metadata):
        path = self.root / f'{name}.npz'
        np.savez(path, features=np.ones((23, channels, 300), dtype=np.float32),
                 soh=np.linspace(1., .9, 23), cycle_indices=np.arange(23),
                 **({'target_kind': 'capacity_ratio'} | metadata))
        return path

    def checkpoint(self, variant, channels):
        cls = bridge.isolated_module(bridge.PROJECTS[variant], 'models.hybrid_model').CNNTCNLSTMAttention
        model = cls(**({'in_channels': channels} if variant == 'spatial' else {}))
        path = self.root / f'{variant}_{channels}.pt'
        torch.save(model.state_dict(), path)
        return path

    def test_legacy_spatial_legacy_can_coexist_with_ems_src(self):
        sys.path.insert(0, str(bridge.ROOT / 'ems_yash'))
        from src.environment.rl_environment import EnergyManagementEnv
        existing_src = sys.modules['src']
        for variant, channels in [('legacy', 3), ('spatial', 4), ('legacy', 3)]:
            self.cell(channels=channels)
            result = bridge.generate_bms_soh_trace('test_cell', data_dir=self.root,
                time_seconds=[0., 2., 7.], variant=variant, checkpoint=self.checkpoint(variant, channels))
            self.assertEqual(result['soh_predicted'].shape, (3,))
            np.testing.assert_array_equal(result['target_cycle_ids'], [20, 21, 22])
            self.assertIs(sys.modules['src'], existing_src)
            self.assertTrue(result['evaluation_scope'].startswith('historical_all_data'))

    def test_source_ids_are_checked_against_timestamp_rows(self):
        path = self.root / 'times.csv'
        with path.open('w', newline='') as f:
            writer = csv.writer(f); writer.writerow(['battery', 'target_cycle_id', 'time_seconds'])
            writer.writerows([['a', 21, 0], ['a', 20, 1]])
        with self.assertRaisesRegex(ValueError, 'cycle IDs'):
            bridge.read_timestamps(path, 'a', [20, 21])
        for times in ([0, 0], [-1, 2], [0, np.nan], [0]):
            with self.assertRaises(ValueError): bridge.validate_times(times, 2)

    def test_halfcycle_proxy_cannot_be_relabelled_as_soh(self):
        self.cell(target_kind='discharge_window_capacity_proxy')
        with self.assertRaisesRegex(ValueError, 'proxies'):
            bridge.generate_bms_soh_trace('test_cell', data_dir=self.root, time_seconds=[0, 1, 2],
                                         acknowledge_unverified_targets=True)

    def test_unknown_target_requires_acknowledgment(self):
        self.cell(target_kind='provided_soh_unverified')
        with self.assertRaisesRegex(ValueError, 'acknowledgment'):
            bridge.generate_bms_soh_trace('test_cell', data_dir=self.root, time_seconds=[0, 1, 2])

    def test_corrected_features_refuse_historical_checkpoint(self):
        self.cell(channels=4, feature_schema='masked_derivatives_v2')
        with self.assertRaisesRegex(ValueError, 'newly fitted fold'):
            bridge.generate_bms_soh_trace('test_cell', data_dir=self.root, variant='spatial', time_seconds=[0, 1, 2])

    def test_existing_trace_is_never_overwritten(self):
        p = self.root / 'trace.npz'; p.write_bytes(b'original')
        with self.assertRaises(FileExistsError): bridge.save_trace({'x': [1]}, p)
        self.assertEqual(p.read_bytes(), b'original')

    def test_nonhybrid_requires_fold(self):
        with self.assertRaisesRegex(ValueError, 'require a frozen fold'):
            bridge.generate_bms_soh_trace('anything', time_seconds=[], model_name='mean')

    def test_mean_fold_respects_saved_history_length(self):
        import json
        cell = self.cell()
        fold = self.root/'run/folds/test_cell'; fold.mkdir(parents=True)
        np.savez(fold/'mean_model.npz', value=.95)
        manifest = dict(status='completed', config=dict(sequence_length=2), models=['mean'],
                        folds={'test_cell': {'status':'completed'}},
                        data_sha256={'test_cell.npz':bridge.sha256(cell)})
        (fold.parent.parent/'manifest.json').write_text(json.dumps(manifest))
        result = bridge.generate_bms_soh_trace('test_cell', data_dir=self.root,
            time_seconds=np.arange(21), fold_dir=fold, model_name='mean')
        np.testing.assert_array_equal(result['target_cycle_ids'], np.arange(2,23))
        self.assertEqual(len(result['soh_predicted']),21)

    def test_saved_ridge_fold_exports_only_its_held_out_cell(self):
        data_dir = bridge.ROOT / 'bms_gtr/data/processed'
        fold = bridge.ROOT / 'research_runs/bms_purged_lobo_seed42/folds/B0005'
        _, truth, ids = bridge.cell_sequences(data_dir, 'B0005')
        result = bridge.generate_bms_soh_trace('B0005', time_seconds=np.arange(len(truth)),
                                              fold_dir=fold, model_name='ridge')
        with (fold / 'predictions.csv').open() as f: rows = list(csv.DictReader(f))
        np.testing.assert_array_equal(result['soh_predicted'], [float(r['ridge']) for r in rows])
        np.testing.assert_array_equal(result['target_cycle_ids'], ids)
        with self.assertRaisesRegex(ValueError, 'holding out'):
            _, y, _ = bridge.cell_sequences(data_dir, 'B0006')
            bridge.generate_bms_soh_trace('B0006', time_seconds=np.arange(len(y)), fold_dir=fold, model_name='ridge')


if __name__ == '__main__':
    unittest.main()
