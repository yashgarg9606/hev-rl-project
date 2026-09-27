"""Research-contract regressions for the uploaded spatial model."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.evaluation.purged_lobo import ProtocolConfig, run_protocol, reconstruct_sequences, make_fold, load_frozen_predictor
from src.features.electrochemical import extract_masked_features, smooth_signal, V_MIN, V_MAX, N_POINTS
from src.data.artifacts import save_new_archive
from scripts import build_hnei_dataset as hnei


class SpatialProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.names = ('K_0001', 'K_0002', 'K_0003')

    def dataset(self, name, changed_holdout=False, mixed_schema=False):
        root = self.root / name; root.mkdir()
        X, y, labels = [], [], []; rng = np.random.default_rng(7)
        for index, cell in enumerate(self.names):
            features = rng.normal(size=(12, 4, 300)).astype(np.float32)
            targets = np.linspace(.98, .8, 12).astype(np.float32)
            if changed_holdout and index == 0: targets = targets[::-1].copy()
            np.savez(root / f'{cell}.npz', features=features, soh=targets, cycle_indices=np.arange(12), target_kind='provided_soh_unverified', feature_schema='other' if mixed_schema and index == 0 else 'masked_derivatives_v2')
            X.extend([features[i:i+2] for i in range(10)]); y.extend(targets[2:]); labels.extend([cell]*10)
        np.savez(root / 'sequences.npz', X=np.asarray(X), y=np.asarray(y), batteries=np.asarray(labels))
        return root

    def test_explicit_semantics_and_family_required(self):
        data = self.dataset('data')
        with self.assertRaisesRegex(ValueError, 'acknowledge'): run_protocol(data, self.root/'out', battery_names=self.names)
        with self.assertRaisesRegex(ValueError, 'one dataset family'): run_protocol(data, self.root/'out', battery_names=('K_0001','H_PL03'), acknowledge_unverified_targets=True)
        self.assertFalse((self.root/'out').exists())

    def test_mixed_feature_schemas_rejected(self):
        data = self.dataset('mixed', mixed_schema=True)
        with self.assertRaisesRegex(ValueError, 'schema'): run_protocol(data, self.root/'out', battery_names=self.names, acknowledge_unverified_targets=True)

    def test_source_rows_and_support_purge(self):
        data = self.dataset('data'); rebuilt = reconstruct_sequences(data, 2, ('K_0003','K_0001'))
        np.testing.assert_array_equal(rebuilt.archive_rows, np.r_[20:30,0:10])
        split = make_fold(rebuilt, 'K_0003')
        self.assertEqual(set(rebuilt.batteries[split['test']]), {'K_0003'})
        self.assertFalse(np.intersect1d(rebuilt.support_cycle_ids[split['train']],rebuilt.support_cycle_ids[split['validation']]).size)

    def test_outer_label_changes_do_not_change_fits_or_predictions(self):
        import torch
        outputs=[]
        for name, changed in [('a',False), ('b',True)]:
            data=self.dataset(name, changed_holdout=changed); out=self.root/(name+'_run')
            run_protocol(data,out,ProtocolConfig(sequence_length=2,max_epochs=2,patience=2,batch_size=4), battery_names=self.names,held_out_batteries=['K_0001'], models=('mean','ridge','hybrid'),acknowledge_unverified_targets=True)
            outputs.append((data,out/'folds/K_0001'))
        X=reconstruct_sequences(outputs[0][0],2,self.names).X[:10]
        for model in ('mean','ridge','hybrid'):
            np.testing.assert_array_equal(load_frozen_predictor(outputs[0][1],model)(X),load_frozen_predictor(outputs[1][1],model)(X))
        for key in ('mean','std'):
            with np.load(outputs[0][1]/'scaler.npz') as a,np.load(outputs[1][1]/'scaler.npz') as b: np.testing.assert_array_equal(a[key],b[key])
        a=torch.load(outputs[0][1]/'hybrid_best.pt',weights_only=True); b=torch.load(outputs[1][1]/'hybrid_best.pt',weights_only=True)
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
        manifest=json.loads((outputs[0][1].parent.parent/'manifest.json').read_text()); self.assertFalse(manifest['target_provenance_verified'])
        with self.assertRaises(FileExistsError): run_protocol(outputs[0][0],outputs[0][1].parent.parent,ProtocolConfig(sequence_length=2), battery_names=self.names,acknowledge_unverified_targets=True)

    def test_mask_excludes_extrapolation_and_derivative_order(self):
        voltage=np.linspace(3.4,4.,50); q=np.linspace(0,1,50); current=np.ones(50)
        f=extract_masked_features(voltage,current,q); grid=np.linspace(V_MIN,V_MAX,N_POINTS); v=smooth_signal(voltage)
        valid=(grid>=v.min())&(grid<=v.max()); np.testing.assert_array_equal(f[3].astype(bool),valid)
        self.assertTrue(np.all(f[:3,~valid]==0)); np.testing.assert_allclose(f[0,valid],1/.6,rtol=1e-10); np.testing.assert_allclose(f[1,valid],.6,rtol=1e-10)
        self.assertLess(f[3].sum(),N_POINTS)

    def test_hnei_canonical_id_and_proxy_metadata(self):
        self.assertEqual(hnei.canonical_cell_id('PL03_SOC_0-60_HalfC'),'PL03')
        with self.assertRaises(ValueError): hnei.canonical_cell_id('unknown')
        voltage=np.linspace(4.1,2.8,50); current=np.ones(50); q=np.linspace(0,1.4,50)
        groups=[(1,1,voltage,current,q),(2,2,voltage,current,q*.6)]
        import pandas as pd
        with patch.object(hnei,'OUTPUT_DIR',self.root),patch.object(hnei.pd,'read_csv',return_value=pd.DataFrame()),patch.object(hnei,'extract_discharge_cycles',return_value=iter(groups)):
            self.assertTrue(hnei.process_cell(self.root/'PL03_SOC_0-60_HalfC.csv','PL03_SOC_0-60_HalfC'))
        with np.load(self.root/'H_PL03.npz') as a:
            self.assertEqual(str(a['target_kind']),'discharge_window_capacity_proxy'); self.assertEqual(str(a['feature_schema']),'masked_derivatives_v2')
            np.testing.assert_allclose(a['capacity_window_ratio'],[1,.6]); np.testing.assert_array_equal(a['source_segment_cycle'],[[1,1],[2,2]])

    def test_archive_writer_preserves_existing_bytes(self):
        p=self.root/'existing.npz';p.write_bytes(b'unchanged')
        with self.assertRaises(FileExistsError): save_new_archive(p,x=[1])
        self.assertEqual(p.read_bytes(),b'unchanged')


if __name__=='__main__': unittest.main()
