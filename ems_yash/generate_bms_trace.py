"""Export an explicitly timed, single-cell BMS trace for offline EMS replay.

This does not infer vehicle-pack health or turn historical all-data predictions
into held-out evidence. Existing trace files are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {'legacy': ROOT / 'bms_gtr', 'spatial': ROOT / 'battery-health-cnn-tcn-lstm-main'}


def isolated_module(project, module):
    """Load a BMS package without replacing EMS's unrelated top-level src."""
    source = Path(project).resolve() / 'src'
    name = '_non2_bms_' + hashlib.sha256(str(source).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source / '__init__.py',
                                                     submodule_search_locations=[str(source)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
    return importlib.import_module(name + '.' + module)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cell_sequences(data_dir, battery, sequence_length=20):
    if not battery or Path(battery).name != battery:
        raise ValueError('Select one battery file stem, never a combined-cell trace')
    with np.load(Path(data_dir) / f'{battery}.npz', allow_pickle=False) as archive:
        features, labels, ids = (archive[k] for k in ('features', 'soh', 'cycle_indices'))
    if features.ndim != 3 or features.shape[2] != 300 or len(features) <= sequence_length or labels.shape != (len(features),) or ids.shape != labels.shape:
        raise ValueError('Invalid per-cell sequence/target shapes')
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError('Features and labels must be finite')
    if not np.issubdtype(ids.dtype, np.integer) or not np.all(np.diff(ids) > 0):
        raise ValueError('Source cycle identifiers must be strictly increasing integers')
    X = np.asarray([features[i-sequence_length:i] for i in range(sequence_length, len(features))], dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite float32 model inputs')
    return X, labels[sequence_length:].astype(np.float64), ids[sequence_length:]


def validate_times(time_seconds, count):
    times = np.asarray(time_seconds, dtype=np.float64)
    if times.shape != (count,) or not count or not np.isfinite(times).all() or times[0] < 0 or not np.all(np.diff(times) > 0):
        raise ValueError('Supply one finite nonnegative, strictly increasing time per prediction')
    return times


def read_timestamps(path, battery, target_cycle_ids):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(target_cycle_ids):
        raise ValueError('Timestamp CSV must contain exactly the selected cell targets')
    if any(row.get('battery') != battery for row in rows):
        raise ValueError('Timestamp CSV battery IDs must match the selected battery')
    if [int(row['target_cycle_id']) for row in rows] != list(target_cycle_ids):
        raise ValueError('Timestamp CSV target cycle IDs do not match inference order')
    return validate_times([float(row['time_seconds']) for row in rows], len(rows))


def fold_sequence_length(fold_dir):
    if fold_dir is None:
        return 20
    manifest = json.loads((Path(fold_dir).resolve().parent.parent / 'manifest.json').read_text())
    length = manifest['config']['sequence_length']
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError('Invalid frozen fold sequence length')
    return length


def generate_bms_soh_trace(battery_name, *, time_seconds, variant='legacy',
                           data_dir=None, checkpoint=None, fold_dir=None,
                           model_name='hybrid', acknowledge_unverified_targets=False):
    """Return paired predictions; caller supplies time mapping, never inferred from window length."""
    if fold_dir is None and model_name != 'hybrid':
        raise ValueError('Mean and ridge exports require a frozen fold directory')
    if variant not in PROJECTS:
        raise ValueError('variant must be legacy or spatial')
    project = PROJECTS[variant]
    data_dir = Path(data_dir) if data_dir is not None else project / 'data/processed'
    X, truth, cycle_ids = cell_sequences(data_dir, battery_name, fold_sequence_length(fold_dir))
    times = validate_times(time_seconds, len(truth))
    source_path = data_dir / f'{battery_name}.npz'
    source_hashes = {str(source_path): sha256(source_path)}
    canonical_nasa = PROJECTS['legacy'] / 'data/processed' / f'{battery_name}.npz'
    with np.load(source_path, allow_pickle=False) as archive:
        target_kind = str(archive['target_kind']) if 'target_kind' in archive.files else 'unknown'
        feature_schema = str(archive['feature_schema']) if 'feature_schema' in archive.files else 'historical_unversioned'
    if canonical_nasa.is_file() and sha256(canonical_nasa) == source_hashes[str(source_path)]:
        target_kind = 'capacity_ratio'
    if battery_name.startswith('H_') or target_kind == 'discharge_window_capacity_proxy':
        raise ValueError('HalfCycle discharge-window proxies cannot be exported as battery SOH truth')
    if target_kind != 'capacity_ratio' and not acknowledge_unverified_targets:
        raise ValueError('Targets require explicit acknowledgment of unverified provenance')
    if target_kind == 'unknown':
        target_kind = 'provided_labels_unverified'
    if fold_dir is not None:
        if checkpoint is not None:
            raise ValueError('A frozen fold export cannot use a separate checkpoint')
        fold_dir = Path(fold_dir).resolve()
        manifest_path = fold_dir.parent.parent / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') != 'completed' or fold_dir.name != battery_name or manifest.get('folds', {}).get(battery_name, {}).get('status') != 'completed':
            raise ValueError('Require a completed fold holding out the selected battery')
        if model_name not in manifest['models']:
            raise ValueError('Model not present in the frozen fold')
        expected = manifest['data_sha256'].get(f'{battery_name}.npz')
        if expected != source_hashes[str(data_dir / f'{battery_name}.npz')]:
            raise ValueError('Selected cell data differ from frozen fold provenance')
        protocol = isolated_module(project, 'evaluation.purged_lobo')
        predictions = protocol.load_frozen_predictor(fold_dir, model_name)(X)
        scope = 'outer_battery_holdout_offline_replay'
        source_hashes.update({str(p): sha256(p) for p in fold_dir.iterdir() if p.is_file()})
        source_hashes[str(manifest_path)] = sha256(manifest_path)
    else:
        if feature_schema == 'masked_derivatives_v2':
            raise ValueError('Corrected masked features require a newly fitted fold; historical checkpoints use a different feature contract')
        import torch
        checkpoint = Path(checkpoint) if checkpoint is not None else project / 'results/best_model.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        channels = int(state['cnn.conv1.weight'].shape[1])
        if X.shape[2] != channels:
            raise ValueError('Processed feature channels do not match the checkpoint; no channels are dropped')
        model_class = isolated_module(project, 'models.hybrid_model').CNNTCNLSTMAttention
        model = model_class(**({'in_channels': channels} if variant == 'spatial' else {})).cpu()
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            predictions = np.concatenate([model(torch.from_numpy(X[i:i+64]))[0].numpy() for i in range(0, len(X), 64)])
        scope = 'historical_all_data_offline_replay_not_independent_test'
        source_hashes[str(checkpoint)] = sha256(checkpoint)
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != truth.shape or not np.isfinite(predictions).all():
        raise ValueError('Predictor returned invalid outputs')
    if np.any((predictions < 0) | (predictions > 1)) or np.any((truth < 0) | (truth > 1)):
        raise ValueError('EMS trace requires SOH in [0, 1]; predictions are never silently clipped')
    error = predictions - truth
    return dict(battery_name=battery_name, time_seconds=times, soh_true=truth,
                soh_predicted=predictions, target_cycle_ids=cycle_ids,
                evaluation_scope=scope, target_kind=target_kind, feature_schema=feature_schema,
                time_basis='caller_supplied_offline_mapping_not_vehicle_pack_alignment',
                source_hashes_json=json.dumps(source_hashes, sort_keys=True),
                mae=float(np.mean(abs(error))), rmse=float(np.sqrt(np.mean(error**2))))


def save_trace(trace_data, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **trace_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=PROJECTS, default='legacy')
    parser.add_argument('--battery', required=True)
    parser.add_argument('--timestamps', type=Path, required=True, help='CSV: battery,target_cycle_id,time_seconds; explicit offline time mapping')
    parser.add_argument('--output', type=Path, required=True, help='Fresh trace path; existing files refused')
    parser.add_argument('--data-dir', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--checkpoint', type=Path)
    group.add_argument('--fold-dir', type=Path)
    parser.add_argument('--model', choices=('mean', 'ridge', 'hybrid'), default='hybrid')
    parser.add_argument('--acknowledge-unverified-targets', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; select a fresh trace path')
    data_dir = args.data_dir or PROJECTS[args.variant] / 'data/processed'
    _, _, ids = cell_sequences(data_dir, args.battery, fold_sequence_length(args.fold_dir))
    times = read_timestamps(args.timestamps, args.battery, ids)
    trace = generate_bms_soh_trace(args.battery, time_seconds=times, variant=args.variant,
        data_dir=data_dir, checkpoint=args.checkpoint, fold_dir=args.fold_dir,
        model_name=args.model, acknowledge_unverified_targets=args.acknowledge_unverified_targets)
    trace['timestamps_sha256'] = sha256(args.timestamps)
    save_trace(trace, args.output)
    print(f"Saved {len(times)} predictions: {args.output}; {trace['evaluation_scope']}")


if __name__ == '__main__':
    main()
