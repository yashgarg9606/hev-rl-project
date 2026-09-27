"""Build one dataset family's histories without mixing target definitions."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.sequence_dataset import load_and_sequence
from src.data.artifacts import save_new_archive


def build(data_dir, output, family, acknowledge_legacy_metadata=False):
    files = sorted(Path(data_dir).glob(f'{family}_*.npz'))
    if family not in ('K', 'H') or not files:
        raise ValueError('Select a nonempty K or H dataset family')
    expected_kind = 'provided_soh_unverified' if family == 'K' else 'discharge_window_capacity_proxy'
    blocks, targets, batteries, schemas = [], [], [], set()
    for path in files:
        with np.load(path, allow_pickle=False) as a:
            if 'target_kind' not in a.files and not acknowledge_legacy_metadata:
                raise ValueError('Legacy target metadata missing; acknowledge explicitly or rebuild into a new directory')
            if 'target_kind' in a.files and str(a['target_kind']) != expected_kind:
                raise ValueError(f'{path}: incompatible target semantics')
            schemas.add(str(a['feature_schema']) if 'feature_schema' in a.files else 'historical_unversioned')
            if a['features'].ndim != 3 or a['features'].shape[1:] != (4, 300):
                raise ValueError(f'{path}: expected four-channel features')
            if not np.isfinite(a['features']).all() or not np.isfinite(a['soh']).all():
                raise ValueError(f'{path}: nonfinite features or targets')
        X, y = load_and_sequence(path, 20)
        if not len(y):
            raise ValueError(f'{path}: not enough retained cycles')
        blocks.append(X); targets.append(y); batteries.extend([path.stem] * len(y))
    if len(schemas) != 1:
        raise ValueError('Do not mix feature schema versions')
    save_new_archive(output, X=np.concatenate(blocks), y=np.concatenate(targets),
        batteries=np.asarray(batteries), target_kind=expected_kind,
        feature_schema=schemas.pop(), target_provenance_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh sequences.npz path in the selected data directory')
    parser.add_argument('--family', choices=('K', 'H'), required=True)
    parser.add_argument('--acknowledge-legacy-metadata', action='store_true')
    a = parser.parse_args()
    build(a.data_dir, a.output, a.family, a.acknowledge_legacy_metadata)


if __name__ == '__main__':
    main()
