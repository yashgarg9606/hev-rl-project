"""Plot paired held-out predictions from a completed purged run into a new directory."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', choices=('mean','ridge','hybrid'), required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir/'manifest.json').read_text())
    if manifest.get('status') != 'completed' or args.model not in manifest.get('models',[]):
        parser.error('Require a completed purged run containing the selected model')
    if manifest.get('protocol') != 'purged_chronological_inner_validation_outer_battery_holdout':
        parser.error('Unsupported evaluation protocol; historical all-data predictions are not held-out')
    with (args.run_dir/'predictions.csv').open() as f: rows=list(csv.DictReader(f))
    if not rows: parser.error('No held-out predictions')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import matplotlib.pyplot as plt
    experimental = manifest.get('target_provenance_verified') is False
    target_label = manifest.get('target_kind','SOH capacity ratio')
    all_errors=[]
    for battery in dict.fromkeys(row['battery'] for row in rows):
        if Path(battery).name != battery: raise ValueError('Invalid battery identifier')
        selected=[r for r in rows if r['battery']==battery]
        x=np.asarray([int(r['target_cycle_id']) for r in selected])
        truth=np.asarray([float(r['soh_true']) for r in selected]);pred=np.asarray([float(r[args.model]) for r in selected])
        if not np.isfinite(truth).all() or not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
        all_errors.extend((pred-truth)*100)
        fig,ax=plt.subplots(figsize=(10,4));ax.plot(x,truth,label='Held-out target');ax.plot(x,pred,label=args.model)
        ax.set(xlabel='Recorded target cycle ID',ylabel=f'{target_label} (fraction)',title=f'{battery}: outer-cell holdout'+(' — unverified targets' if experimental else ''))
        ax.legend();fig.tight_layout();fig.savefig(args.output_dir/f'{battery}.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots();ax.hist(all_errors,bins=30);ax.set(xlabel='Prediction minus target (percentage points)',ylabel='Count',title='Held-out errors'+(' — unverified targets' if experimental else ''))
    fig.tight_layout();fig.savefig(args.output_dir/'errors.png',dpi=150);plt.close(fig)
    (args.output_dir/'scope.json').write_text(json.dumps(dict(run_directory=str(args.run_dir.resolve()),model=args.model,protocol=manifest['protocol'],target_kind=target_label,targets_verified=manifest.get('target_provenance_verified')),indent=2)+'\n')


if __name__=='__main__':main()
