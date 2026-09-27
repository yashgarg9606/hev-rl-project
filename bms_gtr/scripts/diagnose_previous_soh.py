"""Past measured-SOH persistence diagnostic, separate from feature-only models.

This reference has access to a preceding ground-truth capacity measurement from
the evaluated cell. That information is not an explicit input to the trained
feature-only models. It is not a deployable estimator or a fair primary ranking.
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.purged_lobo import BATTERIES, metrics, reconstruct_sequences, sha256, write_json


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    data_dir = PROJECT_ROOT / "data/processed"
    manifest = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        information_protocol="Previous retained discharge measured SOH is available exactly, including on the evaluated cell; privileged past labels, not feature-only inference.",
        formula="prediction[target_retained_index] = processed_soh[target_retained_index - 1]",
        target_dtype="float32 stored benchmark labels", prediction_dtype="float64 processed SOH",
        versions=dict(python=platform.python_version(), numpy=np.__version__),
        source_sha256={str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in
            (Path(__file__).resolve(), PROJECT_ROOT / "src/evaluation/purged_lobo.py")})
    write_json(output_dir / "manifest.json", manifest)
    try:
        data = reconstruct_sequences(data_dir)
        prediction = np.empty(len(data.y), dtype=np.float64)
        per_battery = {}
        for battery in BATTERIES:
            rows = np.flatnonzero(data.batteries == battery)
            with np.load(data_dir / f"{battery}.npz", allow_pickle=False) as archive:
                prediction[rows] = archive["soh"][data.target_retained_indices[rows] - 1]
            per_battery[battery] = metrics(data.y[rows], prediction[rows])
        manifest["data_sha256"] = {path.name: sha256(path) for path in
            [*(data_dir / f"{battery}.npz" for battery in BATTERIES), data_dir / "sequences.npz"]}
        records = [dict(combined_row=i, battery=str(data.batteries[i]),
            target_source_cycle_id=int(data.target_cycle_ids[i]),
            previous_source_cycle_id=int(data.support_cycle_ids[i, -2]),
            target_retained_index=int(data.target_retained_indices[i]),
            soh_true=float(data.y[i]), previous_measured_soh=float(prediction[i]))
            for i in range(len(data.y))]
        with (output_dir / "predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        summary = dict(per_battery=per_battery, pooled=metrics(data.y, prediction),
            macro={name: float(np.mean([row[name] for row in per_battery.values()]))
                   for name in ("mae", "rmse", "r2")})
        write_json(output_dir / "summary.json", summary)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(output_dir / "manifest.json", manifest)
        return summary
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    print(run(parser.parse_args().output_dir)["pooled"])
