"""Validate explicitly supplied one-Hz velocity traces before CSV export."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_velocity_cycle(source_file, expected_samples, maximum_kmh):
    source_file = Path(source_file)
    if not source_file.is_file():
        raise FileNotFoundError(
            f"External velocity source not found: {source_file}. Supply --source "
            "with the original whitespace-separated velocity text file; no substitute is generated."
        )
    velocity = np.asarray([float(value) for value in source_file.read_text().split()], dtype=np.float64)
    if velocity.shape != (expected_samples,):
        raise ValueError(f"Expected {expected_samples} one-Hz samples, received {len(velocity)}.")
    if not np.isfinite(velocity).all() or np.any(velocity < 0.) or np.any(velocity > maximum_kmh):
        raise ValueError(f"Velocities must be finite and within [0, {maximum_kmh}] km/h.")
    return pd.DataFrame(dict(time_s=np.arange(expected_samples, dtype=float),
                             velocity_kmh=velocity, slope_rad=np.zeros(expected_samples)))


def export_from_cli(name, filename, expected_samples, maximum_kmh):
    parser = argparse.ArgumentParser(description=f"Convert an externally supplied {name} one-Hz trace. Existing CSVs are never overwritten.")
    parser.add_argument("--source", type=Path, required=True,
                        help=f"Path to the original {filename} whitespace-separated km/h trace (not bundled in ems_yash)")
    parser.add_argument("--output", type=Path, required=True, help="New CSV path; must not already exist")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}. Choose a new path.")
    try:
        frame = load_velocity_cycle(args.source, expected_samples, maximum_kmh)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as handle:
        frame.to_csv(handle, index=False)
    print(f"Saved {len(frame)} validated-format samples from {args.source} to {args.output}")
