"""Evaluate the spatial model with cell holdout and purged inner validation.

Existing supplied target labels remain unverified. No saved historical result
is upgraded to a held-out score by this command.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.purged_lobo import ProtocolConfig, run_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/processed")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory only")
    parser.add_argument("--batteries", nargs="+", required=True, help="Explicit cells from one family: K_* or H_*")
    parser.add_argument("--held-out", nargs="+")
    parser.add_argument("--models", nargs="+", choices=("mean", "ridge", "hybrid"), default=["mean"])
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acknowledge-unverified-targets", action="store_true", required=True)
    args = parser.parse_args()
    result = run_protocol(args.data_dir, args.output_dir, ProtocolConfig(seed=args.seed, max_epochs=args.max_epochs),
        models=args.models, battery_names=args.batteries, held_out_batteries=args.held_out,
        acknowledge_unverified_targets=args.acknowledge_unverified_targets, progress=print)
    print(result)


if __name__ == "__main__":
    main()
