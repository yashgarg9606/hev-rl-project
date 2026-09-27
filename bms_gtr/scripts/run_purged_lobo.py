"""Run the frozen purged LOBO pilot into an explicitly named fresh directory."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.purged_lobo import BATTERIES, ProtocolConfig, run_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--models", nargs="+", choices=("mean", "ridge", "hybrid"), default=["mean", "ridge", "hybrid"])
    parser.add_argument("--held-out", nargs="+", choices=BATTERIES, default=list(BATTERIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=40, help="Positive epoch budget; pilot default is 40")
    args = parser.parse_args()
    summary = run_protocol(args.data_dir, args.output_dir,
        ProtocolConfig(seed=args.seed, max_epochs=args.max_epochs),
        models=args.models, held_out_batteries=args.held_out,
        progress=lambda message: print(message, flush=True))
    print(f"Completed: {args.output_dir.resolve()}", flush=True)
    print(f"Macro metrics (SOH fraction): {summary['macro']}", flush=True)


if __name__ == "__main__":
    main()
