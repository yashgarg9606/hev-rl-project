"""Fresh destinations and explicit scope for historical figure reproduction."""
import argparse
from pathlib import Path


def historical_plot_directory():
    parser = argparse.ArgumentParser(description="Historical/descriptive figure reproduction, not independent test evidence")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not exist")
    parser.add_argument("--acknowledge-historical-evaluation", action="store_true", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "SCOPE.txt").write_text("Historical all-data or descriptive figures. Supplied targets and dataset provenance remain unverified. Not Oxford/CALCE validation or paper replication.\n")
    return args.output_dir
