from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.nasa_loader import load_battery, get_discharge_cycles
from src.features.electrochemical import extract_features


RAW_DIR = Path("data/raw/NASA/Batteries")
OUTPUT_DIR = Path("data/processed")

BATTERIES = [
    "B0005",
    "B0006",
    "B0007",
    "B0018"
]


def process_battery(battery_name):
    mat_path = RAW_DIR / f"{battery_name}.mat"

    print(f"\nProcessing {battery_name}...")

    records = load_battery(mat_path)
    discharge_cycles = get_discharge_cycles(records)

    print(f"Discharge cycles: {len(discharge_cycles)}")

    feature_data = []
    capacities = []
    soh_values = []
    cycle_indices = []

    for cycle in discharge_cycles:
        try:
            _, features = extract_features(cycle)

            capacity = float(
                np.asarray(cycle["Capacity"]).squeeze()
            )

            feature_data.append(features)
            capacities.append(capacity)
            cycle_indices.append(cycle["cycle_index"])

        except Exception as e:
            print(
                f"Skipping cycle {cycle['cycle_index']}: {e}"
            )

    capacities = np.asarray(capacities)

    if len(capacities) == 0:
        raise RuntimeError(
            f"No usable cycles found for {battery_name}"
        )

    soh_values = capacities / capacities[0]

    feature_data = np.asarray(feature_data)

    output_file = OUTPUT_DIR / f"{battery_name}.npz"

    np.savez_compressed(
        output_file,
        features=feature_data,
        capacities=capacities,
        soh=soh_values,
        cycle_indices=np.asarray(cycle_indices)
    )

    print(f"Saved: {output_file}")
    print(f"Feature shape: {feature_data.shape}")
    print(f"SOH shape: {soh_values.shape}")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for battery in BATTERIES:
        process_battery(battery)

    print("\nDataset construction complete.")


if __name__ == "__main__":
    main()