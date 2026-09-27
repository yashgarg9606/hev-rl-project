import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.artifacts import save_new_archive
from src.features.electrochemical import smooth_signal, interpolate_voltage_domain, calculate_derivative, V_MIN, V_MAX, N_POINTS

RAW_CSV = Path("data/raw/kaggle/cell_level_dataset.csv")
OUTPUT_DIR = Path("data/processed")

def extract_features_from_kaggle(voltage, current, capacity):
    from src.features.electrochemical import extract_masked_features
    return extract_masked_features(voltage, current, capacity)


def main():
    import argparse
    global RAW_CSV, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Build versioned masked features; supplied SOH labels remain unverified")
    parser.add_argument("--raw-csv", type=Path, default=Path(__file__).resolve().parents[1] / RAW_CSV)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory only; historical artifacts are preserved")
    args = parser.parse_args()
    RAW_CSV, OUTPUT_DIR = args.raw_csv, args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    
    print(f"Loading {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV)
    
    cell_ids = df['CellID'].unique()
    print(f"Found {len(cell_ids)} cells.")
    
    for cell_id in cell_ids:
        print(f"\nProcessing CellID {cell_id}...")
        cell_df = df[df['CellID'] == cell_id]
        
        cycles = np.sort(cell_df['Cycle'].unique())
        
        feature_data = []
        capacities = []
        soh_values = []
        cycle_indices = []
        
        for cycle in cycles:
            cycle_df = cell_df[cell_df['Cycle'] == cycle].sort_values(by='Capacity')
            
            # Use raw data for feature extraction
            voltage = cycle_df['Voltage'].values
            current = cycle_df['Current'].values
            capacity = cycle_df['Capacity'].values
            
            try:
                features = extract_features_from_kaggle(voltage, current, capacity)
                
                # Get Max capacity for this cycle
                cycle_max_capacity = capacity.max()
                
                # SoH is already provided
                labels = cycle_df['SoH'].to_numpy(dtype=float)
                if not np.isfinite(labels).all() or np.any((labels < 0) | (labels > 1)):
                    raise ValueError("Invalid supplied SOH labels")
                soh = labels.mean()
                
                feature_data.append(features)
                capacities.append(cycle_max_capacity)
                soh_values.append(soh)
                cycle_indices.append(cycle)
            except Exception as e:
                # Some cycles might not have enough points
                print(f"Skipping cycle {cycle} due to error: {e}")
                
        if len(feature_data) == 0:
            print(f"No usable cycles found for {cell_id}")
            continue
            
        feature_data = np.asarray(feature_data)
        capacities = np.asarray(capacities)
        soh_values = np.asarray(soh_values)
        cycle_indices = np.asarray(cycle_indices)
        
        # Save matching output format
        output_name = f"K_{int(cell_id):04d}"
        output_file = OUTPUT_DIR / f"{output_name}.npz"
        
        save_new_archive(
            output_file,
            features=feature_data,
            capacities=capacities,
            soh=soh_values,
            target_kind="provided_soh_unverified",
            target_provenance_verified=False,
            feature_schema="masked_derivatives_v2",
            source_file=str(RAW_CSV.resolve()),
            cycle_indices=cycle_indices
        )
        print(f"Saved: {output_file}")
        print(f"Feature shape: {feature_data.shape}")
        print(f"SOH shape: {soh_values.shape}")
        
    print("\nKaggle Dataset construction complete.")

if __name__ == "__main__":
    main()
