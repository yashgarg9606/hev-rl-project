"""
Build processed .npz files from HNEI half-cycle CSV data.

Each CSV contains multiple cycling segments for a single battery cell.
This script:
  1. Extracts individual discharge cycles from the raw time-series
  2. Computes a discharge-window capacity proxy, NOT validated battery SOH
  3. Creates 3-channel electrochemical features (dQ/dV, dV/dQ, dI/dV)
  4. Adds a 4th channel: validity mask (1 where data exists, 0 where zero-filled)
  5. Saves per-cell .npz files matching the Kaggle pipeline format
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.artifacts import save_new_archive
from src.features.electrochemical import (
    smooth_signal, interpolate_voltage_domain, calculate_derivative,
    V_MIN, V_MAX, N_POINTS
)

CSV_DIR = Path("data/raw/HalfCycle/csv")
OUTPUT_DIR = Path("data/processed")

# Map cell IDs to their SOC condition for metadata
CELL_SOC_MAP = {
    "PL03": "0-60_HalfC",
    "PL04": "40-60_HalfC",
    "PL05": "40-60_HalfC",
    "PL09": "40-60_2C",
    "PL10": "0-60_HalfC",
    "PL11": "0-100_HalfC",
    "PL12": "0-100_2C",
    "PL13": "0-100_HalfC",
    "PL14": "0-100_2C",
    "PL25": "40-60_2C",
}


def is_full_cycle(soc_condition: str) -> bool:
    """Check if a SOC condition represents a full cycle (0-100%)."""
    return "0-100" in soc_condition


def extract_discharge_cycles(df: pd.DataFrame):
    """
    Extract individual discharge half-cycles from a cell's CSV data.
    
    Discharge is identified by negative current (Current_Amp < 0).
    Each (Segment, Cycle) pair with enough discharge points is one cycle.
    
    Yields:
        (segment, cycle_num, voltage, current, discharge_capacity)
    """
    # Filter for discharge steps (negative current)
    discharge_mask = df['Current_Amp'] < -0.01  # small threshold to avoid noise
    
    if discharge_mask.sum() < 100:
        raise ValueError("Insufficient negative-current discharge samples; current sign is not guessed")
    
    df_dch = df[discharge_mask].copy()
    
    if len(df_dch) == 0:
        return
    
    # Group by Segment and Cycle to get individual discharge cycles
    for (seg, cyc), group in df_dch.groupby(['Segment', 'Cycle']):
        group = group.sort_values('Time_sec')
        
        voltage = group['Voltage_Volt'].values
        current = group['Current_Amp'].values
        
        # Compute capacity from cumulative discharge column
        # The Discharge_Ah column is cumulative within each segment
        discharge_ah = group['Discharge_Ah'].values
        
        # Get per-cycle capacity: difference between end and start
        capacity = discharge_ah - discharge_ah[0]
        
        # Need at least 30 points for meaningful feature extraction
        if len(voltage) < 30:
            continue
        
        # Check voltage range spans at least 0.1V
        v_range = voltage.max() - voltage.min()
        if v_range < 0.1:
            continue
        
        yield int(seg), int(cyc), voltage, np.abs(current), capacity


def extract_features_with_mask(voltage, current, capacity):
    from src.features.electrochemical import extract_masked_features
    return extract_masked_features(voltage, current, capacity)


def canonical_cell_id(name):
    cell = name.split("_SOC_", 1)[0]
    if cell not in CELL_SOC_MAP:
        raise ValueError(f"Unknown cell ID {name!r}; require PLxx or PLxx_SOC_condition")
    return cell


def process_cell(csv_path: Path, cell_name: str):
    """Process a single HNEI cell CSV into an .npz file."""
    cell_name = canonical_cell_id(cell_name)
    soc_condition = CELL_SOC_MAP[cell_name]
    full_cycle = is_full_cycle(soc_condition)
    
    print(f"\n{'='*50}")
    print(f"Processing {cell_name} (SOC: {soc_condition}, "
          f"{'Full' if full_cycle else 'Partial'} cycle)")
    print(f"Loading {csv_path.name}...")
    
    # Read CSV — these files are large, so only load needed columns
    usecols = ['Segment', 'Operation', 'Time_sec', 'Step', 'Cycle',
               'Current_Amp', 'Voltage_Volt', 'Charge_Ah', 'Discharge_Ah']
    df = pd.read_csv(csv_path, usecols=usecols)
    print(f"  Loaded {len(df):,} rows")
    
    # Extract discharge cycles
    feature_list = []
    soh_list = []
    cycle_indices = []
    capacities_list = []
    source_pairs = []
    
    initial_capacity = None
    
    for seg, cyc, voltage, current, capacity in extract_discharge_cycles(df):
        try:
            features = extract_features_with_mask(voltage, current, capacity)
            
            # Get discharge capacity for this cycle
            cycle_capacity = capacity.max()
            
            # Track initial capacity for SOH calculation
            if initial_capacity is None and cycle_capacity > 0.01:
                initial_capacity = cycle_capacity
            
            if initial_capacity is None or initial_capacity < 0.01:
                continue
            
            soh = min(cycle_capacity / initial_capacity, 1.0)
            
            # Skip obvious outliers
            if soh < 0.3 or np.isnan(soh):
                continue
            
            feature_list.append(features)
            soh_list.append(soh)
            capacities_list.append(cycle_capacity)
            source_pairs.append((seg, cyc))
            cycle_indices.append(len(cycle_indices))  # Retained order; original pair recorded separately
            
        except (ValueError, FloatingPointError) as e:
            print(f"Skipping segment={seg} cycle={cyc}: {e}")
            continue
    
    if len(feature_list) == 0:
        print(f"  WARNING: No usable cycles found for {cell_name}")
        return False
    
    feature_data = np.asarray(feature_list, dtype=np.float32)
    soh_values = np.asarray(soh_list, dtype=np.float32)
    capacities = np.asarray(capacities_list, dtype=np.float32)
    cycle_idx = np.asarray(cycle_indices, dtype=np.int32)
    
    # Save
    output_name = f"H_{cell_name}"
    output_file = OUTPUT_DIR / f"{output_name}.npz"
    
    save_new_archive(
        output_file,
        features=feature_data,
        capacities=capacities,
        soh=soh_values,  # Compatibility key: explicitly a proxy, not calibrated health.
        capacity_window_ratio=soh_values,
        target_kind="discharge_window_capacity_proxy",
        target_provenance_verified=False,
        target_definition="window_capacity / first_window_capacity; clipped at 1; ratios below .3 excluded",
        feature_schema="masked_derivatives_v2",
        source_file=str(csv_path.resolve()),
        cycle_indices=cycle_idx,
        source_segment_cycle=np.asarray(source_pairs, dtype=np.int64),
        soc_condition=soc_condition,
        is_full_cycle=full_cycle
    )
    
    print(f"  Saved: {output_file}")
    print(f"  Feature shape: {feature_data.shape}  (cycles, channels, points)")
    print(f"  SOH range: [{soh_values.min():.4f}, {soh_values.max():.4f}]")
    print(f"  Capacity range: [{capacities.min():.4f}, {capacities.max():.4f}] Ah")
    return True


def main():
    import argparse
    global CSV_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Build experimental discharge-window proxies, not calibrated SOH")
    parser.add_argument("--csv-dir", type=Path, default=Path(__file__).resolve().parents[1] / CSV_DIR)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory only")
    parser.add_argument("--acknowledge-proxy-targets", action="store_true", required=True)
    args = parser.parse_args()
    CSV_DIR, OUTPUT_DIR = args.csv_dir, args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    
    csv_files = sorted(CSV_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} HNEI CSV files in {CSV_DIR}")
    
    success = 0
    for csv_path in csv_files:
        cell_name = csv_path.stem  # e.g. "PL03"
        if process_cell(csv_path, cell_name):
            success += 1
    
    print(f"\n{'='*50}")
    print(f"HNEI Dataset construction complete.")
    print(f"Successfully processed {success}/{len(csv_files)} cells.")


if __name__ == "__main__":
    main()
