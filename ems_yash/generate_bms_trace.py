"""
Phase 3B — Generate BMS SOH Prediction Trace

Uses the trained CNN-TCN-LSTM-Attention model to generate SOH predictions
from NASA battery dataset, creating a time-aligned trace for EMS experiments.

This script:
1. Loads the trained BMS model
2. Loads NASA battery B0005 processed sequences
3. Generates SOH predictions
4. Creates time-aligned trace with sample timestamps
5. Saves trace for use in EMS experiments
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add BMS repository to path
bms_root = Path(__file__).resolve().parents[1] / '..' / 'bms_gtr'
sys.path.insert(0, str(bms_root))

from src.models.hybrid_model import CNNTCNLSTMAttention


def generate_bms_soh_trace(
    battery_name='B0005',
    cycles_per_sample=20,
    hours_per_cycle=1.0,
):
    """
    Generate BMS SOH prediction trace from NASA battery data.

    Parameters
    ----------
    battery_name : str
        NASA battery to use (B0005, B0006, B0007, B0018)
    cycles_per_sample : int
        Number of cycles per sequence (20 for BMS model)
    hours_per_cycle : float
        Approximate time per charge/discharge cycle

    Returns
    -------
    trace_data : dict
        Contains time_seconds, soh_true, soh_predicted
    """

    print("=" * 70)
    print("PHASE 3B — BMS SOH PREDICTION TRACE GENERATION")
    print("=" * 70)
    print()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # Load trained BMS model
    print(f"Loading trained BMS model from: {bms_root}/results/best_model.pt")

    model = CNNTCNLSTMAttention(
        cnn_channels=64,
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.2
    )

    model.load_state_dict(
        torch.load(
            bms_root / "results" / "best_model.pt",
            map_location=device,
            weights_only=True
        )
    )

    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    print()

    # Load NASA battery data (use combined sequences file)
    sequences_file = bms_root / "data" / "processed" / "sequences.npz"
    print(f"Loading NASA battery sequences: {sequences_file}")

    data = np.load(sequences_file)
    X = data['X'].astype(np.float32)  # (N_samples, 20, 3, 300)
    y_true = data['y'].astype(np.float32)  # True SOH values
    batteries = data['batteries']  # Battery names per sample

    print(f"Data shape: {X.shape}")
    print(f"Number of samples: {len(X)}")
    print(f"True SOH range: [{y_true.min():.4f}, {y_true.max():.4f}]")
    print(f"Batteries in dataset: {np.unique(batteries)}")

    # Use all data (cross-battery trace)
    print(f"Using all {len(X)} samples from combined dataset")
    print()

    # Generate predictions
    print("Generating BMS predictions...")

    X_tensor = torch.tensor(X).to(device)
    predictions = []

    with torch.no_grad():
        batch_size = 32
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            pred, _ = model(batch)
            predictions.extend(pred.cpu().numpy())

    predictions = np.array(predictions)

    print(f"Predicted SOH range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print()

    # Calculate error metrics
    errors = predictions - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    max_error = np.max(np.abs(errors))

    print("BMS Prediction Quality:")
    print(f"  Mean Absolute Error:  {mae:.6f}")
    print(f"  Root Mean Square Error: {rmse:.6f}")
    print(f"  Maximum Absolute Error: {max_error:.6f}")
    print()

    # Create time-aligned trace
    # Each sample represents 20 consecutive cycles
    # Assume ~1 hour per cycle → 20 hours per sample

    time_hours = np.arange(len(predictions)) * cycles_per_sample * hours_per_cycle
    time_seconds = time_hours * 3600.0

    print("Time Alignment:")
    print(f"  Cycles per sample: {cycles_per_sample}")
    print(f"  Hours per cycle: {hours_per_cycle}")
    print(f"  Time per sample: {cycles_per_sample * hours_per_cycle} hours")
    print(f"  Total duration: {time_hours[-1]:.1f} hours ({time_seconds[-1]/3600:.1f} hours)")
    print()

    # Prepare trace data
    trace_data = {
        'battery_name': 'combined',  # Using all batteries
        'time_seconds': time_seconds,
        'soh_true': y_true,
        'soh_predicted': predictions,
        'batteries': batteries,
        'cycles_per_sample': cycles_per_sample,
        'hours_per_cycle': hours_per_cycle,
        'mae': mae,
        'rmse': rmse,
        'max_error': max_error,
    }

    return trace_data


def save_trace(trace_data, output_path):
    """Save BMS trace to file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        **trace_data
    )

    print(f"Trace saved to: {output_path}")
    print()


def main():
    # Generate trace from combined NASA dataset
    trace_data = generate_bms_soh_trace(
        battery_name='combined',
        cycles_per_sample=20,
        hours_per_cycle=1.0,
    )

    # Save trace
    output_path = Path(__file__).parent / "data" / "bms_soh_trace.npz"
    save_trace(trace_data, output_path)

    print("=" * 70)
    print("TRACE GENERATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Battery: {trace_data['battery_name']}")
    print(f"Samples: {len(trace_data['soh_predicted'])}")
    print(f"Duration: {trace_data['time_seconds'][-1]/3600:.1f} hours")
    print(f"Prediction MAE: {trace_data['mae']:.6f}")
    print()
    print("This trace can now be used in Phase 3B BMS_ESTIMATED experiments.")
    print()


if __name__ == "__main__":
    main()
