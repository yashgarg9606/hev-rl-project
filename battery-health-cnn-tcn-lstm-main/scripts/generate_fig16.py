import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.features.electrochemical import smooth_signal, interpolate_voltage_domain, calculate_derivative, V_MIN, V_MAX, N_POINTS
from src.models.hybrid_model import CNNTCNLSTMAttention

RAW_CSV = Path(__file__).resolve().parents[1] / "data/raw/kaggle/cell_level_dataset_multichem.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "results/best_model.pt"
OUTPUT_DIR = Path("results/plots/paper_figures")
SEQUENCE_LENGTH = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_cell(df, cell_id):
    cell_df = df[df['CellID'] == cell_id]
    cycles = cell_df['Cycle'].unique()
    
    features = []
    soh_values = []
    
    voltage_grid = np.linspace(V_MIN, V_MAX, N_POINTS)
    
    for cycle in cycles:
        cycle_df = cell_df[cell_df['Cycle'] == cycle].sort_values(by='Capacity')
        
        voltage = cycle_df['Voltage'].values
        current = cycle_df['Current'].values
        capacity = cycle_df['Capacity'].values
        soh = cycle_df['SoH'].iloc[0]
        
        try:
            voltage_smooth = smooth_signal(voltage)
            current_smooth = smooth_signal(current)
            capacity_smooth = smooth_signal(capacity)
            
            _, current_interp = interpolate_voltage_domain(voltage_smooth, current_smooth)
            _, capacity_interp = interpolate_voltage_domain(voltage_smooth, capacity_smooth)
            
            dq_dv = calculate_derivative(capacity_interp, voltage_grid)
            dv_dq = calculate_derivative(voltage_grid, capacity_interp)
            di_dv = calculate_derivative(current_interp, voltage_grid)
            
            # Historical checkpoint used all-one validity; corrected v2 features require refitting.
            validity = np.ones_like(dq_dv)
            cycle_features = np.stack([dq_dv, dv_dq, di_dv, validity], axis=0)
            
            features.append(cycle_features)
            soh_values.append(soh)
        except Exception:
            continue
            
    features = np.array(features, dtype=np.float32)
    soh_values = np.array(soh_values, dtype=np.float32)
    
    # Build sequences
    X, y = [], []
    for i in range(len(features) - SEQUENCE_LENGTH):
        X.append(features[i : i + SEQUENCE_LENGTH])
        y.append(soh_values[i + SEQUENCE_LENGTH])
        
    return np.array(X), np.array(y)

def main():
    global OUTPUT_DIR
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.visualization.artifacts import historical_plot_directory
    OUTPUT_DIR = historical_plot_directory()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV)
    
    # Process Cell 1 (NMC) and Cell 13 (LFP)
    print("Processing Cell 1 (NMC)...")
    X_nmc, y_nmc = process_cell(df, 1)
    
    print("Processing Cell 13 (LFP)...")
    X_lfp, y_lfp = process_cell(df, 13)
    
    # Load Model
    model = CNNTCNLSTMAttention().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    def predict(X):
        X_tensor = torch.tensor(X)
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), 64):
                batch = X_tensor[i:i+64].to(DEVICE)
                prediction, _, _ = model(batch)
                predictions.extend(prediction.cpu().numpy())
        return np.asarray(predictions)
        
    print("Predicting NMC...")
    pred_nmc = predict(X_nmc)
    
    print("Predicting LFP...")
    pred_lfp = predict(X_lfp)
    
    # Plot Figure 16
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # NMC Plot
    cycles_nmc = np.arange(1, len(y_nmc) + 1)
    axs[0].plot(cycles_nmc, y_nmc, 'k-', linewidth=2, label="Provided target (unverified)")
    axs[0].plot(cycles_nmc, pred_nmc, 'b--', linewidth=2, label="Predicted SOH")
    axs[0].set_title('Provided multichem CSV: cell 1 (NMC)', fontsize=14)
    axs[0].set_xlabel('Retained prediction index')
    axs[0].set_ylabel('SOH')
    axs[0].grid(True, linestyle=':')
    axs[0].legend()
    
    # LFP Plot
    cycles_lfp = np.arange(1, len(y_lfp) + 1)
    axs[1].plot(cycles_lfp, y_lfp, 'k-', linewidth=2, label="Provided target (unverified)")
    axs[1].plot(cycles_lfp, pred_lfp, 'r--', linewidth=2, label="Predicted SOH")
    axs[1].set_title('Provided multichem CSV: cell 13 (LFP)', fontsize=14)
    axs[1].set_xlabel('Retained prediction index')
    axs[1].grid(True, linestyle=':')
    axs[1].legend()
    
    plt.suptitle("Figure 16: Historical checkpoint on supplied multichemistry CSV (unverified targets)", fontsize=16)
    plt.tight_layout()
    
    out16 = OUTPUT_DIR / "Fig16_Cross_Dataset.png"
    plt.savefig(out16, dpi=300)
    print(f"Saved: {out16}")

if __name__ == "__main__":
    main()
