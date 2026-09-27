import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.hybrid_model import CNNTCNLSTMAttention
from src.features.electrochemical import V_MIN, V_MAX, N_POINTS

DATA_PATH = Path(__file__).resolve().parents[1] / "data/processed/sequences.npz"
MODEL_PATH = Path(__file__).resolve().parents[1] / "results/best_model.pt"
OUTPUT_DIR = Path("results/plots/paper_figures")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global OUTPUT_DIR
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.visualization.artifacts import historical_plot_directory
    OUTPUT_DIR = historical_plot_directory()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    data = np.load(DATA_PATH)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    
    model = CNNTCNLSTMAttention().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # Pick a random sample (e.g., sample 100)
    sample_idx = 100
    sequence = X[sample_idx:sample_idx+1]
    sequence_tensor = torch.tensor(sequence).to(DEVICE)
    
    with torch.no_grad():
        _, temp_attn, spatial_attn = model(sequence_tensor)
        
    # spatial_attn shape: (batch_size, sequence_length, 1, 300)
    spatial_attn = spatial_attn.cpu().numpy()[0] # (20, 1, 300)
    
    # Average across the historical cycles
    mean_spatial_attn = np.mean(spatial_attn, axis=0).squeeze() # (300,)
    
    # Normalize attention for visualization
    mean_spatial_attn = (mean_spatial_attn - mean_spatial_attn.min()) / max(float(np.ptp(mean_spatial_attn)), np.finfo(float).eps)
    
    # Match the attention aggregation: average the same historical cycles.
    ica_curve = sequence[0, :, 0, :].mean(axis=0)
    
    # Normalize ICA
    ica_curve = (ica_curve - ica_curve.min()) / max(float(np.ptp(ica_curve)), np.finfo(float).eps)
    
    voltage_grid = np.linspace(V_MIN, V_MAX, N_POINTS)
    
    # Figure 18 & 20a: Attention mapped on ICA
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Voltage (V)', fontsize=14)
    ax1.set_ylabel('Mean historical dQ/dV (scaled)', color=color, fontsize=14)
    ax1.plot(voltage_grid, ica_curve, color=color, linewidth=2, label="ICA")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Attention Weight (normalized)', color=color, fontsize=14)  
    ax2.plot(voltage_grid, mean_spatial_attn, color=color, linestyle='--', linewidth=2, label="Attention")
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Show only above-threshold locations; do not fill gaps between peaks.
    ax2.fill_between(voltage_grid, 0., 1., where=mean_spatial_attn > .8,
                     transform=ax2.get_xaxis_transform(), color='red', alpha=.2)
    plt.title('Historical learned spatial weights and mean ICA (descriptive)', fontsize=14)

    fig.tight_layout()  
    
    output_path = OUTPUT_DIR / "Fig18_20a_Spatial_Attention.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
