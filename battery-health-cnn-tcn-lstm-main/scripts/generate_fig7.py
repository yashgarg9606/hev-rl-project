import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.features.electrochemical import smooth_signal, interpolate_voltage_domain, calculate_derivative, V_MIN, V_MAX, N_POINTS

RAW_CSV = Path(__file__).resolve().parents[1] / "data/raw/kaggle/cell_level_dataset.csv"
OUTPUT_DIR = Path("results/plots/paper_figures")

def main():
    global OUTPUT_DIR
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.visualization.artifacts import historical_plot_directory
    OUTPUT_DIR = historical_plot_directory()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV)
    
    # Pick a representative cell, e.g., CellID 1
    cell_df = df[df['CellID'] == 1]
    
    # Pick a healthy cycle (early life) and a degraded cycle (late life)
    cycles = cell_df['Cycle'].unique()
    healthy_cycle = cycles[0]
    degraded_cycle = cycles[-1]
    
    healthy_df = cell_df[cell_df['Cycle'] == healthy_cycle].sort_values(by='Capacity')
    degraded_df = cell_df[cell_df['Cycle'] == degraded_cycle].sort_values(by='Capacity')
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    voltage_grid = np.linspace(V_MIN, V_MAX, N_POINTS)
    
    for cycle_df, label, color, linestyle in zip([healthy_df, degraded_df], 
                                                ['Early supplied cycle', 'Late supplied cycle'], 
                                                ['blue', 'orange'], 
                                                ['-', '--']):
        
        voltage = cycle_df['Voltage'].values
        current = cycle_df['Current'].values
        capacity = cycle_df['Capacity'].values
        
        voltage_smooth = smooth_signal(voltage)
        current_smooth = smooth_signal(current)
        capacity_smooth = smooth_signal(capacity)
        
        # Plot (a) Voltage-Capacity Curve
        axs[0, 0].plot(capacity_smooth, voltage_smooth, label=label, color=color, linestyle=linestyle, linewidth=2)
        
        # Calculate differentials
        try:
            _, current_interp = interpolate_voltage_domain(voltage_smooth, current_smooth)
            _, capacity_interp = interpolate_voltage_domain(voltage_smooth, capacity_smooth)
        except ValueError:
            continue
            
        dv_dq = calculate_derivative(voltage_grid, capacity_interp)
        dq_dv = calculate_derivative(capacity_interp, voltage_grid)
        di_dv = calculate_derivative(current_interp, voltage_grid)
        
        # Plot (b) ICA (dQ/dV)
        axs[0, 1].plot(voltage_grid, dq_dv, label=label, color=color, linestyle=linestyle, linewidth=2)
        
        # Plot (c) DV (dV/dQ)
        axs[1, 0].plot(capacity_interp, dv_dq, label=label, color=color, linestyle=linestyle, linewidth=2)
        
        # Plot (d) DC (dI/dV)
        axs[1, 1].plot(voltage_grid, di_dv, label=label, color=color, linestyle=linestyle, linewidth=2)

    # Styling
    axs[0, 0].set_title('(a) Voltage-Capacity Curve', fontsize=14)
    axs[0, 0].set_xlabel('Capacity (Ah)', fontsize=12)
    axs[0, 0].set_ylabel('Voltage (V)', fontsize=12)
    axs[0, 0].grid(True)
    
    axs[0, 1].set_title('(b) Corresponding ICA Curve (dQ/dV)', fontsize=14)
    axs[0, 1].set_xlabel('Voltage (V)', fontsize=12)
    axs[0, 1].set_ylabel('dQ/dV (Ah/V)', fontsize=12)
    axs[0, 1].grid(True)
    
    axs[1, 0].set_title('(c) Differential Voltage (dV/dQ)', fontsize=14)
    axs[1, 0].set_xlabel('Capacity (Ah)', fontsize=12)
    axs[1, 0].set_ylabel('dV/dQ (V/Ah)', fontsize=12)
    axs[1, 0].grid(True)
    
    axs[1, 1].set_title('(d) Differential Current (dI/dV)', fontsize=14)
    axs[1, 1].set_xlabel('Voltage (V)', fontsize=12)
    axs[1, 1].set_ylabel('dI/dV (A/V)', fontsize=12)
    axs[1, 1].grid(True)
    
    # Shared Legend
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    output_path = OUTPUT_DIR / "Fig7_Feature_Extraction.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
