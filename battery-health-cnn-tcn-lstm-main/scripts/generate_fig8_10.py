import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results/predictions.npz"
DATA_PATH = Path(__file__).resolve().parents[1] / "data/processed/sequences.npz"
OUTPUT_DIR = Path("results/plots/paper_figures")

def main():
    global OUTPUT_DIR
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.visualization.artifacts import historical_plot_directory
    OUTPUT_DIR = historical_plot_directory()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    preds = np.load(RESULTS_PATH)
    y_true = preds["y_true"]
    y_pred = preds["y_pred"]
    
    # Load battery labels to separate by cell
    data = np.load(DATA_PATH)
    batteries = data["batteries"]
    if not np.array_equal(y_true, data["y"]) or y_pred.shape != y_true.shape:
        raise ValueError("Predictions and sequence metadata do not align")
    
    unique_bats = np.unique(batteries)
    
    # Figure 8: Global comparison on a representative cell (e.g., first one)
    target_bat = unique_bats[0]
    indices = np.where(batteries == target_bat)[0]
    
    true_traj = y_true[indices]
    pred_traj = y_pred[indices]
    cycles = np.arange(1, len(indices) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(cycles, true_traj, 'k-', linewidth=2, label="Provided target (unverified)")
    plt.plot(cycles, pred_traj, 'r--', linewidth=2, label="Historical hybrid prediction")
    
    plt.xlabel('Retained prediction index', fontsize=14)
    plt.ylabel('Provided target (fraction)', fontsize=14)
    plt.title(f'Figure 8: Historical all-data target trajectory for Cell {target_bat}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Calculate RMSE for the plot text
    rmse = np.sqrt(np.mean((true_traj - pred_traj)**2))
    plt.text(0.05, 0.05, f'RMSE = {rmse:.4f}', transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
             
    plt.tight_layout()
    out8 = OUTPUT_DIR / "Fig8_SOH_Trajectory.png"
    plt.savefig(out8, dpi=300)
    print(f"Saved: {out8}")
    plt.close()
    
    # Figure 10: Multi-panel plot for 4 cells (using the first 4 available)
    if len(unique_bats) >= 4:
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        axs = axs.flatten()
        
        for i, bat in enumerate(unique_bats[:4]):
            idx = np.where(batteries == bat)[0]
            t_traj = y_true[idx]
            p_traj = y_pred[idx]
            cyc = np.arange(1, len(idx) + 1)
            
            axs[i].plot(cyc, t_traj, 'k-', linewidth=2, label="Provided target (proxy/unverified)")
            # For a true Fig 10, we'd plot GRU, BiLSTM, etc. Here we just plot our hybrid model
            axs[i].plot(cyc, p_traj, 'g-', linewidth=2, label="Proposed CNN-TCN-LSTM")
            
            axs[i].set_title(f'Cell {bat}', fontsize=14)
            axs[i].set_xlabel('Retained prediction index')
            axs[i].set_ylabel('Provided target')
            axs[i].grid(True, linestyle=':', alpha=0.6)
            
            # Zoom inset
            # Find the "accelerated degradation" point, roughly last 20% of cycles
            start_zoom = int(len(cyc) * 0.8)
            end_zoom = len(cyc) - 1
            
            if start_zoom < end_zoom:
                # Add an inset axes
                axins = axs[i].inset_axes([0.1, 0.1, 0.4, 0.3])
                axins.plot(cyc, t_traj, 'k-', linewidth=2)
                axins.plot(cyc, p_traj, 'g-', linewidth=2)
                
                # sub region of the original image
                x1, x2 = cyc[start_zoom], cyc[end_zoom]
                y1, y2 = min(t_traj[start_zoom:end_zoom]), max(t_traj[start_zoom:end_zoom])
                # add some margin
                margin = (y2 - y1) * 0.1
                y1, y2 = y1 - margin, y2 + margin
                
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)
                axins.set_xticklabels([])
                axins.set_yticklabels([])
                
                axs[i].indicate_inset_zoom(axins, edgecolor="black")
                
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=14)
        
        plt.suptitle("Historical all-data predictions; targets include discharge-window proxies", fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        out10 = OUTPUT_DIR / "Fig10_MultiPanel_Zoom.png"
        plt.savefig(out10, dpi=300)
        print(f"Saved: {out10}")
        plt.close()

if __name__ == "__main__":
    main()
