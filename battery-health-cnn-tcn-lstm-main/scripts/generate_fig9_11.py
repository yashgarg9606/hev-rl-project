import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results/predictions.npz"
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
    
    error = (y_pred - y_true) * 100 # Convert to percentage error
    
    # Figure 9: Error Distribution
    plt.figure(figsize=(10, 6))
    
    n, bins, patches = plt.hist(error, bins=50, color='blue', alpha=0.7)
    
    # Highlight the ±1.5% bounds
    plt.axvline(x=-1.5, color='red', linestyle='--', linewidth=3)
    plt.axvline(x=1.5, color='red', linestyle='--', linewidth=3)
    
    # Calculate percentage within bounds
    within_bounds = np.sum((error >= -1.5) & (error <= 1.5)) / len(error) * 100
    
    plt.text(0.5, 0.85, f'{within_bounds:.1f}% within ±1.5 percentage points', 
             horizontalalignment='center', verticalalignment='center', 
             transform=plt.gca().transAxes, fontsize=16, color='darkred', fontweight='bold')
             
    plt.xlabel('Prediction error (percentage points)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Figure 9: Error Distribution of Historical all-data target predictions', fontsize=16)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    out9 = OUTPUT_DIR / "Fig9_Error_Distribution.png"
    plt.savefig(out9, dpi=300)
    print(f"Saved: {out9}")
    plt.close()
    
    # Figure 11: Late Stage Aging Analysis
    abs_error = np.abs(y_pred - y_true)
    
    # Create bins for SOH
    bins = np.linspace(float(y_true.min()), np.nextafter(float(y_true.max()), np.inf), 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bin_indices = np.clip(np.digitize(y_true, bins), 1, len(bins)-1)
    
    variances = []
    maes = []
    
    for i in range(1, len(bins)):
        mask = (bin_indices == i)
        if np.sum(mask) > 0:
            variances.append(np.var(error[mask]))
            maes.append(np.mean(abs_error[mask]))
        else:
            variances.append(np.nan)
            maes.append(np.nan)
            
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # (b) Absolute Error vs provided target
    axs[0].scatter(y_true, abs_error, alpha=0.5, color='dodgerblue', s=10)
    axs[0].set_xlabel('Provided target (proxy/unverified)')
    axs[0].set_ylabel('|Prediction Error|')
    axs[0].set_title('(b) Absolute Error vs provided target')
    axs[0].invert_xaxis() # Plot from high SOH to low SOH
    axs[0].grid(True, linestyle=':')
    
    # (c) Error Variance Across target bins
    axs[1].plot(bin_centers, variances, 'o-', linewidth=2, markersize=8)
    axs[1].set_xlabel('Provided target bin center')
    axs[1].set_ylabel('Error Variance (Var(e))')
    axs[1].set_title('(c) Error Variance Across target bins')
    axs[1].invert_xaxis()
    axs[1].legend()
    axs[1].grid(True, linestyle=':')
    
    # (d) Mean Absolute Error Across target bins
    axs[2].plot(bin_centers, maes, 'o-', linewidth=2, markersize=8)
    axs[2].set_xlabel('Provided target bin center')
    axs[2].set_ylabel('Mean Absolute Error')
    axs[2].set_title('(d) Mean Absolute Error Across target bins')
    axs[2].invert_xaxis()
    axs[2].legend()
    axs[2].grid(True, linestyle=':')
    
    plt.suptitle("Historical all-data errors across provided targets (not validated aging)", fontsize=18)
    plt.tight_layout()
    out11 = OUTPUT_DIR / "Fig11_LateStage_Aging.png"
    plt.savefig(out11, dpi=300)
    print(f"Saved: {out11}")
    plt.close()

if __name__ == "__main__":
    main()
