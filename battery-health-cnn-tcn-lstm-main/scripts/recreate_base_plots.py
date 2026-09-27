"""
Recreate ALL base paper figures using ONLY the original Kaggle dataset (3 channels).
Saves everything to results/plots/base_paper_images/
"""

if __name__ == "__main__":
    raise SystemExit("Random-split base-plot training retired. Use run_purged_lobo.py and plot_run.py; this script does not reproduce either paper. Historical artifacts are preserved.")

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.hybrid_model import CNNTCNLSTMAttention
from src.training.trainer import Trainer
from src.features.electrochemical import (
    smooth_signal, interpolate_voltage_domain, calculate_derivative,
    V_MIN, V_MAX, N_POINTS
)

SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 200
PATIENCE = 20
SEQUENCE_LENGTH = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_CSV = Path("data/raw/kaggle/cell_level_dataset.csv")
RAW_CSV_MULTI = Path("data/raw/kaggle/cell_level_dataset_multichem.csv")
OUTPUT_DIR = Path("results/plots/base_paper_images")
BASE_MODEL_PATH = Path("results/base_model_3ch.pt")


def extract_features_3ch(voltage, current, capacity):
    """Extract 3-channel features (original paper format)."""
    voltage_smooth = smooth_signal(voltage)
    current_smooth = smooth_signal(current)
    capacity_smooth = smooth_signal(capacity)

    _, current_interp = interpolate_voltage_domain(voltage_smooth, current_smooth)
    _, capacity_interp = interpolate_voltage_domain(voltage_smooth, capacity_smooth)

    voltage_grid = np.linspace(V_MIN, V_MAX, N_POINTS)
    dq_dv = calculate_derivative(capacity_interp, voltage_grid)
    dv_dq = calculate_derivative(voltage_grid, capacity_interp)
    di_dv = calculate_derivative(current_interp, voltage_grid)

    return np.stack([dq_dv, dv_dq, di_dv], axis=0)


def build_kaggle_features():
    """Build per-cell features from the Kaggle CSV."""
    print(f"Loading {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV)
    cell_ids = df['CellID'].unique()
    print(f"Found {len(cell_ids)} cells.")

    all_features = {}
    for cell_id in cell_ids:
        cell_df = df[df['CellID'] == cell_id]
        cycles = cell_df['Cycle'].unique()

        feat_list, soh_list = [], []
        for cycle in cycles:
            cycle_df = cell_df[cell_df['Cycle'] == cycle].sort_values(by='Capacity')
            try:
                feats = extract_features_3ch(
                    cycle_df['Voltage'].values,
                    cycle_df['Current'].values,
                    cycle_df['Capacity'].values
                )
                feat_list.append(feats)
                soh_list.append(cycle_df['SoH'].mean())
            except Exception:
                continue

        if feat_list:
            all_features[f"K_{int(cell_id):04d}"] = {
                'features': np.array(feat_list, dtype=np.float32),
                'soh': np.array(soh_list, dtype=np.float32)
            }
            print(f"  Cell {cell_id}: {len(feat_list)} cycles")

    return all_features


def build_sequences(all_features):
    """Build sliding window sequences."""
    X_all, y_all, bats_all = [], [], []

    for bat_name, data in all_features.items():
        feats = data['features']
        soh = data['soh']
        for i in range(len(feats) - SEQUENCE_LENGTH):
            X_all.append(feats[i:i + SEQUENCE_LENGTH])
            y_all.append(soh[i + SEQUENCE_LENGTH])
            bats_all.append(bat_name)

    return (
        np.array(X_all, dtype=np.float32),
        np.array(y_all, dtype=np.float32),
        np.array(bats_all)
    )


def train_base_model(X, y):
    """Train a 3-channel base model."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = CNNTCNLSTMAttention(in_channels=3).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    trainer = Trainer(
        model=model, optimizer=optimizer, criterion=criterion,
        scheduler=scheduler, device=DEVICE, patience=PATIENCE
    )
    history = trainer.fit(train_loader, val_loader, EPOCHS)

    # Save model
    torch.save(model.state_dict(), str(BASE_MODEL_PATH))
    print(f"Saved base model to {BASE_MODEL_PATH}")

    return model, history


def predict_all(model, X):
    """Run predictions on all data."""
    model.eval()
    X_tensor = torch.tensor(X)
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 64):
            batch = X_tensor[i:i + 64].to(DEVICE)
            pred, _, _ = model(batch)
            predictions.extend(pred.cpu().numpy())
    return np.asarray(predictions)


# ──────────────────────── Figure Generation ────────────────────────

def generate_fig7():
    """Figure 7: Feature Extraction Visualization."""
    print("Generating Figure 7...")
    df = pd.read_csv(RAW_CSV)
    cell_df = df[df['CellID'] == 1]
    cycles = cell_df['Cycle'].unique()
    healthy_cycle = cycles[0]
    degraded_cycle = cycles[-1]

    healthy_df = cell_df[cell_df['Cycle'] == healthy_cycle].sort_values(by='Capacity')
    degraded_df = cell_df[cell_df['Cycle'] == degraded_cycle].sort_values(by='Capacity')

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    voltage_grid = np.linspace(V_MIN, V_MAX, N_POINTS)

    for cycle_df, label, color, ls in zip(
        [healthy_df, degraded_df], ['Healthy', 'Degraded'],
        ['blue', 'orange'], ['-', '--']
    ):
        v = cycle_df['Voltage'].values
        c = cycle_df['Current'].values
        q = cycle_df['Capacity'].values

        vs = smooth_signal(v)
        cs = smooth_signal(c)
        qs = smooth_signal(q)

        axs[0, 0].plot(qs, vs, label=label, color=color, linestyle=ls, linewidth=2)

        try:
            _, ci = interpolate_voltage_domain(vs, cs)
            _, qi = interpolate_voltage_domain(vs, qs)
        except ValueError:
            continue

        dv_dq = calculate_derivative(voltage_grid, qi)
        dq_dv = calculate_derivative(qi, voltage_grid)
        di_dv = calculate_derivative(ci, voltage_grid)

        axs[0, 1].plot(voltage_grid, dq_dv, label=label, color=color, linestyle=ls, linewidth=2)
        axs[1, 0].plot(qi, dv_dq, label=label, color=color, linestyle=ls, linewidth=2)
        axs[1, 1].plot(voltage_grid, di_dv, label=label, color=color, linestyle=ls, linewidth=2)

    axs[0, 0].set_title('(a) Voltage-Capacity Curve', fontsize=14)
    axs[0, 0].set_xlabel('Capacity (Ah)'); axs[0, 0].set_ylabel('Voltage (V)'); axs[0, 0].grid(True)
    axs[0, 1].set_title('(b) ICA Curve (dQ/dV)', fontsize=14)
    axs[0, 1].set_xlabel('Voltage (V)'); axs[0, 1].set_ylabel('dQ/dV (Ah/V)'); axs[0, 1].grid(True)
    axs[1, 0].set_title('(c) Differential Voltage (dV/dQ)', fontsize=14)
    axs[1, 0].set_xlabel('Capacity (Ah)'); axs[1, 0].set_ylabel('dV/dQ (V/Ah)'); axs[1, 0].grid(True)
    axs[1, 1].set_title('(d) Differential Current (dI/dV)', fontsize=14)
    axs[1, 1].set_xlabel('Voltage (V)'); axs[1, 1].set_ylabel('dI/dV (A/V)'); axs[1, 1].grid(True)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUTPUT_DIR / "Fig7_Feature_Extraction.png", dpi=300)
    plt.close()
    print("  Saved Fig7.")


def generate_fig8(y_true, y_pred, batteries):
    """Figure 8: SOH Prediction Trajectory."""
    print("Generating Figure 8...")
    target = np.unique(batteries)[0]
    idx = np.where(batteries == target)[0]
    t = y_true[idx]; p = y_pred[idx]
    cyc = np.arange(1, len(idx) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(cyc, t, 'k-', lw=2, label="Measured SOH")
    plt.plot(cyc, p, 'r--', lw=2, label="Predicted SOH (Hybrid CNN-TCN-LSTM)")
    plt.xlabel('Cycle Number', fontsize=14); plt.ylabel('SOH', fontsize=14)
    plt.title(f'Figure 8: SOH Prediction Trajectory – Cell {target}', fontsize=16)
    plt.legend(fontsize=12); plt.grid(True, linestyle=':', alpha=0.7)
    rmse = np.sqrt(np.mean((t - p) ** 2))
    plt.text(0.05, 0.05, f'RMSE = {rmse:.4f}', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig8_SOH_Trajectory.png", dpi=300)
    plt.close()
    print("  Saved Fig8.")


def generate_fig9(y_true, y_pred):
    """Figure 9: Error Distribution."""
    print("Generating Figure 9...")
    error = (y_pred - y_true) * 100

    plt.figure(figsize=(10, 6))
    plt.hist(error, bins=50, color='blue', alpha=0.7)
    plt.axvline(x=-1.5, color='red', linestyle='--', linewidth=3)
    plt.axvline(x=1.5, color='red', linestyle='--', linewidth=3)
    within = np.sum((error >= -1.5) & (error <= 1.5)) / len(error) * 100
    plt.text(0.5, 0.85, f'{within:.1f}% within ±1.5%',
             ha='center', va='center', transform=plt.gca().transAxes,
             fontsize=16, color='darkred', fontweight='bold')
    plt.xlabel('Prediction Error (%)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Figure 9: Error Distribution of Predicted SOH', fontsize=16)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig9_Error_Distribution.png", dpi=300)
    plt.close()
    print("  Saved Fig9.")


def generate_fig10(y_true, y_pred, batteries):
    """Figure 10: Multi-Panel Zoom."""
    print("Generating Figure 10...")
    unique_bats = np.unique(batteries)
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    for i, bat in enumerate(unique_bats[:4]):
        idx = np.where(batteries == bat)[0]
        t = y_true[idx]; p = y_pred[idx]
        cyc = np.arange(1, len(idx) + 1)

        axs[i].plot(cyc, t, 'k-', lw=2, label="Real SOH")
        axs[i].plot(cyc, p, 'g-', lw=2, label="Proposed CNN-TCN-LSTM")
        axs[i].set_title(f'Cell {bat}', fontsize=14)
        axs[i].set_xlabel('Cycle Number'); axs[i].set_ylabel('SOH')
        axs[i].grid(True, linestyle=':', alpha=0.6)

        sz = int(len(cyc) * 0.8); ez = len(cyc) - 1
        if sz < ez:
            axins = axs[i].inset_axes([0.1, 0.1, 0.4, 0.3])
            axins.plot(cyc, t, 'k-', lw=2)
            axins.plot(cyc, p, 'g-', lw=2)
            x1, x2 = cyc[sz], cyc[ez]
            y1, y2 = min(t[sz:ez]), max(t[sz:ez])
            m = (y2 - y1) * 0.1
            axins.set_xlim(x1, x2); axins.set_ylim(y1 - m, y2 + m)
            axins.set_xticklabels([]); axins.set_yticklabels([])
            axs[i].indicate_inset_zoom(axins, edgecolor="black")

    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, fontsize=14)
    plt.suptitle("Figure 10: Prediction performance with zoomed insets", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "Fig10_MultiPanel_Zoom.png", dpi=300)
    plt.close()
    print("  Saved Fig10.")


def generate_fig11(y_true, y_pred):
    """Figure 11: Late-Stage Aging."""
    print("Generating Figure 11...")
    error = (y_pred - y_true) * 100
    abs_error = np.abs(y_pred - y_true)

    bins = np.linspace(0.65, 1.0, 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bi = np.digitize(y_true, bins)

    variances, maes = [], []
    for i in range(1, len(bins)):
        mask = (bi == i)
        if np.sum(mask) > 0:
            variances.append(np.var(error[mask]))
            maes.append(np.mean(abs_error[mask]))
        else:
            variances.append(0); maes.append(0)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].scatter(y_true, abs_error, alpha=0.5, color='dodgerblue', s=10)
    axs[0].set_xlabel('True SOH'); axs[0].set_ylabel('|Prediction Error|')
    axs[0].set_title('(b) Absolute Error vs SOH'); axs[0].invert_xaxis(); axs[0].grid(True, linestyle=':')

    axs[1].plot(bin_centers, variances, 'o-', lw=2, ms=8)
    axs[1].axvline(x=0.7, color='red', linestyle='--', label='Late-Life Threshold')
    axs[1].set_xlabel('SOH Bin Center'); axs[1].set_ylabel('Error Variance')
    axs[1].set_title('(c) Error Variance Across SOH'); axs[1].invert_xaxis()
    axs[1].legend(); axs[1].grid(True, linestyle=':')

    axs[2].plot(bin_centers, maes, 'o-', lw=2, ms=8)
    axs[2].axvline(x=0.7, color='red', linestyle='--', label='Late-Life Threshold')
    axs[2].set_xlabel('SOH Bin Center'); axs[2].set_ylabel('MAE')
    axs[2].set_title('(d) MAE Across SOH'); axs[2].invert_xaxis()
    axs[2].legend(); axs[2].grid(True, linestyle=':')

    plt.suptitle("Figure 11: Late-Stage Aging Analysis", fontsize=18)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig11_LateStage_Aging.png", dpi=300)
    plt.close()
    print("  Saved Fig11.")


def generate_fig16(model):
    """Figure 16: Cross-Dataset Generalization (NMC vs LFP)."""
    print("Generating Figure 16...")
    if not RAW_CSV_MULTI.exists():
        print("  Skipping – multichem CSV not found.")
        return

    df = pd.read_csv(RAW_CSV_MULTI)

    def process_cell(cell_id):
        cell_df = df[df['CellID'] == cell_id]
        cycles = cell_df['Cycle'].unique()
        feats, sohs = [], []
        for cycle in cycles:
            cdf = cell_df[cell_df['Cycle'] == cycle].sort_values(by='Capacity')
            try:
                f = extract_features_3ch(cdf['Voltage'].values, cdf['Current'].values, cdf['Capacity'].values)
                feats.append(f); sohs.append(cdf['SoH'].iloc[0])
            except Exception:
                continue
        feats = np.array(feats, dtype=np.float32)
        sohs = np.array(sohs, dtype=np.float32)
        X, y = [], []
        for i in range(len(feats) - SEQUENCE_LENGTH):
            X.append(feats[i:i + SEQUENCE_LENGTH])
            y.append(sohs[i + SEQUENCE_LENGTH])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def pred(X):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                batch = torch.tensor(X[i:i + 64]).to(DEVICE)
                p, _, _ = model(batch)
                preds.extend(p.cpu().numpy())
        return np.asarray(preds)

    print("  Processing Cell 1 (NMC)...")
    X_nmc, y_nmc = process_cell(1)
    print("  Processing Cell 13 (LFP)...")
    X_lfp, y_lfp = process_cell(13)

    pred_nmc = pred(X_nmc)
    pred_lfp = pred(X_lfp)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    c_nmc = np.arange(1, len(y_nmc) + 1)
    axs[0].plot(c_nmc, y_nmc, 'k-', lw=2, label="Measured SOH")
    axs[0].plot(c_nmc, pred_nmc, 'b--', lw=2, label="Predicted SOH")
    axs[0].set_title('Provided multichem CSV: cell 1 (NMC)', fontsize=14)
    axs[0].set_xlabel('Cycle Number'); axs[0].set_ylabel('SOH')
    axs[0].grid(True, linestyle=':'); axs[0].legend()

    c_lfp = np.arange(1, len(y_lfp) + 1)
    axs[1].plot(c_lfp, y_lfp, 'k-', lw=2, label="Measured SOH")
    axs[1].plot(c_lfp, pred_lfp, 'r--', lw=2, label="Predicted SOH")
    axs[1].set_title('Provided multichem CSV: cell 13 (LFP)', fontsize=14)
    axs[1].set_xlabel('Cycle Number')
    axs[1].grid(True, linestyle=':'); axs[1].legend()

    plt.suptitle("Figure 16: Historical supplied multichemistry CSV (unverified targets)", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig16_Cross_Dataset.png", dpi=300)
    plt.close()
    print("  Saved Fig16.")


def generate_fig18_20(model, X):
    """Figures 18 & 20a: Spatial Attention Visualization."""
    print("Generating Figures 18 & 20a...")
    model.eval()
    sample_idx = min(100, len(X) - 1)
    seq = X[sample_idx:sample_idx + 1]
    seq_t = torch.tensor(seq).to(DEVICE)

    with torch.no_grad():
        _, temp_attn, spatial_attn = model(seq_t)

    spatial = spatial_attn.cpu().numpy()[0]
    mean_s = np.mean(spatial, axis=0).squeeze()
    mean_s = (mean_s - mean_s.min()) / (mean_s.max() - mean_s.min())

    ica = seq[0, -1, 0, :]
    ica = (ica - ica.min()) / (ica.max() - ica.min())

    vg = np.linspace(V_MIN, V_MAX, N_POINTS)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Voltage (V)', fontsize=14)
    ax1.set_ylabel('Normalized dQ/dV', color='tab:blue', fontsize=14)
    ax1.plot(vg, ica, color='tab:blue', lw=2, label="ICA")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Attention Weight', color='tab:red', fontsize=14)
    ax2.plot(vg, mean_s, color='tab:red', linestyle='--', lw=2, label="Attention")
    ax2.tick_params(axis='y', labelcolor='tab:red')

    hi = np.where(mean_s > 0.8)[0]
    if len(hi) > 0:
        ax2.axvspan(vg[hi[0]], vg[hi[-1]], color='red', alpha=0.2)
        plt.title(f'Attention on ICA – High Zone ({vg[hi[0]]:.2f}-{vg[hi[-1]]:.2f} V)', fontsize=16)
    else:
        plt.title('Attention Map on ICA Curve', fontsize=16)

    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / "Fig18_20a_Spatial_Attention.png", dpi=300)
    plt.close()
    print("  Saved Fig18_20a.")


# ──────────────────────── Main ────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Step 1: Build features from Kaggle CSV
    all_features = build_kaggle_features()

    # Step 2: Build sequences
    X, y, batteries = build_sequences(all_features)
    print(f"\nTotal sequences: {X.shape[0]}  Shape: {X.shape}")

    # Step 3: Train base model (3 channels)
    print("\n--- Training Base Model (3-channel) ---")
    model, history = train_base_model(X, y)

    # Step 4: Get predictions
    predictions = predict_all(model, X)

    # Step 5: Generate ALL figures
    print("\n--- Generating All Base Paper Figures ---")
    generate_fig7()
    generate_fig8(y, predictions, batteries)
    generate_fig9(y, predictions)
    generate_fig10(y, predictions, batteries)
    generate_fig11(y, predictions)
    generate_fig16(model)
    generate_fig18_20(model, X)

    print(f"\n{'='*50}")
    print(f"ALL base paper images saved to: {OUTPUT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
