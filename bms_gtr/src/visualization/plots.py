import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(y_true,y_pred,output_path):
    plt.figure(figsize=(10,6))

    plt.plot(
        y_true,
        label="Actual SOH"
    )

    plt.plot(
        y_pred,
        label="Predicted SOH"
    )

    plt.xlabel("Sample")
    plt.ylabel("SOH")
    plt.title("Actual vs Predicted SOH")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path,dpi=300)
    plt.close()


def plot_training_history(history,output_path):
    plt.figure(figsize=(10,6))

    plt.plot(
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path,dpi=300)
    plt.close()