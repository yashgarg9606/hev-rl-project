from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR=Path("results")
PLOTS_DIR=RESULTS_DIR/"plots"
EXPERIMENT_DIR=RESULTS_DIR/"experiments"

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def plot_actual_vs_predicted():
    path=RESULTS_DIR/"predictions.npz"

    if not path.exists():
        print("Skipping actual_vs_predicted: predictions.npz not found")
        return

    data=np.load(path)

    y_true=data["y_true"]
    y_pred=data["y_pred"]

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

    plt.title(
        "Actual vs Predicted Battery SOH"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR/"actual_vs_predicted.png",
        dpi=300
    )

    plt.close()

    print("Saved actual_vs_predicted.png")


def plot_prediction_error():
    path=RESULTS_DIR/"predictions.npz"

    if not path.exists():
        print("Skipping prediction_error: predictions.npz not found")
        return

    data=np.load(path)

    y_true=data["y_true"]
    y_pred=data["y_pred"]

    error=y_pred-y_true

    plt.figure(figsize=(10,6))

    plt.plot(
        error,
        label="Prediction Error"
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Sample")
    plt.ylabel("Prediction Error")

    plt.title(
        "SOH Prediction Error"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR/"prediction_error.png",
        dpi=300
    )

    plt.close()

    print("Saved prediction_error.png")


def plot_cross_validation():
    path=EXPERIMENT_DIR/"cross_validation_results.npz"

    if not path.exists():
        print(
            "Skipping cross_validation: "
            "cross_validation_results.npz not found"
        )
        return

    data=np.load(path)

    batteries=data["batteries"]
    mae=data["mae"]
    rmse=data["rmse"]
    r2=data["r2"]

    x=np.arange(len(batteries))

    plt.figure(figsize=(10,6))

    width=0.35

    plt.bar(
        x-width/2,
        mae,
        width,
        label="MAE"
    )

    plt.bar(
        x+width/2,
        rmse,
        width,
        label="RMSE"
    )

    plt.xticks(
        x,
        batteries
    )

    plt.xlabel("Test Battery")
    plt.ylabel("Error")

    plt.title(
        "Leave-One-Battery-Out Evaluation"
    )

    plt.legend()
    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR/"cross_validation_errors.png",
        dpi=300
    )

    plt.close()

    print("Saved cross_validation_errors.png")

    plt.figure(figsize=(10,6))

    plt.bar(
        batteries,
        r2
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Test Battery")
    plt.ylabel("R²")

    plt.title(
        "R² Across Test Batteries"
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR/"cross_validation_r2.png",
        dpi=300
    )

    plt.close()

    print("Saved cross_validation_r2.png")


def plot_training_history():
    path=RESULTS_DIR/"training_history.npy"

    if not path.exists():
        print(
            "Skipping training_history: "
            "training_history.npy not found"
        )
        return

    history=np.load(
        path,
        allow_pickle=True
    )

    if isinstance(history, np.ndarray) and history.dtype==object:
        try:
            history=history.item()
        except:
            print("Could not read training history")
            return

    if isinstance(history, dict):
        train_loss=history.get("train_loss")
        val_loss=history.get("val_loss")
    else:
        print("Unknown training history format")
        return

    if train_loss is None or val_loss is None:
        print("Training history does not contain train_loss/val_loss")
        return

    plt.figure(figsize=(10,6))

    plt.plot(
        train_loss,
        label="Training Loss"
    )

    plt.plot(
        val_loss,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR/"training_history.png",
        dpi=300
    )

    plt.close()

    print("Saved training_history.png")


def main():
    print("Generating plots...")
    print()

    plot_actual_vs_predicted()
    plot_prediction_error()
    plot_cross_validation()
    plot_training_history()

    print()
    print("================================")
    print("Plot generation complete")
    print("================================")
    print()
    print("Plots saved in:")
    print(PLOTS_DIR)


if __name__=="__main__":
    main()