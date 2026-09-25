from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH=Path("results/experiments/cross_validation_predictions.npz")
OUTPUT_DIR=Path("results/plots/cross_validation")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():
    data=np.load(DATA_PATH)

    batteries=data["battery"]
    actual=data["actual"]
    predicted=data["predicted"]

    unique_batteries=np.unique(batteries)

    print("Generating CV prediction plots...")

    for battery in unique_batteries:
        mask=batteries==battery

        y_true=actual[mask]
        y_pred=predicted[mask]

        cycles=np.arange(
            1,
            len(y_true)+1
        )

        plt.figure(figsize=(10,6))

        plt.plot(
            cycles,
            y_true,
            label="Actual SOH"
        )

        plt.plot(
            cycles,
            y_pred,
            label="Predicted SOH"
        )

        plt.xlabel("Cycle")
        plt.ylabel("SOH")

        plt.title(
            f"Leave-One-Battery-Out CV: {battery}"
        )

        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        output_file=(
            OUTPUT_DIR/
            f"{battery}_cv_prediction.png"
        )

        plt.savefig(
            output_file,
            dpi=300
        )

        plt.close()

        print(
            f"Saved: {output_file}"
        )

    print()
    print("CV prediction plots complete.")


if __name__=="__main__":
    main()