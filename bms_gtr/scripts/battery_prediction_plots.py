from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH="data/processed/sequences.npz"
PREDICTIONS_PATH="results/predictions.npz"
OUTPUT_DIR=Path("results/plots/batteries")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():
    data=np.load(DATA_PATH)

    y=data["y"]
    batteries=data["batteries"]

    predictions=np.load(PREDICTIONS_PATH)

    y_pred=predictions["y_pred"]

    if len(y_pred)!=len(y):
        raise ValueError(
            f"Prediction length ({len(y_pred)}) "
            f"does not match dataset length ({len(y)})."
        )

    unique_batteries=np.unique(batteries)

    print("Batteries:",unique_batteries)

    for battery in unique_batteries:
        indices=np.where(
            batteries==battery
        )[0]

        actual=y[indices]
        predicted=y_pred[indices]

        cycles=np.arange(
            1,
            len(indices)+1
        )

        plt.figure(figsize=(10,6))

        plt.plot(
            cycles,
            actual,
            label="Actual SOH"
        )

        plt.plot(
            cycles,
            predicted,
            label="Predicted SOH"
        )

        plt.xlabel("Cycle")
        plt.ylabel("SOH")

        plt.title(
            f"Battery {battery}: Actual vs Predicted SOH"
        )

        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        output_file=(
            OUTPUT_DIR/
            f"{battery}_prediction.png"
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
    print("Battery prediction plots complete.")


if __name__=="__main__":
    main()