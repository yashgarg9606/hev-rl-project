import sys
from pathlib import Path

import numpy as np

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.evaluation.splits import battery_aware_split


def main():
    data=np.load(
        "data/processed/sequences.npz"
    )

    X=data["X"]
    y=data["y"]
    batteries=data["batteries"]

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_batteries,
        test_batteries
    )=battery_aware_split(
        X,
        y,
        batteries,
        test_size=0.25,
        random_state=42
    )

    print("Train:")
    print("X:",X_train.shape)
    print("y:",y_train.shape)
    print(
        "Batteries:",
        np.unique(train_batteries)
    )

    print("\nTest:")
    print("X:",X_test.shape)
    print("y:",y_test.shape)
    print(
        "Batteries:",
        np.unique(test_batteries)
    )


if __name__=="__main__":
    main()