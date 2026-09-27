from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.artifacts import save_new_archive
from src.data.sequence_dataset import load_and_sequence


DATA_DIR=Path("data/processed")
SEQUENCE_LENGTH=20

BATTERIES=["B0005","B0006","B0007","B0018"]


def main():
    import argparse
    global DATA_DIR
    parser = argparse.ArgumentParser(description="Build NASA histories without overwriting existing sequences")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    all_X=[]
    all_y=[]
    all_batteries=[]

    for battery in BATTERIES:
        path=DATA_DIR/f"{battery}.npz"

        X,y=load_and_sequence(
            path,
            SEQUENCE_LENGTH
        )

        print(
            f"{battery}: X={X.shape}, y={y.shape}"
        )

        all_X.append(X)
        all_y.append(y)
        all_batteries.extend(
            [battery]*len(y)
        )

    X=np.concatenate(all_X,axis=0)
    y=np.concatenate(all_y,axis=0)

    save_new_archive(
        args.output,
        X=X,
        y=y,
        batteries=np.asarray(all_batteries)
    )

    print("\nFinal dataset:")
    print("X:",X.shape)
    print("y:",y.shape)


if __name__=="__main__":
    main()