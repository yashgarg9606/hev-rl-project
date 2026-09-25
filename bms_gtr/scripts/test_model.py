import sys
from pathlib import Path

import torch
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.hybrid_model import CNNTCNLSTMAttention


def main():
    data=np.load(
        "data/processed/sequences.npz"
    )

    X=torch.tensor(
        data["X"][:4],
        dtype=torch.float32
    )

    model=CNNTCNLSTMAttention()

    model.eval()

    with torch.no_grad():
        prediction,attention=model(X)

    print("Input:",X.shape)
    print("Prediction:",prediction.shape)
    print("Attention:",attention.shape)
    print("Prediction values:",prediction)


if __name__=="__main__":
    main()