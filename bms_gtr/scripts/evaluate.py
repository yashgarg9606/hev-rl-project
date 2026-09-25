from pathlib import Path
import sys

import numpy as np
import torch

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.models.hybrid_model import CNNTCNLSTMAttention
from src.evaluation.metrics import calculate_metrics
from src.visualization.plots import (
    plot_predictions,
    plot_training_history
)


DEVICE=torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


def main():
    print("Device:",DEVICE)

    data=np.load(
        "data/processed/sequences.npz"
    )

    X=data["X"].astype(np.float32)
    y=data["y"].astype(np.float32)

    model=CNNTCNLSTMAttention()

    model.load_state_dict(
        torch.load(
            "results/best_model.pt",
            map_location=DEVICE
        )
    )

    model=model.to(DEVICE)
    model.eval()

    X_tensor=torch.tensor(X)

    predictions=[]

    with torch.no_grad():
        for i in range(0,len(X_tensor),64):
            batch=X_tensor[i:i+64].to(DEVICE)

            prediction,_=model(batch)

            predictions.extend(
                prediction.cpu().numpy()
            )

    predictions=np.asarray(predictions)

    metrics=calculate_metrics(
        y,
        predictions
    )

    print("\nEvaluation Results")
    print("==================")

    for name,value in metrics.items():
        print(f"{name}: {value:.6f}")

    Path("results/plots").mkdir(
        parents=True,
        exist_ok=True
    )

    plot_predictions(
        y,
        predictions,
        "results/plots/actual_vs_predicted.png"
    )

    history=np.load(
        "results/training_history.npy",
        allow_pickle=True
    ).item()

    plot_training_history(
        history,
        "results/plots/training_history.png"
    )

    np.savez_compressed(
        "results/predictions.npz",
        y_true=y,
        y_pred=predictions
    )

    print("\nSaved:")
    print("results/predictions.npz")
    print("results/plots/actual_vs_predicted.png")
    print("results/plots/training_history.png")


if __name__=="__main__":
    main()