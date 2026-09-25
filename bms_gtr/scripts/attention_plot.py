from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))

from src.models.hybrid_model import CNNTCNLSTMAttention


MODEL_PATH="results/best_model.pt"
DATA_PATH="data/processed/sequences.npz"
OUTPUT_PATH="results/plots/attention_weights.png"

DEVICE=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():
    print("Device:",DEVICE)

    data=np.load(DATA_PATH)

    X=data["X"].astype(np.float32)

    model=CNNTCNLSTMAttention().to(DEVICE)

    checkpoint=torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint,dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    X_tensor=torch.tensor(X)

    attention_values=[]

    with torch.no_grad():
        for start in range(0,len(X_tensor),64):
            batch=X_tensor[start:start+64].to(DEVICE)

            _,attention=model(batch)

            attention_values.append(
                attention.cpu().numpy()
            )

    attention=np.concatenate(
        attention_values,
        axis=0
    )

    attention=attention.squeeze(-1)

    mean_attention=np.mean(
        attention,
        axis=0
    )

    plt.figure(figsize=(10,6))

    plt.bar(
        np.arange(1,len(mean_attention)+1),
        mean_attention
    )

    plt.xlabel("Historical Cycle")
    plt.ylabel("Mean Attention Weight")

    plt.title(
        "Temporal Attention Across 20 Historical Cycles"
    )

    plt.xticks(
        np.arange(1,len(mean_attention)+1)
    )

    plt.grid(
        axis="y"
    )

    plt.tight_layout()

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=300
    )

    plt.close()

    print()
    print("Attention shape:",attention.shape)
    print("Mean attention shape:",mean_attention.shape)
    print()
    print("Saved:")
    print(OUTPUT_PATH)


if __name__=="__main__":
    main()