from pathlib import Path
import sys

import numpy as np
import torch

from torch.utils.data import TensorDataset,DataLoader

from sklearn.model_selection import train_test_split

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.models.hybrid_model import CNNTCNLSTMAttention
from src.training.trainer import Trainer


SEED=42
BATCH_SIZE=64
LEARNING_RATE=0.001
EPOCHS=200
PATIENCE=20


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device=torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:",device)

    data=np.load(
        "data/processed/sequences.npz"
    )

    X=data["X"].astype(np.float32)
    y=data["y"].astype(np.float32)

    X_train,X_val,y_train,y_val=train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED
    )

    train_dataset=TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )

    val_dataset=TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val)
    )

    train_loader=DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader=DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model=CNNTCNLSTMAttention().to(device)

    criterion=torch.nn.MSELoss()

    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    trainer=Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        gradient_clip=5.0,
        patience=PATIENCE
    )

    history=trainer.fit(
        train_loader,
        val_loader,
        EPOCHS
    )

    Path("results").mkdir(
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        "results/best_model.pt"
    )

    np.save(
        "results/training_history.npy",
        history
    )

    print("\nTraining complete.")
    print(
        "Best validation loss:",
        trainer.best_loss
    )


if __name__=="__main__":
    main()