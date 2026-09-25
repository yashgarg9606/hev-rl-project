from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.models.hybrid_model import CNNTCNLSTMAttention
from src.training.trainer import Trainer
from src.evaluation.splits import battery_aware_split
from src.evaluation.metrics import calculate_metrics


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
        random_state=SEED
    )

    print("\nTrain batteries:",np.unique(train_batteries))
    print("Test batteries:",np.unique(test_batteries))

    train_dataset=TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )

    test_dataset=TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test)
    )

    train_loader=DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader=DataLoader(
        test_dataset,
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

    # Use part of the training batteries for validation.
    validation_size=int(len(train_dataset)*0.2)

    training_size=len(train_dataset)-validation_size

    train_subset,val_subset=torch.utils.data.random_split(
        train_dataset,
        [training_size,validation_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader=DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader=DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("\nTraining...")

    history=trainer.fit(
        train_loader,
        val_loader,
        EPOCHS
    )

    print("\nEvaluating on unseen batteries...")

    model.eval()

    predictions=[]
    targets=[]

    with torch.no_grad():
        for X_batch,y_batch in test_loader:
            X_batch=X_batch.to(device)

            prediction,_=model(X_batch)

            predictions.extend(
                prediction.cpu().numpy()
            )

            targets.extend(
                y_batch.numpy()
            )

    predictions=np.asarray(predictions)
    targets=np.asarray(targets)

    metrics=calculate_metrics(
        targets,
        predictions
    )

    print("\nTest Results")
    print("============")

    for name,value in metrics.items():
        print(f"{name}: {value:.6f}")

    Path("results/experiments").mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        "results/experiments/battery_holdout_model.pt"
    )

    np.savez_compressed(
        "results/experiments/battery_holdout_predictions.npz",
        y_true=targets,
        y_pred=predictions,
        test_batteries=test_batteries
    )

    print("\nExperiment complete.")


if __name__=="__main__":
    main()