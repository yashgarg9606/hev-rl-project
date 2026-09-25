from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader,random_split

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.models.hybrid_model import CNNTCNLSTMAttention
from src.training.trainer import Trainer
from src.evaluation.metrics import calculate_metrics


SEED=42
BATCH_SIZE=64
LEARNING_RATE=0.001
EPOCHS=200
PATIENCE=20

BATTERIES=[
    "B0005",
    "B0006",
    "B0007",
    "B0018"
]


def train_and_test(
    X,
    y,
    batteries,
    test_battery,
    device
):
    train_mask=batteries!=test_battery
    test_mask=batteries==test_battery

    X_train=X[train_mask]
    y_train=y[train_mask]

    X_test=X[test_mask]
    y_test=y[test_mask]

    test_indices=np.where(test_mask)[0]

    print("\n========================================")
    print("Test battery:",test_battery)
    print("Train batteries:",np.unique(batteries[train_mask]))
    print("Train samples:",len(X_train))
    print("Test samples:",len(X_test))
    print("========================================")

    train_dataset=TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )

    test_dataset=TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test)
    )

    validation_size=int(len(train_dataset)*0.2)
    training_size=len(train_dataset)-validation_size

    train_subset,val_subset=random_split(
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

    trainer.fit(
        train_loader,
        val_loader,
        EPOCHS
    )

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

    predictions=np.asarray(
        predictions,
        dtype=np.float32
    )

    targets=np.asarray(
        targets,
        dtype=np.float32
    )

    metrics=calculate_metrics(
        targets,
        predictions
    )

    print("\nTest results:")
    print(f"MAE:  {metrics['MAE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"R2:   {metrics['R2']:.6f}")

    return (
        metrics,
        test_indices,
        targets,
        predictions
    )


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

    results={}

    all_prediction_battery=[]
    all_prediction_index=[]
    all_prediction_actual=[]
    all_prediction_predicted=[]

    for test_battery in BATTERIES:
        (
            metrics,
            test_indices,
            targets,
            predictions
        )=train_and_test(
            X,
            y,
            batteries,
            test_battery,
            device
        )

        results[test_battery]=metrics

        all_prediction_battery.extend(
            [test_battery]*len(predictions)
        )

        all_prediction_index.extend(
            test_indices
        )

        all_prediction_actual.extend(
            targets
        )

        all_prediction_predicted.extend(
            predictions
        )

    print("\n\n========================================")
    print("FINAL CROSS-VALIDATION RESULTS")
    print("========================================")

    print(
        f"{'Battery':<12}"
        f"{'MAE':<12}"
        f"{'RMSE':<12}"
        f"{'R2':<12}"
    )

    all_mae=[]
    all_rmse=[]
    all_r2=[]

    for battery in BATTERIES:
        mae=results[battery]["MAE"]
        rmse=results[battery]["RMSE"]
        r2=results[battery]["R2"]

        all_mae.append(mae)
        all_rmse.append(rmse)
        all_r2.append(r2)

        print(
            f"{battery:<12}"
            f"{mae:<12.6f}"
            f"{rmse:<12.6f}"
            f"{r2:<12.6f}"
        )

    print("----------------------------------------")

    print(
        f"{'Average':<12}"
        f"{np.mean(all_mae):<12.6f}"
        f"{np.mean(all_rmse):<12.6f}"
        f"{np.mean(all_r2):<12.6f}"
    )

    output_dir=Path(
        "results/experiments"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        output_dir/"cross_validation_results.npz",
        batteries=np.array(BATTERIES),
        mae=np.array(all_mae),
        rmse=np.array(all_rmse),
        r2=np.array(all_r2)
    )

    np.savez(
        output_dir/"cross_validation_predictions.npz",
        battery=np.array(
            all_prediction_battery
        ),
        dataset_index=np.array(
            all_prediction_index
        ),
        actual=np.array(
            all_prediction_actual
        ),
        predicted=np.array(
            all_prediction_predicted
        )
    )

    print(
        "\nSaved:"
        "\nresults/experiments/cross_validation_results.npz"
        "\nresults/experiments/cross_validation_predictions.npz"
    )


if __name__=="__main__":
    main()