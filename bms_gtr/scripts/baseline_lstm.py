from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader,random_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import calculate_metrics


SEED=42
BATCH_SIZE=64
LEARNING_RATE=0.001
EPOCHS=200
PATIENCE=20

BATTERIES=["B0005","B0006","B0007","B0018"]


class LSTMBaseline(nn.Module):
    def __init__(self,input_size=3*300,hidden_size=64,num_layers=1,dropout=0.2):
        super().__init__()

        self.input_projection=nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm=nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0
        )

        self.regressor=nn.Sequential(
            nn.Linear(hidden_size,32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32,1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,x):
        batch_size,sequence_length,channels,length=x.shape

        x=x.reshape(
            batch_size,
            sequence_length,
            channels*length
        )

        x=self.input_projection(x)

        x,_=self.lstm(x)

        x=x[:,-1,:]

        output=self.regressor(x)

        return output.squeeze(-1)


class Trainer:
    def __init__(self,model,optimizer,criterion,scheduler,device,patience=20):
        self.model=model
        self.optimizer=optimizer
        self.criterion=criterion
        self.scheduler=scheduler
        self.device=device
        self.patience=patience

    def fit(self,train_loader,val_loader,epochs):
        best_loss=float("inf")
        best_state=None
        patience_counter=0

        for epoch in range(epochs):
            self.model.train()

            train_losses=[]

            for X_batch,y_batch in train_loader:
                X_batch=X_batch.to(self.device)
                y_batch=y_batch.to(self.device)

                self.optimizer.zero_grad()

                prediction=self.model(X_batch)

                loss=self.criterion(prediction,y_batch)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    5.0
                )

                self.optimizer.step()

                train_losses.append(loss.item())

            self.model.eval()

            val_losses=[]

            with torch.no_grad():
                for X_batch,y_batch in val_loader:
                    X_batch=X_batch.to(self.device)
                    y_batch=y_batch.to(self.device)

                    prediction=self.model(X_batch)

                    loss=self.criterion(prediction,y_batch)

                    val_losses.append(loss.item())

            train_loss=np.mean(train_losses)
            val_loss=np.mean(val_losses)

            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1:03d}/{epochs} "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f}"
            )

            if val_loss<best_loss:
                best_loss=val_loss
                best_state={
                    k:v.detach().cpu().clone()
                    for k,v in self.model.state_dict().items()
                }
                patience_counter=0
            else:
                patience_counter+=1

            if patience_counter>=self.patience:
                print("Early stopping.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return best_loss


def train_and_test(X,y,batteries,test_battery,device):
    train_mask=batteries!=test_battery
    test_mask=batteries==test_battery

    X_train=X[train_mask]
    y_train=y[train_mask]

    X_test=X[test_mask]
    y_test=y[test_mask]

    print("\n========================================")
    print("LSTM BASELINE")
    print("Test battery:",test_battery)
    print("Train batteries:",np.unique(batteries[train_mask]))
    print("Train samples:",len(X_train))
    print("Test samples:",len(X_test))
    print("========================================")

    train_dataset=TensorDataset(
        torch.tensor(X_train,dtype=torch.float32),
        torch.tensor(y_train,dtype=torch.float32)
    )

    test_dataset=TensorDataset(
        torch.tensor(X_test,dtype=torch.float32),
        torch.tensor(y_test,dtype=torch.float32)
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

    model=LSTMBaseline().to(device)

    criterion=nn.MSELoss()

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
        patience=PATIENCE
    )

    print("\nTraining...")

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

            prediction=model(X_batch)

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

    print("\nLSTM Test Results")
    print("=================")
    print(f"MAE:  {metrics['MAE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"R2:   {metrics['R2']:.6f}")

    return metrics,predictions,targets


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

    print("Dataset:")
    print("X:",X.shape)
    print("y:",y.shape)

    results={}
    all_predictions={}
    all_targets={}

    for test_battery in BATTERIES:
        metrics,predictions,targets=train_and_test(
            X,
            y,
            batteries,
            test_battery,
            device
        )

        results[test_battery]=metrics
        all_predictions[test_battery]=predictions
        all_targets[test_battery]=targets

    print("\n\n========================================")
    print("FINAL LSTM BASELINE RESULTS")
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

    Path("results/experiments").mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        "results/experiments/baseline_lstm_results.npz",
        batteries=np.array(BATTERIES),
        mae=np.array(all_mae),
        rmse=np.array(all_rmse),
        r2=np.array(all_r2)
    )

    np.savez(
        "results/experiments/baseline_lstm_predictions.npz",
        batteries=np.array(BATTERIES),
        **{
            f"{battery}_pred":all_predictions[battery]
            for battery in BATTERIES
        },
        **{
            f"{battery}_target":all_targets[battery]
            for battery in BATTERIES
        }
    )

    print("\nSaved:")
    print("results/experiments/baseline_lstm_results.npz")
    print("results/experiments/baseline_lstm_predictions.npz")


if __name__=="__main__":
    main()