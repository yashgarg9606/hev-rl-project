from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader,random_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import calculate_metrics

SEED=42
BATCH_SIZE=64
LEARNING_RATE=0.001
EPOCHS=200
PATIENCE=20

BATTERIES=["B0005","B0006","B0007","B0018"]


# ============================================================
# CNN
# ============================================================

class CNNBlock(nn.Module):
    def __init__(self,in_channels=3,hidden_channels=64):
        super().__init__()

        self.conv1=nn.Conv1d(in_channels,32,kernel_size=5,padding=2)
        self.bn1=nn.BatchNorm1d(32)

        self.conv2=nn.Conv1d(32,hidden_channels,kernel_size=5,padding=2)
        self.bn2=nn.BatchNorm1d(hidden_channels)

        self.pool=nn.AdaptiveAvgPool1d(1)

    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        x=F.relu(self.bn2(self.conv2(x)))
        x=self.pool(x).squeeze(-1)
        return x


# ============================================================
# TCN
# ============================================================

class TCNBlock(nn.Module):
    def __init__(self,channels,kernel_size=3,dilation=1,dropout=0.2):
        super().__init__()

        padding=(kernel_size-1)*dilation

        self.conv1=nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn1=nn.BatchNorm1d(channels)

        self.conv2=nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn2=nn.BatchNorm1d(channels)

        self.dropout=nn.Dropout(dropout)
        self.padding=padding

    def chomp(self,x):
        if self.padding==0:
            return x
        return x[:,:,:-self.padding]

    def forward(self,x):
        residual=x

        x=self.conv1(x)
        x=self.chomp(x)
        x=F.relu(self.bn1(x))
        x=self.dropout(x)

        x=self.conv2(x)
        x=self.chomp(x)
        x=self.bn2(x)
        x=self.dropout(x)

        return F.relu(x+residual)


class TCN(nn.Module):
    def __init__(self,channels=64,dropout=0.2):
        super().__init__()

        self.blocks=nn.ModuleList([
            TCNBlock(channels,dilation=1,dropout=dropout),
            TCNBlock(channels,dilation=2,dropout=dropout),
            TCNBlock(channels,dilation=4,dropout=dropout),
            TCNBlock(channels,dilation=8,dropout=dropout)
        ])

    def forward(self,x):
        for block in self.blocks:
            x=block(x)
        return x


# ============================================================
# ATTENTION
# ============================================================

class AdditiveAttention(nn.Module):
    def __init__(self,hidden_size=64,attention_size=64):
        super().__init__()

        self.W=nn.Linear(hidden_size,attention_size)
        self.v=nn.Linear(attention_size,1,bias=False)

    def forward(self,x):
        scores=self.v(torch.tanh(self.W(x)))
        weights=torch.softmax(scores,dim=1)
        context=torch.sum(weights*x,dim=1)
        return context,weights


# ============================================================
# COMMON REGRESSOR
# ============================================================

class Regressor(nn.Module):
    def __init__(self,hidden_size=64,dropout=0.2):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(hidden_size,32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.net(x).squeeze(-1)


# ============================================================
# CNN ONLY
# ============================================================

class CNNOnly(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn=CNNBlock()

        self.regressor=Regressor()

    def forward(self,x):
        batch_size,sequence_length,channels,length=x.shape

        features=[]

        for t in range(sequence_length):
            cycle=x[:,t,:,:]
            feature=self.cnn(cycle)
            features.append(feature)

        x=torch.stack(features,dim=1)

        # Average information from all cycles
        x=x.mean(dim=1)

        output=self.regressor(x)

        return output,None


# ============================================================
# CNN + LSTM
# ============================================================

class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn=CNNBlock()

        self.lstm=nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.regressor=Regressor()

    def forward(self,x):
        batch_size,sequence_length,channels,length=x.shape

        features=[]

        for t in range(sequence_length):
            cycle=x[:,t,:,:]
            feature=self.cnn(cycle)
            features.append(feature)

        x=torch.stack(features,dim=1)

        x,_=self.lstm(x)

        x=x[:,-1,:]

        output=self.regressor(x)

        return output,None


# ============================================================
# CNN + TCN + LSTM WITHOUT ATTENTION
# ============================================================

class CNNTCNLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn=CNNBlock()

        self.tcn=TCN()

        self.lstm=nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.regressor=Regressor()

    def forward(self,x):
        batch_size,sequence_length,channels,length=x.shape

        features=[]

        for t in range(sequence_length):
            cycle=x[:,t,:,:]
            feature=self.cnn(cycle)
            features.append(feature)

        x=torch.stack(features,dim=1)

        # [B,20,64] -> [B,64,20]
        x=x.transpose(1,2)

        x=self.tcn(x)

        # [B,64,20] -> [B,20,64]
        x=x.transpose(1,2)

        x,_=self.lstm(x)

        x=x[:,-1,:]

        output=self.regressor(x)

        return output,None


# ============================================================
# FULL CNN + TCN + LSTM + ATTENTION
# ============================================================

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn=CNNBlock()

        self.tcn=TCN()

        self.lstm=nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.attention=AdditiveAttention()

        self.regressor=Regressor()

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module,(nn.Linear,nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,x):
        batch_size,sequence_length,channels,length=x.shape

        features=[]

        for t in range(sequence_length):
            cycle=x[:,t,:,:]

            feature=self.cnn(cycle)

            features.append(feature)

        x=torch.stack(features,dim=1)

        # [B,20,64] -> [B,64,20]
        x=x.transpose(1,2)

        x=self.tcn(x)

        # [B,64,20] -> [B,20,64]
        x=x.transpose(1,2)

        x,_=self.lstm(x)

        context,attention_weights=self.attention(x)

        output=self.regressor(context)

        return output,attention_weights


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        device,
        patience=20
    ):
        self.model=model
        self.optimizer=optimizer
        self.criterion=criterion
        self.scheduler=scheduler
        self.device=device
        self.patience=patience

    def fit(self,train_loader,val_loader,epochs):
        best_loss=float("inf")
        patience_counter=0

        for epoch in range(epochs):
            self.model.train()

            train_losses=[]

            for X_batch,y_batch in train_loader:
                X_batch=X_batch.to(self.device)
                y_batch=y_batch.to(self.device)

                self.optimizer.zero_grad()

                prediction,_=self.model(X_batch)

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

                    prediction,_=self.model(X_batch)

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
                patience_counter=0

                torch.save(
                    self.model.state_dict(),
                    "results/experiments/ablation_best_model.pt"
                )

            else:
                patience_counter+=1

            if patience_counter>=self.patience:
                print("Early stopping.")
                break

        if Path(
            "results/experiments/ablation_best_model.pt"
        ).exists():
            self.model.load_state_dict(
                torch.load(
                    "results/experiments/ablation_best_model.pt",
                    map_location=self.device,
                    weights_only=True
                )
            )

        return best_loss


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    X,
    y,
    batteries,
    test_battery,
    model_class,
    model_name,
    device
):
    train_mask=batteries!=test_battery
    test_mask=batteries==test_battery

    X_train=X[train_mask]
    y_train=y[train_mask]

    X_test=X[test_mask]
    y_test=y[test_mask]

    print("\n========================================")
    print(f"ABLATION: {model_name}")
    print(f"Test battery: {test_battery}")
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

    model=model_class().to(device)

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

    predictions=np.asarray(predictions)
    targets=np.asarray(targets)

    metrics=calculate_metrics(
        targets,
        predictions
    )

    print(
        f"{model_name} | "
        f"{test_battery} | "
        f"MAE={metrics['MAE']:.6f} | "
        f"RMSE={metrics['RMSE']:.6f} | "
        f"R2={metrics['R2']:.6f}"
    )

    return metrics,predictions,targets


# ============================================================
# MAIN
# ============================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Path("results/experiments").mkdir(
        parents=True,
        exist_ok=True
    )

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
    print("Batteries:",np.unique(batteries))

    models={
        "CNN Only":CNNOnly,
        "CNN-LSTM":CNNLSTM,
        "CNN-TCN-LSTM":CNNTCNLSTM,
        "Full Model":FullModel
    }

    results={}

    all_predictions={}

    for model_name,model_class in models.items():

        results[model_name]={}
        all_predictions[model_name]={}

        for test_battery in BATTERIES:

            metrics,predictions,targets=run_experiment(
                X,
                y,
                batteries,
                test_battery,
                model_class,
                model_name,
                device
            )

            results[model_name][test_battery]=metrics

            all_predictions[model_name][test_battery]={
                "predictions":predictions,
                "targets":targets
            }

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("="*100)
    print("FINAL ABLATION RESULTS")
    print("="*100)

    print(
        f"{'Model':<22}"
        f"{'MAE':<14}"
        f"{'RMSE':<14}"
        f"{'R2':<14}"
    )

    summary={}

    for model_name in models:

        mae=[]
        rmse=[]
        r2=[]

        for battery in BATTERIES:
            mae.append(
                results[model_name][battery]["MAE"]
            )

            rmse.append(
                results[model_name][battery]["RMSE"]
            )

            r2.append(
                results[model_name][battery]["R2"]
            )

        summary[model_name]={
            "MAE":np.mean(mae),
            "RMSE":np.mean(rmse),
            "R2":np.mean(r2)
        }

        print(
            f"{model_name:<22}"
            f"{np.mean(mae):<14.6f}"
            f"{np.mean(rmse):<14.6f}"
            f"{np.mean(r2):<14.6f}"
        )

    print("="*100)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    model_names=np.array(list(models.keys()))

    mae=np.array([
        summary[m]["MAE"]
        for m in models
    ])

    rmse=np.array([
        summary[m]["RMSE"]
        for m in models
    ])

    r2=np.array([
        summary[m]["R2"]
        for m in models
    ])

    np.savez(
        "results/experiments/ablation_results.npz",
        models=model_names,
        mae=mae,
        rmse=rmse,
        r2=r2
    )

    # Save predictions
    save_dict={}

    for model_name in models:
        for battery in BATTERIES:

            prefix=f"{model_name}_{battery}"

            save_dict[
                prefix+"_predictions"
            ]=all_predictions[
                model_name
            ][battery]["predictions"]

            save_dict[
                prefix+"_targets"
            ]=all_predictions[
                model_name
            ][battery]["targets"]

    np.savez(
        "results/experiments/ablation_predictions.npz",
        **save_dict
    )

    print("\nSaved:")
    print("results/experiments/ablation_results.npz")
    print("results/experiments/ablation_predictions.npz")

    # ========================================================
    # CREATE PLOTS
    # ========================================================

    try:
        import matplotlib.pyplot as plt

        plot_dir=Path(
            "results/plots/ablation"
        )

        plot_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # RMSE
        plt.figure(figsize=(10,6))

        plt.bar(
            model_names,
            rmse
        )

        plt.ylabel("RMSE")
        plt.title("Ablation Study - RMSE")
        plt.xticks(rotation=20)
        plt.tight_layout()

        plt.savefig(
            plot_dir/"ablation_rmse.png",
            dpi=200
        )

        plt.close()

        # MAE
        plt.figure(figsize=(10,6))

        plt.bar(
            model_names,
            mae
        )

        plt.ylabel("MAE")
        plt.title("Ablation Study - MAE")
        plt.xticks(rotation=20)
        plt.tight_layout()

        plt.savefig(
            plot_dir/"ablation_mae.png",
            dpi=200
        )

        plt.close()

        # R2
        plt.figure(figsize=(10,6))

        plt.bar(
            model_names,
            r2
        )

        plt.ylabel("R2")
        plt.title("Ablation Study - R2")
        plt.xticks(rotation=20)
        plt.tight_layout()

        plt.savefig(
            plot_dir/"ablation_r2.png",
            dpi=200
        )

        plt.close()

        # Prediction trajectories
        for battery in BATTERIES:

            plt.figure(figsize=(12,6))

            targets=all_predictions[
                "Full Model"
            ][battery]["targets"]

            plt.plot(
                targets,
                label="Actual"
            )

            for model_name in models:

                predictions=all_predictions[
                    model_name
                ][battery]["predictions"]

                plt.plot(
                    predictions,
                    label=model_name
                )

            plt.xlabel("Cycle")
            plt.ylabel("SOH")
            plt.title(
                f"Ablation Predictions - {battery}"
            )

            plt.legend()
            plt.tight_layout()

            plt.savefig(
                plot_dir/
                f"{battery}_ablation_prediction.png",
                dpi=200
            )

            plt.close()

        print("\nPlots saved:")
        print(plot_dir)

    except Exception as e:
        print("\nPlot generation failed:")
        print(e)


if __name__=="__main__":
    main()