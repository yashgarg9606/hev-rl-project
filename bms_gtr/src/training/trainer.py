import copy
import torch


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler=None,
        device="cpu",
        gradient_clip=5.0,
        patience=20
    ):
        self.model=model
        self.optimizer=optimizer
        self.criterion=criterion
        self.scheduler=scheduler
        self.device=device
        self.gradient_clip=gradient_clip
        self.patience=patience

        self.best_loss=float("inf")
        self.best_state=None
        self.wait=0

    def train_epoch(self,loader):
        self.model.train()

        total_loss=0.0

        for X,y in loader:
            X=X.to(self.device)
            y=y.to(self.device)

            self.optimizer.zero_grad()

            prediction,_=self.model(X)

            loss=self.criterion(
                prediction,
                y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )

            self.optimizer.step()

            total_loss+=loss.item()*len(X)

        return total_loss/len(loader.dataset)

    @torch.no_grad()
    def validate(self,loader):
        self.model.eval()

        total_loss=0.0

        for X,y in loader:
            X=X.to(self.device)
            y=y.to(self.device)

            prediction,_=self.model(X)

            loss=self.criterion(
                prediction,
                y
            )

            total_loss+=loss.item()*len(X)

        return total_loss/len(loader.dataset)

    def fit(self,train_loader,val_loader,epochs):
        history={
            "train_loss":[],
            "val_loss":[]
        }

        for epoch in range(1,epochs+1):
            train_loss=self.train_epoch(
                train_loader
            )

            val_loss=self.validate(
                val_loader
            )

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            history["train_loss"].append(
                train_loss
            )

            history["val_loss"].append(
                val_loss
            )

            print(
                f"Epoch {epoch:03d}/{epochs} "
                f"train={train_loss:.6f} "
                f"val={val_loss:.6f}"
            )

            if val_loss<self.best_loss:
                self.best_loss=val_loss
                self.best_state=copy.deepcopy(
                    self.model.state_dict()
                )
                self.wait=0
            else:
                self.wait+=1

            if self.wait>=self.patience:
                print("Early stopping.")
                break

        if self.best_state is not None:
            self.model.load_state_dict(
                self.best_state
            )

        return history