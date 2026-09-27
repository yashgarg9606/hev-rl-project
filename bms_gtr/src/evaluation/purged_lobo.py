"""Purged, chronological inner validation with an outer battery holdout.

This protocol uses the existing processed features and unchanged hybrid model.
It neither regenerates raw NASA data nor loads historical fitted checkpoints.
NumPy baselines and split inspection do not import Torch or scikit-learn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


BATTERIES = ("B0005", "B0006", "B0007", "B0018")
RIDGE_ALPHAS = (.001, .01, .1, 1., 10.)
HYBRID_ARCHITECTURE = dict(cnn_channels=64, lstm_hidden=64, lstm_layers=1, dropout=.2)


@dataclass(frozen=True)
class ProtocolConfig:
    seed: int = 42
    sequence_length: int = 20
    validation_fraction: float = .2
    batch_size: int = 64
    max_epochs: int = 40
    patience: int = 10
    learning_rate: float = .001
    gradient_clip: float = 5.
    cpu_threads: int = 2

    def validate(self):
        if self.sequence_length < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("Invalid sequence length or validation fraction")
        if self.max_epochs < 1 or self.patience < 1:
            raise ValueError("Epoch budget and patience must be positive")
        if self.batch_size < 1 or self.cpu_threads < 1:
            raise ValueError("Batch size and CPU thread count must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not np.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("Gradient clip must be finite and positive")


@dataclass
class SequenceData:
    X: np.ndarray
    y: np.ndarray
    batteries: np.ndarray
    local_indices: np.ndarray
    target_retained_indices: np.ndarray
    target_cycle_ids: np.ndarray
    support_cycle_ids: np.ndarray

    def metadata(self):
        return dict(combined_rows=np.arange(len(self.y)), batteries=self.batteries,
                    local_indices=self.local_indices,
                    target_retained_indices=self.target_retained_indices,
                    target_cycle_ids=self.target_cycle_ids,
                    support_cycle_ids=self.support_cycle_ids)


def reconstruct_sequences(data_dir, sequence_length=20, battery_names=BATTERIES):
    """Rebuild each battery independently and verify the legacy archive exactly.

    Local row i uses retained input cycles i..i+L-1 and target cycle i+L.
    Source cycle IDs identify observations; they do not define elapsed time.
    """
    data_dir = Path(data_dir)
    if sequence_length < 1 or len(set(battery_names)) != len(battery_names):
        raise ValueError("Require a positive sequence length and unique batteries")
    blocks = {key: [] for key in ("X", "y", "batteries", "local_indices",
                                  "target_retained_indices", "target_cycle_ids", "support_cycle_ids")}
    for battery in battery_names:
        if Path(battery).name != battery:
            raise ValueError("Battery IDs must be simple file stems")
        with np.load(data_dir / f"{battery}.npz", allow_pickle=False) as archive:
            features, targets, cycles = (archive[key] for key in ("features", "soh", "cycle_indices"))
        if features.ndim != 3 or features.shape[1] != 3 or features.shape[2] < 1:
            raise ValueError(f"{battery}: expected features [cycles, 3, voltage_points]")
        n = len(features) - sequence_length
        if n < 1 or targets.shape != (len(features),) or cycles.shape != targets.shape:
            raise ValueError(f"{battery}: invalid feature/target/cycle lengths")
        if not np.issubdtype(cycles.dtype, np.integer) or not np.all(cycles[1:] > cycles[:-1]):
            raise ValueError(f"{battery}: original cycle IDs must be strictly increasing integers")
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError(f"{battery}: nonfinite source features or targets")
        X = np.asarray([features[i:i + sequence_length] for i in range(n)], dtype=np.float32)
        y = np.asarray(targets[sequence_length:], dtype=np.float32)
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError(f"{battery}: float32 conversion produced nonfinite values")
        blocks["X"].append(X)
        blocks["y"].append(y)
        blocks["batteries"].append(np.full(n, battery))
        blocks["local_indices"].append(np.arange(n))
        blocks["target_retained_indices"].append(np.arange(sequence_length, len(features)))
        blocks["target_cycle_ids"].append(cycles[sequence_length:])
        blocks["support_cycle_ids"].append(np.asarray([cycles[i:i + sequence_length + 1] for i in range(n)]))
    data = SequenceData(**{name: np.concatenate(parts) for name, parts in blocks.items()})
    with np.load(data_dir / "sequences.npz", allow_pickle=False) as archive:
        for name in ("X", "y", "batteries"):
            if not np.array_equal(archive[name], getattr(data, name)):
                raise ValueError(f"Reconstructed {name} does not exactly match sequences.npz")
    return data


def make_fold(data: SequenceData, held_out: str, validation_fraction=.2):
    """Purge L sequence rows between each training prefix and validation tail."""
    if held_out not in data.batteries or not 0 < validation_fraction < 1:
        raise ValueError("Unknown holdout battery or invalid validation fraction")
    length = data.X.shape[1]
    parts = {name: [] for name in ("train", "purged", "validation", "test")}
    for battery in np.unique(data.batteries):
        rows = np.flatnonzero(data.batteries == battery)
        if not np.array_equal(data.local_indices[rows], np.arange(len(rows))):
            raise ValueError("Each battery must retain chronological local sequence rows")
        if battery == held_out:
            parts["test"].extend(rows)
            continue
        cut = int(np.floor((1. - validation_fraction) * len(rows)))
        if cut - length < 1 or cut == len(rows):
            raise ValueError(f"{battery}: too few rows for a nonempty purged split")
        train, purged, validation = rows[:cut - length], rows[cut - length:cut], rows[cut:]
        if data.local_indices[train[-1]] + length >= data.local_indices[validation[0]]:
            raise AssertionError("Retained training and validation supports overlap")
        if np.intersect1d(data.support_cycle_ids[train], data.support_cycle_ids[validation]).size:
            raise AssertionError("Original training and validation cycle supports overlap")
        for name, indices in (("train", train), ("purged", purged), ("validation", validation)):
            parts[name].extend(indices)
    result = {name: np.asarray(rows, dtype=np.int64) for name, rows in parts.items()}
    if not len(result["train"]) or not len(result["validation"]):
        raise ValueError("At least two batteries and nonempty inner partitions are required")
    return result


@dataclass
class ChannelScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 4 or not len(X) or not np.isfinite(X).all():
            raise ValueError("Scaler requires finite, nonempty [N, L, C, W] training inputs")
        mean = X.mean(axis=(0, 1, 3), keepdims=True)
        std = X.std(axis=(0, 1, 3), keepdims=True)
        return cls(mean, np.where(std == 0., 1., std))

    def transform(self, X, dtype=np.float64):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 4 or X.shape[2] != self.mean.shape[2] or not np.isfinite(X).all():
            raise ValueError("Invalid inputs for fitted channel scaler")
        return ((X - self.mean) / self.std).astype(dtype)


@dataclass
class RidgeModel:
    center: np.ndarray
    beta: np.ndarray
    y_mean: float
    alpha: float
    feature_shape: tuple

    def predict(self, standardized_X):
        X = np.asarray(standardized_X, dtype=np.float64)
        if X.shape[1:] != self.feature_shape:
            raise ValueError("Ridge input shape differs from fitted sequence shape")
        U = (X.reshape(len(X), -1) - self.center) / np.sqrt(self.center.size)
        return self.y_mean + U @ self.beta


def select_ridge(train_X, train_y, validation_X, validation_y):
    """Select the frozen alpha grid using inner labels only; do not refit."""
    flat = np.asarray(train_X, dtype=np.float64).reshape(len(train_X), -1)
    center = flat.mean(axis=0)
    U = (flat - center) / np.sqrt(flat.shape[1])
    y_mean = float(np.mean(train_y, dtype=np.float64))
    centered_y = np.asarray(train_y, dtype=np.float64) - y_mean
    kernel = U @ U.T
    identity = np.eye(len(U))
    best_model, best_loss, selection = None, float("inf"), []
    for alpha in RIDGE_ALPHAS:
        dual = np.linalg.solve(kernel + alpha * identity, centered_y)
        model = RidgeModel(center, U.T @ dual, y_mean, alpha, tuple(train_X.shape[1:]))
        loss = float(np.mean((model.predict(validation_X) - validation_y)**2))
        if not np.isfinite(loss):
            raise ValueError("Nonfinite ridge validation MSE")
        selection.append(dict(alpha=alpha, validation_mse=loss))
        if loss < best_loss:  # Ties keep the first alpha in the declared grid.
            best_model, best_loss = model, loss
    return best_model, selection


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim != 1 or not len(y_true) or y_pred.shape != y_true.shape:
        raise ValueError("Metrics require nonempty aligned 1-D targets and predictions")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metrics require finite targets and predictions")
    residual = y_pred - y_true
    denominator = float(np.sum((y_true - y_true.mean())**2))
    return dict(n=len(y_true), mae=float(np.mean(np.abs(residual))),
                rmse=float(np.sqrt(np.mean(residual**2))),
                r2=None if denominator == 0. else float(1. - np.sum(residual**2) / denominator))


def predict_hybrid(model, standardized_X, batch_size=64):
    """Inference uses eval/no_grad, including BatchNorm and dropout layers."""
    import torch
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(standardized_X), batch_size):
            batch = torch.as_tensor(standardized_X[start:start + batch_size], dtype=torch.float32)
            predicted, _ = model(batch)
            outputs.append(predicted.cpu().numpy())
    predictions = np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite hybrid predictions")
    return predictions


def train_hybrid(train_X, train_y, validation_X, validation_y, config,
                 checkpoint_path, history_path=None, progress=None):
    """Fit the existing architecture with static Adam LR and inner stopping."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from ..models.hybrid_model import CNNTCNLSTMAttention

    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    model = CNNTCNLSTMAttention(**HYBRID_ARCHITECTURE).cpu()
    loader = DataLoader(TensorDataset(torch.as_tensor(train_X, dtype=torch.float32),
                                     torch.as_tensor(train_y, dtype=torch.float32)),
                        batch_size=config.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(config.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()
    best_loss, best_state, best_epoch, stale = float("inf"), None, None, 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total = 0.
        for X_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(X_batch)
            loss = criterion(prediction, y_batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite hybrid training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(y_batch)
        validation_prediction = predict_hybrid(model, validation_X, config.batch_size)
        validation_loss = float(np.mean((validation_prediction - validation_y)**2))
        if not np.isfinite(validation_loss):
            raise ValueError("Nonfinite hybrid validation loss")
        history.append(dict(epoch=epoch, train_mse=total / len(train_y), validation_mse=validation_loss))
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
        else:
            stale += 1
        if history_path is not None:
            write_json(history_path, dict(best_epoch=best_epoch, epochs=history))
        if progress:
            progress(f"epoch={epoch} train_mse={history[-1]['train_mse']:.8g} validation_mse={validation_loss:.8g}")
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(best_epoch=best_epoch, best_validation_mse=best_loss, epochs=history)


def load_frozen_predictor(fold_dir, model_name):
    """Load a callable accepting raw, unstandardized sequence arrays."""
    fold_dir = Path(fold_dir)
    if model_name == "mean":
        with np.load(fold_dir / "mean_model.npz", allow_pickle=False) as archive:
            value = float(archive["value"])
        return lambda X: np.full(len(X), value, dtype=np.float64)
    with np.load(fold_dir / "scaler.npz", allow_pickle=False) as archive:
        scaler = ChannelScaler(archive["mean"], archive["std"])
    if model_name == "ridge":
        with np.load(fold_dir / "ridge_model.npz", allow_pickle=False) as archive:
            model = RidgeModel(archive["center"], archive["beta"], float(archive["y_mean"]),
                               float(archive["alpha"]), tuple(archive["feature_shape"].tolist()))
        return lambda X: model.predict(scaler.transform(X))
    if model_name == "hybrid":
        import torch
        from ..models.hybrid_model import CNNTCNLSTMAttention
        settings = json.loads((fold_dir / "hybrid_config.json").read_text())
        model = CNNTCNLSTMAttention(**settings["architecture"]).cpu()
        model.load_state_dict(torch.load(fold_dir / "hybrid_best.pt", map_location="cpu", weights_only=True))
        model.eval()
        def predict_raw(X):
            if np.asarray(X).shape[1:] != tuple(settings["feature_shape"]):
                raise ValueError("Hybrid input shape differs from fitted sequence shape")
            return predict_hybrid(model, scaler.transform(X, np.float32), settings["batch_size"])
        return predict_raw
    raise ValueError(f"Unknown model: {model_name}")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_predictions(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_protocol(data_dir, output_dir, config=None, models=("mean", "ridge", "hybrid"),
                 battery_names=BATTERIES, held_out_batteries=None, progress=None):
    """Run a declared protocol only into a new directory; preserve old artifacts."""
    config = config or ProtocolConfig()
    config.validate()
    models = tuple(models)
    held_out_batteries = tuple(held_out_batteries or battery_names)
    if not models or len(set(models)) != len(models) or not set(models) <= {"mean", "ridge", "hybrid"}:
        raise ValueError("Select unique models from mean, ridge, hybrid")
    if len(set(held_out_batteries)) != len(held_out_batteries) or not set(held_out_batteries) <= set(battery_names):
        raise ValueError("Select unique held-out batteries from the dataset")
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "manifest.json"
    manifest = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                    protocol="purged_chronological_inner_validation_outer_battery_holdout",
                    config=asdict(config), models=list(models), held_out_batteries=list(held_out_batteries),
                    ridge_alphas=list(RIDGE_ALPHAS), ridge_scaling="centered flattened features divided by sqrt(D)",
                    hybrid_architecture=HYBRID_ARCHITECTURE, scheduler=None, refit_after_selection=False,
                    soh_units="fraction", r2_constant_target="null; excluded from macro R2 with count reported",
                    device="cpu", seed_reset="model and DataLoader generator reset per fold",
                    versions=dict(python=platform.python_version(), numpy=np.__version__), folds={})
    write_json(manifest_path, manifest)
    try:
        source_root = Path(__file__).resolve().parents[2]
        paths = sorted(set(source_root.glob("src/**/*.py")) | set(source_root.glob("scripts/*.py")))
        manifest["source_sha256"] = {str(path.relative_to(source_root)): sha256(path) for path in paths}
        data_paths = [data_dir / f"{battery}.npz" for battery in battery_names] + [data_dir / "sequences.npz"]
        manifest["data_directory"] = str(data_dir.resolve())
        manifest["data_sha256"] = {path.name: sha256(path) for path in data_paths}
        write_json(manifest_path, manifest)
        data = reconstruct_sequences(data_dir, config.sequence_length, battery_names)
        np.savez_compressed(output_dir / "sample_metadata.npz", **data.metadata())
        all_rows, fold_metrics, truths = [], {}, []
        predictions_by_model = {name: [] for name in models}
        for held_out in held_out_batteries:
            fold_dir = output_dir / "folds" / held_out
            fold_dir.mkdir(parents=True, exist_ok=False)
            split = make_fold(data, held_out, config.validation_fraction)
            np.savez_compressed(fold_dir / "split_indices.npz", **split)
            manifest["folds"][held_out] = dict(status="running", counts={name: len(rows) for name, rows in split.items()})
            write_json(manifest_path, manifest)
            if progress:
                progress(f"fold={held_out} counts={manifest['folds'][held_out]['counts']}")
            train, validation, test = split["train"], split["validation"], split["test"]
            scaler = ChannelScaler.fit(data.X[train])
            np.savez(fold_dir / "scaler.npz", mean=scaler.mean, std=scaler.std)
            train_X, validation_X = scaler.transform(data.X[train]), scaler.transform(data.X[validation])
            predictions = {}
            if "mean" in models:
                value = float(np.mean(data.y[train], dtype=np.float64))
                np.savez(fold_dir / "mean_model.npz", value=value)
                predictions["mean"] = np.full(len(test), value)
            if "ridge" in models:
                ridge, selection = select_ridge(train_X, data.y[train], validation_X, data.y[validation])
                np.savez(fold_dir / "ridge_model.npz", center=ridge.center, beta=ridge.beta,
                         y_mean=ridge.y_mean, alpha=ridge.alpha, feature_shape=ridge.feature_shape)
                write_json(fold_dir / "ridge_selection.json", selection)
                predictions["ridge"] = ridge.predict(scaler.transform(data.X[test]))
            if "hybrid" in models:
                import torch
                manifest["versions"]["torch"] = torch.__version__
                write_json(manifest_path, manifest)
                model, history = train_hybrid(train_X.astype(np.float32), data.y[train],
                    validation_X.astype(np.float32), data.y[validation], config,
                    fold_dir / "hybrid_best.pt", fold_dir / "hybrid_history.json", progress)
                write_json(fold_dir / "hybrid_history.json", history)
                write_json(fold_dir / "hybrid_config.json", dict(architecture=HYBRID_ARCHITECTURE,
                    batch_size=config.batch_size, feature_shape=list(train_X.shape[1:])))
                predictions["hybrid"] = predict_hybrid(model, scaler.transform(data.X[test], np.float32), config.batch_size)
            # Outer targets are used only here, after inner selection is frozen.
            truth = data.y[test].astype(np.float64)
            truths.append(truth)
            fold_metrics[held_out] = {name: metrics(truth, predictions[name]) for name in models}
            write_json(fold_dir / "metrics.json", fold_metrics[held_out])
            for name in models:
                predictions_by_model[name].append(predictions[name])
            fold_rows = []
            for index, row in enumerate(test):
                fold_rows.append(dict(combined_row=int(row), battery=str(data.batteries[row]),
                    local_sequence_index=int(data.local_indices[row]),
                    target_retained_index=int(data.target_retained_indices[row]),
                    target_cycle_id=int(data.target_cycle_ids[row]), soh_true=float(truth[index]),
                    **{name: float(predictions[name][index]) for name in models}))
            write_predictions(fold_dir / "predictions.csv", fold_rows)
            all_rows.extend(fold_rows)
            manifest["folds"][held_out]["status"] = "completed"
            write_json(manifest_path, manifest)
        write_predictions(output_dir / "predictions.csv", all_rows)
        summary = dict(soh_units="fraction", per_battery=fold_metrics, macro={}, pooled={})
        for name in models:
            values = [fold_metrics[battery][name] for battery in held_out_batteries]
            r2_values = [value["r2"] for value in values if value["r2"] is not None]
            summary["macro"][name] = dict(mae=float(np.mean([v["mae"] for v in values])),
                rmse=float(np.mean([v["rmse"] for v in values])),
                r2=float(np.mean(r2_values)) if r2_values else None, r2_defined_folds=len(r2_values))
            summary["pooled"][name] = metrics(np.concatenate(truths), np.concatenate(predictions_by_model[name]))
        write_json(output_dir / "summary.json", summary)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(manifest_path, manifest)
        return summary
    except Exception as error:
        manifest.update(status="failed", failed_utc=datetime.now(timezone.utc).isoformat(),
                        error=f"{type(error).__name__}: {error}")
        for fold in manifest["folds"].values():
            if fold["status"] == "running":
                fold["status"] = "failed"
        write_json(manifest_path, manifest)
        raise
