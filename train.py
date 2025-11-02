from __future__ import annotations
import os
import json
import csv
import time
import shutil
import numpy as np
from utils import *
from models import TcnAutoencoder
from typing import Optional, Tuple, Dict
from preprocess import load_npz_for_training


# =============================
# Torch / Optuna
# =============================
import torch
import optuna
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


# =============================
# SETTINGS
# =============================
NPZ_PATH = "preprocess_origin/emg_phase1_per_segment.npz"
OUTPUT_DIR = "artifacts"
ENABLE_OPTUNA = True  # Set to False to disable hyperparameter optimization

# Base hyperparameters (used if HPO disabled)
HYPERPARAMS = {
    "filters": 16,
    "kernel_size": 2,
    "dilations": (1, 2, 4, 8),
    "dropout": 0.2,
    "latent_dim": 8,
    "batch_size": 16,
    "epochs": 15,
    "learning_rate": 1e-3,
    "early_stopping": True,
    "patience": 8,
}



def build_val_test_from_splits(splits: Dict, positive: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Validation
    X_vn = splits["val"]["normal"]
    X_vp = splits["val"].get(positive)
    if X_vp is None:
        raise KeyError(f"POSITIVE_SET '{positive}' not found in splits['val'].")
    X_val = np.concatenate([X_vn, X_vp], axis=0) if X_vp.size > 0 else X_vn
    y_val = np.concatenate([
        np.zeros(len(X_vn), dtype=np.int32),
        np.ones(len(X_vp), dtype=np.int32)
    ]) if X_vp.size > 0 else np.zeros(len(X_vn), dtype=np.int32)

    # Test
    X_tn = splits["test"]["normal"]
    X_tp = splits["test"].get(positive)
    if X_tp is None:
        raise KeyError(f"POSITIVE_SET '{positive}' not found in splits['test'].")
    X_test = np.concatenate([X_tn, X_tp], axis=0) if X_tp.size > 0 else X_tn
    y_test = np.concatenate([
        np.zeros(len(X_tn), dtype=np.int32),
        np.ones(len(X_tp), dtype=np.int32)
    ]) if X_tp.size > 0 else np.zeros(len(X_tn), dtype=np.int32)

    return ensure_3d(X_val), y_val, ensure_3d(X_test), y_test

def load_dataset_via_preprocessing(npz_path: str, config: Config = cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Returns X_train, X_val, y_val, X_test, y_test, meta
    """
    X_train, splits, meta = load_npz_for_training(npz_path)
    X_val, y_val, X_test, y_test = build_val_test_from_splits(splits, config.positive_set)
    return ensure_3d(X_train), X_val, y_val, X_test, y_test, meta

# =============================
# 3) THRESHOLD + METRICS
# =============================

def find_best_threshold(errors: np.ndarray, labels: Optional[np.ndarray]) -> Tuple[float, Optional[float]]:
    if labels is None:
        thr = float(np.percentile(errors, 95))
        return thr, None
    if labels.ndim > 1:
        labels = labels.flatten()
    if errors.ndim > 1:
        errors = errors.max(axis=1)
    best_thr, best_f1 = 0.0, 0.0
    lo, hi = float(errors.min()), float(errors.max())
    grid = np.linspace(lo, hi, 100) if hi > lo else np.array([hi])
    for thr in grid:
        pred = (errors > thr).astype(int)
        TP = int(((pred == 1) & (labels == 1)).sum())
        FP = int(((pred == 1) & (labels == 0)).sum())
        FN = int(((pred == 0) & (labels == 1)).sum())
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1

@torch.no_grad()
def predict_reconstruction(model: nn.Module, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(X).to(DEVICE)
    dl = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    outs = []
    for (xb,) in dl:
        yb = model(xb)
        outs.append(yb.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)

def evaluate(X_pred: np.ndarray, X: np.ndarray, y: Optional[np.ndarray], threshold: Optional[float] = None) -> Tuple[float, float, float, float, Dict[str, int]]:
    errs_seq = ((X - X_pred) ** 2).mean(axis=-1)  # (N, T)
    errs = errs_seq.max(axis=1) if errs_seq.ndim == 2 else errs_seq
    if threshold is None:
        threshold, _ = find_best_threshold(errs, y)
    pred = (errs > threshold).astype(int)
    if y is None:
        return threshold, float("nan"), float("nan"), float("nan"), {"TP":0, "FP":0, "FN":0, "TN":0}
    y_samp = y.astype(int) if y.ndim == 1 else (y.max(axis=1) > 0).astype(int)
    TP = int(((pred == 1) & (y_samp == 1)).sum())
    FP = int(((pred == 1) & (y_samp == 0)).sum())
    FN = int(((pred == 0) & (y_samp == 1)).sum())
    TN = int(((pred == 0) & (y_samp == 0)).sum())
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
    return threshold, prec, rec, f1, {"TP":TP, "FP":FP, "FN":FN, "TN":TN}

# =============================
# 4) TRAIN (and optional HPO)
# =============================

@torch.no_grad()
def validate_recon_mse_epoch(model: nn.Module, loader: DataLoader) -> float:
    """
    If optimizer is provided -> train epoch, else eval epoch.
    Loss = mean squared error over all timesteps and features.
    """
    model.eval()
    n_batches = 0
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    for xb, in loader:
        xb = xb.to(DEVICE)
        yb = model(xb)
        loss = criterion(xb, yb)
        total_loss += loss.detach().item()
        n_batches += 1
    return total_loss / max(1, n_batches)

def train_recon_mse_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """
    If optimizer is provided -> train epoch, else eval epoch.
    Loss = mean squared error over all timesteps and features.
    """

    model.train()
    # Define loss
    criterion = nn.MSELoss()

    total_loss = 0.0
    n_batches = 0
    for xb, in loader:
        xb = xb.to(DEVICE)
        optimizer.zero_grad()
        yb = model(xb)
        loss = criterion(xb, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        n_batches += 1

    return total_loss / max(1, n_batches)

def get_data(hp: Dict):
    X_train, X_val, y_val, X_test, y_test, meta = load_dataset_via_preprocessing(NPZ_PATH)

    seq_len, n_feat = X_train.shape[1], X_train.shape[2]
    model = TcnAutoencoder(
        seq_len=seq_len, n_feat=n_feat,
        filters=hp["filters"], kernel_size=hp["kernel_size"],
        dilations=hp["dilations"], dropout=hp["dropout"],
        latent_dim=hp["latent_dim"],
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])

    # Dataloaders
    batch_size = int(hp["batch_size"])
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val)),   batch_size=batch_size, shuffle=False)
    
    return model, optimizer, train_loader, val_loader, X_test, y_test, X_val, y_val

def train_once(hp: dict, logger: Optional[EpochLogger]=None) -> float:
    model, optimizer, train_loader, val_loader, X_test, y_test, _ , _ = get_data(hp)

    # Early stopping
    stopper = EarlyStopping(patience=hp.get("patience", 8)) if hp.get("early_stopping", False) else None

    # Train
    for epoch in range(1, int(hp["epochs"]) + 1):
        train_loss = train_recon_mse_epoch(model, train_loader, optimizer)
        val_loss   = validate_recon_mse_epoch(model, val_loader)
        if logger:
            logger.log(epoch, train_loss, val_loss)

        if stopper:
            should_stop = stopper.step(val_loss, model)
            if should_stop:
                break

    if stopper:
        # Save model with best val loss
        stopper.restore(model)
    if logger:
        logger.save_best_model(model)

    # Evaluate on test
    X_pred = predict_reconstruction(model, X_test, batch_size=128)
    thr, prec, rec, f1, cm = evaluate(X_pred, X_test, y_test)
    if logger:
        logger.save_results(thr, prec, rec, f1, cm)
    
    return f1

@timer
def objective(trial: "optuna.Trial"):
    dilation_options = ["1,2,4,8", "1,2,4,8,16", "1,3,9", "1,2,3,5,8"]
    choice_str = trial.suggest_categorical("dilations", dilation_options)
    dilations = tuple(int(x) for x in choice_str.split(","))

    hp = {}
    hp["filters"]       = trial.suggest_int("filters", 16, 32)
    hp["kernel_size"]   = trial.suggest_int("kernel_size", 6, 10, log=True)
    hp["dropout"]       = trial.suggest_float("dropout", 0.0, 0.5)
    hp["latent_dim"]    = trial.suggest_int("latent_dim", 8, 32, log=True)
    hp["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    hp["batch_size"]    = trial.suggest_categorical("batch_size", [32, 64, 128]) #  [16, 32, 64, 128]
    hp["dilations"]     = dilations

    # Shorter training for HPO
    hp["epochs"] = max(10, HYPERPARAMS["epochs"] // 2)
    f1 = train_once({**HYPERPARAMS, **hp})
    hp["f1"] = f1
    hparams = dict_to_str(hp)
    ic(hparams)
    return f1

def main():

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    ic("Starting training...")
    if ENABLE_OPTUNA:
        cfg.train_hyperparameter = HYPERPARAMS
        logger = EpochLogger(cfg)
        train_once(HYPERPARAMS, logger)
        # return
    
    # For Optuna we need dataset shapes for model creation inside objective
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    print("Best params:", study.best_trial.params, "value=", study.best_value)
    ic(study.best_trial.params)

    # Merge best into HYPERPARAMS and train final
    HYPERPARAMS.update(study.best_trial.params)
    HYPERPARAMS["dilations"] = tuple(int(x) for x in HYPERPARAMS["dilations"].split(","))
    hparams = dict_to_str(HYPERPARAMS)
    ic(hparams)
    cfg.train_hyperparameter = HYPERPARAMS
    logger = EpochLogger(cfg)
    train_once(HYPERPARAMS, logger)

if __name__ == "__main__":
    main()
