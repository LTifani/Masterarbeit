from __future__ import annotations
import os
import json
import csv
import time
from typing import Optional, Tuple, Dict

import importlib
import numpy as np

# =============================
# Torch / Optuna
# =============================
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import optuna

# =============================
# SETTINGS
# =============================
NPZ_PATH = "preprocess/emg_phase1_per_segment.npz"
POSITIVE_SET = "abn_plus_mixed"  # ("abnormal", "mixed", "abn_plus_mixed")
OUTPUT_DIR = "./artifacts"
ENABLE_OPTUNA = False

# Base hyperparameters (used if HPO disabled)
HYPERPARAMS = {
    "filters": 16,
    "kernel_size": 3,
    "dilations": (1, 2, 4, 8),
    "dropout": 0.2,
    "latent_dim": 8,
    "pool_size": 4,        
    "batch_size": 16,
    "epochs": 15,
    "learning_rate": 1e-3,
    "early_stopping": True,
    "patience": 8,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pre = importlib.import_module("preprocess")

# =============================
# 1) DATA LOADING
# =============================
def ensure_3d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1, 1)
    elif x.ndim == 2:
        x = x[..., None]
    return x.astype(np.float32)

def build_val_test_from_splits(splits: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Validation
    X_vn = splits["val"]["normal"]
    X_vp = splits["val"].get(POSITIVE_SET)
    if X_vp is None:
        raise KeyError(f"POSITIVE_SET '{POSITIVE_SET}' not found in splits['val'].")
    X_val = np.concatenate([X_vn, X_vp], axis=0) if X_vp.size > 0 else X_vn
    y_val = np.concatenate([
        np.zeros(len(X_vn), dtype=np.int32),
        np.ones(len(X_vp), dtype=np.int32)
    ]) if X_vp.size > 0 else np.zeros(len(X_vn), dtype=np.int32)

    # Test
    X_tn = splits["test"]["normal"]
    X_tp = splits["test"].get(POSITIVE_SET)
    if X_tp is None:
        raise KeyError(f"POSITIVE_SET '{POSITIVE_SET}' not found in splits['test'].")
    X_test = np.concatenate([X_tn, X_tp], axis=0) if X_tp.size > 0 else X_tn
    y_test = np.concatenate([
        np.zeros(len(X_tn), dtype=np.int32),
        np.ones(len(X_tp), dtype=np.int32)
    ]) if X_tp.size > 0 else np.zeros(len(X_tn), dtype=np.int32)

    return ensure_3d(X_val), y_val, ensure_3d(X_test), y_test

def load_dataset_via_preprocessing(npz_path: str):
    """
    Returns X_train, X_val, y_val, X_test, y_test, meta
    """
    X_train, splits, meta = pre.load_npz_for_training(npz_path)
    X_val, y_val, X_test, y_test = build_val_test_from_splits(splits)
    return ensure_3d(X_train), X_val, y_val, X_test, y_test, meta

# =============================
# 2) MODEL (TCN Autoencoder in Torch)
# =============================

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)

class TcnAutoencoder(nn.Module):
    """
    Same-length TCN Autoencoder: enc stack (dilations) with skip concat -> 1x1 bottleneck ->
    dec stack (dilations) with skip concat -> 1x1 to n_feat; keeps temporal length.
    Note: PyTorch expects (N, C, T); our data is (N, T, C) so we permute.
    """
    def __init__(self, seq_len: int, n_feat: int, filters: int = 32, kernel_size: int = 3,
                 dilations=(1, 2, 4, 8), dropout: float = 0.2, latent_dim: int = 8, pool_size: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.n_feat = n_feat
        self.enc_blocks = nn.ModuleList()
        enc_in = n_feat
        # Encoder dilated stack with skip outputs (concat later)
        for d in dilations:
            self.enc_blocks.append(ConvBlock1D(enc_in, filters, kernel_size, dilation=d, dropout=dropout))
            enc_in = filters  # subsequent blocks see 'filters' channels

        # Bottleneck 1x1 (channel compression)
        self.bottleneck = nn.Conv1d(filters * len(dilations), latent_dim, kernel_size=1)

        # Decoder dilated stack with skip outputs (independent weights)
        self.dec_blocks = nn.ModuleList()
        dec_in = latent_dim
        for d in dilations:
            self.dec_blocks.append(ConvBlock1D(dec_in, filters, kernel_size, dilation=d, dropout=dropout))
            dec_in = filters

        # Final projection back to features
        self.out_conv = nn.Conv1d(filters * len(dilations), n_feat, kernel_size=1)

    def forward(self, x):  # x: (N, T, C)
        # to (N, C, T)
        x = x.permute(0, 2, 1)

        # Encoder with skip collection
        enc_skips = []
        tmp = x
        for blk in self.enc_blocks:
            tmp = blk(tmp)
            enc_skips.append(tmp)
        enc_cat = torch.cat(enc_skips, dim=1) if len(enc_skips) > 1 else enc_skips[0]

        z = self.bottleneck(enc_cat)

        # Decoder with skip collection
        dec_skips = []
        y = z
        for blk in self.dec_blocks:
            y = blk(y)
            dec_skips.append(y)
        dec_cat = torch.cat(dec_skips, dim=1) if len(dec_skips) > 1 else dec_skips[0]

        out = self.out_conv(dec_cat)
        # back to (N, T, C)
        out = out.permute(0, 2, 1)
        # exact length match (safety)
        out = out[:, :self.seq_len, :]
        return out

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

def evaluate(model: nn.Module, X: np.ndarray, y: Optional[np.ndarray], threshold: Optional[float] = None):
    X_pred = predict_reconstruction(model, X, batch_size=128)
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

class EarlyStopping:
    def __init__(self, patience: int = 8):
        self.patience = patience
        self.best = float("inf")
        self.count = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module):
        if val_loss < self.best - 1e-12:
            self.best = val_loss
            self.count = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.count += 1
        return self.count >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

def _write_csv_header(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss"])

def _append_csv(path, epoch, train_loss, val_loss):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}"])

def _recon_mse_epoch(model, loader, optimizer=None):
    """
    If optimizer is provided -> train epoch, else eval epoch.
    Loss = mean squared error over all timesteps and features.
    """
    if optimizer is None:
        model.eval()
        torch.set_grad_enabled(False)
    else:
        model.train()
        torch.set_grad_enabled(True)

    total_loss = 0.0
    n_batches = 0
    for xb, in loader:
        xb = xb.to(DEVICE)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        yb = model(xb)
        loss = ((xb - yb) ** 2).mean()  # average MSE (matches Keras default reduction)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += loss.detach().item()
        n_batches += 1

    torch.set_grad_enabled(True)
    return total_loss / max(1, n_batches)

def train_once(hp: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_train, X_val, y_val, X_test, y_test, meta = load_dataset_via_preprocessing(NPZ_PATH)

    seq_len, n_feat = X_train.shape[1], X_train.shape[2]
    model = TcnAutoencoder(
        seq_len=seq_len, n_feat=n_feat,
        filters=hp["filters"], kernel_size=hp["kernel_size"],
        dilations=hp["dilations"], dropout=hp["dropout"],
        latent_dim=hp["latent_dim"], pool_size=hp["pool_size"],
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])

    # Dataloaders
    bs = int(hp["batch_size"])
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=bs, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val)),   batch_size=bs, shuffle=False)

    # CSV logger
    csv_path = os.path.join(OUTPUT_DIR, "training_history.csv")
    _write_csv_header(csv_path)

    stopper = EarlyStopping(patience=hp.get("patience", 8)) if hp.get("early_stopping", False) else None

    # Train
    for epoch in range(1, int(hp["epochs"]) + 1):
        train_loss = _recon_mse_epoch(model, train_loader, optimizer=optimizer)
        val_loss   = _recon_mse_epoch(model, val_loader, optimizer=None)
        _append_csv(csv_path, epoch, train_loss, val_loss)

        if stopper:
            should_stop = stopper.step(val_loss, model)
            if should_stop:
                break

    if stopper:
        stopper.restore(model)

    # Evaluate on test
    thr, prec, rec, f1, cm = evaluate(model, X_test, y_test)
    results = {
        "meta": meta,
        "params": {
            **hp,
            "device": str(DEVICE),
        },
        "threshold": float(thr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "tcn_autoencoder_model.pt"))
    print("Saved artifacts to:", os.path.abspath(OUTPUT_DIR))
    return f1

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not ENABLE_OPTUNA:
        train_once(HYPERPARAMS)
        return

    # For Optuna we need dataset shapes for model creation inside objective
    X_train, X_val, y_val, X_test, y_test, meta = load_dataset_via_preprocessing(NPZ_PATH)
    seq_len, n_feat = X_train.shape[1], X_train.shape[2]

    def objective(trial: "optuna.Trial"):
        hp = HYPERPARAMS.copy()
        hp["filters"]       = trial.suggest_int("filters", 16, 128)
        hp["kernel_size"]   = trial.suggest_int("kernel_size", 2, 7)
        hp["dropout"]       = trial.suggest_float("dropout", 0.0, 0.5)
        hp["latent_dim"]    = trial.suggest_int("latent_dim", 4, 32)
        hp["pool_size"]     = trial.suggest_categorical("pool_size", [1, 2, 4])
        hp["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
        hp["batch_size"]    = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        hp["dilations"]     = trial.suggest_categorical("dilations", [(1,2,4,8), (1,2,4,8,16), (1,3,9), (1,2,3,5,8)])

        model = TcnAutoencoder(
            seq_len=seq_len, n_feat=n_feat,
            filters=hp["filters"], kernel_size=hp["kernel_size"],
            dilations=hp["dilations"], dropout=hp["dropout"],
            latent_dim=hp["latent_dim"], pool_size=hp["pool_size"]
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])

        # DataLoaders
        bs = int(hp["batch_size"])
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=bs, shuffle=True)
        val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val)),   batch_size=bs, shuffle=False)

        # Shorter training for HPO
        epochs = max(10, HYPERPARAMS["epochs"] // 2)
        for _ in range(epochs):
            _ = _recon_mse_epoch(model, train_loader, optimizer=optimizer)
            _ = _recon_mse_epoch(model, val_loader, optimizer=None)

        # Evaluate on validation (same as Keras version)
        thr, prec, rec, f1, _ = evaluate(model, X_val, y_val)
        # Maximize f1
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    print("Best params:", study.best_trial.params, "value=", study.best_value)

    # Merge best into HYPERPARAMS and train final
    HYPERPARAMS.update(study.best_trial.params)
    train_once(HYPERPARAMS)

if __name__ == "__main__":
    main()
