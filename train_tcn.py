
"""
TCN Autoencoder Training 
- Builds validation/test from provided splits: normal vs abn_plus_mixed (positives)
- Trains TCN Autoencoder, evaluates via reconstruction-error threshold (F1-optimized)
- Saves: model (.h5), training history (CSV), results (JSON)
- Optional Optuna HPO (toggle ENABLE_OPTUNA)
"""
from __future__ import annotations
import os
import json
import numpy as np
import optuna
from typing import Optional, Tuple, Dict
import importlib
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# =============================
# SETTINGS 
# =============================
NPZ_PATH = "preprocess/emg_phase1_per_segment.npz"
POSITIVE_SET = "abn_plus_mixed" 
OUTPUT_DIR = "./artifacts"     
ENABLE_OPTUNA = False          

# Base hyperparameters (used if Hyperparameteroptimierung disabled)
HYPERPARAMS = {
    "filters": 16,
    "kernel_size": 3,
    "dilations": (1, 2, 4, 8),
    "dropout": 0.2,
    "latent_dim": 8,
    "pool_size": 4,
    "batch_size": 64,
    "epochs": 2,
    "learning_rate": 1e-3,
    "early_stopping": True,
    "patience": 8,
}

# =============================
# 1) DATA LOADING 
# =============================

pre = importlib.import_module("preprocess")


def ensure_3d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1, 1)
    elif x.ndim == 2:
        x = x[..., None]
    return x.astype(np.float32)


def build_val_test_from_splits(splits: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X_val, y_val, X_test, y_test from user's splits dict.
    Labels: 0 for normal, 1 for POSITIVE_SET.
    """
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
# 2) MODEL (TCN Autoencoder)
# =============================
def build_tcn_autoencoder(seq_len: int, n_feat: int, filters: int = 32, kernel_size: int = 3,
                          dilations=(1, 2, 4, 8), dropout: float = 0.2,
                          latent_dim: int = 8, pool_size: int = 2) -> Model:
    """
    Same-length TCN Autoencoder (no temporal down/up-sampling) to avoid length mismatches
    and reduce memory spikes from pooling/upsampling. Uses dilations to grow receptive field.
    """
    inp = Input(shape=(seq_len, n_feat), name="input")

    # Encoder: dilated conv stack with skip concat
    x = inp
    enc_skips = []
    for d in dilations:
        x = layers.Conv1D(filters, kernel_size, dilation_rate=d, padding="same", activation="relu")(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
        enc_skips.append(x)
    if len(enc_skips) > 1:
        x = layers.Concatenate(axis=-1, name="enc_skip_concat")(enc_skips)

    # Channel bottleneck only (no time compression)
    z = layers.Conv1D(latent_dim, 1, padding="same", activation=None, name="bottleneck_conv")(x)

    # Decoder: another dilated conv stack (independent weights), same length
    y = z
    dec_skips = []
    for d in dilations:
        y = layers.Conv1D(filters, kernel_size, dilation_rate=d, padding="same", activation="relu")(y)
        if dropout > 0:
            y = layers.Dropout(dropout)(y)
        dec_skips.append(y)
    if len(dec_skips) > 1:
        y = layers.Concatenate(axis=-1, name="dec_skip_concat")(dec_skips)

    out = layers.Conv1D(n_feat, 1, padding="same", activation=None, name="output_conv")(y)
    # Ensure exact length match
    out = layers.Lambda(lambda t: t[:, :seq_len, :], name="fix_length")(out)
    return Model(inp, out, name="TCN_Autoencoder")



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
    grid = np.linspace(lo, hi, 100)
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


def evaluate(model: Model, X: np.ndarray, y: Optional[np.ndarray], threshold: Optional[float] = None):
    X_pred = model.predict(X, verbose=0)
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

def train_once(hp: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_train, X_val, y_val, X_test, y_test, meta = load_dataset_via_preprocessing(NPZ_PATH)

    seq_len, n_feat = X_train.shape[1], X_train.shape[2]
    model = build_tcn_autoencoder(
        seq_len, n_feat,
        filters=hp["filters"],
        kernel_size=hp["kernel_size"],
        dilations=hp["dilations"],
        dropout=hp["dropout"],
        latent_dim=hp["latent_dim"],
        pool_size=hp["pool_size"],
    )
    opt = tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"])
    model.compile(optimizer=opt, loss="mse")

    cbs = [tf.keras.callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "training_history.csv"))]
    if hp.get("early_stopping", False):
        cbs.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=hp.get("patience", 8), restore_best_weights=True))

    val_tuple = (X_val, X_val)
    model.fit(X_train, X_train, epochs=hp["epochs"], batch_size=hp["batch_size"],
              shuffle=True, validation_data=val_tuple, callbacks=cbs, verbose=1)

    thr, prec, rec, f1, cm = evaluate(model, X_test, y_test)
    results = {
        "meta": meta,
        "params": hp,
        "threshold": float(thr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    model.save(os.path.join(OUTPUT_DIR, "tcn_autoencoder_model.h5"))
    print("Saved artifacts to:", os.path.abspath(OUTPUT_DIR))
    return f1


def main():
    if not ENABLE_OPTUNA:
        train_once(HYPERPARAMS)
        return

    
    X_train, X_val, y_val, X_test, y_test, meta = load_dataset_via_preprocessing(NPZ_PATH)
    seq_len, n_feat = X_train.shape[1], X_train.shape[2]

    def objective(trial: "optuna.Trial"):
        hp = HYPERPARAMS.copy()
        hp["filters"] = trial.suggest_int("filters", 16, 128)
        hp["kernel_size"] = trial.suggest_int("kernel_size", 2, 7)
        hp["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        hp["latent_dim"] = trial.suggest_int("latent_dim", 4, 32)
        hp["pool_size"] = trial.suggest_categorical("pool_size", [1, 2, 4])
        hp["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
        hp["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        hp["dilations"] = trial.suggest_categorical("dilations", [(1,2,4,8), (1,2,4,8,16), (1,3,9), (1,2,3,5,8)])

        # Build & train
        model = build_tcn_autoencoder(seq_len, n_feat,
                                      filters=hp["filters"], kernel_size=hp["kernel_size"],
                                      dilations=hp["dilations"], dropout=hp["dropout"],
                                      latent_dim=hp["latent_dim"], pool_size=hp["pool_size"])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"]), loss="mse")
        model.fit(X_train, X_train, epochs=max(10, HYPERPARAMS["epochs"]//2), batch_size=hp["batch_size"],
                  shuffle=True, validation_data=(X_val, X_val), verbose=0)
        # Evaluate on validation
        thr, prec, rec, f1, cm = evaluate(model, X_val, y_val)
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=True)
    print("Best params:", study.best_trial.params, "value=", study.best_value)

    # Merge best into HYPERPARAMS and train final
    HYPERPARAMS.update(study.best_trial.params)
    train_once(HYPERPARAMS)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()

