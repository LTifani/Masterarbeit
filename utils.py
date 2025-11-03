from __future__ import annotations

import os
import csv
import json
import time
import torch
import logging
import numpy as np
from torch import nn
from icecream import ic
from pathlib import Path
from functools import wraps
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Callable, Any

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


LOG_FILE = "output/console.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(filename='output/console.log', filemode='w+', format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG)
# LOG_FILE = open(LOG_FILE, "w")

def dict_to_str(d: Dict) -> str:
    return "\t".join([f"{k}={v}" for k, v in d.items()])

def join_path(*args) -> str:
    """Join paths in a platform-independent way."""
    return os.path.join(*args)

def dual_output(s):
    # print(s)                # Konsole
    logging.info(s)

ic.configureOutput("", includeContext=True, outputFunction=dual_output)

def get_now_str() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

@dataclass
class Config:
    """Configuration for EMG preprocessing pipeline."""
    # Input/Output
    input_dir: str = "Dataset"
    output_dir: str = "preprocess"
    output_name: str = "emg_phase1_augment"
    train_output_dir: str = "artifacts/logs"

    # Segmentation
    target_sample_rate: int = 11025 # Sample rate in Hz / s
    window_duration_seconds: float = 0.01 # 10ms
    hop_duration_seconds: float = 0.005 # 5ms


    # Normalization
    normalization_mode: str = "per_segment"  # 'per_segment' | 'per_file' | 'none'

    # Filtering
    apply_bandpass_filter: bool = True
    bandpass_low: float = 20.0
    bandpass_high: float = 4500.0
    filter_order: int = 4
    apply_notch_filter: bool = True
    notch_frequency: float = 50.0
    notch_quality: float = 30.0

    # Data Sources
    normal_folder: str = "Normal"
    abnormal_folder: str = "Spontanaktivität"
    mixed_folder: str = "Mixed"

    # Mixed Usage
    use_mixed_in_training: bool = False
    include_mixed_in_test: bool = True

    # Splits (segment-based; patient overlap allowed)
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    min_unique_train_patients: int = 5

    # Augmentation (applied to training segments only)
    apply_augmentation: bool = True
    augmentations_per_segment: int = 1
    noise_probability: float = 0.5
    noise_std_range: Tuple[float, float] = (0.002, 0.01)
    time_shift_probability: float = 0.5
    max_time_shift_seconds: float = 0.1
    gain_probability: float = 0.5
    gain_range: Tuple[float, float] = (0.9, 1.1)
    time_stretch_probability: float = 0.3
    time_stretch_range: Tuple[float, float] = (0.95, 1.05)
    mask_probability: float = 0.3
    mask_fraction_range: Tuple[float, float] = (0.05, 0.1)

    # Training Hyperparameters
    create_time: str = get_now_str()
    positive_set: str = "abn_plus_mixed"  # ("abnormal", "mixed", "abn_plus_mixed")
    train_hyperparameter : Dict =  field(default_factory=dict)  # Placeholder for hyperparameters used during training
    best_model_path: str = ""
    
    def get_training_dir(self) -> str:
        dict_helper = {
            'ftr': self.train_hyperparameter.get("filters", 0),
            'ep': self.train_hyperparameter.get("epochs", 0),
            'bs': self.train_hyperparameter.get("batch_size", 0)
        }
        training_dir = f"{self.create_time}__" + "_".join([f"{k}_{v}" for k, v in dict_helper.items()])
        training_dir = join_path(self.train_output_dir, training_dir)
        os.makedirs(training_dir, exist_ok=True)
        return training_dir

    def save(self) -> None:
        config_path = join_path(self.get_training_dir(), "configuration.json")
        json.dump(vars(self), open(config_path, "w"), indent=4)

    def __del__(self):
        try:
            self.save()
        except Exception:
            pass

cfg = Config()


class EpochLogger:
    """
    A simple logger class to log epoch, train_loss, and val_loss to a CSV file.

    Args:
        file_path: Path to the CSV file.
        append: If True, append to existing file; otherwise, overwrite.

    Example:
        logger = EpochLogger("training_history.csv")
        for epoch in range(1, num_epochs + 1):
            # ... train and validate ...
            logger.log(epoch, train_loss, val_loss)
        logger.close()
    """

    def __init__(self, config: Config = cfg) -> None:

        training_dir = config.get_training_dir()
        self.results_path = join_path(training_dir, "results.json")
        self.best_model_path = join_path(training_dir, "best_model.pth")

        self.file = Path(join_path(training_dir, "results.csv")).open("w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["epoch", "train_loss", "val_loss"])

    def log(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """
        Log the values for one epoch.

        Args:
            epoch: Current epoch number.
            train_loss: Training loss.
            val_loss: Validation loss.
        """
        self.writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}"])
        self.file.flush()  # Ensure data is written immediately

    def close(self) -> None:
        """Close the file."""
        self.file.close()
        
    def save_best_model(self, model: torch.nn.Module) -> None:
        """Save the best model to the specified path."""
        torch.save(model.state_dict(), self.best_model_path)

    
    def save_results(self, thr: float, prec: float, rec: float, f1: float, cm: Dict[str, int]) -> None:
        """Save evaluation results to a JSON file."""        
        results = {
            "threshold": float(thr),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm,
        }
        json.dump(results, open(self.results_path, "w"), indent=4)

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

def ensure_3d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1, 1)
    elif x.ndim == 2:
        x = x[..., None]
    return x.astype(np.float32)

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

def timer(func: Callable) -> Callable:
    """Decorator to measure and print execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper

def get_error_per_sample(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Calculates MSE per sample, averaging over specified dimensions (e.g., time, features)."""
    return (target - prediction).square().mean(dim=[1, 2])

# __all__ = ["Config", "cfg", "EpochLogger", "EarlyStopping", "dict_to_str", "join_path", "ensure_3d", "build_val_test_from_splits", "timer"]

if __name__ == "__main__":
    f = Config()