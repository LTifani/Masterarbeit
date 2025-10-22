"""
preprocessing.py — Needle-EMG (audio) preprocessing & augmentation for Autoencoder (Phase 1)

Features:
- Resample to a target sampling rate (default: 11025 Hz)
- Optional bandpass & 50 Hz notch filtering
- Segmentation into fixed windows with hop/overlap
- Per-segment normalization (z-score)
- Data augmentation for Normal-Klasse (additive noise, time-shift, gain, mild time-stretch, time-mask)
- Deterministic splitting (train/val/test)
- Saves arrays ready for Autoencoder training (npz)


Outputs (in output_dir):
  emg_phase1_autoencoder.npz
    X_train           (N_train, win_len)
    X_val_normal      (N_val_norm, win_len)
    X_val_abnormal    (N_val_abn,  win_len)
    X_test_normal     (N_test_norm, win_len)
    X_test_abnormal   (N_test_abn,  win_len)
    meta              dict-like metadata (serialized via numpy object array)
"""
from __future__ import annotations
import os
import glob
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

try:
    import soundfile as sf  # preferred for wav I/O (preserves sr & dtype)
except Exception:
    sf = None

try:
    import librosa  # for resample & time-stretch
except Exception:
    librosa = None







# ----------------------------
# Configuration
# ----------------------------
@dataclass
class Config:
    input_dir: "Dataset"
    output_dir: "preprocess"
    target_sr: int = 11025
    window_sec: float = 2.0
    hop_sec: float = 0.5
    # Filtering
    apply_bandpass: bool = True
    bandpass: Tuple[float, float] = (20.0, 4500.0)
    filter_order: int = 4
    apply_notch: bool = True
    notch_freq: float = 50.0
    notch_q: float = 30.0
    # Splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    # Augmentation
    augment: bool = True
    aug_per_segment: int = 2
    # Augment parameters
    p_noise: float = 0.5
    noise_std: Tuple[float, float] = (0.002, 0.01)
    p_shift: float = 0.5
    max_shift_sec: float = 0.1
    p_gain: float = 0.5
    gain_range: Tuple[float, float] = (0.9, 1.1)
    p_stretch: float = 0.3
    stretch_range: Tuple[float, float] = (0.95, 1.05)
    p_mask: float = 0.3
    mask_frac_range: Tuple[float, float] = (0.05, 0.1)
    # Folder names
    normal_name: str = "Normal"
    abnormal_name: str = "Spontanaktivität"
    mixed: str = "Mixed"

    def output_file(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, "emg_phase1_autoencoder.npz")
    


# ----------------------------
# Utility functions
# ----------------------------

def _read_wav(path: str) -> Tuple[np.ndarray, int]:
    if sf is not None:
        y, sr = sf.read(path, always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y.astype(np.float32), int(sr)
    elif librosa is not None:
        y, sr = librosa.load(path, sr=None, mono=True)
        return y.astype(np.float32), int(sr)
    else:
        raise RuntimeError("Please install 'soundfile' or 'librosa' to read WAV files.")


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    if librosa is not None:
        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    else:
        ratio = target_sr / orig_sr
        idx = (np.arange(int(len(y) * ratio)) / ratio).astype(int)
        idx = np.clip(idx, 0, len(y) - 1)
        return y[idx].astype(np.float32)


def _butter_bandpass_coeffs(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')


def _apply_bandpass(y: np.ndarray, sr: int, band: Tuple[float, float], order: int) -> np.ndarray:
    b, a = _butter_bandpass_coeffs(band[0], band[1], sr, order)
    return filtfilt(b, a, y).astype(np.float32)


def _apply_notch50(y: np.ndarray, sr: int, f0: float, q: float) -> np.ndarray:
    b, a = iirnotch(f0 / (sr / 2.0), Q=q)
    return filtfilt(b, a, y).astype(np.float32)

##sliding window Verfahren
def segment_signal(y: np.ndarray, sr: int, win_sec: float, hop_sec: float) -> np.ndarray:
    win_len = int(round(win_sec * sr))
    hop_len = int(round(hop_sec * sr))
    if len(y) < win_len:
        return np.empty((0, win_len), dtype=np.float32)
    starts = np.arange(0, len(y) - win_len + 1, hop_len, dtype=int)
    return np.stack([y[s:s + win_len] for s in starts], axis=0).astype(np.float32)


def zscore_norm(seg: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean, std = np.mean(seg), np.std(seg)
    if std < eps:
        return np.zeros_like(seg)
    return ((seg - mean) / std).astype(np.float32)

# ----------------------------
# Augmentation functions
# ----------------------------

def aug_add_noise(seg: np.ndarray, std_rel: float) -> np.ndarray:
    noise = np.random.normal(0, std_rel * np.std(seg), size=seg.shape)
    return (seg + noise).astype(np.float32)


def aug_time_shift(seg: np.ndarray, sr: int, max_shift_sec: float) -> np.ndarray:
    shift = np.random.randint(-int(max_shift_sec * sr), int(max_shift_sec * sr) + 1)
    if shift > 0:
        return np.concatenate([np.zeros(shift), seg[:-shift]]).astype(np.float32)
    elif shift < 0:
        return np.concatenate([seg[-shift:], np.zeros(-shift)]).astype(np.float32)
    return seg


def aug_gain(seg: np.ndarray, low: float, high: float) -> np.ndarray:
    return (seg * np.random.uniform(low, high)).astype(np.float32)


def aug_time_stretch(seg: np.ndarray, rate: float) -> np.ndarray:
    """Zeitstreckung kompatibel mit Librosa ≥0.10 und älteren Versionen."""
    import numpy as np
    try:
        # Neue API (librosa >=0.10): erwartet STFT-Input
        D = librosa.stft(seg)
        D_stretch = librosa.effects.time_stretch(D, rate)
        stretched = librosa.istft(D_stretch).astype(np.float32)
    except Exception:
        # Fallback für ältere librosa-Versionen
        stretched = librosa.effects.time_stretch(seg, rate).astype(np.float32)

    # Länge anpassen auf Original
    if len(stretched) == len(seg):
        return stretched
    elif len(stretched) > len(seg):
        start = (len(stretched) - len(seg)) // 2
        return stretched[start:start+len(seg)]
    else:
        pad = len(seg) - len(stretched)
        left = pad // 2
        right = pad - left
        return np.pad(stretched, (left, right), mode="constant")



def aug_time_mask(seg: np.ndarray, low_frac: float, high_frac: float) -> np.ndarray:
    frac = np.random.uniform(low_frac, high_frac)
    n = len(seg)
    k = int(frac * n)
    start = np.random.randint(0, n - k)
    seg_copy = seg.copy()
    seg_copy[start:start + k] = 0.0
    return seg_copy.astype(np.float32)


def apply_augmentations(seg: np.ndarray, sr: int, cfg: Config) -> np.ndarray:
    ops = []
    if np.random.rand() < cfg.p_noise:
        std_rel = np.random.uniform(*cfg.noise_std)
        ops.append(lambda x: aug_add_noise(x, std_rel))
    if np.random.rand() < cfg.p_shift:
        ops.append(lambda x: aug_time_shift(x, sr, cfg.max_shift_sec))
    if np.random.rand() < cfg.p_gain:
        ops.append(lambda x: aug_gain(x, *cfg.gain_range))
    # if np.random.rand() < cfg.p_stretch:
    #     rate = np.random.uniform(*cfg.stretch_range)
    #     ops.append(lambda x: aug_time_stretch(x, rate))
    if np.random.rand() < cfg.p_mask:
        ops.append(lambda x: aug_time_mask(x, *cfg.mask_frac_range))

    np.random.shuffle(ops)
    for op in ops:
        seg = op(seg)
    return seg.astype(np.float32)

# ----------------------------
# Main processing
# ----------------------------

def _collect_files(path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(path, "*.wav")))


def _process_file(path: str, cfg: Config) -> np.ndarray:
    y, sr = _read_wav(path)
    y = _resample(y, sr, cfg.target_sr)
    if cfg.apply_bandpass:
        y = _apply_bandpass(y, cfg.target_sr, cfg.bandpass, cfg.filter_order)
    if cfg.apply_notch:
        y = _apply_notch50(y, cfg.target_sr, cfg.notch_freq, cfg.notch_q)
    segs = segment_signal(y, cfg.target_sr, cfg.window_sec, cfg.hop_sec)
    return np.stack([zscore_norm(s) for s in segs]) if len(segs) else segs


def _split_indices(n: int, ratios: Tuple[float, float, float], seed: int = 42):
    train_r, val_r, test_r = ratios
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    n_train = int(train_r * n)
    n_val = int(val_r * n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def preprocess_dataset(cfg: Config) -> str:
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    normal_files = _collect_files(os.path.join(cfg.input_dir, cfg.normal_name))
    abnorm_files = _collect_files(os.path.join(cfg.input_dir, cfg.abnormal_name))

    normal_segments = [s for f in normal_files for s in [_process_file(f, cfg)] if s.size > 0]
    abnormal_segments = [s for f in abnorm_files for s in [_process_file(f, cfg)] if s.size > 0]

    X_normal = np.concatenate(normal_segments, axis=0)
    X_abn = np.concatenate(abnormal_segments, axis=0) if abnormal_segments else np.empty((0, X_normal.shape[1]))

    i_train, i_val, i_test = _split_indices(len(X_normal), (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio), cfg.seed)
    X_train, X_val_normal, X_test_normal = X_normal[i_train], X_normal[i_val], X_normal[i_test]

    if len(X_abn):
        half = len(X_abn) // 2
        X_val_abn, X_test_abn = X_abn[:half], X_abn[half:]
    else:
        X_val_abn = np.empty((0, X_normal.shape[1]))
        X_test_abn = np.empty((0, X_normal.shape[1]))

    if cfg.augment:
        augments = [zscore_norm(apply_augmentations(s, cfg.target_sr, cfg)) for s in X_train for _ in range(cfg.aug_per_segment)]
        X_train = np.concatenate([X_train, np.stack(augments)], axis=0)
        np.random.default_rng(cfg.seed).shuffle(X_train)

    meta = {
        "config": asdict(cfg),
        "n_train": len(X_train),
        "n_val_normal": len(X_val_normal),
        "n_val_abnormal": len(X_val_abn),
        "n_test_normal": len(X_test_normal),
        "n_test_abnormal": len(X_test_abn),
    }

    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = cfg.output_file()
    np.savez_compressed(out_path,
        X_train=X_train, X_val_normal=X_val_normal, X_val_abnormal=X_val_abn,
        X_test_normal=X_test_normal, X_test_abnormal=X_test_abn,
        meta=np.array(json.dumps(meta), dtype=object))
    print(f"Saved preprocessed data to {out_path}")
    return out_path


def load_npz_for_training(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    X_train = data["X_train"]
    splits = {
        "val": {"normal": data["X_val_normal"], "abnormal": data["X_val_abnormal"]},
        "test": {"normal": data["X_test_normal"], "abnormal": data["X_test_abnormal"]}
    }
    meta = json.loads(str(data["meta"].item()))
    return X_train, splits, meta


# if __name__ == "__main__":
#     import argparse
#     p = argparse.ArgumentParser(description="Needle-EMG preprocessing for Autoencoder (Phase 1)")
#     p.add_argument("--input_dir", type=str, required=True)
#     p.add_argument("--output_dir", type=str, required=True)
#     p.add_argument("--window_sec", type=float, default=1.0)
#     p.add_argument("--hop_sec", type=float, default=0.5)
#     p.add_argument("--sr", type=int, default=11025)
#     p.add_argument("--no_bandpass", action="store_true")
#     p.add_argument("--no_notch", action="store_true")
#     p.add_argument("--augment", action="store_true")
#     p.add_argument("--aug_per_segment", type=int, default=1)
#     args = p.parse_args()

#     cfg = Config(
#         input_dir=args.input_dir,
#         output_dir=args.output_dir,
#         target_sr=args.sr,
#         window_sec=args.window_sec,
#         hop_sec=args.hop_sec,
#         apply_bandpass=not args.no_bandpass,
#         apply_notch=not args.no_notch,
#         augment=args.augment,
#         aug_per_segment=args.aug_per_segment,
#     )
#     preprocess_dataset(cfg)