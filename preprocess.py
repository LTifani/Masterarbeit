
"""
Preprocessing script for Needle-EMG WAV files to LSTM/TCN-ready .npz format.

- Input folders: Normal / Spontanaktivität / Mixed
- Patient-aware splits (overlap allowed), min unique train patients.
- Filtering (bandpass/notch), segmentation, augmentation.
- Normalization modes: 'per_segment' | 'per_file' | 'none'
- Runs standalone: Adjust config below and execute.
"""
import os
import json
import math
import random
import glob
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from dataclasses import asdict
from utils import ic, Config, cfg, timer
from typing import Tuple, List, Dict, Any
from scipy.signal import butter, filtfilt, iirnotch, resample_poly


# =============================
# Utilities: I/O, DSP
# =============================


def read_wav_file(file_path: str, target_sample_rate: int) -> np.ndarray:
    """
    Read WAV file, normalize to float [-1, 1], resample if needed.

    Args:
        file_path: Path to WAV file.
        target_sample_rate: Desired output sample rate.

    Returns:
        1D array of audio signal.
    """
    sample_rate, data = wavfile.read(file_path)

    # Normalize to float32 [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.float32)

    # Resample if necessary
    if sample_rate != target_sample_rate:
        gcd = math.gcd(sample_rate, target_sample_rate)
        up_factor = target_sample_rate // gcd
        down_factor = sample_rate // gcd
        data = resample_poly(data, up_factor, down_factor).astype(np.float32)

    return data


def apply_bandpass_filter(
    signal: np.ndarray, sample_rate: int, low_freq: float, high_freq: float, order: int
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.

    Args:
        signal: Input signal.
        sample_rate: Sample rate in Hz.
        low_freq: Low cutoff frequency.
        high_freq: High cutoff frequency.
        order: Filter order.

    Returns:
        Filtered signal.
    """
    nyquist = 0.5 * sample_rate
    low_norm = max(1.0, low_freq) / nyquist
    high_norm = min(nyquist - 1.0, high_freq) / nyquist
    if not (0 < low_norm < high_norm < 1):
        return signal

    results = butter(order, [low_norm, high_norm], btype='band')
    if results is None:
        return signal

    return filtfilt(results[0], results[1], signal)


def apply_notch_filter(
    signal: np.ndarray, sample_rate: int, notch_freq: float, quality: float
) -> np.ndarray:
    """
    Apply IIR notch filter.

    Args:
        signal: Input signal.
        sample_rate: Sample rate in Hz.
        notch_freq: Notch frequency.
        quality: Quality factor.

    Returns:
        Filtered signal.
    """
    normalized_freq = notch_freq / (sample_rate / 2.0)
    if normalized_freq <= 0 or normalized_freq >= 1:
        return signal

    b, a = iirnotch(normalized_freq, quality)
    return filtfilt(b, a, signal)


def zscore_normalize(signal: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Apply z-score normalization (mean=0, std=1).

    Args:
        signal: Input signal.
        epsilon: Small value to avoid division by zero.

    Returns:
        Normalized signal.
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std < epsilon:
        std = epsilon
    return (signal - mean) / std

zscore_normalize
def segment_audio_signal(
    signal: np.ndarray,
    sample_rate: int,
    window_duration_seconds: float,
    hop_duration_seconds: float
) -> np.ndarray:
    """
    Segment audio into overlapping windows.

    Args:
        signal: 1D input signal.
        sample_rate: Sample rate in Hz.
        window_duration_seconds: Window duration.
        hop_duration_seconds: Hop duration between windows.

    Returns:
        2D array (num_segments, segment_length).
    """
    window_size_samples = int(round(window_duration_seconds * sample_rate))
    hop_size_samples = int(round(hop_duration_seconds * sample_rate))

    if window_size_samples <= 0 or hop_size_samples <= 0 or len(signal) < window_size_samples:
        return np.empty((0, window_size_samples), dtype=np.float32)

    start_indices = np.arange(0, len(signal) - window_size_samples + 1, hop_size_samples)
    segments = np.stack([signal[start:start + window_size_samples] for start in start_indices]).astype(np.float32)

    return segments


# =============================
# Augmentation
# =============================
rng = np.random.default_rng(cfg.random_seed)


def augment_add_noise(segment: np.ndarray, std_range: Tuple[float, float]) -> np.ndarray:
    """Add Gaussian noise to segment."""
    std = rng.uniform(*std_range)
    noise = rng.normal(0.0, std, size=segment.shape).astype(np.float32)
    return segment + noise


def augment_time_shift(segment: np.ndarray, sample_rate: int, max_shift_seconds: float) -> np.ndarray:
    """Apply random time shift with padding."""
    original_length = len(segment)
    max_shift_from_time = int(round(max_shift_seconds * sample_rate))
    max_shift_samples = min(max_shift_from_time, original_length)
    if max_shift_samples <= 0:
        return segment

    shift_samples = rng.integers(-max_shift_samples, max_shift_samples + 1)
    if shift_samples == 0:
        return segment

    pad_value = 1e-6
    if shift_samples > 0:
        return np.concatenate([segment[shift_samples:], rng.normal(0, pad_value, shift_samples).astype(np.float32)])
    else:
        shift_samples = -shift_samples
        return np.concatenate([rng.normal(0, pad_value, shift_samples).astype(np.float32), segment[:-shift_samples]])


def augment_gain(segment: np.ndarray, gain_range: Tuple[float, float]) -> np.ndarray:
    """Apply random gain scaling."""
    gain = rng.uniform(*gain_range)
    return (segment * gain).astype(np.float32)


def augment_time_stretch(segment: np.ndarray, stretch_rate: float) -> np.ndarray:
    """Apply time stretching via resampling."""
    original_length = len(segment)
    new_length = max(8, int(round(original_length / stretch_rate)))
    stretched = resample_poly(segment, new_length, original_length).astype(np.float32)

    if len(stretched) > original_length:
        stretched = stretched[:original_length]
    elif len(stretched) < original_length:
        padding = np.zeros(original_length - len(stretched), dtype=np.float32)
        stretched = np.concatenate([stretched, padding])

    return stretched


def augment_mask(segment: np.ndarray, fraction_range: Tuple[float, float]) -> np.ndarray:
    """Apply random masking."""
    fraction = rng.uniform(*fraction_range)
    length = len(segment)
    mask_length = max(1, int(round(fraction * length)))
    start_index = rng.integers(0, max(1, length - mask_length + 1))
    output = segment.copy()
    output[start_index:start_index + mask_length] = 0.0
    return output


def apply_augmentations(segment: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Apply random augmentations to a segment.

    Args:
        segment: Input segment.
        sample_rate: Sample rate.

    Returns:
        Augmented segment.
    """
    augmented = segment.copy()
    if rng.random() < cfg.noise_probability:
        augmented = augment_add_noise(augmented, cfg.noise_std_range)
    if rng.random() < cfg.time_shift_probability:
        augmented = augment_time_shift(augmented, sample_rate, cfg.max_time_shift_seconds)
    if rng.random() < cfg.gain_probability:
        augmented = augment_gain(augmented, cfg.gain_range)
    if rng.random() < cfg.time_stretch_probability:
        stretch_rate = rng.uniform(*cfg.time_stretch_range)
        augmented = augment_time_stretch(augmented, stretch_rate)
    if rng.random() < cfg.mask_probability:
        augmented = augment_mask(augmented, cfg.mask_fraction_range)
    return augmented


# =============================
# File Processing
# =============================


def preprocess_single_file(file_path: str, config: Config=Config()) -> Dict[str, np.ndarray]:
    """
    Preprocess a single WAV file: filter, normalize, segment.

    Args:
        file_path: Path to WAV file.
        
    Returns:
        Dict with 'segments' key containing (N, T) array.
    """
    signal = read_wav_file(file_path, config.target_sample_rate)

    # Apply filters
    if config.apply_bandpass_filter:
        signal = apply_bandpass_filter(
            signal, config.target_sample_rate, config.bandpass_low, config.bandpass_high, config.filter_order
        )
    if config.apply_notch_filter:
        signal = apply_notch_filter(signal, config.target_sample_rate, config.notch_frequency, config.notch_quality)

    # Normalize per file if specified
    if config.normalization_mode.lower() == "per_file":
        signal = zscore_normalize(signal)

    # Segment
    segments = segment_audio_signal(
        signal, config.target_sample_rate, config.window_duration_seconds, config.hop_duration_seconds
    )
    if segments.size == 0:
        return {"segments": segments}

    # Normalize per segment if specified
    norm_mode = config.normalization_mode.lower()
    if norm_mode == "per_segment":
        segments = np.stack([zscore_normalize(seg) for seg in segments]).astype(np.float32)
    elif norm_mode in ("per_file", "none"):
        segments = segments.astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization mode: {config.normalization_mode}")

    return {"segments": segments}

# =============================
# Splitting
# =============================

def _collect_files(path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(path, "*.wav")))


def _split_indices(n: int, ratios: Tuple[float, float, float], seed: int = 42):
    train_ratio, validation_ratio, test_ratio = ratios
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    n_train = int(train_ratio * n)
    n_val = int(validation_ratio * n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


# =============================
# Main Pipeline
# =============================


def preprocess_dataset(cfg: Config) -> str:
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    normal_files = _collect_files(os.path.join(cfg.input_dir, cfg.normal_name))
    abnorm_files = _collect_files(os.path.join(cfg.input_dir, cfg.abnormal_name))

    normal_segments = [s for f in normal_files for s in [preprocess_single_file(f, cfg)] if s.size > 0]
    abnormal_segments = [s for f in abnorm_files for s in [preprocess_single_file(f, cfg)] if s.size > 0]

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
        augments = [zscore_normalize(apply_augmentations(s, cfg.target_sr, cfg)) for s in X_train for _ in range(cfg.aug_per_segment)]
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



def ensure_3d_shape(data: np.ndarray) -> np.ndarray:
    """Ensure shape (N, T, 1) for channel dimension."""
    return data[..., None].astype(np.float32)


def load_npz_for_training(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    X_train = data["X_train"]
    splits = {
        "val": {"normal": data["X_val_normal"], "abnormal": data["X_val_abnormal"]},
        "test": {"normal": data["X_test_normal"], "abnormal": data["X_test_abnormal"]}
    }
    meta = json.loads(str(data["meta"].item()))
    return X_train, splits, meta


if __name__ == "__main__":
    pass

    
    # normal_segments, normal_patient_ids = [], []
    # all_files = collect_all_files(cfg.input_dir)
    # process_file_group(
    #     all_files.get(cfg.normal_folder, []),
    #     cfg.normal_folder,
    #     normal_segments,
    #     normal_patient_ids
    # )

    # ic(len(normal_segments))
    # ic(type(normal_segments[0]))
    # foo = dict(zip(normal_patient_ids, [i.shape for i in normal_segments]))
    # ic(foo)
    
    
    # Example processing (for testing)
    # example_path = "Dataset/Spontanaktivität/BA0803901.wav"
    # example_results = preprocess_single_file(example_path, "Spontanaktivität")
    # ic(example_results)

