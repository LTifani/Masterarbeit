
"""
Preprocessing script for Needle-EMG WAV files to LSTM/TCN-ready .npz format.

- Input folders: Normal / Spontanaktivität / Mixed
- Patient-aware splits (overlap allowed), min unique train patients.
- Filtering (bandpass/notch), segmentation, augmentation.
- Normalization modes: 'per_segment' | 'per_file' | 'none'
- Runs standalone: Adjust config below and execute.
"""

import json
import math
import random
from pathlib import Path
from dataclasses import asdict
from typing import Tuple, List, Dict, Any

import numpy as np
from icecream import ic
from utils import *
from scipy.io import wavfile
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

    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, signal)


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


def extract_patient_id(file_path: str, folder_type: str) -> str:
    """
    Extract patient ID from filename based on folder type.

    Args:
        file_path: Path to file.
        folder_type: Type of folder ('Normal', 'Spontanaktivität', or 'Mixed').

    Returns:
        Patient ID string.
    """
    filename = Path(file_path).stem
    if folder_type in ("Normal", "Spontanaktivität"):
        import re
        match = re.match(r"^([A-Za-zÄÖÜäöü]+)", filename)
        if match:
            return match.group(1)
        return filename[:1]

    # For Mixed
    import re
    match = re.match(r"^([A-Za-zÄÖÜäöü]+)", filename)
    if match:
        return match.group(1)
    parts = filename.split("_")
    if len(parts) >= 3:
        return parts[-2]
    return parts[0]


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


def preprocess_single_file(file_path: str, folder_type: str, config: Config=Config()) -> Dict[str, np.ndarray]:
    """
    Preprocess a single WAV file: filter, normalize, segment.

    Args:
        file_path: Path to WAV file.
        folder_type: Folder type for patient ID extraction.

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


def collect_all_files(root_dir: str, config: Config=Config()) -> Dict[str, List[str]]:
    """
    Collect WAV files from subfolders.

    Args:
        root_dir: Root directory.

    Returns:
        Dict with keys for normal/abnormal/mixed and lists of file paths.
    """
    subfolders = [config.normal_folder, config.abnormal_folder, config.mixed_folder]
    file_dict = {key: [] for key in subfolders}
    root_path = Path(root_dir)

    for subfolder in subfolders:
        subdir_path = root_path / subfolder
        if not subdir_path.exists():
            continue
        for file_path in subdir_path.rglob("*.wav"):
            file_dict[subfolder].append(str(file_path))

    return file_dict


# =============================
# Splitting
# =============================


def split_segments_by_ratio(
    segments: List[np.ndarray],
    patient_ids: List[str],
    train_ratio: float,
    validation_ratio: float,
    min_unique_train_patients: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split segments into train/val/test ensuring min unique patients in train.

    Args:
        segments: List of segment arrays.
        patient_ids: Corresponding patient IDs.
        train_ratio: Train split ratio.
        validation_ratio: Validation split ratio.
        min_unique_train_patients: Min unique patients in train.
        rng: Random generator for shuffling.

    Returns:
        Tuple of (X_train, X_val, X_test) as stacked arrays.
    """
    num_segments = len(segments)
    indices = np.arange(num_segments)
    rng.shuffle(indices)

    num_train = int(round(train_ratio * num_segments))
    num_validation = int(round(validation_ratio * num_segments))

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_validation]
    test_indices = indices[num_train + num_validation:]

    # Ensure min unique patients in train
    train_patient_ids = np.array(patient_ids)[train_indices]
    unique_train_patients = set(train_patient_ids)

    if len(unique_train_patients) < min_unique_train_patients:
        all_unique_patients = set(patient_ids)
        missing_patients = list(all_unique_patients - unique_train_patients)
        rng.shuffle(missing_patients)

        for patient in missing_patients:
            candidate_indices = np.flatnonzero(np.array(patient_ids) == patient)
            if len(candidate_indices) == 0:
                continue
            selected_index = rng.choice(candidate_indices)
            if selected_index not in train_indices:
                train_indices = np.concatenate((train_indices, [selected_index]))
                unique_train_patients.add(patient)
                if len(unique_train_patients) >= min_unique_train_patients:
                    break

    def stack_segments(indices_array: np.ndarray) -> np.ndarray:
        if len(indices_array) == 0:
            return np.empty((0, *segments[0].shape), dtype=np.float32)
        return np.stack([segments[i] for i in indices_array])

    X_train = stack_segments(train_indices)
    X_val = stack_segments(val_indices)
    X_test = stack_segments(test_indices)

    return X_train, X_val, X_test


# =============================
# Main Pipeline
# =============================


def process_file_group(
    file_list: List[str],
    folder_type: str,
    segments_accumulator: List[np.ndarray],
    patient_ids_accumulator: List[str]
) -> None:
    """
    Process all files in a group, extract segments and patient IDs.

    Args:
        file_list: List of file paths.
        folder_type: Folder type.
        segments_accumulator: List to append segments.
        patient_ids_accumulator: List to append patient IDs.
    """
    for file_path in file_list:
        patient_id = extract_patient_id(file_path, folder_type)
        processed = preprocess_single_file(file_path, folder_type)
        file_segments = processed["segments"]
        
        if file_segments.size == 0:
            continue
        
        segments_accumulator.extend(list(file_segments.astype(np.float32)))
        patient_ids_accumulator.extend([patient_id] * len(file_segments))
        


def ensure_3d_shape(data: np.ndarray) -> np.ndarray:
    """Ensure shape (N, T, 1) for channel dimension."""
    return data[..., None].astype(np.float32)

@timer
def run_preprocessing(config: Config = Config()) -> None:
    """Execute the full preprocessing pipeline."""
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    all_files = collect_all_files(config.input_dir, config)

    # Process each group
    normal_segments, normal_patient_ids = [], []
    abnormal_segments, abnormal_patient_ids = [], []
    mixed_segments, mixed_patient_ids = [], []

    process_file_group(
        all_files.get(config.normal_folder, []),
        config.normal_folder,
        normal_segments,
        normal_patient_ids
    )
    process_file_group(
        all_files.get(config.abnormal_folder, []),
        config.abnormal_folder,
        abnormal_segments,
        abnormal_patient_ids
    )
    process_file_group(
        all_files.get(config.mixed_folder, []),
        config.mixed_folder,
        mixed_segments,
        mixed_patient_ids
    )


    # Split each group
    def split_group(segments_list: List[np.ndarray], patient_ids_list: List[str]) -> Tuple[np.ndarray, ...]:
        if not segments_list:
            empty_shape = (0, int(config.window_duration_seconds * config.target_sample_rate))
            return (np.empty(empty_shape, dtype=np.float32),) * 3
        return split_segments_by_ratio(
            segments_list, patient_ids_list, config.train_ratio, config.validation_ratio,
            config.min_unique_train_patients, rng
        )

    X_train_normal, X_val_normal, X_test_normal = split_group(normal_segments, normal_patient_ids)
    X_train_abnormal, X_val_abnormal, X_test_abnormal = split_group(abnormal_segments, abnormal_patient_ids)
    X_train_mixed, X_val_mixed, X_test_mixed = split_group(mixed_segments, mixed_patient_ids)

    # Combine for training: normal + optional mixed
    X_train = X_train_normal
    if config.use_mixed_in_training and len(X_train_mixed):
        X_train = np.concatenate([X_train, X_train_mixed], axis=0)

    # Apply augmentation to training set
    if config.apply_augmentation and len(X_train):
        ic("Applying augmentation...")
        augmented_segments = []
        segment_length = X_train.shape[1]
        for segment in X_train:
            for _ in range(config.augmentations_per_segment):
                augmented = apply_augmentations(segment, config.target_sample_rate)
                if config.normalization_mode.lower() == "per_segment":
                    augmented = zscore_normalize(augmented)
                augmented_segments.append(augmented.reshape(1, segment_length))
        if augmented_segments:
            X_augmented = np.concatenate(augmented_segments, axis=0).astype(np.float32)
            X_train = np.concatenate([X_train, X_augmented], axis=0)

        # Shuffle training set
        indices = np.arange(len(X_train))
        rng.shuffle(indices)
        X_train = X_train[indices]

    # Prepare val/test sets
    X_val_normal = X_val_normal
    X_test_normal = X_test_normal
    X_val_abnormal = X_val_abnormal
    X_test_abnormal = X_test_abnormal

    if config.include_mixed_in_test and (len(X_val_mixed) or len(X_test_mixed)):
        X_val_mixed = X_val_mixed
        X_test_mixed = X_test_mixed
    else:
        empty_shape = (0, X_train.shape[1])
        X_val_mixed = np.empty(empty_shape, dtype=np.float32)
        X_test_mixed = np.empty(empty_shape, dtype=np.float32)

    # Merged abnormal + mixed for evaluation
    X_val_abnormal_plus_mixed = (
        X_val_abnormal if len(X_val_mixed) == 0
        else np.concatenate([X_val_abnormal, X_val_mixed], axis=0)
    )
    X_test_abnormal_plus_mixed = (
        X_test_abnormal if len(X_test_mixed) == 0
        else np.concatenate([X_test_abnormal, X_test_mixed], axis=0)
    )

    # Ensure 3D shape (N, T, 1)
    X_train = ensure_3d_shape(X_train)
    X_val_normal = ensure_3d_shape(X_val_normal)
    X_test_normal = ensure_3d_shape(X_test_normal)
    X_val_abnormal = ensure_3d_shape(X_val_abnormal)
    X_test_abnormal = ensure_3d_shape(X_test_abnormal)
    X_val_mixed = ensure_3d_shape(X_val_mixed)
    X_test_mixed = ensure_3d_shape(X_test_mixed)
    X_val_abnormal_plus_mixed = ensure_3d_shape(X_val_abnormal_plus_mixed)
    X_test_abnormal_plus_mixed = ensure_3d_shape(X_test_abnormal_plus_mixed)

    # Metadata
    metadata = {
        "config": asdict(config),
        "counts": {
            "train": int(len(X_train)),
            "val_normal": int(len(X_val_normal)),
            "val_abnormal": int(len(X_val_abnormal)),
            "val_mixed": int(len(X_val_mixed)),
            "test_normal": int(len(X_test_normal)),
            "test_abnormal": int(len(X_test_abnormal)),
            "test_mixed": int(len(X_test_mixed)),
        },
        "normalization": config.normalization_mode,
    }

    # Save compressed NPZ
    output_path = Path(config.output_dir) / f"{config.output_name}.npz"
    output_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        output_path,
        X_train=X_train,
        X_val_normal=X_val_normal,
        X_val_abnormal=X_val_abnormal,
        X_val_mixed=X_val_mixed,
        X_test_normal=X_test_normal,
        X_test_abnormal=X_test_abnormal,
        X_test_mixed=X_test_mixed,
        X_val_abn_plus_mixed=X_val_abnormal_plus_mixed,
        X_test_abn_plus_mixed=X_test_abnormal_plus_mixed,
        meta=np.array(json.dumps(metadata), dtype=object),
    )
    ic(f"Saved: {output_path}")


def load_npz_for_training(npz_path: str) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    """
    Load preprocessed NPZ for training.

    Args:
        npz_path: Path to .npz file.

    Returns:
        Tuple of (X_train, splits, meta).
        splits: Dict with 'val'/'test' keys containing normal/abnormal/mixed/abn_plus_mixed.
    """
    data = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(data["meta"].item()))

    splits = {
        "val": {
            "normal": data["X_val_normal"],
            "abnormal": data["X_val_abnormal"],
            "mixed": data["X_val_mixed"],
            "abn_plus_mixed": data["X_val_abn_plus_mixed"],
        },
        "test": {
            "normal": data["X_test_normal"],
            "abnormal": data["X_test_abnormal"],
            "mixed": data["X_test_mixed"],
            "abn_plus_mixed": data["X_test_abn_plus_mixed"],
        }
    }
    return data["X_train"], splits, meta


if __name__ == "__main__":
    pass
    run_preprocessing()
    
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

