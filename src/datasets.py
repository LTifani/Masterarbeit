"""
Dataset Implementation for EMG Signal Processing

Provides PyTorch Dataset classes for loading and preprocessing EMG signals
with support for concatenating multiple directories and reconstruction tasks.
"""

import os
import glob
import torch
import random
import logging
import numpy as np
from scipy.io import wavfile
from typing import List, Tuple, Optional
from scipy.signal import butter, filtfilt, iirnotch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from .config import ExperimentConfig

from .augmentation import AudioAugmenter
from .utils import (
    load_anomaly_annotations,
    generate_audio_segments,
    compute_segment_labels,
    normalize_segment
)


logger = logging.getLogger(__name__)


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


class EMGSegmentDataset(Dataset):
    """
    PyTorch Dataset for EMG audio segments with reconstruction target.
    
    This dataset loads EMG signals, segments them with overlap, and prepares
    them for reconstruction-based anomaly detection. Normal segments are used
    for training the reconstruction model.
    """
    
    def __init__(self,
                 wav_dir: str,
                 anomaly_csv_dir: Optional[str] = None,
                 window_ms: float = 10.0,
                 hop_percentage: float = 0.5,
                 split: str = "train",
                 split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 random_seed: int = 42,
                 min_overlap_threshold: float = 0.2,
                 shuffle_segments: bool = True,
                 enable_augmentation: bool = False,
                 augmentation_config: Optional[dict] = None,
                 apply_normalization: bool = True,
                 return_labels: bool = True,
                 only_normal: bool = False,
                 data_config: Optional[dict] = None,
                 ):
        """
        Initialize EMG segment dataset.
        
        Args:
            wav_dir: Directory containing WAV files
            anomaly_csv_dir: Directory containing anomaly CSV annotations (optional)
            window_ms: Segment window size in milliseconds
            hop_percentage: Overlap percentage between segments (0.0-1.0)
            split: Dataset split ('train', 'val', or 'test')
            split_ratios: Ratios for train/val/test split
            random_seed: Random seed for reproducibility
            min_overlap_threshold: Minimum overlap to label segment as anomaly
            shuffle_segments: Whether to shuffle segments after creation
            enable_augmentation: Enable data augmentation
            augmentation_config: Configuration for augmentation parameters
            apply_normalization: Apply zero-mean, unit-variance normalization
            return_labels: Whether to return labels (False for unlabeled normal data)
            only_normal: If True, filter out anomaly segments (for training)
        """
        super().__init__()
        
        self.wav_dir = wav_dir
        self.anomaly_csv_dir = anomaly_csv_dir
        self.window_ms = window_ms
        self.hop_percentage = hop_percentage
        self.min_overlap_threshold = min_overlap_threshold
        self.split = split
        self.split_ratios = split_ratios
        self.random_seed = random_seed
        self.shuffle_segments = shuffle_segments
        self.enable_augmentation = enable_augmentation and (split == "train")
        self.augmentation_config = augmentation_config or {}
        self.apply_normalization = apply_normalization
        self.return_labels = return_labels
        self.only_normal = only_normal
        self.data_config = data_config or {}
        
        # Initialize augmenter if needed
        self.augmenter = None
        if self.enable_augmentation:
            self.augmenter = AudioAugmenter(random_seed=random_seed)
            logger.info(f"Data augmentation enabled for {split} split")
        
        # Load dataset
        self.segments = self._load_dataset()
        
        if self.return_labels:
            num_anomaly_segments = sum(label.item() for _, _, label in self.segments)
            logger.info(
                f"{split.upper()} dataset from {wav_dir}: {len(self.segments)} segments, "
                f"{num_anomaly_segments} anomalies "
                f"({num_anomaly_segments/max(len(self.segments), 1)*100:.1f}%)"
            )
        else:
            logger.info(f"{split.upper()} dataset from {wav_dir}: {len(self.segments)} segments (unlabeled)")
    
    def _split_files(self) -> List[str]:
        """Split WAV files into train/val/test sets deterministically."""
        wav_files = sorted(glob.glob(os.path.join(self.wav_dir, "*.wav")))
        
        if len(wav_files) == 0:
            logger.warning(f"No WAV files found in {self.wav_dir}")
            return []
        
        # Deterministic shuffle
        rng = random.Random(self.random_seed)
        rng.shuffle(wav_files)
        
        # Calculate split indices
        num_total = len(wav_files)
        num_train = int(num_total * self.split_ratios[0])
        num_val = num_train + int(num_total * self.split_ratios[1])
        
        # Select files for current split
        if self.split == "train":
            selected_files = wav_files[:num_train]
        elif self.split == "val":
            selected_files = wav_files[num_train:num_val]
        else:  # test
            selected_files = wav_files[num_val:]
        
        logger.debug(f"{self.split.upper()}: {len(selected_files)}/{num_total} files from {self.wav_dir}")
        return selected_files
    
    def _process_audio_file(self, wav_path: str) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Process single audio file into labeled segments."""
        filename = os.path.splitext(os.path.basename(wav_path))[0]
        csv_path = None
        if self.anomaly_csv_dir:
            csv_path = os.path.join(self.anomaly_csv_dir, filename + ".csv")
        
        try:
            # Load audio
            sample_rate, audio = self._load_audio_file(wav_path)
            total_duration = len(audio) / sample_rate
            
            # Apply filters 
            if self.data_config.get("apply_bandpass_filter", False):
                audio = apply_bandpass_filter(
                    audio, sample_rate, self.data_config.get("bandpass_low", 20.0), 
                    self.data_config.get("bandpass_high", 4500.0),
                    self.data_config.get("filter_order", 4)
                )
            if self.data_config.get("apply_notch_filter", False):
                audio = apply_notch_filter(audio, sample_rate, self.data_config.get("notch_frequency", 50),
                                          self.data_config.get("notch_quality", 4))

            
            # Generate segments
            window_sec = self.window_ms / 1000.0
            segment_times = generate_audio_segments(
                total_duration, window_sec, self.hop_percentage
            )
            
            # Load anomalies and compute labels if annotations available
            if self.return_labels and csv_path and os.path.exists(csv_path):
                anomaly_ranges = load_anomaly_annotations(csv_path)
                labels = compute_segment_labels(
                    segment_times, anomaly_ranges, self.min_overlap_threshold
                )
            else:
                # No labels available - all segments are normal
                labels = [0] * len(segment_times)
            
            # Extract segments
            target_length = int(window_sec * sample_rate)
            segments = []
            
            for (seg_start, seg_end), label in zip(segment_times, labels):
                # Skip anomaly segments if only_normal is True
                if self.only_normal and label == 1:
                    continue
                
                segment = self._extract_segment(
                    audio, seg_start, seg_end, sample_rate, target_length
                )
                
                # Apply normalization if enabled
                if self.apply_normalization:
                    segment = normalize_segment(segment)
                
                segment_tensor = torch.tensor(segment, dtype=torch.float32)
                label_tensor = torch.tensor(label, dtype=torch.long)
                
                # Store sample rate for augmentation
                segments.append((segment_tensor, segment_tensor.clone(), label_tensor))
            
            return segments
            
        except Exception as e:
            logger.error(f"Error processing {wav_path}: {e}", exc_info=True)
            return []
    
    def _load_audio_file(self, wav_path: str) -> Tuple[int, np.ndarray]:
        """Load audio file and convert to mono float32."""
        sample_rate, audio = wavfile.read(wav_path)
        
        # Convert multi-channel to mono by averaging
        if audio.ndim > 1:
            audio = audio.astype(np.float32).mean(axis=1)
        else:
            audio = audio.astype(np.float32)
        
        return sample_rate, audio
    
    def _extract_segment(self, audio: np.ndarray, start_time: float, end_time: float,
                        sample_rate: int, target_length: int) -> np.ndarray:
        """Extract and pad audio segment to target length."""
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        segment = audio[start_sample:end_sample]
        
        # Pad if segment is shorter than target
        if len(segment) < target_length:
            padding_length = target_length - len(segment)
            segment = np.pad(segment, (0, padding_length), mode='constant')
        
        return segment
    
    def _load_dataset(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Load all segments from split files."""
        files = self._split_files()
        all_segments = []
        
        for wav_path in files:
            segments = self._process_audio_file(wav_path)
            all_segments.extend(segments)
        
        # Store sample rate from first file for augmentation
        if all_segments and self.enable_augmentation and files:
            # Get sample rate from first file
            try:
                sample_rate, _ = wavfile.read(files[0])
                self.sample_rate = sample_rate
            except:
                self.sample_rate = 44100  # Default fallback
        
        # Optional shuffle
        if self.shuffle_segments and all_segments:
            rng = random.Random(self.random_seed)
            rng.shuffle(all_segments)
        
        return all_segments
    
    def __len__(self) -> int:
        """Return number of segments in dataset."""
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get segment and label at index.
        
        For reconstruction tasks, returns (input_segment, target_segment, label) where
        target is the original segment and input may be augmented.
        """
        input_segment, target_segment, label = self.segments[idx]
        
        # Apply augmentation if enabled (only to input, not target)
        if self.enable_augmentation and self.augmenter:
            segment_np = input_segment.numpy()
            segment_np = self.augmenter.augment(
                segment_np, self.sample_rate, self.augmentation_config
            )
            # Re-normalize after augmentation
            if self.apply_normalization:
                segment_np = normalize_segment(segment_np)
            input_segment = torch.from_numpy(segment_np)
        
        return input_segment, target_segment, label


def create_concatenated_dataset(data_dirs: List[str],
                                anomaly_csv_dir: Optional[str] = None,
                                split: str = "train",
                                only_normal: bool = False,
                                **kwargs) -> Dataset:
    """
    Create concatenated dataset from multiple directories.
    
    This is useful when normal and anomaly data are stored separately.
    
    Args:
        data_dirs: List of directories containing WAV files
        anomaly_csv_dir: Directory containing anomaly annotations
        split: Dataset split ('train', 'val', 'test')
        only_normal: If True, filter out anomaly segments
        **kwargs: Additional arguments passed to EMGSegmentDataset
        
    Returns:
        ConcatDataset combining all specified directories
    """
    datasets = []
    
    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            logger.warning(f"Directory not found: {data_dir}")
            continue
        
        dataset = EMGSegmentDataset(
            wav_dir=data_dir,
            anomaly_csv_dir=anomaly_csv_dir,
            split=split,
            only_normal=only_normal,
            **kwargs
        )
        
        if len(dataset) > 0:
            datasets.append(dataset)
    
    if len(datasets) == 0:
        raise ValueError(f"No valid datasets found in directories: {data_dirs}")
    
    concatenated = ConcatDataset(datasets)
    logger.info(f"Created concatenated {split} dataset with {len(concatenated)} total segments")
    
    return concatenated


def emg_collate_fn(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    """
    Custom collate function for EMG segments.
    
    Handles variable-length sequences and reconstruction tasks.
    """
    # Unpack batch: (input, target, label)
    inputs, targets, labels = zip(*batch)
    
    # Find maximum length
    max_len = max(inp.shape[0] for inp in inputs)
    
    # Pad sequences
    padded_inputs = torch.stack([
        torch.nn.functional.pad(inp, (0, max_len - len(inp)))
        for inp in inputs
    ])
    padded_targets = torch.stack([
        torch.nn.functional.pad(tgt, (0, max_len - len(tgt)))
        for tgt in targets
    ])
    
    return padded_inputs, padded_targets, torch.stack(labels)


def create_data_loaders(config: ExperimentConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders from configuration.
    
    Args:
        config: ExperimentConfig instance
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Training dataset: normal data only for reconstruction learning
    train_dataset = create_concatenated_dataset(
        data_dirs=config.data.normal_data_dirs,
        anomaly_csv_dir=config.data.anomaly_csv_dir,
        split="train",
        window_ms=config.data.window_ms,
        hop_percentage=config.data.hop_percentage,
        split_ratios=(config.data.train_ratio, config.data.val_ratio, config.data.test_ratio),
        random_seed=config.random_seed,
        min_overlap_threshold=config.data.min_overlap_threshold,
        shuffle_segments=True,
        enable_augmentation=config.augmentation.enable,
        augmentation_config=config.augmentation.to_dict(),
        data_config=vars(config.data),
        apply_normalization=config.data.apply_normalization,
        only_normal=True,  # Only normal segments for training
        return_labels=True
    )
    
    # Validation dataset: both normal and anomaly data for evaluation
    val_datasets = []
    
    if config.data.normal_data_dirs:
        normal_val = create_concatenated_dataset(
            data_dirs=config.data.normal_data_dirs,
            anomaly_csv_dir=config.data.anomaly_csv_dir,
            split="val",
            window_ms=config.data.window_ms,
            hop_percentage=config.data.hop_percentage,
            split_ratios=(config.data.train_ratio, config.data.val_ratio, config.data.test_ratio),
            random_seed=config.random_seed,
            min_overlap_threshold=config.data.min_overlap_threshold,
            shuffle_segments=False,
            enable_augmentation=False,
            apply_normalization=config.data.apply_normalization,
            only_normal=False,
            data_config=vars(config.data),
            return_labels=True
        )
        val_datasets.append(normal_val)
        
    # Anomaly val data
    if config.data.anomaly_data_dirs:
        anomaly_val = create_concatenated_dataset(
            data_dirs=config.data.anomaly_data_dirs,
            anomaly_csv_dir=config.data.anomaly_csv_dir,
            split="val",
            window_ms=config.data.window_ms,
            hop_percentage=config.data.hop_percentage,
            split_ratios=(config.data.a_train_ratio, config.data.a_val_ratio, config.data.a_test_ratio),
            random_seed=config.random_seed,
            min_overlap_threshold=config.data.min_overlap_threshold,
            shuffle_segments=False,
            enable_augmentation=False,
            apply_normalization=config.data.apply_normalization,
            only_normal=False,
            data_config=vars(config.data),
            return_labels=True
        )
        val_datasets.append(anomaly_val)
    
    # Combine val datasets
    val_dataset = ConcatDataset(val_datasets) if val_datasets else val_datasets[0]
    
         
    # Test dataset: both normal and anomaly data for evaluation
    test_datasets = []
    
    # Normal test data
    if config.data.normal_data_dirs:
        normal_test = create_concatenated_dataset(
            data_dirs=config.data.normal_data_dirs,
            anomaly_csv_dir=config.data.anomaly_csv_dir,
            split="test",
            window_ms=config.data.window_ms,
            hop_percentage=config.data.hop_percentage,
            split_ratios=(config.data.train_ratio, config.data.val_ratio, config.data.test_ratio),
            random_seed=config.random_seed,
            min_overlap_threshold=config.data.min_overlap_threshold,
            shuffle_segments=False,
            enable_augmentation=False,
            apply_normalization=config.data.apply_normalization,
            only_normal=False,
            data_config=vars(config.data),
            return_labels=True
        )
        test_datasets.append(normal_test)
    
    # Anomaly test data
    if config.data.anomaly_data_dirs:
        anomaly_test = create_concatenated_dataset(
            data_dirs=config.data.anomaly_data_dirs,
            anomaly_csv_dir=config.data.anomaly_csv_dir,
            split="test",
            window_ms=config.data.window_ms,
            hop_percentage=config.data.hop_percentage,
            split_ratios=(config.data.a_train_ratio, config.data.a_val_ratio, config.data.a_test_ratio),
            random_seed=config.random_seed,
            min_overlap_threshold=config.data.min_overlap_threshold,
            shuffle_segments=False,
            enable_augmentation=False,
            apply_normalization=config.data.apply_normalization,
            only_normal=False,
            data_config= {},
            return_labels=True
        )
        test_datasets.append(anomaly_test)
    
    # Combine test datasets
    test_dataset = ConcatDataset(test_datasets) if test_datasets else test_datasets[0]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=emg_collate_fn,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=emg_collate_fn,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=emg_collate_fn,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    logger.info(f"Data loaders created - Train: {len(train_loader.dataset)}, "
                f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    return train_loader, val_loader, test_loader