"""
Data Augmentation for EMG Signals

Provides various augmentation techniques to improve model robustness.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.signal import resample_poly


class AudioAugmenter:
    """Handles various audio data augmentation techniques."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize augmenter with random seed for reproducibility.
        
        Args:
            random_seed: Seed for random number generator
        """
        self.rng = np.random.default_rng(random_seed)
    
    def add_noise(self, segment: np.ndarray, 
                  std_range: Tuple[float, float] = (0.001, 0.01)) -> np.ndarray:
        """
        Add Gaussian noise to audio segment.
        
        Args:
            segment: Input audio segment
            std_range: Range for noise standard deviation (min, max)
            
        Returns:
            Augmented audio segment with added noise
        """
        std = self.rng.uniform(*std_range)
        noise = self.rng.normal(0.0, std, size=segment.shape).astype(np.float32)
        return segment + noise
    
    def time_shift(self, segment: np.ndarray, sample_rate: int, 
                   max_shift_seconds: float = 0.001) -> np.ndarray:
        """
        Apply random time shift with padding.
        
        Args:
            segment: Input audio segment
            sample_rate: Audio sample rate in Hz
            max_shift_seconds: Maximum time shift in seconds
            
        Returns:
            Time-shifted audio segment
        """
        original_length = len(segment)
        max_shift_from_time = int(round(max_shift_seconds * sample_rate))
        max_shift_samples = min(max_shift_from_time, original_length)
        
        if max_shift_samples <= 0:
            return segment
        
        shift_samples = self.rng.integers(-max_shift_samples, max_shift_samples + 1)
        if shift_samples == 0:
            return segment
        
        pad_value = 1e-6
        if shift_samples > 0:
            # Shift right: remove from start, pad at end
            padding = self.rng.normal(0, pad_value, shift_samples).astype(np.float32)
            return np.concatenate([segment[shift_samples:], padding])
        else:
            # Shift left: pad at start, remove from end
            shift_samples = -shift_samples
            padding = self.rng.normal(0, pad_value, shift_samples).astype(np.float32)
            return np.concatenate([padding, segment[:-shift_samples]])
    
    def apply_gain(self, segment: np.ndarray, 
                   gain_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Apply random gain scaling to audio segment.
        
        Args:
            segment: Input audio segment
            gain_range: Range for gain factor (min, max)
            
        Returns:
            Gain-adjusted audio segment
        """
        gain = self.rng.uniform(*gain_range)
        return (segment * gain).astype(np.float32)
    
    def time_stretch(self, segment: np.ndarray, stretch_rate: float) -> np.ndarray:
        """
        Apply time stretching via resampling.
        
        Args:
            segment: Input audio segment
            stretch_rate: Stretch factor (>1 speeds up, <1 slows down)
            
        Returns:
            Time-stretched audio segment with original length
        """
        original_length = len(segment)
        new_length = max(8, int(round(original_length / stretch_rate)))
        stretched = resample_poly(segment, new_length, original_length).astype(np.float32)
        
        # Ensure output has original length
        if len(stretched) > original_length:
            stretched = stretched[:original_length]
        elif len(stretched) < original_length:
            padding = np.zeros(original_length - len(stretched), dtype=np.float32)
            stretched = np.concatenate([stretched, padding])
        
        return stretched
    
    def apply_mask(self, segment: np.ndarray, 
                   fraction_range: Tuple[float, float] = (0.05, 0.15)) -> np.ndarray:
        """
        Apply random masking to audio segment.
        
        Args:
            segment: Input audio segment
            fraction_range: Range for mask length as fraction of segment (min, max)
            
        Returns:
            Masked audio segment
        """
        fraction = self.rng.uniform(*fraction_range)
        length = len(segment)
        mask_length = max(1, int(round(fraction * length)))
        start_index = self.rng.integers(0, max(1, length - mask_length + 1))
        
        output = segment.copy()
        output[start_index:start_index + mask_length] = 0.0
        return output
    
    def augment(self, segment: np.ndarray, sample_rate: int,
                augmentation_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Apply random augmentations to audio segment.
        
        Args:
            segment: Input audio segment
            sample_rate: Audio sample rate in Hz
            augmentation_config: Configuration dictionary for augmentation parameters
            
        Returns:
            Augmented audio segment
        """
        if augmentation_config is None:
            augmentation_config = {}
        
        augmented_segment = segment.copy()
        
        # Apply noise augmentation
        if self.rng.random() < augmentation_config.get('noise_prob', 0.3):
            augmented_segment = self.add_noise(
                augmented_segment, 
                augmentation_config.get('noise_std_range', (0.001, 0.01))
            )
        
        # Apply time shift augmentation
        if self.rng.random() < augmentation_config.get('time_shift_prob', 0.3):
            augmented_segment = self.time_shift(
                augmented_segment, 
                sample_rate,
                augmentation_config.get('max_shift_seconds', 0.001)
            )
        
        # Apply gain augmentation
        if self.rng.random() < augmentation_config.get('gain_prob', 0.3):
            augmented_segment = self.apply_gain(
                augmented_segment,
                augmentation_config.get('gain_range', (0.8, 1.2))
            )
        
        # Apply time stretch augmentation
        if self.rng.random() < augmentation_config.get('stretch_prob', 0.2):
            stretch_rate = self.rng.uniform(
                *augmentation_config.get('stretch_range', (0.9, 1.1))
            )
            augmented_segment = self.time_stretch(augmented_segment, stretch_rate)
        
        # Apply masking augmentation
        if self.rng.random() < augmentation_config.get('mask_prob', 0.2):
            augmented_segment = self.apply_mask(
                augmented_segment,
                augmentation_config.get('mask_fraction_range', (0.05, 0.15))
            )
        
        return augmented_segment