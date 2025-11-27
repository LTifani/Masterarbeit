"""
EMG Anomaly Detection Package

Temporal Convolutional Network-based anomaly detection for EMG signals.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .config import ExperimentConfig, create_default_config
from .models import TCNAutoencoder
from .datasets import EMGSegmentDataset, create_data_loaders
from .train import Trainer, train_model
from .augmentation import AudioAugmenter
from .utils import normalize_segment, setup_logging

__all__ = [
    'ExperimentConfig',
    'create_default_config',
    'TCNAutoencoder',
    'EMGSegmentDataset',
    'create_data_loaders',
    'Trainer',
    'train_model',
    'AudioAugmenter',
    'normalize_segment',
    'setup_logging',
]