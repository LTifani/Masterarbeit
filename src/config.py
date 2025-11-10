"""
Configuration Management for EMG Anomaly Detection

This module provides centralized configuration management with
validation, saving, and loading capabilities.
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data-related configuration parameters."""
    
    # Data paths
    normal_data_dirs: List[str] = field(default_factory=lambda: ["./Dataset/Normal"])
    anomaly_data_dirs: List[str] = field(default_factory=lambda: ["./TT/Spontanaktivitäten"])
    anomaly_csv_dir: str = "./TT/Label_Spontanaktivität"
    
    # Segmentation parameters
    window_ms: float = 10.0
    hop_percentage: float = 0.5
    min_overlap_threshold: float = 0.2
    
    # Data split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Preprocessing
    apply_normalization: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.window_ms > 0, "window_ms must be positive"
        assert 0.0 <= self.hop_percentage < 1.0, "hop_percentage must be in [0, 1)"
        assert 0.0 <= self.min_overlap_threshold <= 1.0, "min_overlap must be in [0, 1]"
        
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total_ratio - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {total_ratio}"


@dataclass
class AugmentationConfig:
    """Data augmentation configuration parameters."""
    
    enable: bool = True
    
    # Noise augmentation
    noise_prob: float = 0.3
    noise_std_range: tuple = (0.001, 0.01)
    
    # Time shift augmentation
    time_shift_prob: float = 0.3
    max_shift_seconds: float = 0.001
    
    # Gain augmentation
    gain_prob: float = 0.3
    gain_range: tuple = (0.8, 1.2)
    
    # Time stretch augmentation
    stretch_prob: float = 0.2
    stretch_range: tuple = (0.9, 1.1)
    
    # Masking augmentation
    mask_prob: float = 0.2
    mask_fraction_range: tuple = (0.05, 0.15)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for dataset usage."""
        return {
            'noise_prob': self.noise_prob,
            'noise_std_range': self.noise_std_range,
            'time_shift_prob': self.time_shift_prob,
            'max_shift_seconds': self.max_shift_seconds,
            'gain_prob': self.gain_prob,
            'gain_range': self.gain_range,
            'stretch_prob': self.stretch_prob,
            'stretch_range': self.stretch_range,
            'mask_prob': self.mask_prob,
            'mask_fraction_range': self.mask_fraction_range
        }


@dataclass
class ModelConfig:
    """Model architecture configuration parameters."""
    
    # Architecture
    input_channels: int = 1
    num_filters: int = 64
    kernel_size: int = 7
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    dropout_rate: float = 0.2
    latent_dim: int = 128
    
    # Reconstruction-specific
    reconstruction_type: str = "autoencoder"  # "autoencoder" or "predictive"
    
    def __post_init__(self):
        """Validate model configuration."""
        assert self.num_filters > 0, "num_filters must be positive"
        assert self.kernel_size > 0 and self.kernel_size % 2 == 1, "kernel_size must be odd and positive"
        assert 0.0 <= self.dropout_rate < 1.0, "dropout_rate must be in [0, 1)"
        assert len(self.dilations) > 0, "dilations list cannot be empty"
        assert all(d > 0 for d in self.dilations), "all dilations must be positive"


@dataclass
class TrainingConfig:
    """Training process configuration parameters."""
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    
    # Optimization
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    weight_decay: float = 1e-5
    scheduler: Optional[str] = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", None
    
    # Early stopping
    early_stopping_patience: int = 15
    min_delta: float = 1e-4
    
    # Loss function
    loss_function: str = "mse"  # "mse", "mae", "huber"
    
    # Hardware
    device: str = "cuda"  # "cuda", "cpu", "mps"
    num_workers: int = 0  # IMPORTANT: Keep at 0 to avoid multiprocessing issues
    pin_memory: bool = False  # IMPORTANT: Keep False to avoid CUDA memory issues
    
    def __post_init__(self):
        """Validate training configuration."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.optimizer in ["adam", "adamw", "sgd"], f"Unknown optimizer: {self.optimizer}"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment metadata
    experiment_name: str = "emg_reconstruction"
    random_seed: int = 42
    log_level: str = "INFO"
    
    # Paths
    experiment_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    
    def __post_init__(self):
        """Set up experiment directories."""
        if self.experiment_dir is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = f"experiments/{self.experiment_name}_{timestamp}"
        
        if self.checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration. If None, saves to experiment directory
            
        Returns:
            Path where configuration was saved
        """
        if filepath is None:
            os.makedirs(self.experiment_dir, exist_ok=True)
            filepath = os.path.join(self.experiment_dir, "config.json")
        
        # Convert to dictionary
        config_dict = {
            'data': asdict(self.data),
            'augmentation': asdict(self.augmentation),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'experiment_name': self.experiment_name,
            'random_seed': self.random_seed,
            'log_level': self.log_level,
            'experiment_dir': self.experiment_dir,
            'checkpoint_dir': self.checkpoint_dir
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        logger.info(f"Configuration saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            ExperimentConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct dataclass instances
        config = cls(
            data=DataConfig(**config_dict['data']),
            augmentation=AugmentationConfig(**config_dict['augmentation']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            experiment_name=config_dict['experiment_name'],
            random_seed=config_dict['random_seed'],
            log_level=config_dict['log_level'],
            experiment_dir=config_dict.get('experiment_dir'),
            checkpoint_dir=config_dict.get('checkpoint_dir')
        )
        
        logger.info(f"Configuration loaded from {filepath}")
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': asdict(self.data),
            'augmentation': asdict(self.augmentation),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'experiment_name': self.experiment_name,
            'random_seed': self.random_seed,
            'log_level': self.log_level
        }


def create_default_config() -> ExperimentConfig:
    """
    Create default experiment configuration.
    
    Returns:
        ExperimentConfig with default values
    """
    return ExperimentConfig()


def create_config_from_optuna_params(params: Dict[str, Any], 
                                     base_config: Optional[ExperimentConfig] = None) -> ExperimentConfig:
    """
    Create configuration from Optuna hyperparameters.
    
    Args:
        params: Dictionary of hyperparameters from Optuna trial
        base_config: Base configuration to modify. If None, uses defaults
        
    Returns:
        ExperimentConfig with Optuna parameters applied
    """
    if base_config is None:
        config = create_default_config()
    else:
        # Deep copy to avoid modifying original
        import copy
        config = copy.deepcopy(base_config)
    
    # Update model parameters
    if 'filters' in params:
        config.model.num_filters = params['filters']
    if 'kernel_size' in params:
        config.model.kernel_size = params['kernel_size']
    if 'num_dilation_layers' in params:
        config.model.dilations = [2**i for i in range(params['num_dilation_layers'])]
    if 'dropout' in params:
        config.model.dropout_rate = params['dropout']
    if 'latent_dim' in params:
        config.model.latent_dim = params['latent_dim']
    
    # Update training parameters
    if 'batch_size' in params:
        config.training.batch_size = params['batch_size']
    if 'learning_rate' in params:
        config.training.learning_rate = params['learning_rate']
    
    return config