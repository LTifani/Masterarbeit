"""
Main Entry Point for EMG Anomaly Detection

Runs training with specified configuration.
"""

import os
import sys
import argparse
import torch
import logging

from src.config import ExperimentConfig, create_default_config
from src.datasets import create_data_loaders
from src.train import train_model
from src.utils import setup_logging


def setup_environment():
    """Setup environment for cross-platform compatibility."""
    # Fix Windows encoding issues
    if sys.platform == 'win32':
        try:
            # Set console code page to UTF-8
            os.system('chcp 65001 >nul 2>&1')
            
            # Reconfigure stdout and stderr with UTF-8 encoding
            import codecs
            if hasattr(sys.stdout, 'buffer'):
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            if hasattr(sys.stderr, 'buffer'):
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        except Exception as e:
            print(f"Warning: Could not set UTF-8 encoding: {e}")
        
        # Multiprocessing fix
        import multiprocessing
        multiprocessing.freeze_support()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='EMG Anomaly Detection Training')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration JSON file')
    parser.add_argument('--experiment-name', type=str, default='emg_reconstruction',
                       help='Name of the experiment')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=1500,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main execution function."""
    args = parse_args()
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        config = ExperimentConfig.load(args.config)
    else:
        print("Creating default configuration")
        config = create_default_config()
        config.experiment_name = args.experiment_name
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    config.training.device = args.device
    config.random_seed = args.seed
    
    # Set random seed
    set_seed(config.random_seed)
    
    # Setup logging
    setup_logging(
        log_dir=config.experiment_dir,
        log_filename='training.log',
        # level=config.log_level
        level="debug"
    )
    logger = logging.getLogger(__name__)
    
    # Save configuration
    config.save()
    logger.info(f"Configuration saved to {config.experiment_dir}/config.json")
    
    # Print configuration
    logger.info("="*80)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Random Seed: {config.random_seed}")
    logger.info(f"Device: {config.training.device}")
    logger.info(f"Batch Size: {config.training.batch_size}")
    logger.info(f"Learning Rate: {config.training.learning_rate}")
    logger.info(f"Epochs: {config.training.num_epochs}")
    logger.info(f"Model Filters: {config.model.num_filters}")
    logger.info(f"Kernel Size: {config.model.kernel_size}")
    logger.info(f"Dilations: {config.model.dilations}")
    
    try:
        # Create data loaders
        logger.info("="*80)
        logger.info("LOADING DATA")
        logger.info("="*80)
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        # Train model
        model, results = train_model(config, train_loader, val_loader, test_loader)
        
        # Print final results
        logger.info("="*80)
        logger.info("FINAL RESULTS")
        logger.info("="*80)
        logger.info(f"Best Validation Loss: {results['best_val_loss']:.6f}")
        logger.info(f"Training completed in {results['final_epoch']} epochs")
        
        if results.get('test_metrics'):
            logger.info("Test Set Metrics:")
            for key, value in results['test_metrics'].items():
                logger.info(f"  {key}: {value:.6f}")
        
        logger.info(f"\nResults saved to: {config.experiment_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()