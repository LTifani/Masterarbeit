"""
Hyperparameter Optimization using Optuna

Automates the search for optimal model hyperparameters.
"""

import os
import argparse
import logging
import torch
import optuna
import mlflow
from optuna.trial import Trial
from typing import Optional

from src.config import (
    ExperimentConfig, 
    create_default_config,
    create_config_from_optuna_params
)
from src.datasets import create_data_loaders
from src.train import train_model
from src.utils import setup_logging
from src.utils import _flatten_config_dict

logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

def objective(trial: Trial, 
              base_config: ExperimentConfig,
              device: torch.device) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration to modify
        device: Device for training
        
    Returns:
        Best validation loss (to minimize)
    """
    # Suggest hyperparameters
    params = {
        'filters': trial.suggest_categorical('filters', [32, 64, 128, 256]),
        'kernel_size': trial.suggest_categorical('kernel_size', [5, 7, 9, 11]),
        'num_dilation_layers': trial.suggest_int('num_dilation_layers', 4, 6),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'latent_dim': trial.suggest_categorical('latent_dim', [64, 128, 256, 512]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    }
    
    # Create config with suggested parameters
    config = create_config_from_optuna_params(params, base_config)
    config.experiment_name = f"optuna_trial_{trial.number}"
    config.training.device = str(device)
    config.training.num_epochs = 50 
    
    # Logge Trial-Hyperparameter
    flat_params = _flatten_config_dict(config.to_dict())
    mlflow.log_params(flat_params)
    
    try:
        # Create data loaders
        train_loader, val_loader, _ = create_data_loaders(config)
        
        # Train model
        _, results = train_model(config, train_loader, val_loader, test_loader=None)
        
        # Report intermediate values for pruning
        best_val_loss = results['best_val_loss']
        
        return best_val_loss
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned()


def run_optimization(base_config: ExperimentConfig,
                    n_trials: int = 50,
                    study_name: str = "emg_optimization",
                    storage_path: Optional[str] = None) -> optuna.Study:
    """
    Run hyperparameter optimization study.
    
    Args:
        base_config: Base configuration
        n_trials: Number of trials to run
        study_name: Name of the study
        storage_path: Path to database for persistence
        
    Returns:
        Completed Optuna study
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running optimization on {device}")
    
    # Create study
    storage = f"sqlite:///{storage_path}" if storage_path else None
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',  # Minimize validation loss
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10
        )
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config, device),
        n_trials=n_trials,
        timeout=None,
        show_progress_bar=True
    )
    
    # Log results
    logger.info("="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best validation loss: {study.best_value:.6f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    return study


def train_with_best_params(study: optuna.Study,
                           base_config: ExperimentConfig) -> None:
    """
    Train final model with best hyperparameters.
    
    Args:
        study: Completed Optuna study
        base_config: Base configuration
    """
    logger.info("="*80)
    logger.info("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    logger.info("="*80)
    
    # Create config with best parameters
    config = create_config_from_optuna_params(study.best_params, base_config)
    config.experiment_name = "final_optimized_model"
    config.training.num_epochs = 100  # Full training
    
    # --- MLFLOW INTEGRATION ---
    
    mlflow.set_experiment(config.experiment_name) 

    # Starte den MLflow Run. Wir nutzen eine eindeutige Run-Bezeichnung basierend auf dem besten Trial
    run_name = f"Final_Run_from_Trial_{study.best_trial.number}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Save config
        config.save()
        
        # Logge alle Konfigurationsparameter
        flat_params = _flatten_config_dict(config.to_dict())
        mlflow.log_params(flat_params)
        
        # Logge die gespeicherte Konfigurationsdatei als Artefakt
        config_path = os.path.join(config.experiment_dir, "config.json")
        mlflow.log_artifact(config_path, artifact_path="config")
        
        # Logge Tags zur Nachverfolgung
        mlflow.set_tag("source_best_trial_id", str(study.best_trial.number))
        mlflow.set_tag("training_type", "Final_Model")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        # Train model
        model, results = train_model(config, train_loader, val_loader, test_loader)
        
        mlflow.set_tag("status", "COMPLETED")
        
    logger.info("="*80)
    logger.info("FINAL MODEL RESULTS")
    logger.info("="*80)
    logger.info(f"Best Val Loss: {results['best_val_loss']:.6f}")
    if results.get('test_metrics'):
        for key, value in results['test_metrics'].items():
            logger.info(f"{key}: {value:.6f}")


def main():
    """Main execution for hyperparameter search."""
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--study-name', type=str, default='emg_optimization',
                       help='Name of the study')
    parser.add_argument('--storage', type=str, default='./output/optuna_study.db',
                       help='Path to SQLite database for study persistence')
    parser.add_argument('--train-final', action='store_true',
                       help='Train final model with best parameters after optimization')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_dir='output', log_filename='hyperparameter_search.log', level="debug")
    
    # Create base configuration
    base_config = create_default_config()
    
    # Run optimization
    study = run_optimization(
        base_config=base_config,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage_path=args.storage
    )
    
    # Optionally train final model
    if args.train_final:
        train_with_best_params(study, base_config)


if __name__ == "__main__":
    main()