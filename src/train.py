"""
Training and Evaluation Logic for EMG Reconstruction Model

Handles model training, validation, evaluation, and metric tracking.
"""

import os
import csv
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
import logging
import mlflow

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        """Check if training should stop."""
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


class MetricsLogger:
    """Logs training metrics to CSV file."""
    
    def __init__(self, log_path: str):
        """
        Initialize metrics logger.
        
        Args:
            log_path: Path to CSV file for logging metrics
        """
        self.log_path = log_path
        self.fieldnames = ['epoch', 'train_loss', 'val_loss', 'learning_rate']
        
        # Create CSV file with headers
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def log(self, metrics: Dict[str, Any]):
        """Log metrics to CSV."""
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics)


class Trainer:
    """Handles model training and evaluation."""
    
    def __init__(self, 
                 model: nn.Module,
                 config,
                 device: torch.device):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            config: ExperimentConfig instance
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup loss function
        self.criterion = self._create_loss_function()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.min_delta,
            mode='min'
        )
        
        # Setup metrics logger
        log_path = os.path.join(config.experiment_dir, 'results.csv')
        self.metrics_logger = MetricsLogger(log_path)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        optimizer_name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function from config."""
        loss_name = self.config.training.loss_function.lower()
        
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'mae':
            return nn.L1Loss()
        elif loss_name == 'huber':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler from config."""
        scheduler_name = self.config.training.scheduler
        
        if scheduler_name is None:
            return None
        elif scheduler_name == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.num_epochs
            )
        else:
            logger.warning(f"Unknown scheduler: {scheduler_name}, using None")
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch_data in train_loader:
            inputs, targets, _ = batch_data
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                inputs, targets, _ = batch_data
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def evaluate_anomaly_detection(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate anomaly detection performance.
        
        Uses reconstruction error as anomaly score.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_errors = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                inputs, targets, labels = batch_data
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Compute reconstruction
                outputs = self.model(inputs)
                
                # Compute per-sample reconstruction error
                errors = torch.mean((outputs - targets) ** 2, dim=-1)
                
                all_errors.extend(errors.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_errors = np.array(all_errors)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        try:
            auroc = roc_auc_score(all_labels, all_errors)
            precision, recall, _ = precision_recall_curve(all_labels, all_errors)
            auprc = auc(recall, precision)
            
            # Find optimal threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = all_errors[np.argsort(all_errors)[min(best_idx, len(all_errors)-1)]]
            
            metrics = {
                'auroc': auroc,
                'auprc': auprc,
                'best_threshold': float(best_threshold),
                'mean_error_normal': float(all_errors[all_labels == 0].mean()) if np.any(all_labels == 0) else 0.0,
                'mean_error_anomaly': float(all_errors[all_labels == 1].mean()) if np.any(all_labels == 1) else 0.0,
                'std_error_normal': float(all_errors[all_labels == 0].std()) if np.any(all_labels == 0) else 0.0,
                'std_error_anomaly': float(all_errors[all_labels == 1].std()) if np.any(all_labels == 1) else 0.0
            }
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            metrics = {
                'auroc': 0.0,
                'auprc': 0.0,
                'mean_error': float(all_errors.mean())
            }
        
        return metrics
    
    def train(self, 
             train_loader: DataLoader,
             val_loader: DataLoader,
             test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            
        Returns:
            Dictionary with training results
        """
        logger.info("="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80)
        logger.info(f"Experiment: {self.config.experiment_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            }
            self.metrics_logger.log(metrics)
            
            # --- MLFLOW: Epoch-Metriken loggen ---
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.training.num_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}"
            )
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
                self.model.save_checkpoint(
                    best_model_path,
                    epoch=epoch,
                    optimizer_state=self.optimizer.state_dict(),
                    metrics={'val_loss': val_loss, 'train_loss': train_loss}
                )
                logger.info(f"Best model saved (val_loss: {val_loss:.6f})")

                # --- MLFLOW: Bestes Modell als Artefakt loggen ---
                mlflow.log_artifact(best_model_path, artifact_path="model_checkpoints")
                
            # Check early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save final model
        final_model_path = os.path.join(self.config.checkpoint_dir, 'final_model.pt')
        self.model.save_checkpoint(
            final_model_path,
            epoch=self.current_epoch,
            optimizer_state=self.optimizer.state_dict()
        )
        # --- MLFLOW: Finales Modell als Artefakt loggen ---
        mlflow.log_artifact(final_model_path, artifact_path="model_checkpoints")
        
        # Evaluate on test set if provided
        test_metrics = {}
        if test_loader is not None:
            logger.info("="*80)
            logger.info("EVALUATING ON TEST SET")
            logger.info("="*80)
            test_metrics = self.evaluate_anomaly_detection(test_loader)
            for key, value in test_metrics.items():
                logger.info(f"{key}: {value:.6f}")
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETED")
        logger.info("="*80)
        
        return {
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch + 1,
            'test_metrics': test_metrics
        }


def train_model(config, train_loader, val_loader, test_loader=None):
    """
    Main training function.
    
    Args:
        config: ExperimentConfig instance
        train_loader: Training data loader
        val_loader: Validation data loader  
        test_loader: Test data loader (optional)
        
    Returns:
        Trained model and results
    """
    from .models import TCNAutoencoder
    
    # Setup device
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = TCNAutoencoder(
        input_channels=config.model.input_channels,
        num_filters=config.model.num_filters,
        kernel_size=config.model.kernel_size,
        dilations=config.model.dilations,
        dropout_rate=config.model.dropout_rate,
        latent_dim=config.model.latent_dim
    )
    
    # Create trainer
    trainer = Trainer(model, config, device)
    
    # Train
    results = trainer.train(train_loader, val_loader, test_loader)
    
    return model, results