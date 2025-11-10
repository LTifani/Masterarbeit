"""
Temporal Convolutional Network for EMG Signal Reconstruction

This module implements a TCN-based autoencoder for reconstructing EMG signals.
Anomalies are detected based on reconstruction error.
"""

import os
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class TCNBlock(nn.Module):
    """Single TCN block with dilated convolution, normalization, and residual connection."""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float = 0.2):
        """
        Initialize TCN block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            dilation: Dilation rate
            dropout: Dropout probability
        """
        super().__init__()
        
        # Calculate padding for same output length
        padding = (kernel_size - 1) * dilation // 2
        
        # First conv layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second conv layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN block."""
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        out = F.relu(out + residual)
        return out


class TCNEncoder(nn.Module):
    """TCN-based encoder for signal compression."""
    
    def __init__(self, input_channels: int, num_filters: int, kernel_size: int,
                 dilations: List[int], dropout: float):
        """Initialize TCN encoder."""
        super().__init__()
        
        layers = []
        in_channels = input_channels
        
        for dilation in dilations:
            layers.append(
                TCNBlock(in_channels, num_filters, kernel_size, dilation, dropout)
            )
            in_channels = num_filters
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input signal."""
        return self.network(x)


class TCNDecoder(nn.Module):
    """TCN-based decoder for signal reconstruction."""
    
    def __init__(self, num_filters: int, output_channels: int, kernel_size: int,
                 dilations: List[int], dropout: float):
        """Initialize TCN decoder."""
        super().__init__()
        
        layers = []
        reversed_dilations = list(reversed(dilations))
        
        for i, dilation in enumerate(reversed_dilations):
            out_ch = num_filters if i < len(reversed_dilations) - 1 else output_channels
            layers.append(
                TCNBlock(num_filters, out_ch, kernel_size, dilation, dropout)
            )
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.network(x)


class TCNAutoencoder(nn.Module):
    """
    TCN-based autoencoder for EMG signal reconstruction.
    
    This model learns to reconstruct normal EMG signals. Anomalies are
    detected based on high reconstruction error.
    """
    
    def __init__(self, input_channels: int = 1, num_filters: int = 64,
                 kernel_size: int = 7, dilations: List[int] = [1, 2, 4, 8],
                 dropout_rate: float = 0.2, latent_dim: int = 128):
        """Initialize TCN autoencoder."""
        super().__init__()
        
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = TCNEncoder(input_channels, num_filters, kernel_size, 
                                  dilations, dropout_rate)
        
        # Latent bottleneck
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_filters, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Decoder starting point
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, num_filters * 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Decoder
        self.decoder = TCNDecoder(num_filters, input_channels, kernel_size, 
                                 dilations, dropout_rate)
        
        # Final reconstruction layer
        self.output_conv = nn.Conv1d(input_channels, input_channels, 1)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.encoder(x)
        latent = self.to_latent(x)
        return latent
    
    def decode(self, latent: torch.Tensor, target_length: int) -> torch.Tensor:
        """Decode latent representation to signal."""
        batch_size = latent.size(0)
        
        x = self.from_latent(latent)
        x = x.view(batch_size, self.num_filters, 8)
        
        # Upsample to target length
        x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        
        x = self.decoder(x)
        x = self.output_conv(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode and decode."""
        original_shape = x.shape
        target_length = original_shape[-1]
        
        latent = self.encode(x)
        reconstruction = self.decode(latent, target_length)
        
        if len(original_shape) == 2:
            reconstruction = reconstruction.squeeze(1)
        
        return reconstruction
    
    def compute_reconstruction_error(self, x: torch.Tensor,
                                    reconstruction: Optional[torch.Tensor] = None,
                                    reduction: str = 'none') -> torch.Tensor:
        """Compute reconstruction error (MSE by default)."""
        if reconstruction is None:
            reconstruction = self.forward(x)
        
        error = F.mse_loss(reconstruction, x, reduction='none')
        error = error.view(error.size(0), -1).mean(dim=1)
        
        if reduction == 'mean':
            return error.mean()
        elif reduction == 'sum':
            return error.sum()
        else:
            return error
    
    def save_checkpoint(self, filepath: str, epoch: int,
                       optimizer_state: Optional[Dict] = None,
                       metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_channels': self.input_channels,
                'num_filters': self.num_filters,
                'kernel_size': self.kernel_size,
                'dilations': self.dilations,
                'dropout_rate': self.dropout_rate,
                'latent_dim': self.latent_dim
            }
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, 
                       device: Optional[torch.device] = None) -> tuple:
        """Load model from checkpoint."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"Model loaded from {filepath} (epoch {checkpoint.get('epoch', 'unknown')})")
        
        return model, checkpoint
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            'input_channels': self.input_channels,
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'dilations': self.dilations,
            'dropout_rate': self.dropout_rate,
            'latent_dim': self.latent_dim
        }