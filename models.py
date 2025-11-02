import torch
import optuna
from torch import nn
from utils import *
from typing import Iterable
from torch.utils.data import TensorDataset, DataLoader

# =============================
# 2) MODEL (TCN Autoencoder in Torch)
# =============================

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding='same', dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)

class TcnAutoencoder(nn.Module):
    """
    Same-length TCN Autoencoder: enc stack (dilations) with skip concat -> 1x1 bottleneck ->
    dec stack (dilations) with skip concat -> 1x1 to n_feat; keeps temporal length.
    Note: PyTorch expects (N, C, T); our data is (N, T, C) so we permute.
    """
    def __init__(self, seq_len: int, n_feat: int, filters: int = 32, kernel_size: int = 2,
                 dilations=(1, 2, 4, 8), dropout: float = 0.2, latent_dim: int = 8):
        super().__init__()
        self.seq_len = seq_len
        self.n_feat = n_feat
        self.enc_blocks = nn.ModuleList()
        enc_in = n_feat
        # Encoder dilated stack with skip outputs (concat later)
        for d in dilations:
            self.enc_blocks.append(ConvBlock1D(enc_in, filters, kernel_size, dilation=d, dropout=dropout))
            enc_in = filters  # subsequent blocks see 'filters' channels

        # Bottleneck 1x1 (channel compression)
        self.bottleneck = nn.Conv1d(filters * len(dilations), latent_dim, kernel_size=1)

        # Decoder dilated stack with skip outputs (independent weights)
        self.dec_blocks = nn.ModuleList()
        dec_in = latent_dim
        for d in dilations:
            self.dec_blocks.append(ConvBlock1D(dec_in, filters, kernel_size, dilation=d, dropout=dropout))
            dec_in = filters

        # Final projection back to features
        self.out_conv = nn.Conv1d(filters * len(dilations), n_feat, kernel_size=1)

    def forward(self, x):  # x: (N, T, C)
        # to (N, C, T)
        x = x.permute(0, 2, 1)

        # Encoder with skip collection
        enc_skips = []
        tmp = x
        for blk in self.enc_blocks:
            tmp = blk(tmp)
            enc_skips.append(tmp)
        enc_cat = torch.cat(enc_skips, dim=1) if len(enc_skips) > 1 else enc_skips[0]

        z = self.bottleneck(enc_cat)

        # Decoder with skip collection
        dec_skips = []
        y = z
        for blk in self.dec_blocks:
            y = blk(y)
            dec_skips.append(y)
        dec_cat = torch.cat(dec_skips, dim=1) if len(dec_skips) > 1 else dec_skips[0]

        out = self.out_conv(dec_cat)
        # back to (N, T, C)
        out = out.permute(0, 2, 1)
        # exact length match (safety)
        out = out[:, :self.seq_len, :]
        return out


class Model(nn.Module):
    def __init__(self, seq_len: int, n_feat: int, 
                 config: Config, train_loader: DataLoader, val_loader: DataLoader, **kwargs):
        super().__init__()
        self.config = config
        self.autoencoder = TcnAutoencoder(seq_len, n_feat, **kwargs)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.config.learning_rate)


    def forward(self, x):
        return self.autoencoder(x)
    
    def parameters(self, recurse=True):
        return self.autoencoder.parameters(recurse=recurse)
    
    def save_best_model(self) -> None:
        """Save the best model to the specified path."""
        best_model_path = join_path(self.config.get_training_dir(), "best_model.pth")
        torch.save(self.autoencoder.state_dict(), best_model_path)

    def load_best_model(self, best_model_path: str) -> None:
        """Load the best model from the specified path."""
        self.autoencoder.load_state_dict(torch.load(best_model_path))

    
    def train_one_epoch(self) -> float:
        """
        If optimizer is provided -> train epoch, else eval epoch.
        Loss = mean squared error over all timesteps and features.
        """
        total_loss = 0.0

        self.autoencoder.train()

        for xb, in self.train_loader:
            xb = xb.to(DEVICE)
            self.optimizer.zero_grad()
            yb = self.autoencoder(xb)
            loss = self.criterion(xb, yb)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach().item()

        return total_loss / max(1, len(self.train_loader))
    
    
    @torch.no_grad()
    def validate_one_epoch(self, loader: Optional[DataLoader]=None) -> float:
        """
        If optimizer is provided -> train epoch, else eval epoch.
        Loss = mean squared error over all timesteps and features.
        """
        total_loss = 0.0
        self.autoencoder.eval()
        if loader is None:
            loader = self.val_loader

        for xb, in loader:
            xb = xb.to(DEVICE)
            yb = self.autoencoder(xb)
            loss = self.criterion(xb, yb)
            total_loss += loss.detach().item()
        
        return total_loss / max(1, len(loader))

    @torch.inference_mode()
    def predict_generator(self, input_data: np.ndarray, batch_size: int = 128) -> Iterable[np.ndarray]:
        self.autoencoder.eval()
        tensor_data = torch.from_numpy(input_data).to(DEVICE)
        dataloader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size, shuffle=False)

        for batch, in dataloader:
            pred = self.autoencoder(batch)
            yield pred.cpu().numpy()
    
    
    
    
if __name__ == "__main__":
    # Simple test
    batch_size = 512
    seq_len = 128
    n_feat = 1
    x = torch.randn(batch_size, seq_len, n_feat)
    model = TcnAutoencoder(seq_len, n_feat)
    print(model)
    y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    assert x.shape == y.shape, "Output shape does not match input shape!"