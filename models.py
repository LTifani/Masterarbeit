import torch
import logging
import numpy as np
from torch import nn
from typing import Iterable, Optional
from torch.utils.data import TensorDataset, DataLoader
from utils import cfg, Config, get_error_per_sample, DEVICE, join_path, ic

# =============================
# 2) MODEL (TCN Autoencoder in Torch)
# =============================
logger = logging.getLogger(__name__)


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding="same", dilation=dilation),
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

    def __init__(
        self,
        seq_len: int,
        n_feat: int,
        filters: int = 32,
        kernel_size: int = 2,
        dilations=(1, 2, 4, 8),
        dropout: float = 0.2,
        latent_dim: int = 8,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_feat = n_feat
        self.enc_blocks = nn.ModuleList()
        enc_in = n_feat
        # Encoder dilated stack with skip outputs (concat later)
        for d in dilations:
            self.enc_blocks.append(
                ConvBlock1D(enc_in, filters, kernel_size, dilation=d, dropout=dropout)
            )
            enc_in = filters  # subsequent blocks see 'filters' channels

        # Bottleneck 1x1 (channel compression)
        self.bottleneck = nn.Conv1d(filters * len(dilations), latent_dim, kernel_size=1)

        # Decoder dilated stack with skip outputs (independent weights)
        self.dec_blocks = nn.ModuleList()
        dec_in = latent_dim
        for d in dilations:
            self.dec_blocks.append(
                ConvBlock1D(dec_in, filters, kernel_size, dilation=d, dropout=dropout)
            )
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
        out = out[:, : self.seq_len, :]
        return out


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_feat: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config = Config(),
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.criterion = nn.MSELoss()
        self.autoencoder = TcnAutoencoder(seq_len, n_feat, **kwargs)

        self.val_loader = val_loader
        self.train_loader = train_loader
        lr = config.train_hyperparameter.get("learning_rate", 1e-3)
        self.optimizer = torch.optim.Adam( self.autoencoder.parameters(), lr)

    def forward(self, x):
        return self.autoencoder(x)

    def parameters(self, recurse=True):
        return self.autoencoder.parameters(recurse=recurse)

    def state_dict(self, *arg, **kwargs):
        return self.autoencoder.state_dict(*arg, **kwargs)
    
    def save_best_model(self) -> None:
        best_model_path = join_path(self.config.get_training_dir(), "best_model.pth")
        torch.save(self.autoencoder.state_dict(), best_model_path)

    def load_best_model(self, best_model_path: str) -> None:
        self.autoencoder.load_state_dict(torch.load(best_model_path))

    def train_one_epoch(self) -> float:
        total_loss = 0.0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.autoencoder.train()
        self.autoencoder.to(device)
        for (x_batch,) in self.train_loader:
            x_batch = x_batch.to(device)
            self.optimizer.zero_grad()
            reconstruction = self.autoencoder(x_batch)
            loss = self.criterion(reconstruction, x_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.detach().item()

        return total_loss / max(1, len(self.train_loader))

    @torch.inference_mode()
    def validate_one_epoch(self, loader: Optional[DataLoader] = None) -> float:
        total_loss = 0.0
        self.autoencoder.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.autoencoder.to(device)
        if loader is None:
            loader = self.val_loader

        for (x_batch,) in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            reconstruction = self.autoencoder(x_batch)
            error_per_sample = get_error_per_sample(x_batch, reconstruction)
            total_loss += error_per_sample.mean().detach().item()

        return total_loss / max(1, len(loader))

    @torch.inference_mode()
    def predict_generator(
        self, input_data: np.ndarray | torch.Tensor | DataLoader, 
        batch_size: int = 128
    ) -> Iterable[np.ndarray]:
        self.autoencoder.eval()
        if isinstance(input_data, DataLoader):
            dataloader = input_data
        else:
            if isinstance(input_data, np.ndarray):
                tensor_data = torch.from_numpy(input_data).to(DEVICE)
            elif isinstance(input_data, torch.Tensor):
                tensor_data = input_data.to(DEVICE)
            dataloader = DataLoader(TensorDataset(tensor_data), batch_size, False)

        for (x_batch,) in dataloader:
            pred = self.autoencoder(x_batch)
            yield pred.cpu().numpy()

    @torch.inference_mode()
    def compute_reconstruction_errors(self, data_loader: DataLoader):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        idx = 0
        self.autoencoder.eval()
        self.autoencoder.to(device)
        errors = np.zeros(len(data_loader.dataset), dtype=np.float32)

        for batch in data_loader:
            input_data = batch[0].to(device, non_blocking=True)
            reconstruction = self.autoencoder(input_data)
            batch_errors = get_error_per_sample(input_data, reconstruction).cpu().numpy()

            batch_size = len(input_data)
            errors[idx : idx + batch_size] = batch_errors
            idx += batch_size

        return errors

    def train_model(self, with_eval: bool = True) -> float:
        val_loss = np.inf
        epoches = self.config.train_hyperparameter.get("epochs", 10)
        for epoch in range(epoches):
            train_loss = self.train_one_epoch()
            if with_eval:
                val_loss = self.validate_one_epoch()
                logger.info(f"[{epoch+1}/{epoches}] Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")
        
        if not with_eval:
            val_loss = self.validate_one_epoch()
        return float(val_loss)
    
if __name__ == "__main__":
    # Simple test
    ic("Starting training...")
    batch_size = 5
    seq_len = 128
    n_feat = 1
    input_data = torch.randn(batch_size * 20 + 1, seq_len, n_feat)
    foo = DataLoader(
        TensorDataset(input_data), batch_size=batch_size, shuffle=False, drop_last=False
    )

    model = Model(seq_len, n_feat, foo, foo)

    reconstruction_errrors = model.compute_reconstruction_errors(foo)
    ic(reconstruction_errrors.shape)

    error = model.validate_one_epoch()
    ic(error)
    ic(cfg)

    ic("La vie belle !!!")
    logger.debug("La vie belle !!!")

    all_errors = []
    for batch_errors in model.predict_generator(input_data):
        all_errors.extend(batch_errors)
    # ic(all_errors[:10])

    # ic(model)
    # y = model(x)
    # print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    # assert x.shape == y.shape, "Output shape does not match input shape!"
