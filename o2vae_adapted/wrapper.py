# Adapted pipeline integration based on /raid/scratch/mevans/o2vae/run.py and
# /raid/scratch/mevans/o2vae/models/vae.py
# Upstream commit: 4c70d589954a86d114aba2d27ff7111573ac669c4
# License/attribution: retain upstream O2-VAE attribution and repository license.
# Changes: local dataset preparation, global normalization, circular masking,
# optional odd padding, lazy dependency loading, configurable training, and
# ordered z_mu extraction.

"""Pipeline adapter for the locally adapted O2-VAE implementation."""

from __future__ import annotations

import numpy as np


UPSTREAM_COMMIT = "4c70d589954a86d114aba2d27ff7111573ac669c4"


def _load_dependencies():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from .models.factory import build_model
    except ImportError as exc:
        raise ImportError(
            "autoencoder_method='o2vae' requires torch, e2cnn, and torchgeometry "
            "(upstream-tested versions: torch 1.13.0, e2cnn 0.2.1, "
            "torchgeometry 0.1.2; Python 3.9)."
        ) from exc
    return torch, DataLoader, TensorDataset, build_model


def _prepare_images(dataset, config):
    import torch

    images = dataset.tensors[0] if hasattr(dataset, "tensors") else torch.stack(
        [dataset[i][0] for i in range(len(dataset))]
    )
    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.ndim != 4:
        raise ValueError(f"O2-VAE expects [N,C,H,W] images, got {tuple(images.shape)}")
    image_size = int(getattr(config, "sqrt_pixel_number", images.shape[-1]))
    if tuple(images.shape[-2:]) != (image_size, image_size):
        raise ValueError(
            f"O2-VAE requires native square images of {image_size}x{image_size}; "
            f"got {tuple(images.shape[-2:])}"
        )

    images = images.float()
    if getattr(config, "o2vae_circular_mask", True):
        yy, xx = torch.meshgrid(
            torch.arange(image_size), torch.arange(image_size), indexing="ij"
        )
        radius = (image_size - 1) / 2
        mask = ((yy - radius) ** 2 + (xx - radius) ** 2 <= radius**2).to(images)
        images = images * mask

    lo, hi = images.amin(), images.amax()
    if hi > lo:
        images = (images - lo) / (hi - lo)
    else:
        images = torch.zeros_like(images)

    if getattr(config, "o2vae_pad_to_odd", False):
        images = torch.nn.functional.pad(images, (0, 1, 0, 1))
    return images.contiguous()


def train_autoencoder(dataset, config):
    torch, DataLoader, TensorDataset, build_model = _load_dependencies()
    images = _prepare_images(dataset, config)
    batch_size = max(4, int(getattr(config, "o2vae_batch_size", 4)))
    model = build_model(
        n_channels=images.shape[1],
        latent_dim=int(getattr(config, "o2vae_latent_dim", 64)),
        encoder_channels=getattr(config, "o2vae_encoder_channels", [2, 3, 4, 4, 6, 8]),
        decoder_channels=getattr(config, "o2vae_decoder_channels", [64, 32, 16, 8]),
        image_size=images.shape[-1],
        beta=float(getattr(config, "o2vae_beta", 0.01)),
        align_loss=bool(getattr(config, "o2vae_align_loss", True)),
        align_fourier=bool(getattr(config, "o2vae_align_fourier", True)),
        do_flip=bool(getattr(config, "o2vae_do_flip", True)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(getattr(config, "o2vae_learning_rate", 1e-3)))
    amp = bool(getattr(config, "o2vae_amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    accum = max(1, int(getattr(config, "o2vae_grad_accum_steps", 1)))
    losses = []

    for epoch in range(int(getattr(config, "o2vae_epochs", 41))):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (batch,) in enumerate(loader):
            batch = batch.to(device)
            with torch.cuda.amp.autocast(enabled=amp):
                x, recon, mu, logvar = model(batch)
                loss = model.loss(x, recon, mu, logvar)["loss"] / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0 or step + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += loss.detach().item() * accum * len(batch)
        epoch_loss = running / len(loader.dataset)
        losses.append(epoch_loss)
        print(f"O2-VAE epoch {epoch + 1}/{getattr(config, 'o2vae_epochs', 41)}, loss={epoch_loss:.6f}")

    model.eval()
    encoded = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(images), batch_size=min(batch_size, 256), shuffle=False):
            encoded.append(model.embed_data(batch.to(device)).cpu().numpy())
    return np.concatenate(encoded, axis=0).astype(np.float32), model, losses
