import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import TensorDataset

from autoencoder_dispatch import train_autoencoder


def test_existing_dispatch_preserves_shape():
    if not TORCH_AVAILABLE:
        pytest.skip("existing autoencoder smoke test requires torch")
    images = torch.rand(4, 1, 304, 304)
    config = type("Config", (), {
        "ae_batch_size": 2,
        "ae_epochs": 1,
        "ae_learning_rate": 1e-3,
        "ae_bottleneck_dim": 4,
        "ae_use_bottleneck": True,
    })()
    encoded, _model, losses = train_autoencoder("existing", TensorDataset(images), config)
    assert encoded.shape == (4, 4)
    assert encoded.dtype == np.float32
    assert len(losses) == 1


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None
        for name in ("torch", "e2cnn", "torchgeometry")),
    reason="O2-VAE smoke and rotation checks require torch, e2cnn, and torchgeometry",
)
def test_o2vae_smoke_and_rotation_check():
    config = type("Config", (), {
        "sqrt_pixel_number": 304,
        "o2vae_batch_size": 4,
        "o2vae_epochs": 1,
        "o2vae_learning_rate": 1e-3,
        "o2vae_latent_dim": 4,
        "o2vae_encoder_channels": [2, 2, 2, 2, 2, 2],
        "o2vae_decoder_channels": [8, 4, 2, 1],
        "o2vae_beta": 0.01,
        "o2vae_align_loss": False,
        "o2vae_align_fourier": False,
        "o2vae_do_flip": True,
        "o2vae_circular_mask": True,
        "o2vae_pad_to_odd": False,
        "o2vae_amp": False,
        "o2vae_grad_accum_steps": 1,
    })()
    images = torch.rand(4, 1, 304, 304)
    encoded, _model, losses = train_autoencoder("o2vae", TensorDataset(images), config)
    assert encoded.shape == (4, 4)
    assert encoded.dtype == np.float32
    assert len(losses) == 1
