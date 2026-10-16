"""Select the configured autoencoder without importing optional O2-VAE dependencies."""

from __future__ import annotations

import numpy as np


VALID_AUTOENCODER_METHODS = {"existing", "o2vae"}


def train_autoencoder(method, dataset, config):
    """Train the requested autoencoder and return float32, ordered latent vectors."""
    if method not in VALID_AUTOENCODER_METHODS:
        raise ValueError(
            f"Unknown autoencoder_method={method!r}; expected 'existing' or 'o2vae'."
        )

    if method == "existing":
        from autoencoder import train_autoencoder as train_existing

        encoded, model, losses = train_existing(dataset, config)
    else:
        from o2vae_adapted.wrapper import train_autoencoder as train_o2vae

        encoded, model, losses = train_o2vae(dataset, config)

    encoded = np.asarray(encoded, dtype=np.float32)
    if encoded.ndim != 2 or encoded.shape[0] != len(dataset):
        raise ValueError(
            f"{method} autoencoder returned latents with invalid shape {encoded.shape}; "
            f"expected ({len(dataset)}, dim)."
        )
    return encoded, model, losses
