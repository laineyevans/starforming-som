# Adapted from /raid/scratch/mevans/o2vae/run.py
# Upstream commit: 4c70d589954a86d114aba2d27ff7111573ac669c4
# License/attribution: retain upstream O2-VAE attribution and repository license.
# Changes: local package factory and image-size-derived decoder configuration.

"""Small adapted model factory based on upstream O2-VAE configuration."""

from .decoders.cnn_decoder import CnnDecoder
from .encoders_o2.e2scnn import E2SFCNN
from .vae import VAE


def build_model(
    n_channels,
    latent_dim=64,
    encoder_channels=(2, 3, 4, 4, 6, 8),
    decoder_channels=(64, 32, 16, 8),
    image_size=304,
    beta=0.01,
    align_loss=True,
    align_fourier=True,
    do_flip=True,
):
    """Build the O2-VAE using the upstream encoder and adapted decoder."""
    decoder_channels = tuple(decoder_channels)
    base_size = image_size // (2 ** len(decoder_channels))
    expected_size = base_size * (2 ** len(decoder_channels))
    if image_size not in (expected_size, expected_size + 1):
        raise ValueError(
            f"image_size={image_size} must equal base_size * 2**n_stages "
            f"({expected_size}) or that size plus one for odd padding"
        )
    encoder = E2SFCNN(
        n_channels=n_channels,
        n_classes=2 * latent_dim,
        layer_type="inducedgated_norm",
        N=-3,
        cnn_dims=list(encoder_channels),
    )
    decoder = CnnDecoder(
        zdim=latent_dim,
        cnn_dims=decoder_channels,
        out_channels=n_channels,
        base_size=base_size,
        image_size=image_size,
    )
    return VAE(
        q_net=encoder,
        p_net=decoder,
        zdim=latent_dim,
        do_sigmoid=True,
        loss_kwargs={
            "recon_loss_type": "bce",
            "beta": beta,
            "align_loss": align_loss,
            "align_fourier": align_fourier,
            "do_flip": do_flip,
            "rot_steps": 2,
        },
        prior_kwargs={"prior": "standard"},
    )
