"""Compatibility wrappers for embedding diagnostics."""

from embedding_diagnostics import _fit_tsne, _save_scatter
from embedding_diagnostics import run_embedding_diagnostics


def umap_test(encoded_data, preds, config):
    """Run the diagnostics entry point when legacy callers request UMAP."""
    return run_embedding_diagnostics(
        encoded_data, preds, [""] * len(encoded_data), [""] * len(encoded_data),
        encoded_data, [0.0] * len(encoded_data), config,
    )


def tSNE(encoded_data, preds, config):
    """Retained compatibility name; diagnostics owns the t-SNE sweep."""
    return run_embedding_diagnostics(
        encoded_data, preds, [""] * len(encoded_data), [""] * len(encoded_data),
        encoded_data, [0.0] * len(encoded_data), config,
    )
