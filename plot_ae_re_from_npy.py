"""Generate a polished autoencoder reconstruction error plot from a .npy file."""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def plot_reconstruction_error(ae_re: np.ndarray, output_path: str) -> None:
    ae_re = np.asarray(ae_re).squeeze()
    if ae_re.ndim != 1:
        raise ValueError("Expected a 1D array for reconstruction error values.")

    epochs = np.arange(1, ae_re.size + 1)
    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor="white")

    ax.plot(epochs, ae_re, color="#1f77b4", linewidth=3.2, alpha=0.9)

    
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Mean Reconstruction Error", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=14)
    ax.set_facecolor("white")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.margins(x=0.02)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    ax.legend(["Reconstruction error"], frameon=True, fontsize=14)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AE reconstruction error from a .npy file.")
    parser.add_argument(
        "--input",
        default="/raid/scratch/mevans/autoencoder_results_no_bottle_flip/ae_re.npy",
        help="Path to ae_re.npy",
    )
    parser.add_argument(
        "--output",
        default="/raid/scratch/mevans/autoencoder_results_no_bottle_flip/autoencoder_training_re_pretty.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    ae_re = np.load(args.input)
    plot_reconstruction_error(ae_re, args.output)


if __name__ == "__main__":
    main()
