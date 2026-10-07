#!/usr/bin/env python3
"""Plot continuum, smoothed continuum, and molecular moment-0 maps for a single source.

Default source: 810154
Outputs a 1x5 panel PNG with:
    1. continuum
    2. smoothed continuum
    3. DCN moment-0
    4. SiO moment-0
    5. H2CO moment-0
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import Normalize
from scipy import ndimage


SOURCE_ID = "810154"
BASE_DIR = "/raid/scratch/mevans"
CONTINUUM_DIR = os.path.join(BASE_DIR, "newdata")
SMOOTHED_DIR = os.path.join(BASE_DIR, "smoothed_7mtm1tm2_noiseremoved0")
MOMENT0_DIR = os.path.join(BASE_DIR, "moment0_with_rms")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "source_plots")

LINE_NAMES = ["DCN", "SiO", "H2CO"]


def _load_2d_array(path: str) -> np.ndarray:
    """Load a FITS or NPY file and return a 2D float array."""
    if path.lower().endswith(".fits"):
        with fits.open(path) as hdul:
            data = hdul[0].data
    else:
        data = np.load(path, allow_pickle=False)

    if data is None:
        raise ValueError(f"No data found in {path}")

    arr = np.asarray(data, dtype=float)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
    return np.nan_to_num(arr, nan=0.0)


def _rescale_to_shape(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Rescale a 2D array to the requested shape using bilinear interpolation."""
    if arr.shape == shape:
        return arr
    zoom_y = shape[0] / arr.shape[0]
    zoom_x = shape[1] / arr.shape[1]
    return ndimage.zoom(arr, (zoom_y, zoom_x), order=1)


def _find_continuum_path(source_id: str) -> str:
    path = os.path.join(CONTINUUM_DIR, f"{source_id}_cont_7MTM2TM1_jointdeconv.image.fits")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Continuum file not found: {path}")
    return path


def _find_smoothed_path(source_id: str) -> str:
    candidates = [
        os.path.join(SMOOTHED_DIR, f"{source_id}_smoothed_noiseremoved0sigma.fits"),
        os.path.join(SMOOTHED_DIR, f"{source_id}_smoothed_noiseremoved1sigma.fits"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Smoothed continuum file not found for source {source_id}")


def _find_moment0_path(source_id: str, line: str) -> str:
    source_dir = os.path.join(MOMENT0_DIR, source_id)
    candidates = [source_dir, MOMENT0_DIR]
    for base in candidates:
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            lower = entry.lower()
            if not lower.endswith(".fits"):
                continue
            if source_id.lower() not in lower:
                continue
            if line.lower() not in lower:
                continue
            if "_moment0" not in lower:
                continue
            return os.path.join(base, entry)
    raise FileNotFoundError(f"Moment-0 FITS not found for source {source_id}, line {line}")


def _normalize_0_to_1(arr: np.ndarray) -> np.ndarray:
    """Normalize a 2D image to the [0, 1] range."""
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(arr, dtype=float)

    normed = (arr - vmin) / (vmax - vmin)
    return np.clip(np.nan_to_num(normed, nan=0.0), 0.0, 1.0)


def build_figure(source_id: str, output_dir: str, show: bool = False) -> str:
    continuum_path = _find_continuum_path(source_id)
    smoothed_path = _find_smoothed_path(source_id)
    moment_paths = {line: _find_moment0_path(source_id, line) for line in LINE_NAMES}

    continuum = _load_2d_array(continuum_path)
    smoothed = _load_2d_array(smoothed_path)
    target_shape = continuum.shape
    smoothed = _rescale_to_shape(smoothed, target_shape)
    continuum = _normalize_0_to_1(continuum)
    smoothed = _normalize_0_to_1(smoothed)

    moment_maps = {}
    for line, path in moment_paths.items():
        arr = _load_2d_array(path)
        moment_maps[line] = _normalize_0_to_1(_rescale_to_shape(arr, target_shape))

    panels = [
        ("Continuum", continuum, "plasma", Normalize(vmin=0.0, vmax=1.0)),
        ("Smoothed continuum", smoothed, "plasma", Normalize(vmin=0.0, vmax=1.0)),
        ("DCN moment-0", moment_maps["DCN"], "plasma", Normalize(vmin=0.0, vmax=1.0)),
        ("SiO moment-0", moment_maps["SiO"], "plasma", Normalize(vmin=0.0, vmax=1.0)),
        ("H2CO moment-0", moment_maps["H2CO"], "plasma", Normalize(vmin=0.0, vmax=1.0)),
    ]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{source_id}_continuum_smoothed_moment0.png")

    fig, axes = plt.subplots(1, len(panels), figsize=(24, 5.5), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, img, cmap, norm) in zip(axes, panels):
        im = ax.imshow(img, origin="lower", cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=28)
        ax.set_xticks([])
        ax.set_yticks([])

    #fig.suptitle(f"Source {source_id}: continuum, smoothed continuum, and moment-0 maps", fontsize=18)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved figure to {out_path}")
    print(f"Loaded continuum: {continuum_path}")
    print(f"Loaded smoothed continuum: {smoothed_path}")
    for line, path in moment_paths.items():
        print(f"Loaded {line} moment-0: {path}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot continuum, smoothed continuum, and molecular moment-0 maps.")
    parser.add_argument("--source-id", default=SOURCE_ID, help="Source identifier to plot (default: 810154)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for the saved PNG")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively")
    args = parser.parse_args()

    build_figure(args.source_id, args.output_dir, show=args.show)


if __name__ == "__main__":
    main()
