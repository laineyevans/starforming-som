"""Plot three chosen continuum preprocessed sources using grayscale + plasma overlay.

This script selects 3 random FITS files from:
    /raid/scratch/mevans/Preprocessed_Datas/continuumpreprocessed

For each source:
- the dimmer structure is shown in grayscale
- the brightest emission is overlaid in plasma

The result is saved as a single 3-panel figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import Normalize


CONTINUUM_PREPROCESSED_DIR = Path("/raid/scratch/mevans/Preprocessed_Datas/continuumpreprocessed")
OUTPUT_FILE = Path("/raid/scratch/mevans/colourmaptest.png")
LOW_PCT = 0
HIGH_PCT = 100
BRIGHT_SPLIT_PCT = 99.8
USE_LOG_SCALE = False

TARGET_SOURCE_IDS = [
    "105855",
    "797972",
    "683688",
]


def _find_fits_files(root: Path) -> list[Path]:
    files = list(root.glob("**/*_continuum_preproc.fits"))
    seen = set()
    unique = []
    for path in sorted(files):
        norm = path.resolve()
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(path)
    return unique


def _extract_source_id(path: Path) -> str:
    stem = path.stem
    for marker in ("_cont_", "_continuum_preproc", "_continuum", "_smoothed", "_moment0"):
        if marker in stem:
            return stem.split(marker, 1)[0]
    return stem


def _resolve_target_files(root: Path, target_ids: list[str]) -> list[Path]:
    fits_files = _find_fits_files(root)
    by_source = {}
    for path in fits_files:
        sid = _extract_source_id(path)
        by_source.setdefault(sid, path)

    chosen = []
    missing = []
    for sid in target_ids:
        if sid in by_source:
            chosen.append(by_source[sid])
        else:
            missing.append(sid)

    if missing:
        raise FileNotFoundError(f"Could not find these source IDs in {root}: {', '.join(missing)}")

    return chosen


def _load_image(path: Path) -> np.ndarray:
    with fits.open(path) as hdul:
        data = hdul[0].data
    arr = np.asarray(data, dtype=float)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image in {path}, got shape {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0)
    return arr


def _normalize_percentile(arr: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    lo = float(np.percentile(finite, low_pct))
    hi = float(np.percentile(finite, high_pct))
    if hi <= lo:
        return np.zeros_like(arr)
    norm = (arr - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)


def _log_scale_image(arr: np.ndarray) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    min_val = float(np.min(finite))
    shifted = arr - min_val if min_val <= 0.0 else arr.copy()
    shifted = np.clip(shifted, 0.0, None)
    return np.log1p(shifted)


def _plot_with_overlay(ax, img: np.ndarray, title: str) -> None:
    if USE_LOG_SCALE:
        img = _log_scale_image(img)
    norm = _normalize_percentile(img, 0, 100)
    finite = norm[np.isfinite(norm)]
    split = float(np.percentile(finite, BRIGHT_SPLIT_PCT)) if finite.size else 0.0
    split = max(split, 0.15)

    bright_pixels = norm[norm >= split]
    if bright_pixels.size:
        bright_vmin = float(np.percentile(bright_pixels, 1.0))
        bright_vmax = float(np.percentile(bright_pixels, 99.0))
        if bright_vmax <= bright_vmin:
            bright_vmin = float(bright_pixels.min())
            bright_vmax = float(bright_pixels.max())
    else:
        bright_vmin, bright_vmax = 0.0, 1.0

    # Dimmer structure in grayscale.
    from matplotlib.colors import PowerNorm

    ax.imshow(
        norm,
        cmap="plasma",
        origin="lower",
        norm=PowerNorm(gamma=1)  # <1 brightens mid/low values
    )

    # Bright emission in plasma, rescaled only over the bright part.
    bright = np.ma.masked_less(norm, split)


    ax.imshow(
        bright,
        cmap="plasma",
        origin="lower",
        norm=PowerNorm(
            gamma=0.3,                 # expands top-end contrast
            vmin=bright_vmin,
            vmax=bright_vmax,
        ),
    )

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> int:
    if not CONTINUUM_PREPROCESSED_DIR.is_dir():
        raise FileNotFoundError(f"Directory not found: {CONTINUUM_PREPROCESSED_DIR}")

    chosen = _resolve_target_files(CONTINUUM_PREPROCESSED_DIR, TARGET_SOURCE_IDS)

    fig, axes = plt.subplots(1, len(chosen), figsize=(5 * len(chosen), 5), constrained_layout=True)
    if len(chosen) == 1:
        axes = [axes]

    print("Selected sources:")
    for ax, path in zip(axes, chosen):
        img = _load_image(path)
        _plot_with_overlay(ax, img, path.stem)
        print(f"  {path}")

    fig.suptitle("Random continuum sources with grayscale + plasma overlay", fontsize=13)
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
