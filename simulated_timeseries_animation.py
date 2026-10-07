"""Create a GIF animation for simulated FITS time series data.

The script scans files in a directory such as:

    /raid/scratch/hannahb/simulated_data/

and expects filenames like:

    image_rfl10_update_cnt_0051_1362um_incl_90_phi_-90_1024pixel_jyPixel.fits

It groups images by frame number and viewing angle, then generates an animation
with three side-by-side panels for the requested angle combinations:

    (incl=90, phi=-90)
    (incl=90, phi=0)
    (incl=0, phi=0)

Each animation frame shows the same frame number across all three views.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from astropy.io import fits


TARGET_COMBOS: List[Tuple[int, int]] = [(90, -90), (90, 0), (0, 0)]
TARGET_LABELS: Dict[Tuple[int, int], str] = {
    (90, -90): "incl 90, phi -90",
    (90, 0): "incl 90, phi 0",
    (0, 0): "incl 0, phi 0",
}

FILENAME_RE = re.compile(
    r"_update_cnt_(?P<frame>\d+)_(?P<wavelength>\d+)um_incl_(?P<incl>-?\d+)_phi_(?P<phi>-?\d+)_"
)


@dataclass(frozen=True)
class SimulatedFrame:
    frame: int
    incl: int
    phi: int
    path: str


def parse_filename(path: str) -> Optional[SimulatedFrame]:
    base = os.path.basename(path)
    match = FILENAME_RE.search(base)
    if not match:
        return None
    return SimulatedFrame(
        frame=int(match.group("frame")),
        incl=int(match.group("incl")),
        phi=int(match.group("phi")),
        path=path,
    )


def load_image(path: str) -> np.ndarray:
    with fits.open(path) as hdul:
        data = hdul[0].data
    img = np.asarray(data, dtype=float)
    if img.ndim > 2:
        img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D FITS image, got shape {img.shape} for {path}")
    img = np.nan_to_num(img, nan=0.0)
    return img

def normalize_image(img: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Normalize one image to [0, 1] using robust per-image percentiles."""
    img = np.asarray(img, dtype=float)
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img)

    vmin = float(np.percentile(finite, low_pct))
    vmax = float(np.percentile(finite, high_pct))
    if vmax <= vmin:
        return np.zeros_like(img)

    img = (img - vmin) / (vmax - vmin)
    return np.clip(img, 0.0, 1.0)


def collect_frames(input_dir: str) -> Dict[int, Dict[Tuple[int, int], str]]:
    frame_map: Dict[int, Dict[Tuple[int, int], str]] = {}
    for root, _, files in os.walk(input_dir):
        for name in files:
            if not name.endswith(".fits"):
                continue
            path = os.path.join(root, name)
            parsed = parse_filename(path)
            if parsed is None:
                continue
            key = (parsed.incl, parsed.phi)
            frame_map.setdefault(parsed.frame, {})[key] = path
    return frame_map


def build_animation(frame_map: Dict[int, Dict[Tuple[int, int], str]], output_path: str, fps: int = 4, speed_factor: float = 1.5) -> None:
    existing_frames = sorted(frame_map.keys())
    if not existing_frames:
        raise RuntimeError("No matching frames found.")

    end_frame = existing_frames[-2] if len(existing_frames) >= 2 else existing_frames[-1]
    frames = list(range(existing_frames[0], end_frame + 1))


    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    cmap = "inferno"

    images = []
    for ax, combo in zip(axes, TARGET_COMBOS):
        ax.set_title(TARGET_LABELS[combo])
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(np.zeros((10, 10)), cmap='plasma', origin="lower", vmin=0.0, vmax=1.0)
        images.append(im)

    cbar = fig.colorbar(images[0], ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Intensity")

    suptitle = fig.suptitle("", fontsize=14)

    def update(frame_no: int):
        for ax, combo in zip(axes, TARGET_COMBOS):
            ax.clear()
            ax.set_title(TARGET_LABELS[combo])
            ax.set_xticks([])
            ax.set_yticks([])
            path = frame_map.get(frame_no, {}).get(combo)
            if path is None:
                ax.text(0.5, 0.5, "missing frame", ha="center", va="center", transform=ax.transAxes)
                continue
            try:
                img = load_image(path)
            except Exception as exc:
                ax.text(0.5, 0.5, f"error\n{exc}", ha="center", va="center", transform=ax.transAxes, fontsize=9)
                continue
            img = normalize_image(img, low_pct=1.0, high_pct=99.0)
            ax.imshow(img, cmap=cmap, origin="lower", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"Frame {frame_no:04d}")
        suptitle.set_text(f"Simulated time series frame {frame_no:04d}")
        return axes

    interval_ms = max(1, int(round(250 / speed_factor)))
    writer_fps = max(1, int(round(fps * speed_factor)))
    final_block = frames[-10:] if len(frames) >= 10 else frames
    display_frames = frames[:-len(final_block)] + [frame_no for frame_no in final_block for _ in range(writer_fps)]
    anim = FuncAnimation(fig, update, frames=display_frames, interval=interval_ms, blit=False, repeat=True)
    writer = PillowWriter(fps=writer_fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a GIF animation for the simulated FITS time series.")
    parser.add_argument(
        "--input-dir",
        default="/raid/scratch/hannahb/simulated_data",
        help="Directory containing the simulated FITS files.",
    )
    parser.add_argument(
        "--output",
        default="/raid/scratch/mevans/simulated_time_series.gif",
        help="Output GIF path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second for the GIF.",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=1.5,
        help="Speed multiplier for the animation.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    frame_map = collect_frames(args.input_dir)
    if not frame_map:
        raise RuntimeError(f"No FITS files matching the expected pattern were found in {args.input_dir}")

    missing_summary = []
    for frame_no in sorted(frame_map.keys()):
        combos = frame_map[frame_no]
        missing = [TARGET_LABELS[c] for c in TARGET_COMBOS if c not in combos]
        if missing:
            missing_summary.append((frame_no, missing))

    if missing_summary:
        print("Some frames are missing one or more requested angle combinations:")
        for frame_no, missing in missing_summary[:10]:
            print(f"  frame {frame_no:04d}: missing {', '.join(missing)}")
        if len(missing_summary) > 10:
            print(f"  ... and {len(missing_summary) - 10} more frames")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    build_animation(frame_map, args.output, fps=args.fps, speed_factor=args.speed_factor)
    print(f"Saved GIF to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
