#!/usr/bin/env python3
"""Plot nine source examples per evolutionary stage and preprocessed channel.

Example:
    python plot_evolutionary_stage_examples.py \
        --run-dir /path/to/run \
        --preprocessed-dir /path/to/continuum_preprocessed \
        --smoothed-dir /path/to/smoothed \
        --molecular-dir /path/to/moment_zero_preprocessed

The script writes 25 PNG files to ``evolutionary_stage_examples`` by default:
five stages (Q, P, Y, H, na) times five channels.
"""

import argparse
import csv
import glob
import os
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import PowerNorm


STAGES = ("Q", "P", "Y", "H", "na")
CHANNELS = ("continuum", "smoothed", "H2CO", "SiO", "DCN")
LOW_PCT = 0
HIGH_PCT = 100
BRIGHT_SPLIT_PCT = 99.8
USE_LOG_SCALE = False

# Default paths used by the current pipeline run.  The command can therefore
# be run directly from the repository with no arguments.
DEFAULT_RUN_DIR = "/raid/scratch/mevans/run_14-10_14:00"
DEFAULT_PREPROCESSED_DIR = (
    "/raid/scratch/mevans/Preprocessed_Datas/continuumpreprocessed"
)
DEFAULT_SMOOTHED_DIR = (
    "/raid/scratch/mevans/Preprocessed_Datas/smoothed_preprocessed_island"
)
DEFAULT_MOLECULAR_DIR = (
    "/raid/scratch/mevans/Preprocessed_Datas/moment_zero_preprocessed/island"
)
DEFAULT_STAGE_TABLE = "/raid/scratch/mevans/table1.dat"


def source_id(filename):
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    for marker in ("_cont_", "_continuum_preproc", "_continuum",
                   "_smoothed", "_moment0"):
        if marker in stem:
            return stem.split(marker, 1)[0]
    return stem


def read_assignments(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Assignments CSV not found: {path}")
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            filename = row.get("filename", "")
            if filename:
                rows.append({
                    "filename": filename,
                    "label": row.get("label", ""),
                    "source_id": source_id(filename),
                })
    return rows


def read_stage_table(path):
    """Read source ID and evolutionary stage from the project stage table."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Stage table not found: {path}")
    stages = {}
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 11:
                continue
            sid = fields[3]
            # The project table uses column 17 when present, otherwise 11.
            stage = fields[17] if len(fields) > 17 else fields[10]
            if stage in STAGES:
                stages[sid] = stage
    if not stages:
        raise ValueError(f"No Q/P/Y/H/na source stages found in {path}")
    return stages


def load_run_config(run_dir):
    values = {}
    path = os.path.join(run_dir, "run_info.txt")
    if not os.path.isfile(path):
        return values
    with open(path) as handle:
        for line in handle:
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()
    return values


def apply_flip(image, label):
    if "[hflip]" in label:
        image = np.flipud(image)
    if "[vflip]" in label:
        image = np.fliplr(image)
    if "[hvflip]" in label:
        image = np.flipud(np.fliplr(image))
    if "[rot" in label:
        image = np.rot90(image)
    return image


def load_array(path):
    if path.endswith((".fits", ".fit", ".fts")):
        with fits.open(path) as hdul:
            image = hdul[0].data
    else:
        image = np.load(path, allow_pickle=False)
    image = np.squeeze(np.asarray(image, dtype=float))
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image in {path}, got {image.shape}")
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)


def find_continuum(preprocessed_dir, sid):
    patterns = (
        os.path.join(preprocessed_dir, "**", f"{sid}_continuum_preproc.fits"),
        os.path.join(preprocessed_dir, "**", f"{sid}*_continuum_preproc.fits"),
    )
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def find_smoothed(smoothed_dir, sid):
    if not smoothed_dir:
        return None
    for name in (
        f"{sid}_smoothed_noiseremoved1sigma.npy",
        f"{sid}_smoothed_island.npy",
    ):
        path = os.path.join(smoothed_dir, name)
        if os.path.isfile(path):
            return path
    return None


def find_molecular(molecular_dir, sid, line):
    if not molecular_dir:
        return None
    pattern = os.path.join(
        molecular_dir, "**", f"*{sid}*{line}*_moment0*.npy"
    )
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def channel_path(channel, sid, config, args):
    if channel == "continuum":
        return find_continuum(
            args.preprocessed_dir or config.get("continuum_preprocessed_dir", ""),
            sid,
        )
    if channel == "smoothed":
        return find_smoothed(
            args.smoothed_dir or config.get("smoothed_image_file_dir"), sid
        )
    return find_molecular(
        args.molecular_dir or config.get("molecular_line_file_dir"), sid, channel
    )


def normalize_percentile(image, low_pct=LOW_PCT, high_pct=HIGH_PCT):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image)
    low = float(np.percentile(finite, low_pct))
    high = float(np.percentile(finite, high_pct))
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def log_scale_image(image):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image)
    minimum = float(np.min(finite))
    shifted = image - minimum if minimum <= 0.0 else image.copy()
    return np.log1p(np.clip(shifted, 0.0, None))


def plot_with_overlay(axis, image, title):
    if USE_LOG_SCALE:
        image = log_scale_image(image)
    normalized = normalize_percentile(image)
    finite = normalized[np.isfinite(normalized)]
    split = float(np.percentile(finite, BRIGHT_SPLIT_PCT)) if finite.size else 0.0
    split = max(split, 0.15)

    bright = np.ma.masked_less(normalized, split)
    bright_pixels = normalized[normalized >= split]
    if bright_pixels.size:
        bright_vmin = float(np.percentile(bright_pixels, 1.0))
        bright_vmax = float(np.percentile(bright_pixels, 99.0))
        if bright_vmax <= bright_vmin:
            bright_vmin = float(bright_pixels.min())
            bright_vmax = float(bright_pixels.max())
    else:
        bright_vmin, bright_vmax = 0.0, 1.0

    axis.imshow(
        normalized,
        cmap="plasma",
        origin="lower",
        norm=PowerNorm(gamma=1),
    )
    axis.imshow(
        bright,
        cmap="plasma",
        origin="lower",
        norm=PowerNorm(
            gamma=0.3,
            vmin=bright_vmin,
            vmax=bright_vmax,
        ),
    )
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def plot_stage_channel(stage, channel, entries, stage_by_source, args, config,
                       output_dir):
    candidates = []
    seen = set()
    for entry in entries:
        sid = entry["source_id"]
        if sid in seen or stage_by_source.get(sid) != stage:
            continue
        path = channel_path(channel, sid, config, args)
        if path is None:
            continue
        seen.add(sid)
        candidates.append((entry, path))
        if len(candidates) == 9:
            break

    if len(candidates) < 9:
        warnings.warn(
            f"{stage}/{channel}: found only {len(candidates)} usable sources; "
            "remaining panels will be blank.",
            RuntimeWarning,
        )

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = np.asarray(axes).ravel()
    for axis in axes:
        axis.axis("off")

    for axis, (entry, path) in zip(axes, candidates):
        try:
            image = apply_flip(load_array(path), entry["label"])
        except (OSError, ValueError) as exc:
            warnings.warn(f"Skipping {path}: {exc}", RuntimeWarning)
            continue
        plot_with_overlay(axis, image, entry["source_id"])

    fig.suptitle(f"Evolutionary stage {stage} — {channel} — 9 examples")
    fig.tight_layout()
    output_path = os.path.join(output_dir, f"stage_{stage}_{channel}.png")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--preprocessed-dir", default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--smoothed-dir", default=DEFAULT_SMOOTHED_DIR)
    parser.add_argument("--molecular-dir", default=DEFAULT_MOLECULAR_DIR)
    parser.add_argument("--stage-table", default=DEFAULT_STAGE_TABLE)
    parser.add_argument("--output-dir", default="evolutionary_stage_examples")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_run_config(args.run_dir)
    assignments_path = os.path.join(args.run_dir, "source_cluster_assignments.csv")
    entries = read_assignments(assignments_path)
    stage_by_source = read_stage_table(args.stage_table)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for stage in STAGES:
        for channel in CHANNELS:
            plot_stage_channel(
                stage, channel, entries, stage_by_source, args, config, output_dir
            )
    print(f"Saved evolutionary-stage examples to {output_dir}")


if __name__ == "__main__":
    main()
