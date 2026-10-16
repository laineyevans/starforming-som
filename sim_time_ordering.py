"""Diagnostics for ordering and colour structure in simulated-data runs."""

import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


_FRAME_RE = re.compile(r"update_cnt_(\d+)")
_INCL_RE = re.compile(r"incl_(\d+)")
_PHI_RE = re.compile(r"phi_(-?\d+)")


def parse_sim_metadata(filenames, labels):
    """Parse simulation frame/view/flip metadata without rejecting real data."""
    frames, incls, phis = [], [], []
    for filename in filenames:
        basename = os.path.basename(str(filename))
        frame = _FRAME_RE.search(basename)
        incl = _INCL_RE.search(basename)
        phi = _PHI_RE.search(basename)
        frames.append(float(frame.group(1)) if frame else np.nan)
        incls.append(float(incl.group(1)) if incl else np.nan)
        phis.append(float(phi.group(1)) if phi else np.nan)
    flip_types = []
    for label in labels:
        text = str(label).lower()
        if "[hvflip]" in text:
            flip_types.append("hvflip")
        elif "[hflip]" in text:
            flip_types.append("hflip")
        elif "[vflip]" in text:
            flip_types.append("vflip")
        else:
            flip_types.append("original")
    views = [
        f"incl {int(incl)} phi {int(phi)}" if np.isfinite(incl) and np.isfinite(phi)
        else "incl nan phi nan"
        for incl, phi in zip(incls, phis)
    ]
    tracks = [f"{view} | {flip}" for view, flip in zip(views, flip_types)]
    return {
        "frame": np.asarray(frames, dtype=float),
        "incl": np.asarray(incls, dtype=float),
        "phi": np.asarray(phis, dtype=float),
        "flip_type": np.asarray(flip_types, dtype=object),
        "view": np.asarray(views, dtype=object),
        "track": np.asarray(tracks, dtype=object),
    }


def _save_scatter(path, embedding, values, title, cmap="viridis", categorical=False,
                  labels=None, center_zero=False, categorical_cmap="tab10",
                  cbar_label=None):
    """Local scatter helper for the optional command-line workflow."""
    order = np.random.default_rng(42).permutation(len(embedding))
    fig, ax = plt.subplots(figsize=(7, 6))
    if categorical:
        categories = list(dict.fromkeys(values))
        colors = plt.get_cmap(categorical_cmap)(
            np.arange(len(categories)) % plt.get_cmap(categorical_cmap).N)
        for category, color in zip(categories, colors):
            mask = np.asarray([value == category for value in values])
            selected = order[mask[order]]
            ax.scatter(embedding[selected, 0], embedding[selected, 1], s=6,
                       alpha=.6, color=color, rasterized=True, label=str(category))
        ax.legend(markerscale=2, fontsize=8)
    else:
        scatter = ax.scatter(embedding[order, 0], embedding[order, 1],
                             c=np.asarray(values)[order], cmap=cmap, s=6,
                             alpha=.6, rasterized=True)
        fig.colorbar(scatter, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_sim_colourings(out_dir, key, embedding, meta_subset):
    """Save frame, view, flip, and track colourings for one embedding."""
    os.makedirs(out_dir, exist_ok=True)
    _save_scatter(
        os.path.join(out_dir, key + "_sim_frame.png"), embedding,
        meta_subset["frame"], key + " simulated frame",
        cbar_label="frame (linear in collapse time)",
    )
    _save_scatter(
        os.path.join(out_dir, key + "_sim_viewangle.png"), embedding,
        meta_subset["view"], key + " simulated view angle", categorical=True,
    )
    _save_scatter(
        os.path.join(out_dir, key + "_sim_flip.png"), embedding,
        meta_subset["flip_type"], key + " simulated flip type", categorical=True,
    )
    _save_scatter(
        os.path.join(out_dir, key + "_sim_track.png"), embedding,
        meta_subset["track"], key + " simulated track", categorical=True,
        categorical_cmap="tab20",
    )


def plot_time_ordering(preds, labels, filenames, config, coords, out_dir):
    """Plot coordinate/frame ordering and write per-track Spearman correlations."""
    meta = parse_sim_metadata(filenames, labels)
    coords = np.asarray(coords, dtype=float)
    frame = meta["frame"]
    finite = np.isfinite(frame) & np.all(np.isfinite(coords), axis=1)
    centered = coords[finite] - np.mean(coords[finite], axis=0)
    if len(centered):
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        pc1 = centered @ vh[0]
    else:
        pc1 = np.empty(0)
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.get_cmap("tab10")
    track_indices = {}
    for index, track in enumerate(meta["track"]):
        if finite[index]:
            track_indices.setdefault(track, []).append(index)
    track_indices = {track: indices for track, indices in track_indices.items()
                     if len(indices) >= 5}
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=False)
    flip_order = ("original", "hflip", "vflip", "hvflip")
    rows = []
    for col, flip in enumerate(flip_order):
        for track, indices in track_indices.items():
            if not track.endswith(" | " + flip):
                continue
            idx = np.asarray(indices)
            jitter = np.random.default_rng(42 + col).uniform(-.15, .15, len(idx))
            view = str(meta["view"][idx[0]])
            color = colors(hash(view) % 10)
            rhos = [spearmanr(frame[idx], coords[idx, axis]).statistic for axis in (0, 1)]
            rhos.append(spearmanr(frame[idx], pc1[np.searchsorted(np.flatnonzero(finite), idx)]).statistic)
            rows.append((track, len(idx), *rhos))
            for row, values in enumerate((coords[idx, 0], coords[idx, 1],
                                          pc1[np.searchsorted(np.flatnonzero(finite), idx)])):
                axes[row, col].scatter(frame[idx] + jitter, values, s=10,
                                       alpha=.5, color=color, label=f"{view} (rho={rhos[row]:.2f})")
        for row in range(3):
            axes[row, col].set_title(flip)
            axes[row, col].set_ylabel(("x", "y", "pc1")[row])
            if axes[row, col].get_legend_handles_labels()[0]:
                axes[row, col].legend(fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "time_ordering.png"), dpi=200)
    plt.close(fig)
    with open(os.path.join(out_dir, "time_ordering_spearman.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("track", "n", "rho_x", "rho_y", "rho_pc1"))
        pooled = np.flatnonzero(finite)
        pooled_rhos = [
            spearmanr(frame[pooled], coords[pooled, axis]).statistic for axis in (0, 1)
        ]
        pooled_rhos.append(spearmanr(frame[pooled], pc1).statistic)
        writer.writerow(("pooled-all", len(pooled), *pooled_rhos))
        for row in rows:
            writer.writerow(row)
        if rows:
            values = np.asarray([row[2:] for row in rows], dtype=float)
            writer.writerow(("mean |rho|", "", *np.nanmean(np.abs(values), axis=0)))


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    assignments = os.path.join(args.run_dir, "source_cluster_assignments.csv")
    with open(assignments, newline="") as handle:
        records = list(csv.DictReader(handle))
    filenames = [row["filename"] for row in records]
    labels = [row["label"] for row in records]
    meta = parse_sim_metadata(filenames, labels)
    diag_dir = os.path.join(args.run_dir, "embedding_diagnostics")
    files = sorted(name for name in os.listdir(diag_dir) if name.endswith(".npy"))
    if not files:
        raise SystemExit(f"No saved embeddings found in {diag_dir}")
    for name in files:
        embedding = np.load(os.path.join(diag_dir, name))
        if len(embedding) != len(records):
            raise SystemExit(
                f"Row count mismatch for {name}: embedding has {len(embedding)} rows, "
                f"assignments have {len(records)} rows (possibly diag_max_points subsetting)."
            )
        _save_sim_colourings(diag_dir, os.path.splitext(name)[0], embedding, meta)


if __name__ == "__main__":
    _cli()
