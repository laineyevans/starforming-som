"""Compute silhouette-weighted average image for a specific neuron.

This script reads source_cluster_assignments.csv from a given run, filters
rows for one neuron, loads the corresponding FITS images, rescales them to a
common shape, and computes a silhouette-weighted mean image. The result is
saved as a PNG for quick inspection.
"""

import csv
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import zoom


# Paths and constants (adjust if you run a different checkpoint)
RUN_DIR = "/raid/scratch/mevans/run_24-03_13:19"
CSV_PATH = os.path.join(RUN_DIR, "source_cluster_assignments.csv")
DATA_ROOT = "/raid/scratch/mevans/newdata"
PREPROCESSED_PATH = "/raid/scratch/mevans/preprocessed_data/preprocessed_data.npz"
USE_PREPROCESSED = True  # toggle: True -> use preprocessed NPZ; False -> raw FITS under DATA_ROOT
TARGET_SHAPE = (305, 305)
NEURON_ID = None  # unused in batch mode; kept for legacy compatibility


def _find_image_path(filename: str, label: str) -> Optional[str]:
	"""Return the first existing FITS path for this sample.

	Tries absolute paths first; otherwise resolves under DATA_ROOT.
	"""
	candidates: List[str] = []

	for name in (filename, label):
		if not name:
			continue
		if os.path.isabs(name):
			candidates.append(name)
		else:
			candidates.append(os.path.join(DATA_ROOT, name))

	for path in candidates:
		if os.path.isfile(path):
			return path
	return None


def _load_preprocessed_map(npz_path: str) -> dict:
	"""Load preprocessed continuum arrays and map by basename for quick lookup."""
	if not os.path.isfile(npz_path):
		raise FileNotFoundError(f"Preprocessed NPZ not found: {npz_path}")
	data = np.load(npz_path, allow_pickle=True)
	# Prefer processed_2d (FOV-matched & resized); fall back to raw_datas if absent
	if "processed_2d" in data.files:
		arrs = data["processed_2d"]
	elif "raw_datas" in data.files:
		arrs = data["raw_datas"]
	else:
		raise KeyError("No processed_2d or raw_datas in preprocessed NPZ")
	files = data["filenames"]
	lookup = {}
	for fname, arr in zip(files, arrs):
		base = os.path.basename(str(fname))
		lookup[base] = np.nan_to_num(np.squeeze(arr), nan=0.0)
	return lookup


_PREPROCESSED_CACHE: Optional[dict] = None


def _get_image_from_row(fname: str, label: str) -> Optional[np.ndarray]:
	"""Return image from preprocessed NPZ (if enabled) else raw FITS."""
	global _PREPROCESSED_CACHE
	if USE_PREPROCESSED:
		if _PREPROCESSED_CACHE is None:
			_PREPROCESSED_CACHE = _load_preprocessed_map(PREPROCESSED_PATH)
		base = os.path.basename(fname)
		return _PREPROCESSED_CACHE.get(base)
	else:
		path = _find_image_path(fname, label)
		if not path:
			return None
		return _load_fits_2d(path)


def _load_fits_2d(path: str) -> np.ndarray:
	with fits.open(path) as hdul:
		data = hdul[0].data
	arr = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0)
	if arr.ndim > 2:
		arr = np.squeeze(arr)
	if arr.ndim != 2:
		raise ValueError(f"Unexpected array shape {arr.shape} in {path}")
	return arr


def _rescale_to(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
	if arr.shape == target_shape:
		return arr
	zoom_y = target_shape[0] / arr.shape[0]
	zoom_x = target_shape[1] / arr.shape[1]
	# Match preprocessing: bilinear zoom (order=1)
	return zoom(arr, (zoom_y, zoom_x), order=1)


def _normalize_01(arr: np.ndarray) -> np.ndarray:
	"""Scale array to [0, 1]; if flat, return zeros."""
	amin = float(np.nanmin(arr))
	amax = float(np.nanmax(arr))
	if not np.isfinite(amin) or not np.isfinite(amax):
		return np.zeros_like(arr)
	span = amax - amin
	if span == 0.0:
		return np.zeros_like(arr)
	return (arr - amin) / span



def load_cluster_rows(csv_path: str = CSV_PATH) -> dict:
	"""Return mapping cluster_id -> list of assignment rows."""
	if not os.path.isfile(csv_path):
		raise FileNotFoundError(f"Cluster assignment CSV not found: {csv_path}")
	clusters = {}
	with open(csv_path, newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			try:
				cid = int(row.get("cluster", -1))
			except ValueError:
				continue
			clusters.setdefault(cid, []).append(row)
	return clusters


def compute_weighted_mean(rows: List[dict], neuron_id: int):
	if not rows:
		raise RuntimeError(f"No rows found for neuron {neuron_id} in {CSV_PATH}")

	acc = None
	weight_sum = 0.0
	loaded = 0
	used = 0  # images with positive silhouette weight
	neg_rows = []  # rows with non-positive silhouettes, used only if no positives

	for row in rows:
		fname = row.get("filename", "").strip()
		label = row.get("label", "").strip()
		sil_raw = row.get("silhouette_score", "")
		try:
			sil = float(sil_raw)
		except ValueError:
			sil = np.nan
		# Clamp negatives to zero so only non-negative silhouettes contribute
		if np.isfinite(sil):
			sil = max(sil, 0.0)
		else:
			continue  # skip missing / invalid scores

		if sil <= 0.0:
			neg_rows.append(row)
			continue  # skip zero/negative now; may reuse if no positives

		img = _get_image_from_row(fname, label)
		if img is None:
			print(f"Skipping {fname}: no image found (preprocessed={USE_PREPROCESSED})")
			continue

		img = _rescale_to(img, TARGET_SHAPE)
		img = _normalize_01(img)

		if acc is None:
			acc = np.zeros(TARGET_SHAPE, dtype=float)

		acc += img * sil
		weight_sum += sil
		loaded += 1
		used += 1

	# If no positives were accumulated, fall back to non-positives shifted by +1
	if loaded == 0 and neg_rows:
		print("No positive silhouettes found; using non-positive silhouettes shifted by +1.")
		for row in neg_rows:
			fname = row.get("filename", "").strip()
			label = row.get("label", "").strip()
			sil_raw = row.get("silhouette_score", "")
			try:
				sil = float(sil_raw)
			except ValueError:
				continue
			sil = sil + 1.0  # shift up to be positive

			path = _find_image_path(fname, label)
			if not path:
				print(f"Skipping {fname}: no FITS found (checked {DATA_ROOT})")
				continue

			try:
				img = _load_fits_2d(path)
			except Exception as exc:
				print(f"Skipping {path}: {exc}")
				continue

			img = _rescale_to(img, TARGET_SHAPE)
			img = _normalize_01(img)

			if acc is None:
				acc = np.zeros(TARGET_SHAPE, dtype=float)

			acc += img * sil
			weight_sum += sil
			loaded += 1
			used += 1

	if loaded == 0 or acc is None:
		raise RuntimeError(f"No images loaded for neuron {neuron_id}")

	if weight_sum == 0:
		print("Total silhouette weight is zero; falling back to simple mean.")
		weight_sum = float(loaded)

	mean_img = acc / weight_sum
	return mean_img, used, rows


def load_top_examples(rows: List[dict], max_examples: int = 9) -> List[Tuple[np.ndarray, float, str]]:
	"""Load up to `max_examples` highest-silhouette images for the cluster.

	Returns list of (img_normalized, silhouette_score, label_for_plot).
	"""
	def _sil(row: dict) -> float:
		try:
			return float(row.get("silhouette_score", "nan"))
		except ValueError:
			return float("nan")

	sorted_rows = sorted(rows, key=_sil, reverse=True)
	examples: List[Tuple[np.ndarray, float, str]] = []
	for row in sorted_rows:
		sil = _sil(row)
		if not np.isfinite(sil):
			continue
		fname = row.get("filename", "").strip()
		label = row.get("label", "").strip()
		path = _find_image_path(fname, label)
		if not path:
			continue
		try:
			img = _get_image_from_row(fname, label)
		except Exception:
			continue
		img = _rescale_to(img, TARGET_SHAPE)
		img = _normalize_01(img)
		display_label = os.path.basename(path)
		examples.append((img, sil, display_label))
		if len(examples) >= max_examples:
			break
	return examples


def plot_mean_and_examples(mean_img: np.ndarray, used: int, examples: List[Tuple[np.ndarray, float, str]], neuron_id: int, out_path: str):
	"""Plot mean image and top example thumbnails side-by-side in one PNG."""
	fig = plt.figure(figsize=(12, 6))
	gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2])

	# Left: silhouette-weighted mean
	ax_mean = fig.add_subplot(gs[0, 0])
	mean_vmin, mean_vmax = float(np.nanmin(mean_img)), float(np.nanmax(mean_img))
	im_mean = ax_mean.imshow(mean_img, origin="lower", cmap="plasma", vmin=mean_vmin, vmax=mean_vmax)
	ax_mean.set_title(f"Neuron {neuron_id} silhouette-weighted mean (n={used})")
	fig.colorbar(im_mean, ax=ax_mean, fraction=0.046, pad=0.04, label="Intensity")

	# Right: up to 9 highest-silhouette originals
	ax_grid = fig.add_subplot(gs[0, 1])
	ax_grid.axis("off")

	if examples:
		rows = cols = 3
		inner = gs[0, 1].subgridspec(rows, cols, wspace=0.15, hspace=0.25)
		for idx, (img, sil, label) in enumerate(examples[: rows * cols]):
			i = idx // cols
			j = idx % cols
			ax = fig.add_subplot(inner[i, j])
			ax.imshow(img, origin="lower", cmap="plasma", vmin=0.0, vmax=1.0)
			title = f"S={sil:.3f}\n{label}"
			ax.set_title(title, fontsize=8)
			ax.axis("off")
	else:
		ax_grid.text(0.5, 0.5, "No examples available", ha="center", va="center")

	plt.tight_layout()
	plt.savefig(out_path, dpi=200)
	plt.close(fig)
	print(f"Saved mean + top examples for neuron {neuron_id} to {out_path} (n={used}, examples={len(examples)})")


if __name__ == "__main__":
	clusters = load_cluster_rows(CSV_PATH)
	# Pick top 5 clusters by sample count
	top = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
	if not top:
		raise RuntimeError("No clusters found in CSV")
	for cid, rows in top:
		out_png = os.path.join(RUN_DIR, f"neuron_{cid:03d}_silhouette_weighted_mean.png")
		mean_img, used, rows_for_cluster = compute_weighted_mean(rows, cid)
		examples = load_top_examples(rows_for_cluster, max_examples=9)
		plot_mean_and_examples(mean_img, used, examples, cid, out_png)