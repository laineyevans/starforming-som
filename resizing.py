"""Resize FITS images from smoothed outputs to 304x304 using ndimage.zoom.

Usage: python resizing.py
"""

import os
import numpy as np
from astropy.io import fits
from scipy import ndimage


# Target side length for square images after resizing
TARGET = 304

# Molecular line moment-0 map root directories
MOMENT_INPUT_ROOT = "/raid/scratch/mevans/spectral_cube_test"
MOMENT_OUTPUT_ROOT = "/raid/scratch/mevans/moment_zero_preprocessed"


def resize_fits(src_path, dst_dir, base_name, target=TARGET):
	with fits.open(src_path) as hdul:
		data = hdul[0].data

	if data is None or data.ndim < 2:
		print(f"Skipping {src_path}: data is empty or not at least 2D")
		return False

	ny, nx = data.shape[-2], data.shape[-1]
	zoom_y = target / ny
	zoom_x = target / nx

	zoom_factors = [1] * (data.ndim - 2) + [zoom_y, zoom_x]
	resized = ndimage.zoom(data, zoom_factors, order=1)

	# Ensure exact target shape on the last two axes
	resized_shape = list(resized.shape)
	if resized_shape[-2] != target or resized_shape[-1] != target:
		resized_shape[-2] = target
		resized_shape[-1] = target
		resized = resized[..., :target, :target]
	# Replace NaNs with zero
	resized = np.nan_to_num(resized, nan=0.0)

	# Normalize to 0..1 across the array
	amin = resized.min()
	amax = resized.max()
	if np.isfinite(amin) and np.isfinite(amax) and (amax > amin):
		resized = (resized - amin) / (amax - amin)
	else:
		# fallback: make a zero array if no dynamic range
		resized = np.zeros_like(resized)

	dst_path = os.path.join(dst_dir, f"{base_name}.npy")
	np.save(dst_path, resized)
	return True


def main():
	# Process molecular line moment-0 maps organised by source ID
	if not os.path.isdir(MOMENT_INPUT_ROOT):
		print(f"Skipping moment-0 processing: {MOMENT_INPUT_ROOT} does not exist")
		return

	os.makedirs(MOMENT_OUTPUT_ROOT, exist_ok=True)

	source_ids = sorted([d for d in os.listdir(MOMENT_INPUT_ROOT)
					 if os.path.isdir(os.path.join(MOMENT_INPUT_ROOT, d))])
	if not source_ids:
		print(f"No source subdirectories found in {MOMENT_INPUT_ROOT}")
		return

	for sid in source_ids:
		src_sub = os.path.join(MOMENT_INPUT_ROOT, sid)
		dst_sub = os.path.join(MOMENT_OUTPUT_ROOT, sid)
		os.makedirs(dst_sub, exist_ok=True)

		fits_files = [f for f in sorted(os.listdir(src_sub))
					 if f.lower().endswith("moment0.fits")]
		if not fits_files:
			continue

		for fname in fits_files:
			src_path = os.path.join(src_sub, fname)
			base_name, _ = os.path.splitext(fname)
			ok = resize_fits(src_path, dst_sub, base_name)
			if ok:
				print(f"[moment0] {sid}: wrote {os.path.join(dst_sub, base_name + '.npy')}")
			else:
				print(f"[moment0] {sid}: failed to resize {src_path}")


if __name__ == "__main__":
	main()
