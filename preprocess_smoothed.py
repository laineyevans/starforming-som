"""
preprocess_smoothed.py

Preprocesses all smoothed FITS files in smoothed_7mtm1tm2_noiseremoved0
using the same logic as preprocessing.py, but:
  - FOV_common is derived from the original continuum images in newdata
    (so smoothed outputs share the same physical field of view as the
     preprocessed continuum images)
    - Applies the same FOV trimming rule as preprocessing.py
        (drop 10 smallest + 10 largest when enough samples)

Outputs two directories:
  smoothed_preprocessed_island/    - noise removal + island segmentation + FOV match
  smoothed_preprocessed_nosisland/ - noise removal only + FOV match (no segmentation)

Each source is saved as a 2-D float32 .npy file:
  {source_id}_smoothed_island.npy
  {source_id}_smoothed_noisland.npy
"""

import os
import glob
import numpy as np
from astropy.io import fits
from scipy import ndimage
from skimage.morphology import closing
from skimage.measure import label as skimage_label

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SMOOTHED_DIR    = "/raid/scratch/mevans/smoothed_7mtm1tm2_noiseremoved0"
CONTINUUM_PREPROC_DIR = "/raid/scratch/mevans/Preprocessed_Datas/continuumpreprocessed"
CONTINUUM_DIR   = "/raid/scratch/mevans/newdata"
ID_INFO_TABLE   = "/raid/scratch/mevans/imageproperties.dat"
OUT_ISLAND      = "/raid/scratch/mevans/Preprocessed_Datas/smoothed_preprocessed_island"
OUT_NOISLAND    = "/raid/scratch/mevans/Preprocessed_Datas/smoothed_preprocessed_nosisland"

SQRT_PIXEL_NUMBER   = 304   # output image side length (pixels)
NOISE_MULTIPLIER    = 1     # sigma multiplier for noise threshold
THRESHOLD           = 0.025 # island segmentation threshold (post-normalisation)
MIN_ISLAND_SIZE     = 150   # starting minimum island size (pixels)
MIN_ISLAND_SIZE_FLOOR = 120 # fallback floor for iterative island search
ISLAND_SIZE_STEP    = 10    # step size for iterative fallback

# ---------------------------------------------------------------------------
# Helpers (mirrored from preprocessing.py)
# ---------------------------------------------------------------------------

def extract_distance(source_id, id_info_table):
    """Return heliocentric distance [pc] for source_id from the info table."""
    source_id = str(source_id).strip()
    with open(id_info_table, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            if parts[3] == source_id:
                try:
                    return float(parts[8])
                except ValueError:
                    return None
    return None


def pc_per_pixel(cdelt_deg, distance_pc):
    """Physical scale [pc/pixel] via the small-angle formula."""
    if cdelt_deg is None or distance_pc is None:
        return None
    rad_per_pixel = abs(cdelt_deg) * (np.pi / 180.0)
    return rad_per_pixel * distance_pc


def crop_to_FOV(data, s_pc, FOV_common, minpix=0.0):
    """Crop or zero-pad *data* so it covers FOV_common parsecs."""
    N_target = int(round(FOV_common / s_pc))
    N_in = data.shape[0]
    if N_in >= N_target:
        start = (N_in - N_target) // 2
        return data[start : start + N_target, start : start + N_target]
    out = np.full((N_target, N_target), minpix, dtype=float)
    start = (N_target - N_in) // 2
    out[start : start + N_in, start : start + N_in] = data
    return out


def rescale_to_fixed_size(data, N_target):
    """Zoom *data* to an N_target × N_target array."""
    zoom_factor = N_target / data.shape[0]
    return ndimage.zoom(data, zoom_factor, order=1)


def island_segmentation(data, threshold, minimum_island_size):
    """Return a list of boolean masks for islands above threshold."""
    mask = closing(data > threshold)
    labels_map, n_labels = skimage_label(mask, return_num=True, connectivity=2)
    islands = []
    for i in range(1, n_labels + 1):
        region = labels_map == i
        if np.sum(region) >= minimum_island_size:
            islands.append(region)
    return islands


def extract_source_id(path):
    """Extract the numeric / alphanumeric source ID from a smoothed filename."""
    base = os.path.basename(path)
    if "_smoothed_" in base:
        return base.split("_smoothed_")[0]
    return os.path.splitext(base)[0]


def find_continuum_file(source_id, cont_dir):
    """Return the path of the continuum FITS for *source_id*, or None."""
    pattern = os.path.join(cont_dir, f"{source_id}_cont_*.fits")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    # fallback: any file starting with source_id
    for f in os.listdir(cont_dir):
        if f.startswith(source_id) and f.endswith(".fits"):
            return os.path.join(cont_dir, f)
    return None


def load_continuum_metadata(source_id):
    """Load continuum preprocessed FITS header to retrieve FOV_common and target side.

    Returns (hdr, path) or (None, None) if not found.
    """
    # Layout: <continuumpreprocessed>/<full_cont_filename>/<full_cont_filename>_continuum_preproc.fits
    pattern = os.path.join(CONTINUUM_PREPROC_DIR, "*", "*_continuum_preproc.fits")
    for candidate in glob.glob(pattern):
        base = os.path.basename(candidate)
        if base.startswith(source_id):
            with fits.open(candidate) as hdul:
                hdr = hdul[0].header.copy()
            return hdr, candidate

    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_ISLAND, exist_ok=True)
    os.makedirs(OUT_NOISLAND, exist_ok=True)

    smoothed_files = sorted(glob.glob(os.path.join(SMOOTHED_DIR, "*.fits")))
    print(f"Found {len(smoothed_files)} smoothed FITS files")

    # ------------------------------------------------------------------
    # Pass 1: gather per-source metadata using continuum preprocessed headers
    # ------------------------------------------------------------------
    entries = []   # (spath, source_id, s_pc_smoothed)
    fov_commons = []
    target_sides = []

    for spath in smoothed_files:
        source_id = extract_source_id(spath)

        dist = extract_distance(source_id, ID_INFO_TABLE)
        if dist is None:
            print(f"  [skip] no distance entry for source '{source_id}'")
            continue

        # Smoothed pixel scale
        with fits.open(spath) as hdul:
            s_cdelt = hdul[0].header.get("CDELT1")
        s_pc_smoothed = pc_per_pixel(s_cdelt, dist)
        if s_pc_smoothed is None:
            print(f"  [skip] cannot compute s_pc for smoothed '{source_id}'")
            continue

        # Continuum preprocessing metadata (source-specific)
        c_hdr, c_path = load_continuum_metadata(source_id)
        if c_hdr is None:
            print(f"  [skip] no continuum preproc FITS found for '{source_id}'")
            continue

        FOV_common = c_hdr.get("FOVCOMM")
        target_side = c_hdr.get("TARGNPIX")
        if FOV_common is None or target_side is None:
            print(f"  [skip] missing FOV/TARGNPIX in continuum header for '{source_id}'")
            continue

        fov_commons.append(float(FOV_common))
        target_sides.append(int(target_side))
        entries.append((spath, source_id, s_pc_smoothed, float(FOV_common), int(target_side)))

    if not entries:
        raise RuntimeError("No valid smoothed entries with continuum metadata found; run continuum preprocessing first.")

    # Enforce consistent FOV_common and target_side across all entries
    FOV_common = float(np.median(fov_commons))
    target_side = int(np.median(target_sides))
    print(f"FOV_common from continuum headers: {FOV_common:.4f} pc; target_side: {target_side}")
    print(f"Processing {len(entries)} sources...\n")

    # ------------------------------------------------------------------
    # Pass 2: process each smoothed image
    # ------------------------------------------------------------------
    n_ok = 0
    n_skip = 0

    for spath, source_id, s_pc_smoothed, _, _ in entries:

        try:
            with fits.open(spath) as hdul:
                data_raw = np.squeeze(hdul[0].data).astype(float)
                hdr = hdul[0].header

            # Noise value — prefer AGSTDRES_SMO, fall back to AGSTDRES
            noise = hdr.get("AGSTDRES_SMO") or hdr.get("AGSTDRES")
            if noise is None:
                print(f"  [skip] no noise header for '{source_id}'")
                n_skip += 1
                continue

            data = np.nan_to_num(data_raw, nan=0.0)

            # ---- Noise removal and normalisation ----
            data[data <= NOISE_MULTIPLIER * noise] = 0
            max_val = np.nanmax(data)
            if max_val == 0 or not np.isfinite(max_val):
                print(f"  [skip] zero / invalid max after noise removal for '{source_id}'")
                n_skip += 1
                continue
            data = data / max_val   # normalise to [0, 1]

            # ---- Version WITHOUT island segmentation ----
            data_noisland = data.copy()

            # ---- Version WITH island segmentation ----
            min_size = MIN_ISLAND_SIZE
            islands = []
            while min_size >= MIN_ISLAND_SIZE_FLOOR:
                islands = island_segmentation(data, THRESHOLD, min_size)
                if islands:
                    break
                min_size -= ISLAND_SIZE_STEP

            if islands:
                mask = np.zeros_like(data, dtype=bool)
                for isl in islands:
                    mask |= isl
                data_island = data * mask
            else:
                print(f"  [warn] no islands found for '{source_id}'; island version = full noise-removed image")
                data_island = data.copy()

            # ---- FOV matching ----
            # Use the smoothed map pixel scale here (same rule as preprocessing.py:
            # crop/pad by each image's own pc-per-pixel to the continuum FOV_common).
            for data_proc, out_dir, suffix in [
                (data_island,   OUT_ISLAND,   "smoothed_island"),
                (data_noisland, OUT_NOISLAND, "smoothed_noisland"),
            ]:
                finite_vals = data_proc[np.isfinite(data_proc)]
                minpix_val = float(np.median(finite_vals)) if finite_vals.size else 0.0
                img_fov = crop_to_FOV(data_proc, s_pc_smoothed, FOV_common, minpix=minpix_val)
                img_scaled = rescale_to_fixed_size(img_fov, target_side)
                out_path = os.path.join(out_dir, f"{source_id}_{suffix}.npy")
                np.save(out_path, img_scaled.astype(np.float32))

            n_ok += 1

        except Exception as e:
            print(f"  [error] {source_id}: {e}")
            n_skip += 1

    print(f"\nDone. Saved {n_ok} sources, skipped {n_skip}.")
    print(f"  Island output    : {OUT_ISLAND}")
    print(f"  No-island output : {OUT_NOISLAND}")


if __name__ == "__main__":
    main()
