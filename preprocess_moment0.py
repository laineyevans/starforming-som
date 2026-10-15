"""
preprocess_moment0.py

Preprocesses moment-zero FITS maps stored in per-source subdirectories:
  /raid/scratch/mevans/moment0_with_rms/<source_id>/*.fits

Uses the same logic as preprocess_smoothed.py:
  - FOV_common is derived from original continuum images in newdata
    - Applies the same FOV trimming rule as preprocessing.py
        (drop 10 smallest + 10 largest when enough samples)
  - Two outputs are written:
      1) island-segmented version
      2) noise-removed-only version

Output layout preserves source subdirectories and creates one .npy per input FITS:
  /raid/scratch/mevans/moment_zero_preprocessed/island/<source_id>/<base>_island.npy
  /raid/scratch/mevans/moment_zero_preprocessed/noisland/<source_id>/<base>_noisland.npy
"""

import os
import glob
import numpy as np
from astropy.io import fits
from scipy import ndimage
from skimage.morphology import closing
from skimage.measure import label as skimage_label


MOMENT0_DIR     = "/raid/scratch/mevans/moment0_maps_with_rms"
CONTINUUM_PREPROC_DIR = "/raid/scratch/mevans/Preprocessed_Datas/continuumpreprocessed"
CONTINUUM_DIR   = "/raid/scratch/mevans/newdata"
ID_INFO_TABLE   = "/raid/scratch/mevans/imageproperties.dat"

OUT_ISLAND      = "/raid/scratch/mevans/Preprocessed_Datas/moment_zero_preprocessed/island_segmented"
OUT_NOISLAND    = "/raid/scratch/mevans/Preprocessed_Datas/moment_zero_preprocessed/noisland_segmented"

SQRT_PIXEL_NUMBER     = 304   # output image side length (pixels)
NOISE_MULTIPLIER      = 1     # sigma multiplier for noise threshold
THRESHOLD             = 0.075 # island threshold (post-normalisation)
MIN_ISLAND_SIZE       = 150   # starting minimum island size (pixels)
MIN_ISLAND_SIZE_FLOOR = 120   # fallback floor for iterative island search
ISLAND_SIZE_STEP      = 10    # step size for iterative fallback


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
    """Physical scale [pc/pixel] via small-angle formula."""
    if cdelt_deg is None or distance_pc is None:
        return None
    rad_per_pixel = abs(cdelt_deg) * (np.pi / 180.0)
    return rad_per_pixel * distance_pc


def crop_to_FOV(data, s_pc, FOV_common, minpix=0.0):
    """Crop or pad data to match FOV_common parsecs."""
    N_target = int(round(FOV_common / s_pc))
    N_in = data.shape[0]

    if N_in >= N_target:
        start = (N_in - N_target) // 2
        return data[start:start + N_target, start:start + N_target]

    out = np.full((N_target, N_target), minpix, dtype=float)
    start = (N_target - N_in) // 2
    out[start:start + N_in, start:start + N_in] = data
    return out


def rescale_to_fixed_size(data, N_target):
    """Zoom data to an N_target x N_target array."""
    zoom_factor = N_target / data.shape[0]
    return ndimage.zoom(data, zoom_factor, order=1)


def island_segmentation(data, threshold, minimum_island_size):
    """Return list of boolean masks for islands above threshold."""
    mask = closing(data > threshold)
    labels_map, n_labels = skimage_label(mask, return_num=True, connectivity=2)
    islands = []
    for i in range(1, n_labels + 1):
        region = labels_map == i
        if np.sum(region) >= minimum_island_size:
            islands.append(region)
    return islands


def find_continuum_file(source_id, cont_dir):
    """Return path to continuum FITS for source_id, or None."""
    pattern = os.path.join(cont_dir, f"{source_id}_cont_*.fits")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    for fname in os.listdir(cont_dir):
        if fname.startswith(source_id) and fname.endswith(".fits"):
            return os.path.join(cont_dir, fname)
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

def get_source_dirs(root_dir):
    """Return sorted list of per-source directories under root_dir."""
    out = []
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            out.append(entry.path)
    return sorted(out)


def get_source_fits(source_dir):
    """Return FITS files inside one source directory."""
    fits_files = sorted(glob.glob(os.path.join(source_dir, "*.fits")))
    return fits_files


def write_preprocessing_info(out_dir, variant_name, fov_common, n_source_dirs, n_valid_sources, n_ok, n_skip):
    """Write preprocessing settings and run summary to preprocessing_info.txt."""
    info_path = os.path.join(out_dir, "preprocessing_info.txt")
    with open(info_path, "w") as f:
        f.write("Moment-zero preprocessing configuration\n")
        f.write("=====================================\n")
        f.write(f"output_variant: {variant_name}\n")
        f.write(f"moment0_dir: {MOMENT0_DIR}\n")
        f.write(f"continuum_dir: {CONTINUUM_DIR}\n")
        f.write(f"id_info_table: {ID_INFO_TABLE}\n")
        f.write(f"output_dir: {out_dir}\n")
        f.write("\n")
        f.write(f"sqrt_pixel_number: {SQRT_PIXEL_NUMBER}\n")
        f.write(f"noise_multiplier: {NOISE_MULTIPLIER}\n")
        f.write(f"threshold: {THRESHOLD}\n")
        f.write(f"minimum_island_size: {MIN_ISLAND_SIZE}\n")
        f.write(f"min_island_size_floor: {MIN_ISLAND_SIZE_FLOOR}\n")
        f.write(f"island_size_step: {ISLAND_SIZE_STEP}\n")
        f.write(f"FOV_common_pc: {fov_common:.6f}\n")
        f.write("\n")
        f.write(f"source_dirs_found: {n_source_dirs}\n")
        f.write(f"valid_sources_processed: {n_valid_sources}\n")
        f.write(f"files_saved: {n_ok}\n")
        f.write(f"files_skipped: {n_skip}\n")



def main():
    os.makedirs(OUT_ISLAND, exist_ok=True)
    os.makedirs(OUT_NOISLAND, exist_ok=True)

    source_dirs = get_source_dirs(MOMENT0_DIR)
    print(f"Found {len(source_dirs)} source directories")


    entries = []   # (source_id, moment0_fits)
    fov_commons = []
    target_sides = []

    for sdir in source_dirs:
        source_id = os.path.basename(sdir)
        fits_list = get_source_fits(sdir)

        if not fits_list:
            print(f"  [skip] no FITS files in '{source_id}'")
            continue

        dist = extract_distance(source_id, ID_INFO_TABLE)
        if dist is None:
            print(f"  [skip] no distance entry for source '{source_id}'")
            continue

        # Continuum preprocessing metadata
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
        entries.append((source_id, fits_list, float(FOV_common), int(target_side)))

    if not entries:
        raise RuntimeError("No valid moment0 entries with continuum metadata found — run continuum preprocessing first.")

    FOV_common = float(np.median(fov_commons))
    target_side = int(np.median(target_sides))
    print(f"FOV_common from continuum headers: {FOV_common:.4f} pc; target_side: {target_side}")
    print(f"Processing {len(entries)} sources...\n")

    n_ok = 0
    n_skip = 0

    for source_id, fits_list, _, _ in entries:
        out_src_island = os.path.join(OUT_ISLAND, source_id)
        out_src_noisland = os.path.join(OUT_NOISLAND, source_id)
        os.makedirs(out_src_island, exist_ok=True)
        os.makedirs(out_src_noisland, exist_ok=True)

        for fpath in fits_list:
            base = os.path.splitext(os.path.basename(fpath))[0]

            try:
                with fits.open(fpath) as hdul:
                    data_raw = np.squeeze(hdul[0].data).astype(float)
                    hdr = hdul[0].header

                # Use each moment0 map's own pixel scale for FOV matching
                dist = extract_distance(source_id, ID_INFO_TABLE)
                m_cdelt = hdr.get("CDELT1")
                s_pc_moment0 = pc_per_pixel(m_cdelt, dist)
                if s_pc_moment0 is None:
                    print(f"  [skip] cannot compute moment0 s_pc for '{source_id}' file '{base}'")
                    n_skip += 1
                    continue

                # Noise value: prefer AGSTDRES_SMO, then AGSTDRES
                noise = hdr.get("AGSTDRES_SMO") or hdr.get("AGSTDRES")
                if noise is None:
                    print(f"  [skip] no noise header for '{source_id}' file '{base}'")
                    n_skip += 1
                    continue

                data = np.nan_to_num(data_raw, nan=0.0)

                # ---- Noise removal + normalisation ----
                data[data <= NOISE_MULTIPLIER * noise] = 0
                max_val = np.nanmax(data)
                if max_val == 0 or not np.isfinite(max_val):
                    print(f"  [skip] zero/invalid max after noise removal for '{source_id}' file '{base}'")
                    n_skip += 1
                    continue
                data = data / max_val

                # ---- No-island version ----
                data_noisland = data.copy()

                # ---- Island version with fallback ----
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
                    print(
                        f"  [warn] no islands found for '{source_id}' file '{base}'; "
                        "island version = full noise-removed image"
                    )
                    data_island = data.copy()

                # ---- FOV match + resize, then save both ----
                for data_proc, out_dir, suffix in [
                    (data_island, out_src_island, "island"),
                    (data_noisland, out_src_noisland, "noisland"),
                ]:
                    finite_vals = data_proc[np.isfinite(data_proc)]
                    minpix_val = float(np.median(finite_vals)) if finite_vals.size else 0.0
                    img_fov = crop_to_FOV(data_proc, s_pc_moment0, FOV_common, minpix=minpix_val)
                    img_scaled = rescale_to_fixed_size(img_fov, target_side)
                    out_path = os.path.join(out_dir, f"{base}_{suffix}.npy")
                    np.save(out_path, img_scaled.astype(np.float32))

                n_ok += 1

            except Exception as e:
                print(f"  [error] {source_id} / {base}: {e}")
                n_skip += 1

    print(f"\nDone. Saved {n_ok} files, skipped {n_skip}.")
    print(f"  Island output    : {OUT_ISLAND}")
    print(f"  No-island output : {OUT_NOISLAND}")

    write_preprocessing_info(
        out_dir=OUT_ISLAND,
        variant_name="island",
        fov_common=FOV_common,
        n_source_dirs=len(source_dirs),
        n_valid_sources=len(entries),
        n_ok=n_ok,
        n_skip=n_skip,
    )
    write_preprocessing_info(
        out_dir=OUT_NOISLAND,
        variant_name="noisland",
        fov_common=FOV_common,
        n_source_dirs=len(source_dirs),
        n_valid_sources=len(entries),
        n_ok=n_ok,
        n_skip=n_skip,
    )


if __name__ == "__main__":
    main()
