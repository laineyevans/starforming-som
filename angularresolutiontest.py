import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def extract_distance(source_id, id_info_table):
    source_id = str(source_id).strip()
    with open(id_info_table, 'r') as f:
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
    if cdelt_deg is None or distance_pc is None:
        return None
    arcsec = abs(cdelt_deg) * 3600.0
    return arcsec * (distance_pc / 206265.0)


def collect_image_properties(directory, id_info_table):
    props = []
    for filename in os.listdir(directory):
        if not filename.endswith(".fits"):
            continue

        path = os.path.join(directory, filename)
        with fits.open(path) as hd:
            hdr = hd[0].header
            data = hd[0].data

        if data is None:
            continue

        N = data.shape[0]
        source = hdr.get("OBJECT", "").strip()
        dist = extract_distance(source, id_info_table)
        cdelt = hdr.get("CDELT1")
        s_pc = pc_per_pixel(cdelt, dist)
        if s_pc is None:
            continue

        FOV_pc = N * s_pc
        props.append((filename, s_pc, FOV_pc, N, dist))
    return props


# crop central region to the smallest physical FOV
def crop_to_FOV(data, s_pc, FOV_common, minpix=0.0):
    """Crop or pad the central region to match a target physical FOV.

    If the image is larger than the target (in pixels) we take a central crop.
    If it is smaller, we pad with `minpix` to reach the target size.
    """
    # target number of pixels at this image's linear resolution
    N_target = int(round(FOV_common / s_pc))
    N_in = data.shape[0]

    if N_target <= 0:
        return data.copy()

    if N_in >= N_target:
        start = (N_in - N_target) // 2
        return data[start:start + N_target, start:start + N_target]

    # image smaller than target -> pad with minpix
    out = np.full((N_target, N_target), minpix, dtype=float)
    start = (N_target - N_in) // 2
    out[start:start + N_in, start:start + N_in] = data
    return out


# resample cropped region to 305x305 
def rescale_to_305(data, N_target=305):
    zoom_factor = N_target / data.shape[0]
    return zoom(data, zoom_factor, order=1)


def process_single_image(path, s_pc, FOV_common, dist):
    with fits.open(path) as hd:
        hdr = hd[0].header
        data = hd[0].data.astype(float)

    finite = np.isfinite(data)

    if np.any(finite):
        bg = float(np.median(data[finite]))
    else:
        bg = 0.0

    data = np.nan_to_num(data, nan=bg, posinf=bg, neginf=bg)

    cropped = crop_to_FOV(data, s_pc, FOV_common, minpix=bg)
    processed = rescale_to_305(cropped, N_target=305)

    return data, processed


def plot_pair(original, processed, label, fname, FOV, outpath):
    vmin = np.nanpercentile(original, 1)
    vmax = np.nanpercentile(original, 99)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].imshow(original, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax[0].set_title("Original")

    ax[1].imshow(processed, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax[1].set_title("Cropped to min FOV + Resized to 305×305")

    fig.suptitle(f"{label} — {fname}\nFOV = {FOV:.3f} pc")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def compare_representative_images(directory, id_info_table, outdir):
    props = collect_image_properties(directory, id_info_table)
    if len(props) < 25:
        print("Not enough images after trimming.")
        return

    # --- sort by FOV ascending ---
    props_sorted = sorted(props, key=lambda p: p[2])

    # --- remove top 10 smallest and top 10 largest ---
    props_trimmed = props_sorted[20:-20]

    print(f"Original count: {len(props)}")
    print(f"Trimmed count:  {len(props_trimmed)} (removed 10 smallest + 10 largest)\n")

    # --- recompute FOV array for trimmed set ---
    FOVs = np.array([p[2] for p in props_trimmed])
    sorted_idx = np.argsort(FOVs)

    # representative indices within trimmed set
    idx_small = sorted_idx[0]
    idx_q1 = sorted_idx[len(FOVs) // 4]
    idx_med = sorted_idx[len(FOVs) // 2]
    idx_q3 = sorted_idx[3 * len(FOVs) // 4]
    idx_large = sorted_idx[-1]

    key_list = [
        ("smallest_trimmed", idx_small),
        ("first_quartile_trimmed", idx_q1),
        ("median_trimmed", idx_med),
        ("third_quartile_trimmed", idx_q3),
        ("largest_trimmed", idx_large),
    ]

    # new common FOV based on trimmed distribution
    FOV_common = float(np.percentile(FOVs, 75))

    os.makedirs(outdir, exist_ok=True)

    for label, idx in key_list:
        fname, s_pc, FOV, N, dist = props_trimmed[idx]
        path = os.path.join(directory, fname)

        original, processed = process_single_image(path, s_pc, FOV_common, dist)

        outname = f"{label}_{fname}_processed.png"
        plot_pair(
            original,
            processed,
            label,
            fname,
            FOV,
            os.path.join(outdir, outname)
        )

    print("Saved updated plots after removing extreme FOVs.")


if __name__ == "__main__":
    compare_representative_images(
        directory="/raid/scratch/mevans/newdata",
        id_info_table="/raid/scratch/mevans/imageproperties.dat",
        outdir="/raid/scratch/mevans/representative_plots"
    )
