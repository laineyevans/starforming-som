import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits
import os
import glob
import numpy as np
from skimage.morphology import closing
from skimage.measure import label
from scipy import ndimage
import sys


# ------------------------------------------------------------------
# Simple FOV check mode
# Plots images as they are and as they enter the pipeline.
# ------------------------------------------------------------------
USE_SIMPLE_FOV_CHECK = True

if USE_SIMPLE_FOV_CHECK and __name__ == "__main__":
    SOURCE_ID = "778940"
    # "smoothed" or "moment0"
    FOV_CHECK_DATASET = "moment0"

    OUT_DIR = "/raid/scratch/mevans/imageplotter"
    CONTINUUM_DIR = "/raid/scratch/mevans/newdata"
    SMOOTHED_DIR = "/raid/scratch/mevans/smoothed_7mtm1tm2_noiseremoved0"
    SMOOTHED_PREPROCESSED_DIR = "/raid/scratch/mevans/Preprocessed_Datas/smoothed_preprocessed_island"
    MOMENT0_PREPROCESSED_DIR = "/raid/scratch/mevans/Preprocessed_Datas/moment_zero_preprocessed/island3"
    MOMENT0_DIRS = [
        "/raid/scratch/mevans/moment0_with_rms",
        "/raid/scratch/mevans/moment0_with_rms2",
    ]
    CONTINUUM_PREPROC_DIR = "/raid/scratch/mevans/Preprocessed_Datas/continuumpreprocessed"
    # Optional hint to choose one moment0 map when multiple exist (e.g. "H2CO", "SiO", "DCN")
    MOMENT0_LINE_HINT = "H2CO"
    PIPELINE_SIZE = 304

    def _sid(name):
        base = os.path.basename(str(name))
        if "_cont_" in base:
            return base.split("_cont_", 1)[0]
        if "_smoothed_" in base:
            return base.split("_smoothed_", 1)[0]
        return os.path.splitext(base)[0]

    def _to2d(arr):
        arr = np.asarray(arr, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D, got {arr.shape}")
        return arr

    def _norm01(arr):
        arr = _to2d(arr)
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        if mx <= mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def _rescale_to_pipeline(arr, n=PIPELINE_SIZE):
        arr = _to2d(arr)
        if arr.shape == (n, n):
            return arr
        zy = n / arr.shape[0]
        zx = n / arr.shape[1]
        return ndimage.zoom(arr, (zy, zx), order=1)

    def _first_match(directory, pattern):
        matches = sorted(glob.glob(os.path.join(directory, pattern)))
        return matches[0] if matches else None

    def _load_fits_data(path):
        with fits.open(path) as hdul:
            return np.squeeze(hdul[0].data)

    def _load_continuum_preproc(source_id):
        pattern = os.path.join(CONTINUUM_PREPROC_DIR, "*", "*_continuum_preproc.fits")
        for candidate in glob.glob(pattern):
            base = os.path.basename(candidate)
            if base.startswith(source_id):
                return _to2d(_load_fits_data(candidate)), candidate
        return None, None

    os.makedirs(OUT_DIR, exist_ok=True)

    # Continuum original FITS
    cont_orig_path = _first_match(CONTINUUM_DIR, f"{SOURCE_ID}_cont_*.fits")
    if not cont_orig_path:
        raise FileNotFoundError(f"Missing continuum FITS for {SOURCE_ID} in {CONTINUUM_DIR}")
    cont_orig = _to2d(_load_fits_data(cont_orig_path))

    # Continuum preprocessed FITS
    cont_preproc, cont_preproc_path = _load_continuum_preproc(SOURCE_ID)
    if cont_preproc is None:
        raise FileNotFoundError(f"Missing continuum preproc FITS for {SOURCE_ID} under {CONTINUUM_PREPROC_DIR}")

    # Choose dataset to plot (smoothed or moment0)
    if FOV_CHECK_DATASET.lower() == "smoothed":
        orig_path = _first_match(SMOOTHED_DIR, f"{SOURCE_ID}_smoothed_*.fits")
        if not orig_path:
            raise FileNotFoundError(f"Missing smoothed FITS for {SOURCE_ID} in {SMOOTHED_DIR}")
        orig_img = _to2d(_load_fits_data(orig_path))

        preproc_path = os.path.join(SMOOTHED_PREPROCESSED_DIR, f"{SOURCE_ID}_smoothed_island.npy")
        if not os.path.isfile(preproc_path):
            preproc_path = os.path.join(SMOOTHED_PREPROCESSED_DIR, f"{SOURCE_ID}_smoothed_noisland.npy")
        if not os.path.isfile(preproc_path):
            raise FileNotFoundError(f"Missing smoothed preprocessed .npy for {SOURCE_ID} in {SMOOTHED_PREPROCESSED_DIR}")
        preproc_img = _to2d(np.load(preproc_path, allow_pickle=False))
        dataset_label = "Smoothed"
    elif FOV_CHECK_DATASET.lower() == "moment0":
        line_tag = MOMENT0_LINE_HINT or "H2CO"
        orig_path = None
        for root in MOMENT0_DIRS:
            cand = os.path.join(root, str(SOURCE_ID), f"{SOURCE_ID}_{line_tag}_moment0.fits")
            if os.path.isfile(cand):
                orig_path = cand
                break
        if not orig_path:
            raise FileNotFoundError(f"Missing moment0 FITS for {SOURCE_ID} ({line_tag}) in {MOMENT0_DIRS}")
        orig_img = _to2d(_load_fits_data(orig_path))

        preproc_dir = os.path.join(MOMENT0_PREPROCESSED_DIR, SOURCE_ID)
        if not os.path.isdir(preproc_dir):
            raise FileNotFoundError(f"Missing moment0 preprocessed dir for {SOURCE_ID}: {preproc_dir}")
        preproc_path = os.path.join(preproc_dir, f"{SOURCE_ID}_{line_tag}_moment0_island.npy")
        if not os.path.isfile(preproc_path):
            raise FileNotFoundError(f"Missing moment0 preprocessed .npy for {SOURCE_ID} ({line_tag}) in {preproc_dir}")
        preproc_img = _to2d(np.load(preproc_path, allow_pickle=False))
        dataset_label = "Moment0"
    else:
        raise ValueError("FOV_CHECK_DATASET must be 'smoothed' or 'moment0'")

    cont_pipeline = _rescale_to_pipeline(cont_preproc, PIPELINE_SIZE)
    aux_pipeline = _rescale_to_pipeline(preproc_img, PIPELINE_SIZE)

    imgs = [
        (_norm01(cont_orig),      f"Continuum original\n{os.path.basename(cont_orig_path)}\nshape={cont_orig.shape}"),
        (_norm01(cont_pipeline),  f"Continuum preprocessed\n{os.path.basename(cont_preproc_path)}\nshape={cont_preproc.shape}"),
        (_norm01(orig_img),       f"{dataset_label} original\n{os.path.basename(orig_path)}\nshape={orig_img.shape}"),
        (_norm01(aux_pipeline),   f"{dataset_label} preprocessed\n{os.path.basename(preproc_path)}\nshape={preproc_img.shape}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (img, title) in zip(axes.ravel(), imgs):
        im = ax.imshow(img, origin='lower', cmap='plasma', vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, orientation='vertical', label='Normalized Intensity')

    out = os.path.join(OUT_DIR, f"fov_check_{SOURCE_ID}_orig_vs_preproc.png")
    plt.tight_layout()
    plt.savefig(out, dpi=250)
    plt.close(fig)
    print(f"Saved: {out}")
    sys.exit(0)

outdir = "/raid/scratch/mevans/imageplotter"
os.makedirs(outdir, exist_ok=True)


FILE_TO_PLOT = "/raid/scratch/mevans/smoothed_preprocessed_island/778940_smoothed_island.npy"
# Optional second file; when provided with PLOT_TWO_SIDE_BY_SIDE=True, both are plotted in one PNG.
SECOND_FILE_TO_PLOT = None
PLOT_TWO_SIDE_BY_SIDE = True
USE_PREPROCESSED_CONTINUUM_SECOND = True
PREPROCESSED_NPZ = "/raid/scratch/mevans/preprocessed_data/preprocessed_data.npz"
SOURCE_ID_FOR_CONT = "778940"
GRID_SIZE = 305
# If you want to override the header AGSTDRES, set a numeric value here (or None to use header)
AGSTDRES_VALUE = None
# Integer multiple to subtract (e.g. 1 => subtract AGSTDRES once)
AGSTDRES_MULTIPLE = 1
# Whether to attempt removal (if False, only original image is plotted)
REMOVE_AGSTDRES = False
# Toggle island segmentation (applies after noise removal)
ENABLE_ISLAND_SEGMENTATION = True
# Island segmentation settings (same style as preprocessing)
ISLAND_THRESHOLD = 0.2
MINIMUM_ISLAND_SIZE = 300
MIN_ISLAND_SIZE_FLOOR = 120
MIN_ISLAND_SIZE_STEP = 10
# ------------------------------------------

# collect files (either explicit paths or all FITS/.npy in input dir)
second_from_npz = None
if USE_PREPROCESSED_CONTINUUM_SECOND and os.path.isfile(PREPROCESSED_NPZ):
    try:
        npz = np.load(PREPROCESSED_NPZ, allow_pickle=True)
        # Prefer FOV-matched continuum images saved by preprocessing.py
        fov_items = npz["fov_matched"].tolist() if "fov_matched" in npz else []
        fov_map = dict(fov_items) if fov_items else {}
        # match by source id prefix before '_cont_'
        def _sid_from_fname(fname):
            base = os.path.basename(fname)
            return base.split("_cont_", 1)[0]
        for fname, arr in fov_map.items():
            if _sid_from_fname(fname) == SOURCE_ID_FOR_CONT:
                second_from_npz = arr
                SECOND_FILE_TO_PLOT = fname
                break

        # Fallback to raw_datas only if fov_matched was unavailable
        if second_from_npz is None and "filenames" in npz and "raw_datas" in npz:
            filenames_npz = npz["filenames"].tolist()
            raw_datas_npz = npz["raw_datas"]
            for fname, raw in zip(filenames_npz, raw_datas_npz):
                if _sid_from_fname(fname) == SOURCE_ID_FOR_CONT:
                    second_from_npz = raw
                    SECOND_FILE_TO_PLOT = fname
                    break
    except Exception as exc:
        print(f"Failed to load preprocessed continuum NPZ: {exc}")

# collect files (either explicit paths or all FITS/.npy in input dir)
if PLOT_TWO_SIDE_BY_SIDE and FILE_TO_PLOT and (second_from_npz is not None or SECOND_FILE_TO_PLOT):
    files = [FILE_TO_PLOT, SECOND_FILE_TO_PLOT or ""]
elif PLOT_TWO_SIDE_BY_SIDE and FILE_TO_PLOT and SECOND_FILE_TO_PLOT:
    files = [FILE_TO_PLOT, SECOND_FILE_TO_PLOT]
elif FILE_TO_PLOT:
    files = [FILE_TO_PLOT]
else:
    patterns = [os.path.join(input_dir, "*.fits"), os.path.join(input_dir, "*.npy")]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))

print(f'Found {len(files)} files')


def _load_image(path):
    ext = os.path.splitext(path)[1].lower()
    header = None
    wcs = None
    data = None

    if ext == '.npy':
        data = np.load(path, allow_pickle=False)
    else:
        hdul = fits.open(path)
        hdu = hdul[0]
        data = np.squeeze(hdu.data)
        header = hdu.header
        try:
            wcs = WCS(header)
        except Exception:
            wcs = None
    return data, header, wcs


def clean_and_normalize(img):
    img = np.array(img, dtype=float)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img)
    minval = img[finite].min()
    img[~finite] = minval
    minv = img[finite].min()
    maxv = img[finite].max()
    if maxv > minv:
        img = (img - minv) / (maxv - minv)
    else:
        img = np.zeros_like(img)
    return img


def place_on_grid(img, grid_size=GRID_SIZE, fill_value=0.0):
    """Center-crop or pad image to a fixed grid without resizing content."""
    arr = np.asarray(img, dtype=float)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {arr.shape}")

    h, w = arr.shape
    # Central crop if larger than grid
    if h > grid_size:
        y0 = (h - grid_size) // 2
        arr = arr[y0:y0 + grid_size, :]
        h = grid_size
    if w > grid_size:
        x0 = (w - grid_size) // 2
        arr = arr[:, x0:x0 + grid_size]
        w = grid_size

    out = np.full((grid_size, grid_size), fill_value, dtype=float)
    y = (grid_size - h) // 2
    x = (grid_size - w) // 2
    out[y:y + h, x:x + w] = arr
    return out


if PLOT_TWO_SIDE_BY_SIDE and len(files) == 2:
    a_path, b_path = files
    try:
        a_data, _, _ = _load_image(a_path)
        if second_from_npz is not None:
            b_data = second_from_npz
        else:
            b_data, _, _ = _load_image(b_path)
    except Exception as exc:
        print(f"Failed to load pair: {exc}")
        exit(1)

    a_shape = np.asarray(a_data).shape
    b_shape = np.asarray(b_data).shape
    a_grid = place_on_grid(clean_and_normalize(a_data), GRID_SIZE)
    b_grid = place_on_grid(clean_and_normalize(b_data), GRID_SIZE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, img, path, shp in zip(axes, [a_grid, b_grid], [a_path, b_path], [a_shape, b_shape]):
        im = ax.imshow(img, origin='lower', cmap='plasma', vmin=0, vmax=1)
        ax.set_title(f"{os.path.basename(path)}\norig shape={shp}")
        ax.set_xlim(0, GRID_SIZE - 1)
        ax.set_ylim(0, GRID_SIZE - 1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(im, ax=ax, orientation='vertical', label='Normalized Intensity')

    b_base = os.path.splitext(os.path.basename(b_path))[0] if b_path else SOURCE_ID_FOR_CONT
    outpair = os.path.join(outdir, f"pair_{os.path.splitext(os.path.basename(a_path))[0]}_vs_{b_base}_grid{GRID_SIZE}.png")
    plt.tight_layout()
    plt.savefig(outpair, dpi=300)
    plt.close()
    print('Saved pair plot:', outpair)
    exit(0)

def island_segmentation(data, threshold, minimum_island_size):
    mask = closing(data > threshold)
    labels_map, n_labels = label(mask, return_num=True, connectivity=2)
    islands = []
    for i in range(1, n_labels + 1):
        region = (labels_map == i)
        if np.sum(region) >= minimum_island_size:
            islands.append(region)
    return islands

def segment_islands_with_fallback(noise_removed_img):
    if noise_removed_img is None:
        return None

    seg_data = np.array(noise_removed_img, dtype=float)
    seg_data = np.nan_to_num(seg_data, nan=0.0)

    max_val = np.nanmax(seg_data)
    if (not np.isfinite(max_val)) or max_val <= 0:
        return None

    seg_data = seg_data / max_val

    min_size = int(MINIMUM_ISLAND_SIZE)
    islands = []
    while min_size >= int(MIN_ISLAND_SIZE_FLOOR):
        islands = island_segmentation(seg_data, ISLAND_THRESHOLD, min_size)
        if len(islands) > 0:
            break
        min_size -= int(MIN_ISLAND_SIZE_STEP)

    if len(islands) == 0:
        return None

    mask = np.zeros_like(seg_data, dtype=bool)
    for isl in islands:
        mask |= isl

    return seg_data * mask

for f in files:
    print('Processing:', f)
    ext = os.path.splitext(f)[1].lower()
    header = None
    wcs = None
    data = None

    if ext == '.npy':
        try:
            data = np.load(f, allow_pickle=False)
        except Exception as e:
            print(f'  Failed to load .npy file {f}: {e}')
            continue
    else:
        try:
            hdul = fits.open(f)
            hdu = hdul[0]
            data = np.squeeze(hdu.data)
            header = hdu.header
            try:
                wcs = WCS(header)
            except Exception:
                wcs = None
        except Exception as e:
            print(f'  Failed to open FITS {f}: {e}')
            continue

    # AGSTDRES only meaningful for FITS with header
    agstdres = None
    if header is not None:
        agstdres_header = header.get('AGSTDRES')
        agstdres = AGSTDRES_VALUE if AGSTDRES_VALUE is not None else agstdres_header
        if AGSTDRES_VALUE is not None:
            print('  Using explicit AGSTDRES value from code:', AGSTDRES_VALUE)
        else:
            print('  AGSTDRES in header:', agstdres_header)
        print('  AGSTDRES:', agstdres)

    # Prepare noise-removed image if requested and AGSTDRES exists
    noise_removed = None
    island_segmented = None
    mult = 10
    if REMOVE_AGSTDRES and (agstdres is not None):
        try:
            mult = int(AGSTDRES_MULTIPLE)
        except Exception:
            mult = 10
        threshold = mult * agstdres
        noise_removed = np.where(data >= threshold, data, np.nan)
        print(f'  Noise threshold: {threshold:.6e} ({mult} x AGSTDRES={agstdres:.6e})')
        n_masked = np.sum(data < threshold)
        print(f'  Pixels masked as noise: {n_masked} / {data.size} ({100*n_masked/data.size:.1f}%)')

        if ENABLE_ISLAND_SEGMENTATION:
            island_segmented = segment_islands_with_fallback(noise_removed)
            if island_segmented is None:
                print('  Island segmentation found no valid islands (with fallback).')
            else:
                print('  Island segmentation applied successfully.')

    data_clean = clean_and_normalize(data)
    noise_clean = clean_and_normalize(noise_removed) if noise_removed is not None else None
    island_clean = clean_and_normalize(island_segmented) if island_segmented is not None else None

    base = os.path.basename(f)
    out_base = os.path.splitext(base)[0]

    # Build up to 3 panels: original, noise-removed, noise-removed+island-segmented
    panel_images = [data_clean]
    panel_titles = [f"Original (0-1): {base}"]

    if noise_clean is not None:
        panel_images.append(noise_clean)
        panel_titles.append(f"Noise-removed (0-1) (AGSTDRES x {mult}): {base}")

    if island_clean is not None:
        panel_images.append(island_clean)
        panel_titles.append(f"Noise-removed + island segmented (0-1): {base}")

    n_panels = len(panel_images)
    fig = plt.figure(figsize=(8 * n_panels, 8))
    axes = []
    for i in range(n_panels):
        if wcs is not None:
            ax = fig.add_subplot(1, n_panels, i + 1, projection=wcs)
        else:
            ax = fig.add_subplot(1, n_panels, i + 1)
        axes.append(ax)

    for ax, img_plot, title in zip(axes, panel_images, panel_titles):
        im = ax.imshow(img_plot, origin='lower', cmap='plasma', vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("Right Ascension" if wcs is not None else "X")
        ax.set_ylabel("Declination" if wcs is not None else "Y")
        plt.colorbar(im, ax=ax, orientation='vertical', label='Normalized Intensity')

    if n_panels == 1:
        outpath = os.path.join(outdir, out_base + ".png")
    elif n_panels == 2:
        outpath = os.path.join(outdir, out_base + '_orig_vs_noise.png')
    else:
        outpath = os.path.join(outdir, out_base + '_orig_vs_noise_vs_island.png')

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print('  Saved:', outpath)
