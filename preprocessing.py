
import os
import glob
import numpy as np
from astropy.io import fits
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from skimage.morphology import closing
from skimage.measure import label


#The FOV distances for each image are calculated, and then matched to a common distance. First the heliocentric 
#distances are extracted from the relevant table.
def extract_distance(source_id, id_info_table):
    source_id = str(source_id).strip()
    with open(id_info_table, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9:     #skip any lines with wrong number of columns
                continue
            if parts[3] == source_id:
                try:
                    return float(parts[8]) #heliocentric distance pc
                except ValueError:
                    return None
    return None
#The linear parsec per pixel is calculated using the small angle formula from the angular deg per parsec CDELT
def pc_per_pixel(cdelt_deg, distance_pc):
    if cdelt_deg is None or distance_pc is None:
        return None
    rad_per_pixel = abs(cdelt_deg) * (np.pi / 180.0)
    # pc per pixel small angle formula
    return rad_per_pixel * distance_pc
#Images are matched to a common FOV, by either padding with zeroes or cropping centrally
def crop_to_FOV(data, s_pc, FOV_common, minpix=0.0):
    N_target = int(round(FOV_common / s_pc)) #target number of pixels for the same FOV
    N_in = data.shape[0]
    #cropping/padding to target fov
    if N_in >= N_target:
        start = (N_in - N_target) // 2
        return data[start:start+N_target, start:start+N_target]
    out = np.full((N_target, N_target), minpix, dtype=float)
    start = (N_target - N_in) // 2
    out[start:start+N_in, start:start+N_in] = data
    return out
#Images are rescaled to a common pixel size so they are fed into the SOM at the same size
def rescale_to_fixed_size(data, N_target):
    zoom_factor = N_target / data.shape[0]
    return ndimage.zoom(data, zoom_factor, order=1)
#Island segmentation function adapted from PINK, looks for connected pixels of value above the threshold and 
#creates a mask. Any other pixels are removed
def island_segmentation(data, threshold, minimum_island_size):
    mask = closing(data > threshold)
    labels_map, n_labels = label(mask, return_num=True, connectivity=2)
    islands = []
    for i in range(1, n_labels + 1):
        region = (labels_map == i)
        if np.sum(region) >= minimum_island_size:
            islands.append(region)
    return islands
#Single file processing helper so preprocessing can be parallised. Includes the noise filtering and
# normalising and the function  calls for island segmentation and stuff.
def process_single_file(args):
    filename, data_raw, config, s_pc, noise = args
    try:
        if data_raw is None or noise is None:
            return [], [], None
        data = data_raw.copy()
        # noise threshold + normalisation
        data[data <= config.noise_multiplier * noise] = 0
        max_val = np.nanmax(data)
        if max_val == 0 or not np.isfinite(max_val):
            return [], [], None
        data = data / max_val
        # iterative fallback island segmentation (same logic as original)
        min_size_start = config.minimum_island_size
        min_size_end   = 120
        step           = 10
        min_size = min_size_start
        islands = []
        while min_size >= min_size_end:
            islands = island_segmentation(
                data,
                config.threshold,
                min_size
            )
            if len(islands) > 0:
                break
            min_size -= step
        # remove image if no islands after fallback
        if len(islands) == 0:
            return [], [], None
        # merge islands into a single mask
        mask = np.zeros_like(data, dtype=bool)
        for isl in islands:
            mask |= isl
        masked = data * mask
        return [masked], [filename], s_pc
    except Exception:
        return [], [], None
#The common FOV is selected as the median value and images r rescaled
def worker_fov(args):
    img, s_pc, FOV_common, N_target = args
    minpix = float(np.median(img[np.isfinite(img)]))
    crop_side = int(round(FOV_common / s_pc))
    crop_mode = "crop" if img.shape[0] > crop_side else ("pad" if img.shape[0] < crop_side else "none")
    img_fov = crop_to_FOV(img, s_pc, FOV_common, minpix=minpix)
    img_rescaled = rescale_to_fixed_size(img_fov, N_target)
    zoom_factor = N_target / img_fov.shape[0]
    meta = {
        "crop_mode": crop_mode,
        "crop_side": img_fov.shape[0],
        "pad_value": minpix,
        "zoom_factor": zoom_factor,
    }
    return img_rescaled.flatten(), meta


#Func to actually run the preprocessing. This extracts the fits info needed 
#(cdelt, ASTDRES), finds the FOV of each image in pc, removes the ten smallest 
#and ten largest fov images,noise removes and island segments in parallel, flips 
#and rotates if requested,  makes them all a common fov and makes them all a common
#pixel scale in parallel, scales with standard scaler and plops out a nice output.

def run_preprocessing(config):
    """
    Returns:
        datas_scaled  : standardized masked/processed vectors
        labels        : image labels
        filenames     : original FITS filenames (unaugmented)
        raw_datas     : processed but un-FOV-matched masked arrays (for plotting)
        fov_matched   : dict of FOV-matched original images
    """
    files = sorted(glob.glob(os.path.join(
        config.data_dir,
        "*",
        "images",
        "combined",
        "7MTM2TM1",
        "almagal",
        "*_cont_7MTM2TM1_jointdeconv.image.fits",
    )))
    if not files:
        raise FileNotFoundError(
            "No continuum FITS files found under "
            f"{config.data_dir}/<sourcename>/images/combined/7MTM2TM1/almagal"
        )
    total_sources = len(files)
    n_cores = max(1, cpu_count() - 2)
    print(f"Processing {len(files)} FITS files on {n_cores} cores...")
    props = []      # (filename, s_pc, FOV_pc)
    loaded = []     # (filename, data_raw, s_pc, noise)
    for fname in files:
        try:
            with fits.open(fname) as hd:
                hdr = hd[0].header
                data = hd[0].data
        except Exception as e:
            print(f"Skipping file (not a valid FITS?): {fname} — {e}")
            continue
        if data is None:
            continue
        data = np.nan_to_num(data, nan=0)
        N = data.shape[0]
        source = hdr.get("OBJECT", "").strip()
        dist = extract_distance(source, config.id_info_table)
        cdelt = hdr.get("CDELT1")
        s_pc = pc_per_pixel(cdelt, dist)
        if s_pc is None:
            continue
        FOV_pc = N * s_pc
        props.append((fname, s_pc, FOV_pc))
        noise = hdr.get("AGSTDRES")
        loaded.append((fname, data.copy(), s_pc, noise))
    # remove top 10 smallest and top 10 largest FOVs
    props_sorted = sorted(props, key=lambda x: x[2])
    if len(props_sorted) > 20:
        props_filtered = props_sorted[10:-10]
    else:
        props_filtered = props_sorted
    valid_files = set([p[0] for p in props_filtered])
    s_pc_map = {p[0]: p[1] for p in props_filtered}
    # restrict to valid files
    loaded_filtered = [entry for entry in loaded if entry[0] in valid_files]
    files = [f for f in files if f in valid_files]
    # FOV_common from unprocessed FOVs (median)
    FOV_common = float(np.median([p[2] for p in props_filtered]))
    # FOV-matched originals for plotting (no noise cut, no normalisation)
    fov_matched = {}
    for fname, data_raw, s_pc, noise in loaded_filtered:
        if fname in valid_files:
            fov_image = crop_to_FOV(data_raw, s_pc, FOV_common, minpix=0.0)
            fov_matched[fname] = fov_image
    # Parallel noise removal + island segmentation
    print("Performing noise removal and island segmentation...")
    args_seg = [(fname, data_raw, config, s_pc, noise)
                for (fname, data_raw, s_pc, noise) in loaded_filtered]
    with Pool(n_cores) as p:
        results = p.map(process_single_file, args_seg)
    masked_images = []
    filenames = []
    labels = []
    s_pcs_used = []
    for imgs, fns, s_pc in results:
        if len(imgs) == 0:
            continue
        for img, fn in zip(imgs, fns):
            base_label = os.path.basename(fn)
            # original masked image
            masked_images.append(img)
            filenames.append(fn)
            labels.append(base_label + "_masked")
            s_pcs_used.append(s_pc)
            # flipped images
            if config.add_flipped_images:
                hf = np.flipud(img)
                masked_images.append(hf)
                filenames.append(fn)
                labels.append(base_label + "_masked[hflip]")
                s_pcs_used.append(s_pc)
                vf = np.fliplr(img)
                masked_images.append(vf)
                filenames.append(fn)
                labels.append(base_label + "_masked[vflip]")
                s_pcs_used.append(s_pc)
                hv = np.flip(img)
                masked_images.append(hv)
                filenames.append(fn)
                labels.append(base_label + "_masked[hvflip]")
                s_pcs_used.append(s_pc)
            # rotated images
            if config.add_rotated_images:
                for angle in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
                    rot = ndimage.rotate(img, angle, reshape=False, order=1)
                    rot = np.nan_to_num(rot, nan=0)
                    masked_images.append(rot)
                    filenames.append(fn)
                    labels.append(base_label + f"_masked[rot{angle}]")
                    s_pcs_used.append(s_pc)

    masked_images = np.array(masked_images, dtype=object)
    fixed_raw = []
    for img in masked_images:
        if img.shape[0] != config.sqrt_pixel_number:
            zoom_factor = config.sqrt_pixel_number / img.shape[0]
            img = ndimage.zoom(img, zoom_factor, order=1)
        fixed_raw.append(img)
    raw_datas = np.array(fixed_raw)

    # Parallel post-segmentationa FOV matching and rescaling
    print("Performing FOV matching and rescaling...")
    args_fov = []
    for idx, img in enumerate(masked_images):
        fname = filenames[idx]
        s_pc = s_pc_map[fname]
        args_fov.append((img, s_pc, FOV_common, config.sqrt_pixel_number))
    with Pool(n_cores) as p:
        results_fov = p.map(worker_fov, args_fov)

    processed_flat = []
    fov_meta = []
    for idx, (flat_img, meta) in enumerate(results_fov):
        fname = filenames[idx]
        orig_side = masked_images[idx].shape[0]
        orig_s_pc = s_pc_map[fname]
        orig_fov = orig_side * orig_s_pc
        final_side = config.sqrt_pixel_number
        final_s_pc = FOV_common / final_side
        meta.update({
            "filename": fname,
            "orig_side": orig_side,
            "orig_s_pc": orig_s_pc,
            "orig_fov": orig_fov,
            "final_side": final_side,
            "final_s_pc": final_s_pc,
            "FOV_common": FOV_common,
        })
        processed_flat.append(flat_img)
        fov_meta.append(meta)

    processed = np.array(processed_flat)
    processed_2d = processed.reshape((-1, config.sqrt_pixel_number, config.sqrt_pixel_number))
    scaler = StandardScaler()
    datas_scaled = scaler.fit_transform(processed)
    datas_scaled = np.real(datas_scaled)
    print("Preprocessing complete.")

    fov_info = {
        "FOV_common": FOV_common,
        "target_side": config.sqrt_pixel_number,
        "meta": fov_meta,
    }

    processed_sources = len(set(filenames))
    skipped_sources = total_sources - processed_sources
    print(
        f"Continuum preprocessing complete: {processed_sources} sources preprocessed, "
        f"{skipped_sources} skipped."
    )
    if skipped_sources:
        print("Some sources were skipped because their FITS data, metadata, or usable emission was invalid.")

    return datas_scaled, labels, filenames, raw_datas, fov_matched, processed_2d, fov_info

def _save_continuum_fits(processed_2d, labels, filenames, fov_info, config):
    if processed_2d is None or fov_info is None:
        return

    if not getattr(config, "save_continuum_fits", True):
        return

    base_dir = getattr(
        config,
        "continuum_preprocessed_dir",
        os.path.join(getattr(config, "run_base_dir", "/raid/scratch/mevans"), "continuumpreprocessed")
    )
    os.makedirs(base_dir, exist_ok=True)

    skip_tags = ("[hflip]", "[vflip]", "[hvflip]", "[rot")
    meta_entries = fov_info.get("meta", []) if isinstance(fov_info, dict) else []
    FOV_common = fov_info.get("FOV_common") if isinstance(fov_info, dict) else None

    for img, fname, lbl, meta in zip(processed_2d, filenames, labels, meta_entries):
        if any(tag in lbl for tag in skip_tags):
            continue

        base = os.path.basename(fname)
        source_id = os.path.splitext(base)[0]
        src_dir = os.path.join(base_dir, source_id)
        os.makedirs(src_dir, exist_ok=True)

        out_name = f"{source_id}_continuum_preproc.fits"
        out_path = os.path.join(src_dir, out_name)

        orig_side = int(meta.get("orig_side", img.shape[0]))
        orig_s_pc = float(meta.get("orig_s_pc", np.nan))
        orig_fov = float(meta.get("orig_fov", np.nan))
        crop_side = int(meta.get("crop_side", img.shape[0]))
        pad_value = float(meta.get("pad_value", np.nan))
        zoom_factor = float(meta.get("zoom_factor", 1.0))
        final_side = int(meta.get("final_side", img.shape[0]))
        final_s_pc = float(meta.get("final_s_pc", np.nan))
        crop_mode = meta.get("crop_mode", "unknown")

        hdr = fits.Header()
        hdr["ORGFNAME"] = (base, "Original continuum FITS filename")
        hdr["ORIGNPIX"] = (orig_side, "Original side length (pixels)")
        hdr["ORIGSPC"] = (orig_s_pc, "Original pc per pixel")
        hdr["ORIGFOV"] = (orig_fov, "Original field of view (pc)")
        hdr["FOVCOMM"] = (float(FOV_common) if FOV_common is not None else np.nan, "Matched FOV (pc)")
        hdr["CROPMOD"] = (crop_mode, "crop/pad/none to reach FOV_common")
        hdr["CROPNPIX"] = (crop_side, "Side after crop/pad (pixels)")
        hdr["PADVAL"] = (pad_value, "Padding value used if padded")
        hdr["ZOOMFAC"] = (zoom_factor, "Zoom factor to final target size")
        hdr["TARGNPIX"] = (final_side, "Final side length after resize (pixels)")
        hdr["FINLSPC"] = (final_s_pc, "Final pc per pixel after resize")
        hdr["FOVUNIT"] = ("pc", "Units for FOV metrics")

        hdu = fits.PrimaryHDU(data=np.asarray(img, dtype=np.float32), header=hdr)
        hdu.writeto(out_path, overwrite=True)


def save_preprocessed_global(save_dir, datas_scaled, labels, filenames, raw_datas, fov_matched, config, processed_2d=None, fov_info=None):
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "preprocessing_info.txt"), "w") as f:
        f.write("Preprocessing configuration\n")
        f.write("============================\n")
        for key, val in config.__dict__.items():
            if key in [
                "sqrt_pixel_number",
                "noise_multiplier",
                "threshold",
                "minimum_island_size",
                "min_island_size_floor",
                "add_flipped_images",
                "add_rotated_images",
            ]:
                f.write(f"{key}: {val}\n")
        f.write(f"\nTotal processed samples: {len(datas_scaled)}\n")

    _save_continuum_fits(processed_2d, labels, filenames, fov_info, config)

def load_preprocessed_global(save_dir):
    pattern = os.path.join(save_dir, "**", "*_continuum_preproc.fits")
    paths = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        raise FileNotFoundError(f"No preprocessed FITS found under {save_dir}")

    raw_datas = []
    labels = []
    filenames = []
    metas = []
    fov_values = []

    for path in paths:
        try:
            with fits.open(path) as hd:
                data = hd[0].data
                hdr = hd[0].header
        except Exception as e:
            print(f"Skipping unreadable FITS: {path} — {e}")
            continue

        if data is None:
            continue

        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            print(f"Skipping non-2D FITS: {path} (shape {arr.shape})")
            continue
        arr = np.nan_to_num(arr, nan=0.0)

        orig_fname = hdr.get("ORGFNAME", os.path.basename(path))
        base_label = os.path.basename(orig_fname)

        meta = {
            "filename": orig_fname,
            "orig_side": hdr.get("ORIGNPIX"),
            "orig_s_pc": hdr.get("ORIGSPC"),
            "orig_fov": hdr.get("ORIGFOV"),
            "crop_mode": hdr.get("CROPMOD"),
            "crop_side": hdr.get("CROPNPIX"),
            "pad_value": hdr.get("PADVAL"),
            "zoom_factor": hdr.get("ZOOMFAC"),
            "final_side": hdr.get("TARGNPIX", arr.shape[0]),
            "final_s_pc": hdr.get("FINLSPC"),
            "FOV_common": hdr.get("FOVCOMM"),
        }

        if meta["FOV_common"] is not None:
            try:
                fov_values.append(float(meta["FOV_common"]))
            except (TypeError, ValueError):
                pass

        raw_datas.append(arr)
        labels.append(f"{base_label}_masked")
        filenames.append(orig_fname)
        metas.append(meta)

    if not raw_datas:
        raise RuntimeError(f"No usable preprocessed FITS found under {save_dir}")

    raw_datas = np.asarray(raw_datas, dtype=np.float32)
    processed_2d = np.asarray(raw_datas, dtype=np.float32)

    flat = processed_2d.reshape(processed_2d.shape[0], -1)
    scaler = StandardScaler()
    datas_scaled = scaler.fit_transform(flat).astype(np.float32)

    fov_common = float(np.median(fov_values)) if fov_values else None
    fov_info = {
        "FOV_common": fov_common,
        "target_side": processed_2d.shape[1],
        "meta": metas,
    }

    fov_matched = {fn: img for fn, img in zip(filenames, raw_datas)}

    return datas_scaled, labels, filenames, raw_datas, fov_matched, processed_2d, fov_info
