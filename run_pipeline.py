from config import Config
import preprocessing
import autoencoder
import som_module
import gsom_module
import soumap_module
import plotting
import cluster_diagnostics
import data_visualisation
import embedding_diagnostics
import save_results
import os
import datetime
from astropy.io import fits
import torch
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
import glob
import traceback


def save_run_info(config, save_dir):
    """
    Writes all attributes of the config object to run_info.txt
    """
    path = os.path.join(save_dir, "run_info.txt")

    with open(path, "w") as f:
        for key, value in config.__dict__.items():
            f.write(f"{key}: {value}\n")


def _extract_source_id_from_filename(path):
    """Extract the ALMAGAL source identifier from a continuum FITS filename.

    Examples
    --------
    100132_cont_7MTM2TM1_jointdeconv.image.fits -> 100132
    G010.3208-00.1570B_cont_7MTM2TM1_jointdeconv.image.fits -> G010.3208-00.1570B
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]

    # Common suffixes in this pipeline
    for marker in ("_cont_", "_continuum_preproc", "_continuum", "_smoothed", "_moment0"):
        if marker in stem:
            return stem.split(marker, 1)[0]

    if "_cont_" in base:
        return base.split("_cont_", 1)[0]
    return stem


def _save_source_cluster_assignments(filenames, labels, preds, config, sil_scores=None, pred_coords=None):
    """Save per-source cluster assignments in CSV format.

    Output columns:
        source_id,flip_type,cluster,node_x,node_y,filename,label,silhouette_score
    """
    assign_path = os.path.join(config.run_dir, "source_cluster_assignments.csv")
    def _flip_type_from_label(lbl):
        if not lbl:
            return "original"
        if "[hflip]" in lbl:
            return "hflip"
        if "[vflip]" in lbl:
            return "vflip"
        if "[hvflip]" in lbl:
            return "hvflip"
        if "[rot" in lbl:
            return "rot"
        return "original"

    with open(assign_path, "w") as f:
        f.write("source_id,flip_type,cluster,node_x,node_y,filename,label,silhouette_score\n")
        for idx, (fname, lbl, cluster) in enumerate(zip(filenames, labels, preds)):
            source_id = _extract_source_id_from_filename(fname)
            flip_type = _flip_type_from_label(lbl)
            if pred_coords is not None and len(pred_coords) > idx and pred_coords[idx] is not None:
                node_x, node_y = pred_coords[idx]
            else:
                node_x, node_y = "", ""
            if sil_scores is not None and len(sil_scores) > idx:
                sil = float(sil_scores[idx])
            else:
                sil = np.nan
            f.write(f"{source_id},{flip_type},{int(cluster)},{node_x},{node_y},{os.path.basename(fname)},{lbl},{sil:.6f}\n")

    print(f"Saved source cluster assignments to {assign_path}")


def _load_and_rescale_array(arr, H, W):
    """Ensure a 2D array of shape (H, W), rescaling if needed."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array after squeeze, got shape {arr.shape}")
    if arr.shape != (H, W):
        zoom_y = H / arr.shape[0]
        zoom_x = W / arr.shape[1]
        arr = ndimage.zoom(arr, (zoom_y, zoom_x), order=1)
    return np.nan_to_num(arr, nan=0.0)


def _load_preprocessed_fits(preprocessed_dir):
    """Load preprocessed continuum FITS products written by preprocessing (no resizing)."""
    pattern = os.path.join(preprocessed_dir, "**", "*_continuum_preproc.fits")
    paths = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        raise FileNotFoundError(f"No preprocessed FITS found under {preprocessed_dir}")

    raw_datas = []
    labels = []
    filenames = []
    metas = []
    fov_values = []

    for path in paths:
        try:
            with fits.open(path) as hdul:
                data = hdul[0].data
                hdr = hdul[0].header
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
            "orig_fov": hdr.get("ORIGFOV"),
            "FOV_common": hdr.get("FOVCOMM"),
        }
        metas.append(meta)

        try:
            if meta["FOV_common"] is not None:
                fov_values.append(float(meta["FOV_common"]))
        except (TypeError, ValueError):
            pass

        raw_datas.append(arr)
        labels.append(f"{base_label}_masked")
        filenames.append(orig_fname)

    if not raw_datas:
        raise RuntimeError(f"No usable preprocessed FITS found under {preprocessed_dir}")

    raw_datas = np.asarray(raw_datas, dtype=np.float32)
    processed_2d = raw_datas

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


def _load_simulated_fits(simulated_dir, target_side):
    """Load simulated FITS images, normalize to [0,1], then standard-scale."""
    pattern = os.path.join(simulated_dir, "**", "*.fits")
    paths = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        raise FileNotFoundError(f"No FITS files found under {simulated_dir}")

    raw_datas = []
    labels = []
    filenames = []

    for path in paths:
        try:
            with fits.open(path) as hdul:
                data = hdul[0].data
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

        arr = _load_and_rescale_array(arr, target_side, target_side).astype(np.float32)
        amin = float(np.nanmin(arr))
        amax = float(np.nanmax(arr))
        if np.isfinite(amin) and np.isfinite(amax) and amax > amin:
            arr = (arr - amin) / (amax - amin)
        else:
            arr = np.zeros_like(arr)

        raw_datas.append(arr)
        labels.append(f"{os.path.basename(path)}_sim")
        filenames.append(path)

    if not raw_datas:
        raise RuntimeError(f"No usable simulated FITS found under {simulated_dir}")

    raw_datas = np.asarray(raw_datas, dtype=np.float32)
    processed_2d = raw_datas

    flat = processed_2d.reshape(processed_2d.shape[0], -1)
    scaler = StandardScaler()
    datas_scaled = scaler.fit_transform(flat).astype(np.float32)

    fov_info = {
        "FOV_common": None,
        "target_side": processed_2d.shape[1],
        "meta": [],
    }

    return datas_scaled, labels, filenames, raw_datas, {}, processed_2d, fov_info


def _build_multi_channel_images(raw_datas, labels, filenames, config):
    """Construct multi-channel images per sample.

    Always includes continuum; additional channels are controlled by config:

        config.include_smoothed_channel
        config.include_molecular_line_channels

    When both toggles are True, the channel order is:

        0: continuum (segmented mask from preprocessing)
        1: smoothed continuum (noise-removed, zoomed)
        2: H2CO moment-0 map
        3: SiO  moment-0 map
        4: DCN  moment-0 map

    If smoothed and/or molecular lines are disabled, those channels are
    simply omitted and the total channel count is reduced accordingly.
    Missing enabled auxiliary maps are filled with zeros so that the
    channel count remains constant across the dataset.
    Flips applied during preprocessing are also applied to all channels.
    """
    H = config.sqrt_pixel_number
    W = config.sqrt_pixel_number
    n_samples = raw_datas.shape[0]

    include_smoothed = config.include_smoothed_channel
    include_lines = config.include_molecular_line_channels

    # Decide channel layout
    channel_names = ["continuum"]
    if include_smoothed:
        channel_names.append("smoothed")
    if include_lines:
        channel_names.extend(["H2CO", "SiO", "DCN"])
    C = len(channel_names)
    idx = {name: i for i, name in enumerate(channel_names)}

    # Caches to avoid re-reading the same files many times
    smoothed_cache = {}
    line_cache = {}

    def get_smoothed(source_id):
        if source_id in smoothed_cache:
            return smoothed_cache[source_id]
        if not hasattr(config, "smoothed_image_file_dir"):
            smoothed_cache[source_id] = None
            return None
        # Only accept the specific smoothed filename pattern produced by preprocessing
        fname = f"{source_id}_smoothed_island.npy"
        fpath = os.path.join(config.smoothed_image_file_dir, fname)
        if not os.path.isfile(fpath):
            smoothed_cache[source_id] = None
            return None
        arr = np.load(fpath, allow_pickle=False)
        arr = _load_and_rescale_array(arr, H, W)
        smoothed_cache[source_id] = arr.astype(np.float32)
        return smoothed_cache[source_id]

    def get_line_maps(source_id):
        if source_id in line_cache:
            return line_cache[source_id]
        maps = {
            "H2CO": None,
            "SiO": None,
            "DCN": None,
        }
        for line in maps.keys():
            path = plotting._find_molecular_line_path(config, source_id, line)
            if not path:
                continue
            try:
                data = np.load(path, allow_pickle=False)
            except Exception:
                continue
            maps[line] = _load_and_rescale_array(data, H, W).astype(np.float32)
        line_cache[source_id] = maps
        return maps

    # Allocate (N, C, H, W)
    multi_rows = []
    keep_indices = []

    missing_smoothed_sources = set()
    missing_molecular_sources = set()

    for i in range(n_samples):
        base_fname = filenames[i]
        label = labels[i]
        source_id = _extract_source_id_from_filename(base_fname)

        cont = raw_datas[i]
        cont = _load_and_rescale_array(cont, H, W).astype(np.float32)

        # Optional smoothed and line maps
        if include_smoothed:
            smoothed = get_smoothed(source_id)
            if smoothed is None:
                missing_smoothed_sources.add(source_id)
                continue
        else:
            smoothed = None

        if include_lines:
            line_maps = get_line_maps(source_id)
            h2co = line_maps.get("H2CO")
            sio = line_maps.get("SiO")
            dcn = line_maps.get("DCN")
            if h2co is None or sio is None or dcn is None:
                missing_molecular_sources.add(source_id)
                continue
        else:
            h2co = sio = dcn = None

        # Apply the same flips used in preprocessing to all channels
        if "[hflip]" in label:
            if smoothed is not None:
                smoothed = np.flipud(smoothed)
            if h2co is not None:
                h2co = np.flipud(h2co)
            if sio is not None:
                sio = np.flipud(sio)
            if dcn is not None:
                dcn = np.flipud(dcn)
        if "[vflip]" in label:
            if smoothed is not None:
                smoothed = np.fliplr(smoothed)
            if h2co is not None:
                h2co = np.fliplr(h2co)
            if sio is not None:
                sio = np.fliplr(sio)
            if dcn is not None:
                dcn = np.fliplr(dcn)
        if "[hvflip]" in label:
            if smoothed is not None:
                smoothed = np.flip(np.flip(smoothed, 0), 1)
            if h2co is not None:
                h2co = np.flip(np.flip(h2co, 0), 1)
            if sio is not None:
                sio = np.flip(np.flip(sio, 0), 1)
            if dcn is not None:
                dcn = np.flip(np.flip(dcn, 0), 1)

        stacked = np.zeros((C, H, W), dtype=np.float32)
        stacked[idx["continuum"]] = cont
        if include_smoothed:
            stacked[idx["smoothed"]] = smoothed
        if include_lines:
            stacked[idx["H2CO"]] = h2co
            stacked[idx["SiO"]] = sio
            stacked[idx["DCN"]] = dcn

        multi_rows.append(stacked)
        keep_indices.append(i)

    total_checked = n_samples
    kept = len(multi_rows)
    skipped = total_checked - kept

    if kept == 0:
        raise RuntimeError("No samples remain after channel-availability filtering; check input data paths.")

    if skipped > 0:
        msg_parts = [f"Skipped {skipped} samples after channel checks (kept {kept} of {total_checked})."]
        if include_smoothed and missing_smoothed_sources:
            msg_parts.append(f"Missing smoothed for {len(missing_smoothed_sources)} unique sources.")
        if include_lines and missing_molecular_sources:
            msg_parts.append(f"Missing molecular lines for {len(missing_molecular_sources)} unique sources.")
        print(" ".join(msg_parts))

    multi = np.stack(multi_rows, axis=0)

    channel_summary = "  ".join(f"[{i}] {name}" for i, name in enumerate(channel_names))
    print(f"Channels included ({C} total, {kept} samples): {channel_summary}")

    # Standardize all channels jointly, similar to preprocessing.datas_scaled
    flat = multi.reshape(len(multi_rows), -1)
    scaler = StandardScaler()
    flat_scaled = scaler.fit_transform(flat)
    multi_scaled = flat_scaled.reshape(len(multi_rows), C, H, W).astype(np.float32)
    return multi_scaled, keep_indices


def _augment_with_flips(raw_datas, processed_2d, labels, filenames, fov_matched, target_side):
    """Augment datasets with horizontal/vertical flips using in-memory arrays.

    Only base (non-augmented) samples are flipped; augmented labels (with
    [hflip]/[vflip]/[hvflip]/[rot]) are skipped to avoid double flips.
    Returns updated (raw_datas, processed_2d, labels, filenames) and a freshly
    scaled flat array built from the augmented processed_2d.
    """
    skip_tags = ("[hflip]", "[vflip]", "[hvflip]", "[rot")

    def is_augmented(lbl):
        return any(tag in lbl for tag in skip_tags)

    flips = [
        ("[hflip]", np.flipud),
        ("[vflip]", np.fliplr),
        ("[hvflip]", lambda x: np.flip(np.flip(x, 0), 1)),
    ]

    new_raw = list(raw_datas)
    new_proc = list(processed_2d) if processed_2d is not None else None
    new_labels = list(labels)
    new_filenames = list(filenames)

    for idx, lbl in enumerate(labels):
        if is_augmented(lbl):
            continue

        base_raw = raw_datas[idx]
        base_proc = processed_2d[idx] if processed_2d is not None else None
        fname = filenames[idx]

        for tag, fn in flips:
            flipped_raw = fn(base_raw)
            new_raw.append(flipped_raw)
            if new_proc is not None:
                new_proc.append(fn(base_proc))
            new_labels.append(f"{lbl}{tag}")
            new_filenames.append(fname)

    # Recompute standardized flats from augmented processed_2d (if provided)
    datas_scaled = None
    if new_proc is not None:
        proc_array = np.asarray(new_proc, dtype=float)
        flat = proc_array.reshape(proc_array.shape[0], -1)
        scaler = StandardScaler()
        flat_scaled = scaler.fit_transform(flat)
        datas_scaled = flat_scaled.astype(np.float32)

    return (np.asarray(new_raw),
            np.asarray(new_proc) if new_proc is not None else None,
            new_labels,
            new_filenames,
            datas_scaled)


def main():

    config = Config()

    # flags
    config.run_preprocessing = False     # run preprocessing or not
    config.load_preprocessed = True    # load from disk instead 
    config.save_continuum_fits = True   
    config.run_island_segmentation = False
    config.cluster_info_saving = True
    config.add_flip_augmentation = True  # apply flips after loading preprocessed data

    config.use_simulated_data = False  # if True, ignore real data paths and load simulated FITS from config.simulated_fits_dir
    config.run_autoencoder = True
    config.perform_som_training = True
    config.run_embedding_diagnostics = True
    config.diag_max_points = 6000
    config.diag_seed_stability = False
    
    # plotting flags
    config.plot_quantization_error = False
    config.plot_per_neuron_examples = False
    config.plot_top_populated_neuron_samples = True
    config.plot_fov_matched_originals = False
    config.plot_processed_per_neuron_examples = True
    config.plot_neuron_image_map = False
    config.plot_smoothed_per_neuron_examples = True
    config.plot_molecular_per_neuron_examples = True
    config.plot_autoencoder_recons = True
    config.plot_representative_originals = False
    config.plot_representative_processed = True
    config.plot_representative_smoothed = True
    config.plot_representative_molecular = True
    config.plot_representative_raw_moment0 = False
    config.plot_evolutionary_stage = True
    config.evo_neuron_to_plot = 0
    config.plot_gsom_growth_animation = True   # GIF of topology + hit proportions as nodes are added
    config.soumap_label_prototypes = False
    config.mpec_enabled = True
    config.mpec_walktrap_n_steps = "auto"
    config.mpec_kernel_w_cadj = 1.0
    config.mpec_kernel_w_high_d = 1.0
    config.mpec_kernel_w_low_d = 1.0
    config.mpec_kernel_power = 2.0
    config.mpec_kernel_min_value = 0.01
    config.mpec_kernel_min_nhbs = 3
    config.mpec_kernel_support = "dense"
    config.mpec_walktrap_max_steps = 15
    config.mpec_remove_empty_protos = True
    config.mpec_verbose = True
    config.mpec_plot_which = "all"
    config.mpec_plot_format = "png"
    config.mpec_show_graph = False
    config.mpec_output_dir = None


    # preprocessing (not needed anymore)
    config.sqrt_pixel_number = 304
    config.noise_multiplier = 0
    config.threshold = 0.025
    config.minimum_island_size = 150
    config.min_island_size_floor = 100
    config.add_flipped_images = True
    config.add_rotated_images = False

    # autoencoder
    config.ae_epochs = 400
    config.ae_batch_size = 32
    config.ae_learning_rate = 1e-3
    config.ae_bottleneck_dim =256
    config.ae_use_bottleneck = False
    config.use_premade_autoencoder_results = True
    config.premade_autoencoder_results_dir = "/raid/scratch/mevans/autoencoder_results_no_molecular_2048"
    # Set True if encoded_data.npy was already standardized before saving.
    config.premade_encoded_already_scaled = False
    # whether to save autoencoder outputs)
    config.save_autoencoder_results = False

    # SOM toggle: "som" (fixed grid) or "gsom" (growing) or "soumap" (UMAP+SOM)
    config.som_type = "soumap"  # "som" | "gsom" | "soumap"
    config.init_method = "pca"        # "pca" | "kmeans++" | "random", shared default for som, gsom and soumap
    config.som_init = None            # optional per-method override; None -> init_method
    config.gsom_init = None
    config.soumap_init = None

     # SOM
    config.init_pca_span = 2.0
    config.init_seed = 42
    config.n_num = 6
    config.m_num = 6
    config.rough_epochs_num = 40000
    config.fine_epochs_num = 80000
    config.rough_learning_rate = 0.2
    config.fine_learning_rate = 0.05
    config.rough_sigma = 2.5
    config.fine_sigma = 1.0
    config.fine_lr_end = 0.005
    config.fine_sigma_end = 1.0
    config.som_fine_method = "random"
    config.som_block_size = 1000
    config.train_in_blocks_flag = True

    # GSOM 
    config.gsom_spread_factor = 0.6
    config.gsom_learning_rate = 0.2
    config.gsom_iterations = 120000  # old value: 400000
    config.gsom_max_nodes = 64
    config.gsom_growth_phase_ratio = 1 / 3  # old value: 0.3; mirrors the SOM's 40k rough / 80k fine split
    config.gsom_lr_end = 0.005  # matches the SOM's final learning rate; with init_lr 0.2 this is k of about 3.69
    config.gsom_nb_lr_fraction = 0.6  # at SOM sigma=1.0 adjacent-neuron weight is about 0.61; old GSOM value was 0.3; GSOM updates only four direct neighbours (no diagonals)
    config.gsom_min_hits_to_grow = 5 # require repeat BMU hits before a node can grow
    config.gsom_min_active_neighbours_to_grow = 3  # blocks one-neighbour chain growth (long tails)
    config.gsom_init_grid_size = 5       # initial seed grid: int -> NxN square, or (rows, cols) tuple

    # SOUMAP (source defaults from soumap's parameter dataclasses)
    config.soumap_prototypes = 36
    config.soumap_random_state = 42
    config.soumap_som_rho_0 = 2.0
    config.soumap_som_rho_f = 1.0
    config.soumap_som_nbr_topo_alpha_0 = 0.5
    config.soumap_som_nbr_topo_alpha_f = 0.5
    config.soumap_som_coord_topo = "delaunay"
    config.soumap_som_proto_topo = "CONN"
    config.soumap_som_compute_dr_metrics = False
    config.soumap_umap_update_freq = 5
    config.soumap_umap_n_epochs = 200
    config.soumap_umap_min_dist_0 = 0.05
    config.soumap_umap_min_dist_f = 0.05
    config.soumap_umap_spread_0 = 1.5
    config.soumap_umap_spread_f = 1.5
    config.soumap_umap_lrate_0 = 1.0
    config.soumap_umap_lrate_f = 0.1
    config.soumap_umap_negative_sample_rate = 5
    config.soumap_umap_gamma = 1.0
    config.soumap_umap_use_standard_umap = False
    config.soumap_ctrl_total_epochs = 200
    config.soumap_ctrl_min_nhbs = 3
    config.soumap_ctrl_fill_val = 0.5
    config.soumap_ctrl_coord_init = "pca"
    config.soumap_ctrl_embedding_range = (-3.0, 3.0)
    config.soumap_ctrl_n_jobs = None
    config.soumap_ctrl_random_state = 42
    config.soumap_ctrl_verbose = True
    config.soumap_ctrl_plot_every = 0

    # paths
    config.data_dir = "/raid/scratch/mevans/newdata"
    config.id_info_table = "/raid/scratch/mevans/imageproperties.dat"
    config.run_base_dir = "/raid/scratch/mevans"
    config.core_table = "/raid/scratch/mevans/table3.dat"
    config.continuum_preprocessed_dir = "/raid/scratch/mevans/Preprocessed_Datas/continuumpreprocessed"
    config.preprocessed_dir = config.continuum_preprocessed_dir
    config.smoothed_image_file_dir = "/raid/scratch/mevans/Preprocessed_Datas/smoothed_preprocessed_island"
    config.molecular_line_file_dir = "/raid/scratch/mevans/Preprocessed_Datas/moment_zero_preprocessed/island"  # expects subdirs per source with line maps as .npy files
    config.raw_moment0_dir = "/raid/scratch/mevans/moment0_with_rms2"
    config.simulated_fits_dir = "/raid/scratch/hannahb/simulated_data"
    # channel toggles
    config.include_smoothed_channel = True
    config.include_molecular_line_channels = False

    # create run directory
    ts = datetime.datetime.now().strftime("%d-%m_%H:%M")
    config.run_dir = os.path.join(config.run_base_dir, f"run_{ts}")
    os.makedirs(config.run_dir, exist_ok=True)
    # Classification results will be generated per run in the run directory
    config.classification_result_file = os.path.join(
        config.run_dir,
        "COMBINEDclassification_info.txt"
    )

    # save run info file
    save_run_info(config, config.run_dir)



    # run preprocessing
    if config.run_preprocessing and config.load_preprocessed:
        raise ValueError("Cannot run preprocessing and load preprocessed data at the same time.")

    if config.use_simulated_data:
        config.run_preprocessing = False
        config.load_preprocessed = False
        config.run_island_segmentation = False
        config.save_continuum_fits = False
        config.include_smoothed_channel = False
        config.include_molecular_line_channels = False
        config.plot_smoothed_per_neuron_examples = False
        config.plot_molecular_per_neuron_examples = False
        config.plot_processed_per_neuron_examples = False
        config.plot_representative_smoothed = False
        config.plot_representative_molecular = False
        config.plot_representative_raw_moment0 = False
        config.plot_representative_processed = False
        config.plot_fov_matched_originals = False
        config.add_flip_augmentation = True
        config.use_premade_autoencoder_results = True
        config.save_autoencoder_results = False
        config.premade_autoencoder_results_dir = "/raid/scratch/mevans/autoencoder_results_simulated"

        print(f"Loading simulated FITS from {config.simulated_fits_dir}...")
        datas_scaled, labels, filenames, raw_datas, fov_matched, processed_2d, fov_info = _load_simulated_fits(
            config.simulated_fits_dir,
            config.sqrt_pixel_number,
        )

    elif config.load_preprocessed:
        print(f"Loading preprocessed continuum FITS from {config.continuum_preprocessed_dir}...")
        datas_scaled, labels, filenames, raw_datas, fov_matched, processed_2d, fov_info = _load_preprocessed_fits(
            config.continuum_preprocessed_dir
        )

    elif config.run_preprocessing:
        datas_scaled, labels, filenames, raw_datas, fov_matched, processed_2d, fov_info = (
            preprocessing.run_preprocessing(config)
        )

        preprocessing.save_preprocessed_global(
            config.preprocessed_dir,
            datas_scaled,
            labels,
            filenames,
            raw_datas,
            fov_matched,
            config,
            processed_2d=processed_2d,
            fov_info=fov_info
        )

    else:
        raise RuntimeError("Either run_preprocessing or load_preprocessed must be True.")

    # Optional flip augmentation applied post-preprocessing so it works when loading saved data
    if getattr(config, "add_flip_augmentation", False):
        raw_datas, processed_2d, labels, filenames, datas_scaled = _augment_with_flips(
            raw_datas,
            processed_2d,
            labels,
            filenames,
            fov_matched,
            config.sqrt_pixel_number,
        )
        if datas_scaled is None:
            # rebuild datas_scaled from raw_datas if processed_2d unavailable
            flat = raw_datas.reshape(raw_datas.shape[0], -1)
            scaler = StandardScaler()
            datas_scaled = scaler.fit_transform(flat).astype(np.float32)

    # autoencoder

    if config.run_autoencoder:

        # Build multi-channel images: continuum + smoothed + molecular lines
        imgs_np, keep_indices = _build_multi_channel_images(raw_datas, labels, filenames, config)

        # Filter auxiliary arrays to match kept samples
        raw_datas = raw_datas[keep_indices]
        labels = [labels[i] for i in keep_indices]
        filenames = [filenames[i] for i in keep_indices]
        if 'fov_matched' in locals():
            fov_matched = {fn: fov_matched[fn] for fn in filenames if fn in fov_matched}

        # N, C, H, W
        imgs = torch.tensor(imgs_np, dtype=torch.float32)

        # build dataset ONCE
        dataset = TensorDataset(imgs)

        # train AE or load premade AE outputs
        if getattr(config, "use_premade_autoencoder_results", False):
            ae_dir = getattr(config, "premade_autoencoder_results_dir", "")
            enc_path = os.path.join(ae_dir, "encoded_data.npy")
            if not os.path.isfile(enc_path):
                raise FileNotFoundError(
                    f"Premade encoded data not found: {enc_path}. "
                    "Set use_premade_autoencoder_results=False to train a new autoencoder."
                )

            encoded_data = np.load(enc_path, allow_pickle=False)
            if encoded_data.shape[0] != len(dataset):
                raise ValueError(
                    f"Premade encoded sample count ({encoded_data.shape[0]}) does not match current dataset "
                    f"({len(dataset)}). Ensure preprocessing/filtering matches the premade run."
                )

            ae_re = None
            ae_re_path = os.path.join(ae_dir, "ae_re.npy")
            if os.path.isfile(ae_re_path):
                ae_re = np.load(ae_re_path, allow_pickle=False)

            model = None
            model_path = os.path.join(ae_dir, "ae_model_state.pth")
            if os.path.isfile(model_path):
                state_dict = torch.load(model_path, map_location="cpu")

                sample = dataset[0][0]
                if sample.dim() == 2:
                    in_channels = 1
                    H, W = sample.shape
                elif sample.dim() == 3:
                    in_channels, H, W = sample.shape
                else:
                    raise ValueError(
                        f"Unexpected sample shape {sample.shape}; expected (H,W) or (C,H,W)"
                    )

                # Infer checkpoint architecture to avoid bottleneck mismatch crashes.
                has_bottleneck = ("fc_enc.weight" in state_dict and "fc_dec.weight" in state_dict)
                ckpt_bottleneck_dim = None
                if has_bottleneck:
                    ckpt_bottleneck_dim = int(state_dict["fc_enc.weight"].shape[0])

                model = autoencoder.ConvAE(
                    H,
                    W,
                    bottleneck_dim=ckpt_bottleneck_dim,
                    in_channels=in_channels,
                )
                try:
                    model.load_state_dict(state_dict, strict=True)
                    model.eval()
                except RuntimeError as e:
                    print(f"Warning: failed to load AE checkpoint for recon plots ({e}).")
                    model = None

            print(f"Loaded premade autoencoder outputs from {ae_dir}")
        else:
            encoded_data, model, ae_re = autoencoder.train_autoencoder(dataset, config)

        from plotting import plot_autoencoder_qe
        if ae_re is not None:
            plot_autoencoder_qe(ae_re, config)
        else:
            print("No AE training curve found; skipping AE QE plot.")

        if not (getattr(config, "use_premade_autoencoder_results", False)
                and getattr(config, "premade_encoded_already_scaled", True)):
            scaler = StandardScaler()
            encoded_data = scaler.fit_transform(encoded_data)
        else:
            print("Using premade encoded vectors as-is (already scaled).")

        # If requested, save AE outputs now that encoded_data is available
        if getattr(config, "save_autoencoder_results", False) and not getattr(config, "use_premade_autoencoder_results", False):
            ae_dir = "/raid/scratch/mevans/autoencoder_results"
            os.makedirs(ae_dir, exist_ok=True)
            try:
                np.save(os.path.join(ae_dir, "encoded_data.npy"), encoded_data)
            except Exception:
                print("Warning: failed to save encoded_data.npy")
            try:
                np.save(os.path.join(ae_dir, "ae_re.npy"), np.asarray(ae_re))
            except Exception:
                print("Warning: failed to save ae_re.npy")
            try:
                if model is not None:
                    torch.save(model.state_dict(), os.path.join(ae_dir, "ae_model_state.pth"))
            except Exception:
                print("Warning: failed to save AE model state")

            # Write a simple text info file with autoencoder settings
            info_path = os.path.join(ae_dir, "ae_settings.txt")
            try:
                with open(info_path, "w") as info_f:
                    info_f.write(f"saved_time: {datetime.datetime.now().isoformat()}\n")
                    info_f.write(f"run_dir: {config.run_dir}\n")
                    info_f.write(f"ae_epochs: {config.ae_epochs}\n")
                    info_f.write(f"ae_batch_size: {config.ae_batch_size}\n")
                    info_f.write(f"ae_learning_rate: {config.ae_learning_rate}\n")
                    info_f.write(f"ae_bottleneck_dim: {config.ae_bottleneck_dim}\n")
                    info_f.write(f"ae_use_bottleneck: {config.ae_use_bottleneck}\n")
                    info_f.write(f"premade_source: {getattr(config, 'premade_autoencoder_results_dir', '')}\n")
                print(f"Saved autoencoder results to {ae_dir}")
            except Exception:
                print("Warning: failed to write AE settings info file")
        # optional reconstruction plots
        if config.plot_autoencoder_recons and model is not None:
            from plotting import plot_autoencoder_reconstructions
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # take a small batch for recon plotting
            sample_imgs = imgs[:16].to(device)
            sample_labels = labels[:16]
            plot_autoencoder_reconstructions(model, sample_imgs, sample_labels, device, config.run_dir)
        elif config.plot_autoencoder_recons and model is None:
            print("AE model checkpoint not found in premade directory; skipping reconstruction plots.")

    else:
        encoded_data = datas_scaled
        model = None

    # SOM / GSOM
    if config.perform_som_training:

        if config.som_type == "gsom":
            # ---- Growing SOM ----
            gsom, preds_coords, qe = gsom_module.train_gsom(encoded_data, labels, config)
            # For downstream code, convert (x,y) coords to integer node indices
            preds = gsom.predict_indices(encoded_data)
            plotting.plot_gsom_hits_map(gsom, encoded_data, config, labels=labels)
            gsom_hits = gsom.hits_map(encoded_data, labels=labels)
            active_nodes = len(gsom_hits)
            total_nodes = len(gsom.nodes)
            gsom_transform = gsom.transform(encoded_data)
            gsom_node_coords = [(n.x, n.y) for n in sorted(gsom.nodes, key=lambda n: (n.x, n.y))]
            sil_proxy, sil_samples = cluster_diagnostics.run_cluster_diagnostics(
                encoded_data,
                preds,
                gsom_transform,
                config,
                gsom_node_coords=gsom_node_coords,
            )
            _save_source_cluster_assignments(
                filenames,
                labels,
                preds,
                config,
                sil_scores=sil_samples,
                pred_coords=preds_coords,
            )
            print(f"GSOM quantization error: {qe:.4f}")
            print(f"GSOM node counts: total={total_nodes}, active(with hits)={active_nodes}")

            if config.plot_gsom_growth_animation:
                plotting.animate_gsom_growth(gsom, encoded_data, config, labels=labels)

            # Representative image grids on the GSOM topology
            if config.plot_representative_originals:
                plotting.plot_gsom_representative_originals(gsom, preds, encoded_data, filenames, labels, config, sil_scores=sil_samples)
            if config.plot_representative_processed:
                plotting.plot_gsom_representative_processed(gsom, preds, encoded_data, raw_datas, labels, config, sil_scores=sil_samples)
            if config.plot_representative_smoothed and config.include_smoothed_channel:
                plotting.plot_gsom_representative_smoothed(gsom, preds, encoded_data, filenames, labels, config, sil_scores=sil_samples)
            if config.plot_representative_molecular and config.include_molecular_line_channels:
                plotting.plot_gsom_representative_molecular(gsom, preds, encoded_data, filenames, labels, config, sil_scores=sil_samples)
            if getattr(config, "plot_representative_raw_moment0", False) and config.include_molecular_line_channels:
                plotting.plot_gsom_representative_raw_moment0(gsom, preds, encoded_data, filenames, labels, config, sil_scores=sil_samples)

            if getattr(config, "plot_top_populated_neuron_samples", True):
                plotting.plot_top_populated_neuron_samples(
                    preds,
                    raw_datas,
                    filenames,
                    labels,
                    sil_samples,
                    config,
                    mode_tag="gsom",
                    top_k=active_nodes,
                    n_show=9,
                )

            if config.cluster_info_saving:
                class_file = getattr(
                    config,
                    "classification_result_file",
                    os.path.join(config.run_dir, "COMBINEDclassification_info.txt")
                )
                number_labels, distances_og, classifications_og = save_results.read_cluster_info()
                classifications_ordered, matched_clustering, matched_files = save_results.order_values(
                    filenames, number_labels, distances_og, classifications_og, preds, config
                )
                save_results.write_classification_info(
                    classifications_ordered,
                    matched_clustering,
                    class_file,
                    cluster_no=len(gsom.nodes),
                    config=config,
                )

            if getattr(config, "run_embedding_diagnostics", False):
                try:
                    fov_by_filename = {
                        meta["filename"]: meta["orig_fov"]
                        for meta in (fov_info or {}).get("meta", [])
                        if "filename" in meta and "orig_fov" in meta
                    }
                    embedding_diagnostics.run_embedding_diagnostics(
                        encoded_data, preds, labels, filenames, raw_datas,
                        sil_samples, config, som=None,
                        fov_by_filename=fov_by_filename,
                    )
                except Exception:
                    traceback.print_exc()

            if getattr(config, "plot_evolutionary_stage", False):
                class_file = getattr(
                    config,
                    "classification_result_file",
                    os.path.join(config.run_dir, "COMBINEDclassification_info.txt")
                )

                if os.path.isfile(class_file):
                    plotting.plot_evolutionary_stage(
                        class_file,
                        config,
                        neuron_to_plot=getattr(config, "evo_neuron_to_plot", None),
                        gsom=gsom,
                    )
                else:
                    print(f"Classification file {class_file} not found; skipping GSOM evolutionary stage plot.")

        elif config.som_type == "som":
            # ---- Classic fixed-grid SOM ----
            som, preds, qe_r, qe_f, hits = som_module.train_som(encoded_data, labels, config)

            plotting.plot_u_matrix(som, config)
            plotting.plot_hits_map(som, encoded_data, config, labels=labels, preds=preds, exclude_augmented=False)

            som_transform = som.transform(encoded_data)
            sil_proxy, sil_samples = cluster_diagnostics.run_cluster_diagnostics(encoded_data, preds, som_transform, config)

            if config.plot_quantization_error:
                plotting.plot_qe(qe_r, qe_f, config)

            sil_scores = sil_samples
            if config.plot_per_neuron_examples:
                plotting.plot_original_examples(som, preds, filenames, labels, sil_scores, config)

            if getattr(config, "plot_top_populated_neuron_samples", True):
                plotting.plot_top_populated_neuron_samples(
                    preds,
                    raw_datas,
                    filenames,
                    labels,
                    sil_scores,
                    config,
                    mode_tag="som",
                    top_k=5,
                    n_show=9,
                )

            if config.plot_fov_matched_originals:
                plotting.plot_fov_matched_originals(som, preds, fov_matched, filenames, labels, sil_scores, config)



            if config.plot_processed_per_neuron_examples:
                plotting.plot_processed_examples(som, preds, raw_datas, labels, sil_scores, config)

            if config.plot_smoothed_per_neuron_examples and config.include_smoothed_channel:
                plotting.plot_smoothed_examples(som, preds, filenames, labels, sil_scores, config)

            if config.plot_molecular_per_neuron_examples and config.include_molecular_line_channels:
                plotting.plot_molecular_examples(som, preds, filenames, labels, sil_scores, config)

            if config.plot_representative_originals:
                plotting.plot_representative_grid_originals(preds, filenames, labels, sil_scores, config)

            if config.plot_representative_processed:
                plotting.plot_representative_grid_processed(preds, raw_datas, labels, sil_scores, config)

            if config.plot_representative_smoothed and config.include_smoothed_channel:
                plotting.plot_representative_grid_smoothed(preds, filenames, labels, sil_scores, config)

            if config.plot_representative_molecular and config.include_molecular_line_channels:
                plotting.plot_representative_grid_molecular(preds, filenames, labels, sil_scores, config)

            _save_source_cluster_assignments(filenames, labels, preds, config, sil_scores=sil_scores)

            if config.cluster_info_saving:
                save_results.save_all_results(filenames, preds, sil_proxy, sil_samples, config)

            if getattr(config, "run_embedding_diagnostics", False):
                try:
                    fov_by_filename = {
                        meta["filename"]: meta["orig_fov"]
                        for meta in (fov_info or {}).get("meta", [])
                        if "filename" in meta and "orig_fov" in meta
                    }
                    embedding_diagnostics.run_embedding_diagnostics(
                        encoded_data, preds, labels, filenames, raw_datas,
                        sil_samples, config, som=som,
                        fov_by_filename=fov_by_filename,
                    )
                except Exception:
                    traceback.print_exc()

            if getattr(config, "plot_evolutionary_stage", False):
                class_file = getattr(
                    config,
                    "classification_result_file",
                    os.path.join(config.run_dir, "COMBINEDclassification_info.txt")
                )

                if os.path.isfile(class_file):
                    plotting.plot_evolutionary_stage(
                        class_file,
                        config,
                        neuron_to_plot=getattr(config, "evo_neuron_to_plot", None)
                    )
                else:
                    print(f"Classification file {class_file} not found; skipping evolutionary stage plot.")

        elif config.som_type == "soumap":
            # SOUMAP consumes the same encoded_data matrix as SOM and GSOM.
            M = getattr(config, "soumap_prototypes", None)
            M = int(M if M is not None else config.n_num * config.m_num)
            print(f"SOUMAP selected: using M={M} prototypes.")
            soumap_result = soumap_module.train_soumap(encoded_data, labels, config)
            preds = soumap_result["bmu"]
            sil_proxy, sil_samples = cluster_diagnostics.run_cluster_diagnostics(
                encoded_data, preds, soumap_result["distances"], config
            )
            soumap_module.write_outputs(
                soumap_result, filenames, labels, sil_samples, config
            )
            if config.plot_per_neuron_examples:
                plotting.plot_original_examples(
                    None, preds, filenames, labels, sil_samples, config
                )
            if getattr(config, "plot_top_populated_neuron_samples", True):
                plotting.plot_top_populated_neuron_samples(
                    preds, raw_datas, filenames, labels, sil_samples, config,
                    mode_tag="soumap", top_k=min(5, soumap_result["M"]), n_show=9,
                )
            if config.plot_processed_per_neuron_examples:
                plotting.plot_processed_examples(
                    None, preds, raw_datas, labels, sil_samples, config
                )
            if config.plot_smoothed_per_neuron_examples and config.include_smoothed_channel:
                plotting.plot_smoothed_examples(
                    None, preds, filenames, labels, sil_samples, config
                )
            if config.plot_molecular_per_neuron_examples and config.include_molecular_line_channels:
                plotting.plot_molecular_examples(
                    None, preds, filenames, labels, sil_samples, config
                )
            print("SOUMAP skipped lattice-only plots: U-matrix, rectangular hits map, "
                  "neuron image map, and GSOM growth animation.")
            print("SOUMAP neighbour overlap is not computed; only direct assignments are available.")
            if getattr(config, "mpec_enabled", False):
                mpec = soumap_module.run_mpec(soumap_result, config)
                plotting.plot_soumap_mpec_fused_graph(soumap_result, mpec, config)

        else:
            raise ValueError(
                f"Unknown som_type={config.som_type!r}; expected 'som', 'gsom', or 'soumap'."
            )
        



if __name__ == "__main__":
    main()
