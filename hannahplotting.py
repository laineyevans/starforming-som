import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import torch
import re

AUGMENTATION_TAGS = ["[hflip]", "[vflip]", "[hvflip]", "[rot]"]

def plot_autoencoder_qe(ae_re, config):
    """
    Plot autoencoder reconstruction error (mean reconstruction error per epoch).
    """
    plt.figure()
    plt.plot(ae_re, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("AE Reconstruction Error")


    out = os.path.join(config.run_dir, "autoencoder_training_re.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved AE RE plot to {out}")

def plot_autoencoder_reconstructions(model, imgs, labels, device, save_dir, num_samples=4):
    model.eval()

    # only keep indices of non-augmented images
    augmentation_tags = ["[hflip]", "[vflip]", "[hvflip]", "[rot"]
    base_indices = [
        i for i, lbl in enumerate(labels)
        if not any(tag in lbl for tag in augmentation_tags)
    ]

    if len(base_indices) == 0:
        print("No non-augmented images available for AE reconstruction plot.")
        return

    # choose unique originals only
    choose_from = np.random.choice(base_indices, num_samples, replace=False)

    sample = imgs[choose_from].to(device)

    with torch.no_grad():
        recon = model(sample).cpu()

    sample = sample.cpu()

    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))

    for i in range(num_samples):
        axes[0, i].imshow(sample[i, 0], cmap='plasma')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        axes[1, i].imshow(recon[i, 0], cmap='plasma')
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis('off')

    plt.tight_layout()

    out = os.path.join(save_dir, "ae_reconstructions.png")
    plt.savefig(out, dpi=150)
    plt.close()



def plot_qe(qe_rough, qe_fine, config):
    qe = qe_rough + qe_fine

    plt.figure()
    plt.plot(qe)
    plt.axvline(len(qe_rough), color='r', linestyle='--')
    plt.xlabel("Block")
    plt.ylabel("Quantization Error")
    out = os.path.join(config.run_dir, "quantization_error.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved QE plot to {out}")

def plot_u_matrix(som, config):
    U = som.u_matrix()

    plt.figure(figsize=(6, 6))
    plt.imshow(U, cmap="viridis", origin="upper")
    plt.colorbar(label="Neighbour distance")

    out = os.path.join(config.run_dir, "u_matrix.png")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved U-matrix to {out}")

def plot_hits_map(som, encoded_data, config, labels=None, preds=None, exclude_augmented=True):
    """
    Plot SOM hits map with neuron coordinates:
    x: left → right (0 … m-1)
    y: bottom → top (0 … n-1)

    If labels/preds are provided and exclude_augmented is True, augmented samples
    are omitted to keep consistency with per-neuron example plots.
    """

    if exclude_augmented and labels is not None and preds is not None:
        hits = np.zeros((som.m, som.n), dtype=int)
        for lbl, p in zip(labels, preds):
            if lbl and any(tag in lbl for tag in AUGMENTATION_TAGS):
                continue
            r, c = som._locations[int(p)]
            hits[r, c] += 1
    else:
        hits = som.hits_map(encoded_data)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Use origin='upper' so row 0 appears at the top-left, matching neuron_id -> row mapping
    im = ax.imshow(
        hits,
        cmap="plasma",
        origin="upper",
        extent=(-0.5, config.m_num - 0.5,
                -0.5, config.n_num - 0.5)
    )

    ax.set_xticks(np.arange(config.m_num))
    ax.set_yticks(np.arange(config.n_num))

    ax.set_xlabel("Neuron x-dimension")
    ax.set_ylabel("Neuron y-dimension")

    fig.colorbar(im, ax=ax, label="Number of samples")

    ax.set_aspect("equal")

    out = os.path.join(config.run_dir, "hits_map.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)

    print(f"Saved hits map to {out}")


def _build_neuron_assignment(som_results, labels, config):
    """
    Build a dict mapping neuron_id → [sample indices]
    Only include non-augmented images.
    """
    neuron_map = {idx: [] for idx in range(config.n_num * config.m_num)}

    for i, lbl in enumerate(labels):
        if lbl and any(tag in lbl for tag in AUGMENTATION_TAGS):
            continue
        node = som_results[i]
        neuron_map[int(node)].append(i)

    return neuron_map


def _best_per_neuron_indices(som_results, labels, sil_scores, config):
    """Return best-silhouette sample index per neuron (non-augmented only)."""
    neuron_map = _build_neuron_assignment(som_results, labels, config)
    sil_scores = np.asarray(sil_scores)
    best = {}
    for nid, indices in neuron_map.items():
        if len(indices) == 0:
            continue
        best_idx = max(indices, key=lambda idx: sil_scores[idx])
        best[nid] = best_idx
    return best


def _extract_source_id_from_filename(config, path):
    """Extract source ID from continuum FITS filename.

    Examples
    --------
    100132_cont_7MTM2TM1_jointdeconv.image.fits -> 100132
    G010.3208-00.1570B_cont_7MTM2TM1_jointdeconv.image.fits -> G010.3208-00.1570B
    """
    base = os.path.basename(path)
    if config.data_is_simulated == False:
        if "_cont_" in base:
            return base.split("_cont_", 1)[0]
    else:
        if "_cnt_" in base:
            timestep = base.split("_cnt_", 1)[1].split("_", 1)[0]
            incl = base.split("_incl_", 1)[1].split("_", 1)[0]
            phi = base.split("_phi_", 1)[1].split("_", 1)[0]
            return f'{timestep}_{incl}_{phi}'
    return os.path.splitext(base)[0]


def _clean_label(label):
    if label is None:
        return ""
    cleaned = label
    for tag in AUGMENTATION_TAGS:
        cleaned = cleaned.replace(tag, "")
    return cleaned.strip()


def _infer_source_id(config, sample_idx, filenames=None, labels=None):
    def _try_extract(value):
        if not value:
            return None
        sid = _extract_source_id_from_filename(config, value)
        return sid if sid else None

    if filenames is not None:
        try:
            candidate = _try_extract(filenames[sample_idx])
            if candidate:
                return candidate
        except (IndexError, TypeError):
            pass

    if labels is not None:
        try:
            candidate = _try_extract(_clean_label(labels[sample_idx]))
            if candidate:
                return candidate
        except (IndexError, TypeError):
            pass

    return f"sample_{sample_idx:04d}"


def _format_image_subheading(sample_idx, sil_scores, config, filenames=None, labels=None):
    score = float(sil_scores[sample_idx])
    sid = _infer_source_id(config, sample_idx, filenames=filenames, labels=labels)
    return f"{sid} | S={score:.2f}"




def plot_original_examples(som, som_results, filenames, labels, sil_scores, config):
    """
    Plot the top-silhouette-score images per neuron.
    Only non-augmented images are included.
    """
    print("Plotting examples...")
    out_dir = os.path.join(config.run_dir, "neuron_original_examples")
    low_out_dir = os.path.join(config.run_dir, "neuron_lowest_original_examples") 
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(low_out_dir, exist_ok=True)

    neuron_map = _build_neuron_assignment(som_results, labels, config)

    sil_scores = np.asarray(sil_scores)

    for neuron_id, indices in neuron_map.items():
        if len(indices) == 0:
            continue

        # sort the indices belonging to this neuron by silhouette score (descending)
        ordered = sorted(indices, key=lambda i: sil_scores[i], reverse=True)

        # choose the best 16
        chosen = ordered[:16]
        if len(ordered) >= 32:
            lowest = ordered[-16:]
            fig = plt.figure(figsize=(16, 16))

            for ax_idx in range(16):
                ax = fig.add_subplot(4, 4, ax_idx + 1)
                ax.axis("off")

                if ax_idx >= len(lowest):
                    continue

                sample_idx = lowest[ax_idx]
                filename = filenames[sample_idx]

                try:
                    with fits.open(filename) as hdul:
                        img = hdul[0].data

                    img = np.nan_to_num(img, nan=0)
                    vmin_local = np.nanmin(img)
                    vmax_local = np.nanmax(img)

                    ax.imshow(img, cmap="plasma", origin="lower",
                            vmin=vmin_local, vmax=vmax_local)

                    ax.set_title(
                        _format_image_subheading(
                            sample_idx,
                            sil_scores,
                            config,
                            filenames=filenames,
                            labels=labels,
                        ),
                        fontsize=8
                    )


                except Exception as e:
                    print(f"Error plotting original image {filename}: {e}")


            
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            out_file = os.path.join(low_out_dir, f"neuron_{neuron_id:03d}_lowest_originals.png")
            fig.savefig(out_file)
            plt.close(fig)

        fig = plt.figure(figsize=(16, 16))

        for ax_idx in range(16):
            ax = fig.add_subplot(4, 4, ax_idx + 1)
            ax.axis("off")

            if ax_idx >= len(chosen):
                continue

            sample_idx = chosen[ax_idx]
            filename = filenames[sample_idx]

            try:
                with fits.open(filename) as hdul:
                    if config.data_is_simulated == True:
                        img = hdul[0].data[0]
                    else:
                        img = hdul[0].data

                    if config.data_is_simulated == True:
                        c = 40000000
                        c = 25500 / np.log(1 + np.nanmax(img)) * 0.1 #0.65
                        img = c * (np.log(img + 1))
                        img = np.array(img, dtype = np.uint8)

                img = np.nan_to_num(img, nan=0)
                vmin_local = np.nanmin(img)
                vmax_local = np.nanmax(img)

                ax.imshow(img, cmap="plasma", origin="lower",
                          vmin=vmin_local, vmax=vmax_local)

                ax.set_title(
                    _format_image_subheading(
                        sample_idx,
                        sil_scores,
                        config,
                        filenames=filenames,
                        labels=labels
                    ),
                    fontsize=8
                )


            except Exception as e:
                print(f"Error plotting original image {filename}: {e}")


        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_file = os.path.join(out_dir, f"neuron_{neuron_id:03d}_originals.png")
        fig.savefig(out_file)
        plt.close(fig)

        print(f"Saved silhouette-ranked originals for neuron {neuron_id} to {out_file}")


def plot_fov_matched_originals(som, som_results, fov_matched, filenames, labels, sil_scores, config):
    out_dir = os.path.join(config.run_dir, "neuron_fov_matched_originals")
    os.makedirs(out_dir, exist_ok=True)

    neuron_map = _build_neuron_assignment(som_results, labels, config)
    sil_scores = np.asarray(sil_scores)

    for neuron_id, indices in neuron_map.items():
        if len(indices) == 0:
            continue

        ordered = sorted(indices, key=lambda i: sil_scores[i], reverse=True)
        chosen = ordered[:16]

        fig = plt.figure(figsize=(16, 16))

        for ax_idx in range(16):
            ax = fig.add_subplot(4, 4, ax_idx + 1)
            ax.axis("off")

            if ax_idx >= len(chosen):
                continue

            sample_idx = chosen[ax_idx]
            fname = filenames[sample_idx]

            img = fov_matched[fname]
            img = np.nan_to_num(img, nan=0)

            vmin_local = np.nanmin(img)
            vmax_local = np.nanmax(img)

            ax.imshow(img, cmap="plasma", origin="lower",
                      vmin=vmin_local, vmax=vmax_local)

            ax.set_title(
                _format_image_subheading(
                    sample_idx,
                    sil_scores,
                    config,
                    filenames=filenames,
                    labels=labels
                ),
                fontsize=10
            )

        plt.suptitle(
            f"Top-Silhouette FOV-Matched Images for Neuron {neuron_id}",
            fontsize=12
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_file = os.path.join(out_dir, f"neuron_{neuron_id:03d}_fov_matched_originals.png")
        fig.savefig(out_file)
        plt.close(fig)

        print(f"Saved silhouette-ranked FOV-matched originals for neuron {neuron_id} to {out_file}")


def plot_processed_examples(som, som_results, raw_datas, labels, sil_scores, config):
    out_dir = os.path.join(config.run_dir, "neuron_processed_examples")
    os.makedirs(out_dir, exist_ok=True)

    neuron_map = _build_neuron_assignment(som_results, labels, config)
    sil_scores = np.asarray(sil_scores)

    for neuron_id, indices in neuron_map.items():
        if len(indices) == 0:
            continue

        ordered = sorted(indices, key=lambda i: sil_scores[i], reverse=True)
        chosen = ordered[:9]

        fig = plt.figure(figsize=(9, 9))

        for ax_idx in range(9):
            ax = fig.add_subplot(3, 3, ax_idx + 1)
            ax.axis("off")

            if ax_idx >= len(chosen):
                continue

            sample_idx = chosen[ax_idx]
            img = raw_datas[sample_idx].reshape(config.sqrt_pixel_number,
                                                config.sqrt_pixel_number)

            if config.data_is_simulated == True:
                    c = 40000000
                    c = 25500 / np.log(1 + np.nanmax(img)) * 0.1
                    img = c * (np.log(img + 1))
                    img = np.array(img, dtype = np.uint8)

            ax.imshow(np.nan_to_num(img), origin="lower", cmap="plasma",
                      vmin=0, vmax=1)

            ax.set_title(
                _format_image_subheading(
                    sample_idx,
                    sil_scores,
                    config,
                    labels=labels
                ),
                fontsize=10
            )

    
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_file = os.path.join(out_dir, f"neuron_{neuron_id:03d}_processed.png")
        fig.savefig(out_file)
        plt.close(fig)


def plot_smoothed_examples(som, som_results, filenames, labels, sil_scores, config):
    """Per-neuron examples for smoothed continuum images (from .npy files)."""
    if not hasattr(config, "smoothed_image_file_dir"):
        print("Config has no smoothed_image_file_dir; skipping smoothed examples.")
        return

    out_dir = os.path.join(config.run_dir, "neuron_smoothed_examples")
    os.makedirs(out_dir, exist_ok=True)

    neuron_map = _build_neuron_assignment(som_results, labels, config)
    sil_scores = np.asarray(sil_scores)

    for neuron_id, indices in neuron_map.items():
        if len(indices) == 0:
            continue

        ordered = sorted(indices, key=lambda i: sil_scores[i], reverse=True)
        chosen = ordered[:9]

        fig = plt.figure(figsize=(9, 9))

        for ax_idx in range(9):
            ax = fig.add_subplot(3, 3, ax_idx + 1)
            ax.axis("off")

            if ax_idx >= len(chosen):
                continue

            sample_idx = chosen[ax_idx]
            fname = filenames[sample_idx]
            sid = _extract_source_id_from_filename(config, fname)
            npy_name = f"{sid}_smoothed_noiseremoved1sigma.npy"
            npy_path = os.path.join(config.smoothed_image_file_dir, npy_name)
            if not os.path.isfile(npy_path):
                continue

            try:
                img = np.load(npy_path, allow_pickle=False)
            except Exception as e:
                print(f"Error loading smoothed image {npy_path}: {e}")
                continue

            img = np.nan_to_num(img, nan=0.0)
            ax.imshow(img, origin="lower", cmap="plasma", vmin=0.0, vmax=1.0)
            ax.set_title(
                _format_image_subheading(
                    sample_idx,
                    sil_scores,
                    config,
                    filenames=filenames,
                    labels=labels
                ),
                fontsize=10
            )

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_file = os.path.join(out_dir, f"neuron_{neuron_id:03d}_smoothed.png")
        fig.savefig(out_file)
        plt.close(fig)
        print(f"Saved smoothed examples for neuron {neuron_id} to {out_file}")


def plot_molecular_examples(som, som_results, filenames, labels, sil_scores, config):
    """Per-neuron examples for molecular-line moment-0 maps (H2CO, SiO, DCN)."""
    if not hasattr(config, "molecular_line_file_dir"):
        print("Config has no molecular_line_file_dir; skipping molecular examples.")
        return

    out_dir = os.path.join(config.run_dir, "neuron_molecular_examples")
    os.makedirs(out_dir, exist_ok=True)

    low_out_dir = os.path.join(config.run_dir, "neuron_molecular_lowest_examples")
    os.makedirs(low_out_dir, exist_ok=True)


    lines = ["H2CO", "SiO", "DCN"]
    line_dirs = {}
    for line in lines:
        line_path = os.path.join(out_dir, line)
        os.makedirs(line_path, exist_ok=True)
        line_dirs[line] = line_path
    low_line_dirs = {}
    for line in lines:
        line_path = os.path.join(low_out_dir, line)
        os.makedirs(line_path, exist_ok=True)
        low_line_dirs[line] = line_path

    neuron_map = _build_neuron_assignment(som_results, labels, config)
    sil_scores = np.asarray(sil_scores)

    max_examples = 9

    for neuron_id, indices in neuron_map.items():
        if len(indices) == 0:
            continue

        ordered = sorted(indices, key=lambda i: sil_scores[i], reverse=True)
        chosen = ordered[:max_examples]
        lowest = []
        if len(ordered) >= 2 * max_examples:
            lowest = ordered[-max_examples:]
            lowest_maps = _collect_sample_maps(lowest, filenames, labels, config, lines)
            for line in lines:
                line_samples = [idx for idx in lowest if lowest_maps.get(idx, {}).get(line) is not None]
                if not line_samples:
                    continue

                ncols = min(3, len(line_samples))
                nrows = int(np.ceil(len(line_samples) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
                axes = np.atleast_2d(axes).reshape(nrows, ncols)

                for ax in axes.flat:
                    ax.axis("off")

                for plot_idx, sample_idx in enumerate(line_samples):
                    ax = axes.flat[plot_idx]
                    img = lowest_maps[sample_idx][line]
                    ax.imshow(img, origin="lower", cmap="plasma", vmin=0.0, vmax=1.0)
                    ax.set_title(
                        _format_image_subheading(
                            sample_idx,
                            sil_scores,
                            config,
                            filenames=filenames,
                            labels=labels
                        ),
                        fontsize=9
                    )

                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                out_file = os.path.join(low_line_dirs[line], f"neuron_{neuron_id:03d}_{line}_lowest.png")
                fig.savefig(out_file)
                plt.close(fig)

        sample_maps = _collect_sample_maps(chosen, filenames, labels, config, lines)

        if not sample_maps:
            continue

        for line in lines:
            line_samples = [idx for idx in chosen if sample_maps.get(idx, {}).get(line) is not None]
            if not line_samples:
                continue

            ncols = min(3, len(line_samples))
            nrows = int(np.ceil(len(line_samples) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
            axes = np.atleast_2d(axes).reshape(nrows, ncols)

            for ax in axes.flat:
                ax.axis("off")

            for plot_idx, sample_idx in enumerate(line_samples):
                ax = axes.flat[plot_idx]
                img = sample_maps[sample_idx][line]
                ax.imshow(img, origin="lower", cmap="plasma", vmin=0.0, vmax=1.0)
                ax.set_title(
                    _format_image_subheading(
                        sample_idx,
                        sil_scores,
                        config,
                        filenames=filenames,
                        labels=labels
                    ),
                    fontsize=9
                )

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_file = os.path.join(line_dirs[line], f"neuron_{neuron_id:03d}_{line}.png")
            fig.savefig(out_file)
            plt.close(fig)
            print(f"Saved {line} examples for neuron {neuron_id} to {out_file}")

def _collect_sample_maps(sample_indices, filenames, labels, config, lines):
    sample_maps = {}
    for sample_idx in sample_indices:
        fname = filenames[sample_idx]
        sid = _extract_source_id_from_filename(config, fname)
        if not sid:
            sid = _infer_source_id(config, sample_idx, filenames=filenames, labels=labels)

        src_dir = os.path.join(config.molecular_line_file_dir, str(sid))
        if not os.path.isdir(src_dir):
            continue

        maps = {}
        try:
            for npy_name in os.listdir(src_dir):
                if not npy_name.endswith("_moment0_island.npy"):
                    continue
                full = os.path.join(src_dir, npy_name)
                arr = np.load(full, allow_pickle=False)
                arr = np.nan_to_num(arr, nan=0.0)

                for line in lines:
                    if line in npy_name:
                        maps[line] = arr
                        break
        except Exception as e:
            print(f"Error loading molecular maps for {src_dir}: {e}")
            continue

        if maps:
            sample_maps[sample_idx] = maps

    return sample_maps

def _make_grid_canvas(config):
    fig, axes = plt.subplots(
        config.n_num,
        config.m_num,
        figsize=(config.m_num * 3, config.n_num * 3)
    )
    if config.n_num == 1 and config.m_num == 1:
        axes = np.array([[axes]])
    elif config.n_num == 1:
        axes = axes.reshape(1, -1)
    elif config.m_num == 1:
        axes = axes.reshape(-1, 1)
    return fig, axes


def _grid_iter_axes(axes, config):
    for i in range(config.n_num):
        for j in range(config.m_num):
            yield i, j, axes[i, j], i * config.m_num + j


def plot_representative_grid_processed(som_results, raw_datas, labels, sil_scores, config):
    out_file = os.path.join(config.run_dir, "som_representative_grid_processed.png")
    best = _best_per_neuron_indices(som_results, labels, sil_scores, config)
    fig, axes = _make_grid_canvas(config)

    sil_scores = np.asarray(sil_scores)

    for i, j, ax, neuron_id in _grid_iter_axes(axes, config):
        ax.axis("off")
        if neuron_id not in best:
            continue
        idx = best[neuron_id]
        img = raw_datas[idx].reshape(config.sqrt_pixel_number, config.sqrt_pixel_number)
        if config.data_is_simulated == True:
            c = 40000000
            c = 25500 / np.log(1 + np.nanmax(img)) * 0.1
            img = c * (np.log(img + 1))
            img = np.array(img, dtype = np.uint8)
        img = np.nan_to_num(img)
        vmin_local = np.nanmin(img)
        vmax_local = np.nanmax(img)
        ax.imshow(img, cmap="plasma", origin="lower", vmin=vmin_local, vmax=vmax_local)
        ax.set_title(_format_image_subheading(idx, sil_scores,config, labels=labels), fontsize=10)

    plt.suptitle("Silhouette-Best Processed Image per SOM Neuron", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved processed representative SOM grid to {out_file}")


def plot_representative_grid_originals(som_results, filenames, labels, sil_scores, config):
    out_file = os.path.join(config.run_dir, "som_representative_grid_originals.png")
    best = _best_per_neuron_indices(som_results, labels, sil_scores, config)
    fig, axes = _make_grid_canvas(config)
    sil_scores = np.asarray(sil_scores)

    for i, j, ax, neuron_id in _grid_iter_axes(axes, config):
        ax.axis("off")
        if neuron_id not in best:
            continue
        idx = best[neuron_id]
        fname = filenames[idx]
        try:
            with fits.open(fname) as hdul:
                img = np.nan_to_num(hdul[0].data[0], nan=0)
                if config.data_is_simulated == True:
                    c = 40000000
                    c = 25500 / np.log(1 + np.nanmax(img)) * 0.1
                    img = c * (np.log(img + 1))
                    img = np.array(img, dtype = np.uint8)
            vmin_local = np.nanmin(img)
            vmax_local = np.nanmax(img)
            ax.imshow(img, cmap="plasma", origin="lower", vmin=vmin_local, vmax=vmax_local)
            ax.set_title(_format_image_subheading(idx, sil_scores,config, filenames=filenames, labels=labels), fontsize=8)
        except Exception as e:
            print(f"Error loading original image {fname}: {e}")
            continue

    plt.suptitle("Silhouette-Best Original Image per SOM Neuron", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved original representative SOM grid to {out_file}")


def plot_representative_grid_smoothed(som_results, filenames, labels, sil_scores, config):
    if not hasattr(config, "smoothed_image_file_dir"):
        print("Config has no smoothed_image_file_dir; skipping smoothed representative grid.")
        return
    out_file = os.path.join(config.run_dir, "som_representative_grid_smoothed.png")
    best = _best_per_neuron_indices(som_results, labels, sil_scores, config)
    fig, axes = _make_grid_canvas(config)
    sil_scores = np.asarray(sil_scores)

    for i, j, ax, neuron_id in _grid_iter_axes(axes, config):
        ax.axis("off")
        if neuron_id not in best:
            continue
        idx = best[neuron_id]
        fname = filenames[idx]
        sid = _extract_source_id_from_filename(config, fname)
        npy_name = f"{sid}_smoothed_noiseremoved1sigma.npy"
        npy_path = os.path.join(config.smoothed_image_file_dir, npy_name)
        if not os.path.isfile(npy_path):
            continue
        try:
            img = np.load(npy_path, allow_pickle=False)
        except Exception as e:
            print(f"Error loading smoothed image {npy_path}: {e}")
            continue
        img = np.nan_to_num(img, nan=0.0)
        ax.imshow(img, origin="lower", cmap="plasma", vmin=0.0, vmax=1.0)
        ax.set_title(_format_image_subheading(idx, sil_scores,config, filenames=filenames, labels=labels), fontsize=9)

    plt.suptitle("Silhouette-Best Smoothed Image per SOM Neuron", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved smoothed representative SOM grid to {out_file}")


def plot_representative_grid_molecular(som_results, filenames, labels, sil_scores, config):
    if not hasattr(config, "molecular_line_file_dir"):
        print("Config has no molecular_line_file_dir; skipping molecular representative grid.")
        return
    lines = ["H2CO", "SiO", "DCN"]
    best = _best_per_neuron_indices(som_results, labels, sil_scores, config)
    sil_scores = np.asarray(sil_scores)

    for line in lines:
        out_file = os.path.join(config.run_dir, f"som_representative_grid_{line}.png")
        fig, axes = _make_grid_canvas(config)
        for i, j, ax, neuron_id in _grid_iter_axes(axes, config):
            ax.axis("off")
            if neuron_id not in best:
                continue
            idx = best[neuron_id]
            fname = filenames[idx]
            sid = _extract_source_id_from_filename(config, fname)
            src_dir = os.path.join(config.molecular_line_file_dir, str(sid))
            if not os.path.isdir(src_dir):
                continue
            img = None
            try:
                for npy_name in os.listdir(src_dir):
                    if npy_name.endswith("_moment0_island.npy") and line in npy_name:
                        img = np.load(os.path.join(src_dir, npy_name), allow_pickle=False)
                        break
            except Exception as e:
                print(f"Error loading molecular map for {sid}: {e}")
                continue
            if img is None:
                continue
            img = np.nan_to_num(img, nan=0.0)
            ax.imshow(img, origin="lower", cmap="plasma", vmin=0.0, vmax=1.0)
            ax.set_title(_format_image_subheading(idx, sil_scores,config, filenames=filenames, labels=labels), fontsize=9)

        plt.suptitle(f"Silhouette-Best {line} Map per SOM Neuron", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_file)
        plt.close(fig)
        print(f"Saved {line} representative SOM grid to {out_file}")


# Backwards-compatible alias
def plot_representative_grid(som, som_results, raw_datas, labels, sil_scores, config):
    plot_representative_grid_processed(som_results, raw_datas, labels, sil_scores, config)


def plot_evolutionary_stage(classification_file, config, neuron_to_plot=None):
    """Plot dominant evolutionary stage per neuron and a per-neuron observed/expected bar chart.

    classification_file should match the format written by save_results.save_all_results
    (COMBINEDclassification_info.txt). If the file is missing, the plots are skipped.
    """
    if not os.path.isfile(classification_file):
        print(f"Classification file not found: {classification_file}; skipping evolutionary stage plot.")
        return

    clusters = {}

    with open(classification_file, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        m = re.match(r"cluster (\d+): Q:(\d+) P:(\d+) Y:(\d+) H:(\d+) na:(\d+)", line)
        if m:
            idx = int(m.group(1))
            obs = np.array(list(map(int, m.groups()[1:])))

            exp_line = lines[i + 1] if i + 1 < len(lines) else ""
            exp_vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", exp_line)
            exp = np.array(list(map(float, exp_vals))) if exp_vals else np.zeros_like(obs, dtype=float)

            chi_line = lines[i + 2] if i + 2 < len(lines) else ""
            chi_match = re.search(r"Chi-square:\s*([\d\.]+)", chi_line)
            p_match = re.search(r"p-value:\s*([\d\.]+)", chi_line)
            chi2 = float(chi_match.group(1)) if chi_match else np.nan
            pval = float(p_match.group(1)) if p_match else np.nan

            clusters[idx] = {
                "observed": obs,
                "expected": exp,
                "chi2": chi2,
                "p": pval
            }

            i += 3
        else:
            i += 1

    if not clusters:
        print(f"No cluster entries found in {classification_file}; skipping evolutionary stage plot.")
        return

    stages = ["Q", "P", "Y", "H", "na"]

    def _text_color(stage):
        return "white" if stage == "Q" else "black"

    plasma = plt.get_cmap("plasma")
    stage_colors = {
        "Q": plasma(0.15),
        "P": plasma(0.35),
        "Y": plasma(0.55),
        "H": plasma(0.75),
        "na": plasma(0.95),
    }

    from matplotlib.patches import Polygon, Rectangle

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    som_nx = config.m_num
    som_ny = config.n_num

    for idx, d in clusters.items():
        y = idx // som_nx
        x = idx % som_nx

        obs = d["observed"].astype(float)
        total = obs.sum()
        if total == 0:
            continue
        fracs = obs / total

        order = np.argsort(fracs)[::-1]
        primary = stages[order[0]]
        secondary = stages[order[1]] if len(order) > 1 else primary
        f1 = fracs[order[0]]
        f2 = fracs[order[1]] if len(order) > 1 else 0.0

        x0, x1 = x - 0.5, x + 0.5
        y0, y1 = y - 0.5, y + 0.5

        if primary == "na":
            tri_top = Polygon([(x0, y1), (x1, y1), (x1, y0)], closed=True,
                              facecolor=stage_colors["na"], edgecolor="none")
            ax.add_patch(tri_top)

            tri_bot = Polygon([(x0, y1), (x0, y0), (x1, y0)], closed=True,
                              facecolor=stage_colors[secondary], edgecolor="none")
            ax.add_patch(tri_bot)

            ax.text(x + 1/6, y + 1/6, f"na\n{f1:.2f}",
                    ha="center", va="center", fontsize=10, color="black", clip_on=True)
            ax.text(
                x - 1/6, y - 1/6,
                f"{secondary}\n{f2:.2f}",
                ha="center", va="center",
                fontsize=10, color=_text_color(secondary), clip_on=True
            )

        else:
            ax.add_patch(Rectangle((x0, y0), 1, 1, facecolor=stage_colors[primary], edgecolor="none"))
            ax.text(x, y, f"{primary}\n{f1:.2f}",
                    ha="center", va="center", fontsize=11, color="black", clip_on=True)

    ax.set_xlim(-0.5, som_nx - 0.5)
    ax.set_ylim(-0.5, som_ny - 0.5)
    ax.set_xticks(range(som_nx))
    ax.set_yticks(range(som_ny))
    ax.set_xlabel("SOM x", fontsize=16)
    ax.set_ylabel("SOM y", fontsize=16)
    ax.set_aspect("equal")

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=stage_colors[s], label=s)
        for s in stages
    ]
    ax.legend(
        handles=handles,
        title="Stage",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=12,
        title_fontsize=14,
        handlelength=1.6,
        handleheight=1.6
    )

    plt.tight_layout()
    grid_out = os.path.join(config.run_dir, "evolutionary_stage_grid.png")
    plt.savefig(grid_out, dpi=200)
    plt.close()
    print(f"Saved evolutionary stage grid to {grid_out}")

    if neuron_to_plot is None or neuron_to_plot not in clusters:
        return

    d = clusters[neuron_to_plot]
    labels = ["Q", "P", "Y", "H", "na"]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 7))
    plt.bar(x - width/2, d["observed"], width, label="Observed", color="rebeccapurple")
    plt.bar(x + width/2, d.get("expected", np.zeros_like(d["observed"])), width, label="Expected", color="lightsteelblue")

    plt.xticks(x, labels, fontsize=14)
    plt.ylabel("Count", fontsize=16)
    p_val = d.get("p", np.nan)
    title_text = rf"$p = {p_val:.4f}$" if np.isfinite(p_val) else "p = nan"
    plt.title(title_text, fontsize=16)
    plt.tick_params(axis="both", labelsize=12)
    plt.legend()
    plt.tight_layout()
    bar_out = os.path.join(config.run_dir, f"evolutionary_stage_neuron_{neuron_to_plot}.png")
    plt.savefig(bar_out, dpi=200)
    plt.close()
    print(f"Saved observed vs expected plot for neuron {neuron_to_plot} to {bar_out}")

# ------------------------------------------------------------------
# GSOM hit map
# ------------------------------------------------------------------

def plot_gsom_hits_map(gsom, encoded_data, config, labels=None):
    """
    Plot a hit map for a Growing SOM.

    Because the GSOM grid is irregular (nodes at arbitrary integer x,y
    positions), we render the map as a scatter plot sized and coloured
    by hit count.
    """
    hits = gsom.hits_map(encoded_data, labels=labels)

    # Build full node coordinate list (includes zero-hit nodes)
    all_coords = [(n.x, n.y) for n in gsom.nodes]
    if not all_coords:
        print("No GSOM nodes to plot.")
        return

    x_all = np.array([c[0] for c in all_coords], dtype=float)
    y_all = np.array([c[1] for c in all_coords], dtype=float)
    counts_all = np.array([hits.get(c, 0) for c in all_coords], dtype=float)

    # Adapt figure size to GSOM bounds so elongated maps stay readable.
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    x_span = max(1.0, x_max - x_min + 1.0)
    y_span = max(1.0, y_max - y_min + 1.0)
    aspect = x_span / y_span
    base = 9.0
    if aspect >= 1.0:
        fig_w = min(16.0, max(8.0, base * aspect))
        fig_h = max(7.0, fig_w / aspect)
    else:
        fig_h = min(16.0, max(8.0, base / aspect))
        fig_w = max(7.0, fig_h * aspect)

    # Keep empty nodes only when they border at least two non-empty neighbours.
    active_mask = counts_all > 0
    zero_mask = counts_all == 0
    zero_keep_mask = np.zeros_like(zero_mask, dtype=bool)
    for i, (x, y) in enumerate(all_coords):
        if not zero_mask[i]:
            continue
        n_non_empty_nb = 0
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if hits.get((x + dx, y + dy), 0) > 0:
                n_non_empty_nb += 1
        if n_non_empty_nb >= 2:
            zero_keep_mask[i] = True

    plot_mask = active_mask | zero_keep_mask
    plot_coords = {c for c, keep in zip(all_coords, plot_mask) if keep}

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw topology edges only for plotted nodes.
    drawn_edges = set()
    for node in gsom.nodes:
        if (node.x, node.y) not in plot_coords:
            continue
        for nb in (node.right, node.up):
            if nb is None or (nb.x, nb.y) not in plot_coords:
                continue
            key = tuple(sorted(((node.x, node.y), (nb.x, nb.y))))
            if key in drawn_edges:
                continue
            drawn_edges.add(key)
            ax.plot([node.x, nb.x], [node.y, nb.y], color="gray", linewidth=1.0, alpha=0.8, zorder=1)

    # Plot selected zero-hit nodes in light gray
    if np.any(zero_keep_mask):
        ax.scatter(
            x_all[zero_keep_mask],
            y_all[zero_keep_mask],
            s=120,
            c="lightgray",
            edgecolors="black",
            linewidths=0.5,
            marker="s",
            zorder=2,
            label="0 hits",
        )

    # Plot active nodes (hit > 0) with colour map and size by hit count
    active_count = int(np.sum(active_mask))
    n_nodes = len(gsom.nodes)
    base_size = float(np.clip(220.0 - 0.35 * n_nodes, 30.0, 140.0))
    active_gain = float(np.clip(520.0 - 0.55 * n_nodes, 90.0, 380.0))
    if np.any(active_mask):
        c_active = counts_all[active_mask]
        max_count = max(float(np.max(c_active)), 1.0)
        sizes = base_size + active_gain * (c_active / max_count)
        sc = ax.scatter(
            x_all[active_mask],
            y_all[active_mask],
            c=c_active,
            s=sizes,
            cmap="plasma",
            edgecolors="black",
            linewidths=0.7,
            marker="s",
            zorder=3,
            label=">0 hits",
        )
        fig.colorbar(sc, ax=ax, label="Number of samples")

    # Avoid unreadable clutter on large maps.
    if n_nodes <= 180:
        for i, node in enumerate(gsom.nodes):
            if (node.x, node.y) not in plot_coords:
                continue
            cnt = int(hits.get((node.x, node.y), 0))
            ax.annotate(
                f"{i}\n({cnt})",
                xy=(node.x, node.y),
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                zorder=5,
            )
    elif n_nodes <= 400:
        for i, node in enumerate(gsom.nodes):
            if (node.x, node.y) not in plot_coords:
                continue
            cnt = int(hits.get((node.x, node.y), 0))
            if cnt <= 0:
                continue
            ax.annotate(
                f"{cnt}",
                xy=(node.x, node.y),
                ha="center",
                va="center",
                fontsize=6,
                color="black",
                zorder=5,
            )

    total_nodes = len(gsom.nodes)
    ax.set_xlabel("Node x")
    ax.set_ylabel("Node y")
    ax.set_title(f"GSOM topology + hits (total nodes={total_nodes}, active nodes={active_count})")
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Put y increasing upward so the initial square appears as (0,0)-(1,1)
    ax.invert_yaxis()

    # Axis padding so edge nodes are not clipped
    if plot_coords:
        x_plot = np.array([c[0] for c in plot_coords], dtype=float)
        y_plot = np.array([c[1] for c in plot_coords], dtype=float)
        x_min, x_max = float(np.min(x_plot)), float(np.max(x_plot))
        y_min, y_max = float(np.min(y_plot)), float(np.max(y_plot))
    ax.set_xlim(x_min - 0.8, x_max + 0.8)
    ax.set_ylim(y_max + 0.8, y_min - 0.8)

    ax.legend(loc="best", fontsize=8)

    out = os.path.join(config.run_dir, "gsom_hits_map.png")
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close(fig)
    print(f"Saved GSOM hits map to {out} (total nodes={total_nodes}, active nodes={active_count})")


# ------------------------------------------------------------------
# GSOM representative image grids
# ------------------------------------------------------------------

def _gsom_best_per_node(gsom, preds, labels, encoded_data, sil_score):
    """Pick one representative sample per active GSOM node.

    For each node that has at least one non-augmented sample mapped to it,
    choose the sample closest to the node's weight vector (lowest quantization
    error).  Returns dict  node_index → sample_index.
    """
    from collections import defaultdict
    node_samples = defaultdict(list)
    for i, node_idx in enumerate(preds):
        lbl = labels[i] if labels is not None else ""
        if lbl and any(tag in lbl for tag in AUGMENTATION_TAGS):
            continue
        node_samples[int(node_idx)].append(i)

    encoded_data = np.asarray(encoded_data)
    best = {}
    for nid, indices in node_samples.items():
        if not indices:
            continue
        node = gsom.nodes[nid]
        # pick sample with smallest distance to the node's weights
        #dists = [float(np.sum((encoded_data[i] - node.weights) ** 2)) for i in indices]
        #best[nid] = indices[int(np.argmin(dists))]


        node_sil_scores = [sil_score[i] for i in indices]
        best_idx = int(np.argmax(node_sil_scores))
        best[nid] = indices[best_idx]
    return best

def _gsom_image_grid(gsom, best, load_image_fn, title, out_path, thumb_px=64):
    """Render a GSOM grid with one thumbnail per active node.

    Parameters
    ----------
    gsom : GSOM object
    best : dict  node_index → sample_index  (from _gsom_best_per_node)
    load_image_fn : callable(sample_idx) → 2D numpy array or None
    title : str
    out_path : str   file path to save to
    thumb_px : int   uniform thumbnail side length in pixels (default 64)
    """
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from scipy.ndimage import zoom as ndizoom

    if not best:
        print(f"No samples to plot for {title}; skipping.")
        return

    # Collect only nodes that have a best sample AND a loadable image.
    # Resize every image to (thumb_px, thumb_px) so all thumbnails are uniform.
    items = []   # (node_x, node_y, img_array)
    for nid, sample_idx in best.items():
        node = gsom.nodes[nid]
        img = load_image_fn(sample_idx)
        if img is None:
            continue
        img = np.nan_to_num(np.asarray(img, dtype=float), nan=0.0)
        if img.ndim > 2:
            img = np.squeeze(img)
        if img.ndim != 2:
            continue
        # Resize to uniform thumbnail
        if img.shape[0] != thumb_px or img.shape[1] != thumb_px:
            zy = thumb_px / max(img.shape[0], 1)
            zx = thumb_px / max(img.shape[1], 1)
            img = ndizoom(img, (zy, zx), order=1)
        items.append((node.x, node.y, img))

    if not items:
        print(f"No loadable images for {title}; skipping.")
        return

    xs = np.array([it[0] for it in items], dtype=float)
    ys = np.array([it[1] for it in items], dtype=float)

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_span = max(1.0, x_max - x_min + 1.0)
    y_span = max(1.0, y_max - y_min + 1.0)

    # Figure sizing: allocate compact space per cell so images are smaller
    cell = 0.8
    fig_w = max(6.0, min(48.0, x_span * cell + 2.0))
    fig_h = max(5.0, min(48.0, y_span * cell + 2.0))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Topology edges (all nodes, faint background)
    drawn = set()
    for node in gsom.nodes:
        for nb in (node.right, node.up):
            if nb is None:
                continue
            key = tuple(sorted(((node.x, node.y), (nb.x, nb.y))))
            if key in drawn:
                continue
            drawn.add(key)
            ax.plot([node.x, nb.x], [node.y, nb.y],
                    color="lightgray", linewidth=0.5, alpha=0.5, zorder=0)

    # Compute zoom so that each thumbnail fills roughly one grid cell.
    # Use a reasonable figure DPI and compute axis pixel width from it.
    dpi = 75
    ax_width_px = fig_w * dpi
    px_per_unit = ax_width_px / max(x_span + 1.6, 1.0)  # pixels per data-unit
    target_px = px_per_unit * 0.88  # thumbnail target size in figure pixels
    zoom = float(np.clip(target_px / float(max(1, thumb_px)), 0.25, 2.5))

    for nx, ny, img in items:
        oi = OffsetImage(img, zoom=zoom, cmap="plasma", origin="lower")
        ab = AnnotationBbox(oi, (nx, ny), frameon=True,
                            bboxprops=dict(edgecolor="black", linewidth=0.4),
                            pad=0.0)
        ax.add_artist(ab)

    ax.set_xlim(x_min - 0.8, x_max + 0.8)
    ax.set_ylim(y_max + 0.8, y_min - 0.8)  # invert y
    ax.set_aspect("equal")
    ax.set_xlabel("Node x", fontsize=16)
    ax.set_ylabel("Node y", fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.grid(True, linewidth=0.2, alpha=0.3)

    plt.tight_layout()
    # Crop extra whitespace and use a very small pad to minimize output size
    fig.savefig(out_path, dpi=int(dpi), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {title} to {out_path}")


def plot_gsom_representative_originals(gsom, preds, encoded_data, filenames, labels, config, sil_score):
    """Original FITS images placed on the GSOM topology grid."""
    best = _gsom_best_per_node(gsom, preds, labels, encoded_data, sil_score)

    def load_fn(idx):
        try:
            with fits.open(filenames[idx]) as hdul:
                img = np.nan_to_num(hdul[0].data[0], nan=0)
                if config.data_is_simulated == True:
                    c = 40000000
                    c = 25500 / np.log(1 + np.nanmax(img)) * 0.1 #0.65
                    img = c * (np.log(img + 1))
                    img = np.array(img, dtype = np.uint8)
                    return img
                return np.nan_to_num(hdul[0].data, nan=0.0)
        except Exception:
            return None

    _gsom_image_grid(
        gsom, best, load_fn,
        title="GSOM Representative Originals",
        out_path=os.path.join(config.run_dir, "gsom_representative_originals.png"),
    )


def plot_gsom_representative_processed(gsom, preds, encoded_data, raw_datas, labels, config, sil_score):
    """Preprocessed (segmented) images placed on the GSOM topology grid."""
    best = _gsom_best_per_node(gsom, preds, labels, encoded_data, sil_score)
    H = config.sqrt_pixel_number

    def load_fn(idx):
        try:
            return raw_datas[idx].reshape(H, H)
        except Exception:
            return None

    _gsom_image_grid(
        gsom, best, load_fn,
        title="GSOM Representative Processed",
        out_path=os.path.join(config.run_dir, "gsom_representative_processed.png"),
    )


def plot_gsom_representative_smoothed(gsom, preds, encoded_data, filenames, labels, config, sil_score):
    """Smoothed noise-removed images placed on the GSOM topology grid."""
    if not hasattr(config, "smoothed_image_file_dir"):
        print("Config has no smoothed_image_file_dir; skipping GSOM smoothed grid.")
        return
    best = _gsom_best_per_node(gsom, preds, labels, encoded_data, sil_score)

    def load_fn(idx):
        sid = _extract_source_id_from_filename(config, filenames[idx])
        npy_path = os.path.join(
            config.smoothed_image_file_dir,
            f"{sid}_smoothed_noiseremoved1sigma.npy",
        )
        if not os.path.isfile(npy_path):
            return None
        try:
            return np.load(npy_path, allow_pickle=False)
        except Exception:
            return None

    _gsom_image_grid(
        gsom, best, load_fn,
        title="GSOM Representative Smoothed",
        out_path=os.path.join(config.run_dir, "gsom_representative_smoothed.png"),
    )


def plot_gsom_representative_molecular(gsom, preds, encoded_data, filenames, labels, config, sil_score):
    """Molecular line moment-0 maps placed on the GSOM topology grid (one plot per line)."""
    if not hasattr(config, "molecular_line_file_dir"):
        print("Config has no molecular_line_file_dir; skipping GSOM molecular grids.")
        return
    best = _gsom_best_per_node(gsom, preds, labels, encoded_data, sil_score)

    for line in ("H2CO", "SiO", "DCN"):
        def load_fn(idx, _line=line):
            sid = _extract_source_id_from_filename(config, filenames[idx])
            src_dir = os.path.join(config.molecular_line_file_dir, str(sid))
            if not os.path.isdir(src_dir):
                return None
            try:
                for name in os.listdir(src_dir):
                    if name.endswith("_moment0_island.npy") and _line in name:
                        return np.load(os.path.join(src_dir, name), allow_pickle=False)
            except Exception:
                return None
            return None

        _gsom_image_grid(
            gsom, best, load_fn,
            title=f"GSOM Representative {line}",
            out_path=os.path.join(config.run_dir, f"gsom_representative_{line}.png"),
        )

def animate_gsom_growth(gsom, encoded_data, config, labels=None):
    """Produce a GIF showing GSOM topology and hit-proportion evolution.

    Each frame corresponds to the map state immediately after a new node was
    inserted.  Two panels are shown per frame:

        Left  — U-matrix: nodes coloured by mean weight-distance to direct
                neighbours (reveals cluster boundaries forming).
        Right — Hit proportions: nodes sized and coloured by the share of
                data points they attract (using the weights at that snapshot).

    Requires Pillow:  pip install Pillow
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    snapshots = gsom.growth_snapshots
    if not snapshots:
        print("No GSOM growth snapshots recorded; skipping growth animation.")
        return

    augmentation_tags = ["[hflip]", "[vflip]", "[hvflip]", "[rot"]
    X = np.asarray(encoded_data, dtype=np.float64)

    # Build a boolean mask to exclude augmented samples
    valid_mask = np.ones(len(X), dtype=bool)
    if labels is not None:
        for i, lbl in enumerate(labels):
            if lbl and any(tag in lbl for tag in augmentation_tags):
                valid_mask[i] = False
    X_valid = X[valid_mask]

    print(f"  Computing hit maps for {len(snapshots)} GSOM snapshots "
          f"({len(X_valid)} valid samples)…")

    # Pre-compute hit counts for every snapshot using that snapshot's weights.
    # Vectorised over samples: (N, K, dim) → (N, K) distances → argmin.
    all_hits = []
    for snap in snapshots:
        W = snap["weights"].astype(np.float64)          # (K, dim)
        diff = X_valid[:, np.newaxis, :] - W[np.newaxis, :, :]   # (N, K, dim)
        dists = np.sum(diff ** 2, axis=2)               # (N, K)
        bmu_idx = np.argmin(dists, axis=1)              # (N,)
        hits = np.bincount(bmu_idx, minlength=snap["n_nodes"])
        all_hits.append(hits)

    # Track which frames correspond to topology changes (new node added)
    node_counts = [s["n_nodes"] for s in snapshots]
    topology_change_frames = set(
        i for i in range(1, len(snapshots))
        if node_counts[i] != node_counts[i - 1]
    )

    # Fixed spatial bounds across all frames so the layout stays stable
    all_xs = [p[0] for snap in snapshots for p in snap["positions"]]
    all_ys = [p[1] for snap in snapshots for p in snap["positions"]]
    x_pad = max(0.8, (max(all_xs) - min(all_xs)) * 0.18)
    y_pad = max(0.8, (max(all_ys) - min(all_ys)) * 0.18)
    x_lim = (min(all_xs) - x_pad, max(all_xs) + x_pad)
    y_lim = (min(all_ys) - y_pad, max(all_ys) + y_pad)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    fig.subplots_adjust(top=0.88, bottom=0.04, left=0.04, right=0.96)

    def update(frame_idx):
        snap = snapshots[frame_idx]
        hits = all_hits[frame_idx]
        positions = snap["positions"]
        xs = np.array([p[0] for p in positions], dtype=float)
        ys = np.array([p[1] for p in positions], dtype=float)
        total = max(int(hits.sum()), 1)
        proportions = hits / total

        ax.clear()
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect("equal")
        ax.axis("off")
        # topology edges
        for (i, j) in snap["edges"]:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                    color="gray", lw=1.0, alpha=0.6, zorder=1)

        # Hit proportions — size and colour encode share of data
        sizes = 60 + 500 * proportions
        ax.scatter(xs, ys, c=proportions, s=sizes, cmap="plasma",
                   vmin=0, edgecolors="black", linewidths=0.6, zorder=2)
        ax.set_title(f"Hit proportions  ({snap['n_nodes']} nodes)", fontsize=11)

        fig.suptitle(
            f"GSOM training  —  checkpoint {frame_idx + 1} / {len(snapshots)}"
            + (f"  [+1 node → {snap['n_nodes']}]" if frame_idx in topology_change_frames else f"  ({snap['n_nodes']} nodes)"),
            fontsize=12, fontweight="bold",
        )

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=700, repeat=True)

    save_path = os.path.join(config.run_dir, "gsom_growth_animation.gif")
    try:
        anim.save(save_path, writer=PillowWriter(fps=2))
        print(f"Saved GSOM growth animation → {save_path}")
    except Exception as e:
        print(f"Warning: failed to save GSOM growth animation ({e}). "
              "Ensure Pillow is installed: pip install Pillow")
    finally:
        plt.close(fig)

