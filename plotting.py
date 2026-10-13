import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import torch
import re
import glob
from plot_coordinates import (
    configure_fixed_som_axes,
    fixed_som_subplot_index,
)

AUGMENTATION_TAGS = ["[hflip]", "[vflip]", "[hvflip]", "[rot]"]
SIM_FRAME_RE = re.compile(
    r"_update_cnt_(?P<frame>\d+).*_incl_(?P<incl>-?\d+)_phi_(?P<phi>-?\d+)_"
)


def _ensure_plot_subdir(config, *parts):
    out_dir = os.path.join(config.run_dir, *parts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_autoencoder_qe(ae_re, config):
    """
    Plot autoencoder reconstruction error (mean reconstruction error per epoch).
    """
    os.makedirs(config.run_dir, exist_ok=True)
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
    base_indices = [
        i for i, lbl in enumerate(labels)
        if not any(tag in lbl for tag in AUGMENTATION_TAGS)
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
    plt.imshow(U, cmap="viridis_r", origin="lower")
    plt.colorbar(label="Neighbour distance")
    configure_fixed_som_axes(plt.gca(), som.n, som.m)

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

    im = ax.imshow(
        hits,
        cmap="plasma",
        origin="lower",
        extent=(-0.5, config.m_num - 0.5,
                -0.5, config.n_num - 0.5)
    )

    configure_fixed_som_axes(ax, config.m_num, config.n_num)

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
    Include all samples (original + augmented).
    """
    if getattr(config, "som_type", None) == "soumap":
        return _build_neuron_assignment_from_preds(som_results, labels)
    neuron_map = {idx: [] for idx in range(config.n_num * config.m_num)}

    for i, lbl in enumerate(labels):
        node = som_results[i]
        neuron_map[int(node)].append(i)

    return neuron_map


def _build_neuron_assignment_from_preds(preds, labels):
    """Build prototype → sample-index assignments from BMU predictions.

    Unlike the fixed-grid helper, this supports an arbitrary number of
    prototypes (and therefore SOUMAP's non-lattice prototype coordinates).
    """
    preds = np.asarray(preds).reshape(-1)
    neuron_map = {}
    for sample_idx, prototype in enumerate(preds):
        try:
            prototype = int(prototype)
        except (TypeError, ValueError):
            continue
        neuron_map.setdefault(prototype, []).append(sample_idx)
    return neuron_map


def plot_soumap_mpec_fused_graph(soumap_result, mpec, config):
    """Plot MPEC's weighted fused graph with its final prototype labels."""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.colors import Normalize

    graph = getattr(mpec, "fused_graph_", None)
    coords = np.asarray(soumap_result.get("coords", []), dtype=float)
    active = np.asarray(getattr(mpec, "active_indices_", []), dtype=int).reshape(-1)
    labels = np.asarray(getattr(mpec, "labels_", []))
    if graph is None:
        graph_size = len(active) if len(active) else len(labels)
    else:
        graph_size = int(graph.shape[0])
    if graph_size == 0:
        active = np.empty(0, dtype=int)
    elif len(active) != graph_size:
        active = np.arange(graph_size, dtype=int)
    if len(active) and coords.ndim == 2 and coords.shape[0] > int(active.max()):
        positions = coords[active, :2]
    else:
        positions = np.empty((graph_size, 2), dtype=float)
    valid_positions = (
        positions.shape == (graph_size, 2) and np.isfinite(positions).all()
    )
    if not valid_positions:
        # A deterministic layout keeps diagnostics useful when coordinates are
        # absent, malformed, or contain NaNs.
        if graph_size:
            theta = np.linspace(0, 2 * np.pi, graph_size, endpoint=False)
            positions = np.column_stack((np.cos(theta), np.sin(theta)))
        else:
            positions = np.empty((0, 2))

    edge_segments, edge_weights = [], []
    if graph is not None and graph_size:
        coo = graph.tocoo()
        for left, right, weight in zip(coo.row, coo.col, coo.data):
            if left >= right or not np.isfinite(weight):
                continue
            edge_segments.append([positions[left], positions[right]])
            edge_weights.append(float(weight))

    output_dir = _ensure_plot_subdir(
        config, getattr(config, "mpec_output_dir", None) or "mpec_plots"
    )
    out_file = os.path.join(output_dir, "mpec_fused_graph.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    if edge_weights:
        lo, hi = min(edge_weights), max(edge_weights)
        if lo == hi:
            delta = max(abs(lo) * 0.05, 1e-12)
            lo, hi = lo - delta, hi + delta
        norm = Normalize(vmin=lo, vmax=hi)
        collection = LineCollection(
            edge_segments, cmap="viridis", norm=norm, linewidths=1.5,
            alpha=0.8,
        )
        collection.set_array(np.asarray(edge_weights))
        ax.add_collection(collection)
        fig.colorbar(collection, ax=ax, label="MPEC fused edge weight")

    max_active = int(active.max()) if len(active) else -1
    cluster_values = labels[active] if len(labels) > max_active else np.full(graph_size, -1)
    normalized_clusters = []
    for value in cluster_values:
        try:
            value = int(value)
        except (TypeError, ValueError, OverflowError):
            value = -1
        normalized_clusters.append(value if value >= 0 else -1)
    cluster_values = np.asarray(normalized_clusters, dtype=int)
    unique_clusters = sorted(set(cluster_values[cluster_values >= 0].tolist()))
    cmap = plt.get_cmap("tab20", max(len(unique_clusters), 1))
    colors = {cluster: cmap(i) for i, cluster in enumerate(unique_clusters)}
    for cluster in unique_clusters:
        mask = cluster_values == cluster
        ax.scatter(positions[mask, 0], positions[mask, 1], s=70,
                   color=colors[cluster], edgecolor="black", linewidth=0.4)
    if graph_size and not unique_clusters:
        ax.scatter(positions[:, 0], positions[:, 1], s=70, color="lightgray",
                   edgecolor="black", linewidth=0.4)
    if unique_clusters:
        ax.legend(
            handles=[Line2D([0], [0], marker="o", linestyle="", color="w",
                            markerfacecolor=colors[c], markeredgecolor="black",
                            label=f"Cluster {c}") for c in unique_clusters],
            title="Prototype cluster", loc="best",
        )
    ax.set_title("SOUMAP MPEC fused graph")
    ax.set_xlabel("Prototype coordinate 1")
    ax.set_ylabel("Prototype coordinate 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved SOUMAP MPEC fused graph to {out_file}")
    return out_file


def _best_per_neuron_indices(som_results, labels, sil_scores, config):
    """Return best-silhouette sample index per neuron (non-augmented only)."""
    neuron_map = _build_neuron_assignment(som_results, labels, config)
    sil_scores = np.asarray(sil_scores)
    best = {}
    for nid, indices in neuron_map.items():
        if len(indices) == 0:
            continue
        indices = _prefer_original_indices(
            indices,
            labels,
            allow_augmented=getattr(config, "use_simulated_data", False),
        )
        if len(indices) == 0:
            continue
        best_idx = max(indices, key=lambda idx: sil_scores[idx])
        best[nid] = best_idx
    return best


def _extract_source_id_from_filename(path):
    """Extract source ID from continuum FITS filename.

    Examples
    --------
    100132_cont_7MTM2TM1_jointdeconv.image.fits -> 100132
    G010.3208-00.1570B_cont_7MTM2TM1_jointdeconv.image.fits -> G010.3208-00.1570B
    """
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]

    for marker in ("_cont_", "_continuum_preproc", "_continuum", "_smoothed", "_moment0"):
        if marker in stem:
            return stem.split(marker, 1)[0]

    return stem


def _resolve_original_fits_path(fname, config, cache=None):
    """Resolve a continuum FITS path from stored filename metadata.

    Only searches original-data locations (no preprocessed fallback).
    """
    key = str(fname)
    if cache is not None and key in cache:
        return cache[key]

    base = os.path.basename(key)
    candidates = [key]

    data_dir = getattr(config, "data_dir", None)
    if data_dir:
        candidates.extend([
            os.path.join(data_dir, key),
            os.path.join(data_dir, base),
        ])

    for path in candidates:
        if path and os.path.isfile(path):
            if cache is not None:
                cache[key] = path
            return path

    if data_dir and os.path.isdir(data_dir):
        matches = glob.glob(os.path.join(data_dir, "**", base), recursive=True)
        if matches:
            if cache is not None:
                cache[key] = matches[0]
            return matches[0]

    if cache is not None:
        cache[key] = None
    return None


def _load_smoothed_image(config, source_id):
    """Load one smoothed continuum map for a source (or None)."""
    smoothed_dir = getattr(config, "smoothed_image_file_dir", None)
    if not smoothed_dir:
        return None

    candidates = [
        os.path.join(smoothed_dir, f"{source_id}_smoothed_noiseremoved1sigma.npy"),
        os.path.join(smoothed_dir, f"{source_id}_smoothed_island.npy"),
    ]
    for npy_path in candidates:
        if not os.path.isfile(npy_path):
            continue
        try:
            img = np.load(npy_path, allow_pickle=False)
        except Exception:
            continue
        img = np.nan_to_num(np.asarray(img), nan=0.0)
        if img.ndim > 2:
            img = np.squeeze(img)
        if img.ndim != 2:
            continue
        return img
    return None


def _build_node_assignment_from_preds(preds, labels=None):
    """Build node -> sample-index mapping from prediction indices (all samples)."""
    node_map = {}
    for i, node_idx in enumerate(preds):
        node_map.setdefault(int(node_idx), []).append(i)
    return node_map


def _save_ranked_panel(indices, sil_scores, labels, filenames, out_path, title, image_loader, config=None,
                       overlay_cores=False):
    fig = plt.figure(figsize=(9, 9))
    for ax_idx in range(9):
        ax = fig.add_subplot(3, 3, ax_idx + 1)
        ax.axis("off")
        if ax_idx >= len(indices):
            continue
        sample_idx = indices[ax_idx]
        img = image_loader(sample_idx)
        if img is None:
            continue
        img = np.nan_to_num(np.asarray(img), nan=0.0)
        if img.ndim > 2:
            img = np.squeeze(img)
        if img.ndim != 2:
            continue
        simulated = config is not None and getattr(config, "use_simulated_data", False)
        if simulated:
            img = _normalize_simulated_plot(img)
        cmap = "plasma"
        if overlay_cores and not simulated:
            _imshow_processed_with_core_overlay(ax, img, cmap=cmap)
        else:
            ax.imshow(img, origin="lower", cmap=cmap)
        ax.set_title(
            _format_image_subheading(
                sample_idx,
                sil_scores,
                filenames=filenames,
                labels=labels,
                config=config,
            ),
            fontsize=9,
        )

    plt.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_populated_neuron_samples(preds, raw_datas, filenames, labels, sil_scores, config, mode_tag="som", top_k=5, n_show=9):
    """Save top/bottom silhouette panels for top-populated neurons, split by channel and node."""
    node_map = _build_node_assignment_from_preds(preds, labels)
    if not node_map:
        print("No valid neuron assignments found for top-populated plots.")
        return

    ranked_nodes = sorted(node_map.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_k]
    sil_scores = np.asarray(sil_scores)

    per_neuron_root = _ensure_plot_subdir(config, "per_neuron_plots")
    base_dir = os.path.join(per_neuron_root, f"{mode_tag}_top_populated_neuron_samples")
    os.makedirs(base_dir, exist_ok=True)

    channel_loaders = {
        "continuum": lambda idx: raw_datas[idx],
    }

    if getattr(config, "include_smoothed_channel", False) and hasattr(config, "smoothed_image_file_dir"):
        def _load_smoothed(idx):
            sid = _extract_source_id_from_filename(filenames[idx])
            candidates = [
                os.path.join(config.smoothed_image_file_dir, f"{sid}_smoothed_noiseremoved1sigma.npy"),
                os.path.join(config.smoothed_image_file_dir, f"{sid}_smoothed_island.npy"),
            ]
            for path in candidates:
                if not os.path.isfile(path):
                    continue
                try:
                    img = np.load(path, allow_pickle=False)
                    return _apply_label_flip(img, labels[idx] if labels is not None else "")
                except Exception:
                    continue
            return None
        channel_loaders["smoothed"] = _load_smoothed

    if getattr(config, "include_molecular_line_channels", False):
        for line in ("H2CO", "SiO", "DCN"):
            channel_loaders[line] = (lambda ln: (lambda idx: _apply_label_flip(
                _load_molecular_line_image(
                    config,
                    _extract_source_id_from_filename(filenames[idx]),
                    ln,
                ),
                labels[idx] if labels is not None else "",
            )))(line)

    for rank, (neuron_id, indices) in enumerate(ranked_nodes, start=1):
        indices = _prefer_original_indices(
            indices,
            labels,
            allow_augmented=getattr(config, "use_simulated_data", False),
        )
        ordered = sorted(indices, key=lambda idx: sil_scores[idx], reverse=True)
        highest = ordered[:n_show]
        lowest = list(reversed(ordered[-n_show:]))
        pop = len(indices)

        node_dir = os.path.join(base_dir, f"node_{neuron_id:03d}_rank_{rank:02d}_n{pop}")
        hi_dir = os.path.join(node_dir, "highest")
        lo_dir = os.path.join(node_dir, "lowest")
        os.makedirs(hi_dir, exist_ok=True)
        os.makedirs(lo_dir, exist_ok=True)

        for channel_name, loader in channel_loaders.items():
            hi_path = os.path.join(hi_dir, f"{channel_name}_top.png")
            lo_path = os.path.join(lo_dir, f"{channel_name}_bottom.png")

            _save_ranked_panel(
                highest,
                sil_scores,
                labels,
                filenames,
                hi_path,
                title=(
                    f"Node {neuron_id} (rank {rank}, n={pop}) | {channel_name} | "
                    f"top {min(n_show, len(highest))} silhouette"
                ),
                image_loader=loader,
                overlay_cores=(channel_name == "continuum"),
                config=config,
            )
            _save_ranked_panel(
                lowest,
                sil_scores,
                labels,
                filenames,
                lo_path,
                title=(
                    f"Node {neuron_id} (rank {rank}, n={pop}) | {channel_name} | "
                    f"bottom {min(n_show, len(lowest))} silhouette"
                ),
                image_loader=loader,
                overlay_cores=(channel_name == "continuum"),
                config=config,
            )

    print(f"Saved top-populated neuron panels to {base_dir}")


def _clean_label(label):
    if label is None:
        return ""
    cleaned = label
    for tag in AUGMENTATION_TAGS:
        cleaned = cleaned.replace(tag, "")
    return cleaned.strip()


def _is_augmented_label(label):
    if not label:
        return False
    return any(tag in label for tag in AUGMENTATION_TAGS)


def _prefer_original_indices(indices, labels, allow_augmented=False):
    """Drop flipped/augmented duplicates when the original exists.

    Keeps at most one sample per base label, preferring non-augmented.
    Preserves the original order of `indices`.
    """
    if allow_augmented or labels is None:
        return list(indices)

    choices = {}
    for idx in indices:
        try:
            label = labels[idx]
        except Exception:
            label = None
        base = _clean_label(label)
        key = base if base else f"idx_{idx}"
        is_aug = _is_augmented_label(label)
        if key not in choices:
            choices[key] = (idx, is_aug)
        else:
            prev_idx, prev_aug = choices[key]
            if prev_aug and not is_aug:
                choices[key] = (idx, is_aug)

    chosen = {idx for idx, _ in choices.values()}
    return [idx for idx in indices if idx in chosen]


def _apply_label_flip(img, label):
    """Apply label-indicated flip operations so augmented samples plot in augmented orientation."""
    if img is None:
        return None
    arr = np.asarray(img)
    if label and "[hflip]" in label:
        arr = np.flipud(arr)
    if label and "[vflip]" in label:
        arr = np.fliplr(arr)
    if label and "[hvflip]" in label:
        arr = np.flip(np.flip(arr, 0), 1)
    return arr


def _infer_source_id(sample_idx, filenames=None, labels=None):
    def _try_extract(value):
        if not value:
            return None
        sid = _extract_source_id_from_filename(value)
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


def _format_simulated_label(sample_idx, filenames=None, labels=None):
    candidates = []
    if filenames is not None:
        try:
            candidates.append(str(filenames[sample_idx]))
        except (IndexError, TypeError):
            pass
    if labels is not None:
        try:
            candidates.append(_clean_label(labels[sample_idx]))
        except (IndexError, TypeError):
            pass

    for value in candidates:
        match = SIM_FRAME_RE.search(value)
        if match:
            frame = match.group("frame")
            incl = match.group("incl")
            phi = match.group("phi")
            return f"frame {frame} | incl {incl} | phi {phi}"

    return f"frame {sample_idx:04d}"


def _normalize_simulated_plot(img, low_pct=2.0, high_pct=98.0):
    img = np.asarray(img, dtype=float)
    img = np.nan_to_num(img, nan=0.0)
    c = 25500 / np.log(1 + np.nanmax(img)) * 0.2
    img = c * (np.log(img + 1))
    return np.array(img, dtype=np.uint8)


def _compose_core_overlay(img, top_pct=99.8, core_low_pct=1.0, core_high_pct=99.0, gamma=0.3):
    """Prepare a core-enhanced overlay for processed images.

    Returns (base, overlay) where:
      base    : full image clipped to [0, 1]
      overlay : masked and gamma-stretched core intensities (NaN outside cores)
    """
    arr = np.nan_to_num(np.asarray(img, dtype=float), nan=0.0)
    base = np.clip(arr, 0.0, 1.0)

    try:
        thresh = np.percentile(base, top_pct)
    except Exception:
        return base, None

    mask = base >= thresh
    if not np.any(mask):
        return base, None

    core_vals = base[mask]
    if core_vals.size == 0:
        return base, None

    lo = float(np.percentile(core_vals, core_low_pct))
    hi = float(np.percentile(core_vals, core_high_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = float(np.max(core_vals))
        if not np.isfinite(hi) or hi <= lo:
            return base, None
        if hi == lo:
            hi = lo + 1e-6

    normed = (base - lo) / (hi - lo)
    normed = np.clip(normed, 0.0, 1.0)
    if gamma is not None and gamma > 0:
        normed = np.power(normed, gamma)

    overlay = np.where(mask, normed, np.nan)
    return base, overlay


def _imshow_processed_with_core_overlay(ax, img, cmap="plasma", gamma=0.3,
                                        top_pct=99.8, core_low_pct=1.0, core_high_pct=99.0):
    base, overlay = _compose_core_overlay(
        img,
        top_pct=top_pct,
        core_low_pct=core_low_pct,
        core_high_pct=core_high_pct,
        gamma=gamma,
    )
    ax.imshow(base, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
    if overlay is not None:
        ax.imshow(overlay, origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)


def _compose_core_overlay_rgb(img, cmap="plasma", gamma=0.3,
                              top_pct=99.8, core_low_pct=1.0, core_high_pct=99.0):
    base, overlay = _compose_core_overlay(
        img,
        top_pct=top_pct,
        core_low_pct=core_low_pct,
        core_high_pct=core_high_pct,
        gamma=gamma,
    )
    cmap_obj = plt.get_cmap(cmap)
    base_rgb = cmap_obj(base)[..., :3]
    if overlay is None:
        return base_rgb
    mask = np.isfinite(overlay)
    overlay_rgb = cmap_obj(np.nan_to_num(overlay, nan=0.0))[..., :3]
    comp = base_rgb.copy()
    comp[mask] = overlay_rgb[mask]
    return comp


def _format_image_subheading(sample_idx, sil_scores, filenames=None, labels=None, config=None):
    if config is not None and getattr(config, "use_simulated_data", False):
        return _format_simulated_label(sample_idx, filenames=filenames, labels=labels)
    score = float(sil_scores[sample_idx])
    sid = _infer_source_id(sample_idx, filenames=filenames, labels=labels)
    return f"{sid} | S={score:.2f}"


def _molecular_line_roots(config):
    """Yield concrete directories that might hold line outputs."""
    candidates = []
    base_dir = getattr(config, "molecular_line_file_dir", None)
    if base_dir:
        candidates.append(base_dir)
        parent_dir = os.path.dirname(os.path.normpath(base_dir))
        if parent_dir and parent_dir != os.path.normpath(base_dir):
            candidates.append(parent_dir)

    run_base = getattr(config, "run_base_dir", None)
    if run_base:
        fallback_root = os.path.join(run_base, "moment_zero_preprocessed")
        candidates.append(fallback_root)
        for variant in ("island2", "noisland2", "island", "noisland"):
            candidates.append(os.path.join(fallback_root, variant))

    seen = set()
    roots = []
    for path in candidates:
        if not path:
            continue
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isdir(norm):
            roots.append(norm)
    return roots


def _candidate_molecular_source_dirs(config, source_id):
    """Return candidate directories that might contain a source's line maps."""
    source_id = str(source_id)
    roots = _molecular_line_roots(config)
    seen = set()
    dirs = []

    def add(path):
        if not path:
            return
        norm = os.path.normpath(path)
        if norm in seen:
            return
        seen.add(norm)
        if os.path.isdir(norm):
            dirs.append(norm)

    for root in roots:
        add(root)
        add(os.path.join(root, source_id))
        parent = os.path.dirname(root)
        if parent and parent != root:
            for variant in ("island2", "noisland2", "island", "noisland"):
                add(os.path.join(parent, variant, source_id))
        try:
            for entry in os.listdir(root):
                add(os.path.join(root, entry, source_id))
        except OSError:
            continue
    return dirs


def _guess_preferred_variant(config):
    text = (getattr(config, "molecular_line_file_dir", "") or "").lower()
    if "noisland" in text:
        return "noisland"
    if "island" in text:
        return "island"
    return None


def _find_molecular_line_path(config, source_id, line):
    """Return one matching numpy map path for a line (or None)."""
    line_lower = line.lower()
    source_lower = str(source_id).lower()
    preferred_variant = _guess_preferred_variant(config)

    def _matches(entry, variant=None):
        lower = entry.lower()
        if not lower.endswith(".npy"):
            return False
        if source_lower not in lower:
            return False
        if line_lower not in lower:
            return False
        if "_moment0" not in lower:
            return False
        if variant and variant not in lower:
            return False
        return True

    def _search(variant=None):
        for src_dir in _candidate_molecular_source_dirs(config, source_id):
            try:
                for entry in os.listdir(src_dir):
                    if _matches(entry, variant):
                        return os.path.join(src_dir, entry)
            except OSError:
                continue
        return None

    if preferred_variant:
        path = _search(preferred_variant)
        if path:
            return path
    return _search()


def _load_molecular_line_image(config, source_id, line):
    """Return a nan-safe numpy array for a single molecular line, if available."""
    path = _find_molecular_line_path(config, source_id, line)
    if path is None:
        return None
    try:
        return np.nan_to_num(np.load(path, allow_pickle=False), nan=0.0)
    except Exception as exc:
        print(f"Error loading {line} map for {source_id} at {path}: {exc}")
        return None


def _find_raw_moment0_path(config, source_id, line):
    raw_root = getattr(config, "raw_moment0_dir", "/raid/scratch/mevans/moment0_with_rms2")
    source_dir = os.path.join(raw_root, str(source_id))
    candidates = [source_dir, raw_root]
    for base in candidates:
        if not base or not os.path.isdir(base):
            continue
        try:
            for entry in os.listdir(base):
                lower = entry.lower()
                if not lower.endswith(".fits"):
                    continue
                if str(source_id).lower() not in lower:
                    continue
                if line.lower() not in lower:
                    continue
                if "_moment0" not in lower:
                    continue
                return os.path.join(base, entry)
        except OSError:
            continue
    return None


def _load_raw_moment0_map(config, source_id, line):
    path = _find_raw_moment0_path(config, source_id, line)
    if path is None:
        return None
    try:
        with fits.open(path) as hdul:
            data = hdul[0].data
        return np.nan_to_num(data, nan=0.0)
    except Exception as exc:
        print(f"Error loading raw moment0 {line} for {source_id} at {path}: {exc}")
        return None




def plot_original_examples(som, som_results, filenames, labels, sil_scores, config):
    """
    Plot the top-silhouette-score images per neuron.
    Only non-augmented images are included.
    """
    out_dir = os.path.join(_ensure_plot_subdir(config, "per_neuron_plots"), "neuron_original_examples")
    os.makedirs(out_dir, exist_ok=True)

    neuron_map = _build_neuron_assignment(som_results, labels, config)

    sil_scores = np.asarray(sil_scores)
    resolved_path_cache = {}

    for neuron_id, indices in neuron_map.items():
        if len(indices) == 0:
            continue

        # sort the indices belonging to this neuron by silhouette score (descending)
        indices = _prefer_original_indices(
            indices,
            labels,
            allow_augmented=getattr(config, "use_simulated_data", False),
        )
        if len(indices) == 0:
            continue
        ordered = sorted(indices, key=lambda i: sil_scores[i], reverse=True)

        # choose the best 16
        chosen = ordered[:16]

        fig = plt.figure(figsize=(16, 16))

        for ax_idx in range(16):
            ax = fig.add_subplot(4, 4, ax_idx + 1)
            ax.axis("off")

            if ax_idx >= len(chosen):
                continue

            sample_idx = chosen[ax_idx]
            filename = filenames[sample_idx]

            try:
                resolved = _resolve_original_fits_path(filename, config, cache=resolved_path_cache)
                if resolved is None:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
                    continue

                with fits.open(resolved) as hdul:
                    img = hdul[0].data

                img = np.nan_to_num(img, nan=0)
                img = _apply_label_flip(img, labels[sample_idx] if labels is not None else "")
                if img.ndim > 2:
                    img = np.squeeze(img)
                if img.ndim != 2:
                    ax.text(0.5, 0.5, "invalid", ha="center", va="center", fontsize=8)
                    continue
                vmin_local = np.nanmin(img)
                vmax_local = np.nanmax(img)
                if not np.isfinite(vmin_local) or not np.isfinite(vmax_local) or vmin_local == vmax_local:
                    vmin_local, vmax_local = None, None

                if getattr(config, "use_simulated_data", False):
                    img = _normalize_simulated_plot(img)
                    vmin_local = np.nanmin(img)
                    vmax_local = np.nanmax(img)
                    cmap = "plasma"
                else:
                    cmap = "plasma"

                ax.imshow(img, cmap=cmap, origin="lower",
                          vmin=vmin_local, vmax=vmax_local)

                ax.set_title(
                    _format_image_subheading(
                        sample_idx,
                        sil_scores,
                        filenames=filenames,
                        labels=labels,
                        config=config,
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
    out_dir = os.path.join(_ensure_plot_subdir(config, "per_neuron_plots"), "neuron_fov_matched_originals")
    os.makedirs(out_dir, exist_ok=True)

    neuron_map = _build_neuron_assignment(som_results, labels, config)
    sil_scores = np.asarray(sil_scores)

    for neuron_id, indices in neuron_map.items():
        if len(indices) == 0:
            continue

        indices = _prefer_original_indices(
            indices,
            labels,
            allow_augmented=getattr(config, "use_simulated_data", False),
        )
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
                    filenames=filenames,
                    labels=labels,
                    config=config,
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
    out_dir = os.path.join(_ensure_plot_subdir(config, "per_neuron_plots"), "neuron_processed_examples")
    os.makedirs(out_dir, exist_ok=True)

    neuron_map = _build_neuron_assignment(som_results, labels, config)
    sil_scores = np.asarray(sil_scores)

    for neuron_id, indices in neuron_map.items():
        if len(indices) == 0:
            continue

        indices = _prefer_original_indices(
            indices,
            labels,
            allow_augmented=getattr(config, "use_simulated_data", False),
        )
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

            if getattr(config, "use_simulated_data", False):
                img = _normalize_simulated_plot(img)
                ax.imshow(img, origin="lower", cmap="plasma")
            else:
                _imshow_processed_with_core_overlay(ax, np.nan_to_num(img), cmap="plasma")

            ax.set_title(
                _format_image_subheading(
                    sample_idx,
                    sil_scores,
                    labels=labels,
                    config=config,
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

    out_dir = os.path.join(_ensure_plot_subdir(config, "per_neuron_plots"), "neuron_smoothed_examples")
    os.makedirs(out_dir, exist_ok=True)

    neuron_map = _build_neuron_assignment(som_results, labels, config)
    sil_scores = np.asarray(sil_scores)

    for neuron_id, indices in neuron_map.items():
        if len(indices) == 0:
            continue

        indices = _prefer_original_indices(
            indices,
            labels,
            allow_augmented=getattr(config, "use_simulated_data", False),
        )
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
            sid = _extract_source_id_from_filename(fname)
            candidates = [
                os.path.join(config.smoothed_image_file_dir, f"{sid}_smoothed_noiseremoved1sigma.npy"),
                os.path.join(config.smoothed_image_file_dir, f"{sid}_smoothed_island.npy"),
            ]
            npy_path = next((p for p in candidates if os.path.isfile(p)), None)
            if npy_path is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
                continue

            try:
                img = np.load(npy_path, allow_pickle=False)
            except Exception as e:
                print(f"Error loading smoothed image {npy_path}: {e}")
                continue

            img = np.nan_to_num(img, nan=0.0)
            img = _apply_label_flip(img, labels[sample_idx] if labels is not None else "")
            if img.ndim > 2:
                img = np.squeeze(img)
            if img.ndim != 2:
                ax.text(0.5, 0.5, "invalid", ha="center", va="center", fontsize=8)
                continue

            vmin_local = np.nanmin(img)
            vmax_local = np.nanmax(img)
            if not np.isfinite(vmin_local) or not np.isfinite(vmax_local) or vmin_local == vmax_local:
                vmin_local, vmax_local = None, None

            ax.imshow(img, origin="lower", cmap="plasma", vmin=vmin_local, vmax=vmax_local)
            ax.set_title(
                _format_image_subheading(
                    sample_idx,
                    sil_scores,
                    filenames=filenames,
                    labels=labels,
                    config=config,
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

    out_dir = os.path.join(_ensure_plot_subdir(config, "per_neuron_plots"), "neuron_molecular_examples")
    os.makedirs(out_dir, exist_ok=True)

    lines = ["H2CO", "SiO", "DCN"]
    line_dirs = {}
    for line in lines:
        line_path = os.path.join(out_dir, line)
        os.makedirs(line_path, exist_ok=True)
        line_dirs[line] = line_path

    neuron_map = _build_neuron_assignment(som_results, labels, config)
    sil_scores = np.asarray(sil_scores)

    max_examples = 9

    for neuron_id, indices in neuron_map.items():
        if len(indices) == 0:
            continue

        indices = _prefer_original_indices(
            indices,
            labels,
            allow_augmented=getattr(config, "use_simulated_data", False),
        )
        if len(indices) == 0:
            continue
        ordered = sorted(indices, key=lambda i: sil_scores[i], reverse=True)
        chosen = ordered[:max_examples]

        sample_maps = {}
        for sample_idx in chosen:
            fname = filenames[sample_idx]
            sid = _extract_source_id_from_filename(fname)
            if not sid:
                sid = _infer_source_id(sample_idx, filenames=filenames, labels=labels)

            maps = {}
            for line in lines:
                img = _load_molecular_line_image(config, sid, line)
                if img is not None:
                    maps[line] = _apply_label_flip(img, labels[sample_idx] if labels is not None else "")

            if maps:
                sample_maps[sample_idx] = maps

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
                        filenames=filenames,
                        labels=labels,
                        config=config,
                    ),
                    fontsize=9
                )

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_file = os.path.join(line_dirs[line], f"neuron_{neuron_id:03d}_{line}.png")
            fig.savefig(out_file)
            plt.close(fig)
            print(f"Saved {line} examples for neuron {neuron_id} to {out_file}")


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
    for y in range(config.n_num):
        for x in range(config.m_num):
            row, column = fixed_som_subplot_index(
                x, y, config.m_num, config.n_num
            )
            yield x, y, axes[row, column], y * config.m_num + x


def plot_representative_grid_processed(som_results, raw_datas, labels, sil_scores, config):
    rep_dir = _ensure_plot_subdir(config, "representative_plots")
    out_file = os.path.join(rep_dir, "som_representative_grid_processed.png")
    best = _best_per_neuron_indices(som_results, labels, sil_scores, config)
    fig, axes = _make_grid_canvas(config)

    sil_scores = np.asarray(sil_scores)

    for i, j, ax, neuron_id in _grid_iter_axes(axes, config):
        ax.axis("off")
        if neuron_id not in best:
            continue
        idx = best[neuron_id]
        img = raw_datas[idx].reshape(config.sqrt_pixel_number, config.sqrt_pixel_number)
        img = np.nan_to_num(img)
        if getattr(config, "use_simulated_data", False):
            img = _normalize_simulated_plot(img)
            ax.imshow(img, origin="lower", cmap="plasma")
        else:
            _imshow_processed_with_core_overlay(ax, img, cmap="plasma")
        ax.set_title(_format_image_subheading(idx, sil_scores, labels=labels, config=config), fontsize=10)

    plt.suptitle("Silhouette-Best Processed Image per SOM Neuron", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved processed representative SOM grid to {out_file}")


def plot_representative_grid_originals(som_results, filenames, labels, sil_scores, config):
    rep_dir = _ensure_plot_subdir(config, "representative_plots")
    out_file = os.path.join(rep_dir, "som_representative_grid_originals.png")
    best = _best_per_neuron_indices(som_results, labels, sil_scores, config)
    fig, axes = _make_grid_canvas(config)
    sil_scores = np.asarray(sil_scores)
    resolved_path_cache = {}

    for i, j, ax, neuron_id in _grid_iter_axes(axes, config):
        ax.axis("off")
        if neuron_id not in best:
            continue
        idx = best[neuron_id]
        fname = filenames[idx]
        try:
            resolved = _resolve_original_fits_path(fname, config, cache=resolved_path_cache)
            if resolved is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
                continue

            with fits.open(resolved) as hdul:
                img = np.nan_to_num(hdul[0].data, nan=0)

            img = _apply_label_flip(img, labels[idx] if labels is not None else "")

            if img.ndim > 2:
                img = np.squeeze(img)
            if img.ndim != 2:
                ax.text(0.5, 0.5, "invalid", ha="center", va="center", fontsize=8)
                continue

            if getattr(config, "use_simulated_data", False):
                img = _normalize_simulated_plot(img)
                vmin_local = np.nanmin(img)
                vmax_local = np.nanmax(img)
            else:
                vmin_local = np.nanmin(img)
                vmax_local = np.nanmax(img)
                if not np.isfinite(vmin_local) or not np.isfinite(vmax_local) or vmin_local == vmax_local:
                    vmin_local, vmax_local = None, None
            ax.imshow(img, cmap="plasma", origin="lower", vmin=vmin_local, vmax=vmax_local)
            ax.set_title(
                _format_image_subheading(
                    idx,
                    sil_scores,
                    filenames=filenames,
                    labels=labels,
                    config=config,
                ),
                fontsize=8,
            )
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
    rep_dir = _ensure_plot_subdir(config, "representative_plots")
    out_file = os.path.join(rep_dir, "som_representative_grid_smoothed.png")
    best = _best_per_neuron_indices(som_results, labels, sil_scores, config)
    fig, axes = _make_grid_canvas(config)
    sil_scores = np.asarray(sil_scores)

    for i, j, ax, neuron_id in _grid_iter_axes(axes, config):
        ax.axis("off")
        if neuron_id not in best:
            continue
        idx = best[neuron_id]
        fname = filenames[idx]
        sid = _extract_source_id_from_filename(fname)
        img = _load_smoothed_image(config, sid)
        if img is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
            continue

        img = _apply_label_flip(img, labels[idx] if labels is not None else "")

        vmin_local = np.nanmin(img)
        vmax_local = np.nanmax(img)
        if not np.isfinite(vmin_local) or not np.isfinite(vmax_local) or vmin_local == vmax_local:
            vmin_local, vmax_local = None, None
        ax.imshow(img, origin="lower", cmap="plasma", vmin=vmin_local, vmax=vmax_local)
        ax.set_title(
            _format_image_subheading(
                idx,
                sil_scores,
                filenames=filenames,
                labels=labels,
                config=config,
            ),
            fontsize=9,
        )

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

    rep_dir = _ensure_plot_subdir(config, "representative_plots")
    for line in lines:
        out_file = os.path.join(rep_dir, f"som_representative_grid_{line}.png")
        fig, axes = _make_grid_canvas(config)
        for i, j, ax, neuron_id in _grid_iter_axes(axes, config):
            ax.axis("off")
            if neuron_id not in best:
                continue
            idx = best[neuron_id]
            fname = filenames[idx]
            sid = _extract_source_id_from_filename(fname)
            img = _load_molecular_line_image(config, sid, line)
            if img is None:
                continue
            img = _apply_label_flip(img, labels[idx] if labels is not None else "")
            ax.imshow(img, origin="lower", cmap="plasma", vmin=0.0, vmax=1.0)
            ax.set_title(
                _format_image_subheading(
                    idx,
                    sil_scores,
                    filenames=filenames,
                    labels=labels,
                    config=config,
                ),
                fontsize=9,
            )

        plt.suptitle(f"Silhouette-Best {line} Map per SOM Neuron", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_file)
        plt.close(fig)
        print(f"Saved {line} representative SOM grid to {out_file}")


# Backwards-compatible alias
def plot_representative_grid(som, som_results, raw_datas, labels, sil_scores, config):
    plot_representative_grid_processed(som_results, raw_datas, labels, sil_scores, config)


def _parse_evolutionary_stage_clusters(classification_file):
    """Parse save_results cluster output into a dict of cluster_id -> stats."""
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

            clusters[idx] = {
                "observed": obs,
                "expected": exp,
            }

            i += 2
        else:
            i += 1

    return clusters


def _plot_gsom_evolutionary_stage(clusters, gsom, config, neuron_to_plot=None):
    """Plot GSOM evolutionary stage as a sparse node-position map."""
    if not clusters:
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

    node_positions = [(n.x, n.y) for n in gsom.nodes]
    if not node_positions:
        print("No GSOM nodes available for evolutionary stage plot.")
        return

    xs = np.array([p[0] for p in node_positions], dtype=float)
    ys = np.array([p[1] for p in node_positions], dtype=float)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_span = max(1.0, x_max - x_min + 1.0)
    y_span = max(1.0, y_max - y_min + 1.0)

    fig_w = max(7.0, min(16.0, 1.0 + 0.65 * x_span))
    fig_h = max(6.0, min(16.0, 1.0 + 0.65 * y_span))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for idx, d in clusters.items():
        if idx >= len(gsom.nodes):
            continue

        node = gsom.nodes[idx]
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

        x0, x1 = node.x - 0.5, node.x + 0.5
        y0, y1 = node.y - 0.5, node.y + 0.5

        

        if primary == "na" and len(order) > 1:
            # Upper-right triangle: na
            tri_top = Polygon(
                [(x0, y1), (x1, y1), (x1, y0)],
                closed=True,
                facecolor=stage_colors["na"],
                edgecolor="none",
                zorder=3,
            )
            ax.add_patch(tri_top)

            # Lower-left triangle: second-most abundant category
            tri_bot = Polygon(
                [(x0, y1), (x0, y0), (x1, y0)],
                closed=True,
                facecolor=stage_colors[secondary],
                edgecolor="none",
                zorder=3,
            )
            ax.add_patch(tri_bot)

            # Black outline around the full neuron box
            ax.add_patch(
                Polygon(
                    [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                    closed=True,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=5,
                )
            )

            # na label: upper-right triangle centroid
            ax.text(
                node.x + 1/6,
                node.y + 1/6,
                f"na\n{f1:.2f}",
                ha="center",
                va="center",
                fontsize=6 if len(gsom.nodes) > 80 else 7,
                color="black",
                clip_on=True,
                zorder=4,
            )

            # Secondary category label: lower-left triangle centroid
            ax.text(
                node.x - 1/6,
                node.y - 1/6,
                f"{secondary}\n{f2:.2f}",
                ha="center",
                va="center",
                fontsize=6 if len(gsom.nodes) > 80 else 7,
                color=_text_color(secondary),
                clip_on=True,
                zorder=4,
            )
        else:
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    1,
                    1,
                    facecolor=stage_colors[primary],
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=3,
                )
            )
            if len(gsom.nodes) <= 120:
                ax.text(
                    node.x,
                    node.y,
                    f"{primary}\n{f1:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6 if len(gsom.nodes) > 80 else 7,
                    color=_text_color(primary),
                    clip_on=True,
                    zorder=4,
                )

    ax.set_xlim(x_min - 0.8, x_max + 0.8)
    ax.set_ylim(y_min - 0.8, y_max + 0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Node x", fontsize=14)
    ax.set_ylabel("Node y", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Sparse ticks keep the figure readable as the grid grows.
    if x_span > 12:
        step_x = 2 if x_span <= 30 else 5
        ax.set_xticks(np.arange(int(np.floor(x_min)), int(np.ceil(x_max)) + 1, step_x))
    if y_span > 12:
        step_y = 2 if y_span <= 30 else 5
        ax.set_yticks(np.arange(int(np.floor(y_min)), int(np.ceil(y_max)) + 1, step_y))

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=stage_colors[s], label=s)
        for s in stages
    ]
    ax.legend(
        handles=handles,
        title="Stage",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=11,
        title_fontsize=12,
        handlelength=1.4,
        handleheight=1.4,
    )

    plt.tight_layout()
    evo_dir = _ensure_plot_subdir(config, "evolutionary_stage")
    grid_out = os.path.join(evo_dir, "gsom_evolutionary_stage_grid.png")
    plt.savefig(grid_out, dpi=200)
    plt.close(fig)
    print(f"Saved GSOM evolutionary stage grid to {grid_out}")


def plot_evolutionary_stage(classification_file, config, neuron_to_plot=None, gsom=None):
    """Plot the dominant evolutionary stage for each neuron.

    classification_file should match the format written by save_results.save_all_results
    (COMBINEDclassification_info.txt). If the file is missing, the plots are skipped.
    """
    if not os.path.isfile(classification_file):
        print(f"Classification file not found: {classification_file}; skipping evolutionary stage plot.")
        return

    clusters = _parse_evolutionary_stage_clusters(classification_file)

    if not clusters:
        print(f"No cluster entries found in {classification_file}; skipping evolutionary stage plot.")
        return

    if gsom is not None:
        _plot_gsom_evolutionary_stage(clusters, gsom, config, neuron_to_plot=neuron_to_plot)
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
    evo_dir = _ensure_plot_subdir(config, "evolutionary_stage")
    grid_out = os.path.join(evo_dir, "evolutionary_stage_grid.png")
    plt.savefig(grid_out, dpi=200)
    plt.close()
    print(f"Saved evolutionary stage grid to {grid_out}")


def plot_gsom_hits_map(gsom, encoded_data, config, labels=None):
    """
    Plot a hit map for a Growing SOM.

    Because the GSOM grid is irregular (nodes at arbitrary integer x,y
    positions), we render the map as a scatter plot sized and coloured
    by hit count.
    """
    # For the hit map only, count all samples including flipped augmentations.
    # Other GSOM plots can still exclude augmented samples where appropriate.
    hits = gsom.hits_map(encoded_data, labels=None)

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

    # Show the full topology so connectivity is visible even when many nodes
    # have zero hits. Empty nodes are still drawn, but in a lighter colour.
    active_mask = counts_all > 0
    zero_mask = counts_all == 0
    plot_coords = set(all_coords)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw topology edges for the full map.
    drawn_edges = set()
    for node in gsom.nodes:
        for nb in (node.right, node.up):
            if nb is None:
                continue
            key = tuple(sorted(((node.x, node.y), (nb.x, nb.y))))
            if key in drawn_edges:
                continue
            drawn_edges.add(key)
            ax.plot([node.x, nb.x], [node.y, nb.y], color="gray", linewidth=1.0, alpha=0.8, zorder=1)

    # Plot zero-hit nodes in light gray, so they remain visible but de-emphasised.
    if np.any(zero_mask):
        ax.scatter(
            x_all[zero_mask],
            y_all[zero_mask],
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
    if n_nodes <= 250:
        for i, node in enumerate(gsom.nodes):
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
    total_samples = int(len(encoded_data))
    ax.set_title(
        f"GSOM topology + hits (total nodes={total_nodes}, active nodes={active_count}, total samples={total_samples})"
    )
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # Axis padding so edge nodes are not clipped
    if plot_coords:
        x_plot = np.array([c[0] for c in plot_coords], dtype=float)
        y_plot = np.array([c[1] for c in plot_coords], dtype=float)
        x_min, x_max = float(np.min(x_plot)), float(np.max(x_plot))
        y_min, y_max = float(np.min(y_plot)), float(np.max(y_plot))
    ax.set_xlim(x_min - 0.8, x_max + 0.8)
    ax.set_ylim(y_min - 0.8, y_max + 0.8)

    ax.legend(loc="best", fontsize=8)

    out = os.path.join(config.run_dir, "gsom_hits_map.png")
    plt.tight_layout()
    plt.savefig(out, dpi=170)
    plt.close(fig)
    print(
        f"Saved GSOM hits map to {out} "
        f"(total nodes={total_nodes}, active nodes={active_count}, total samples={total_samples})"
    )


def _gsom_best_per_node(gsom, preds, labels, encoded_data, sil_scores=None, allow_augmented=False):
    """Pick one representative sample per active GSOM node.

    For each node that has at least one sample mapped to it,
    choose the sample with highest silhouette score if available, otherwise
    fallback to lowest quantization error. Returns dict node_index -> sample_index.
    """
    from collections import defaultdict
    node_samples = defaultdict(list)
    for i, node_idx in enumerate(preds):
        node_samples[int(node_idx)].append(i)

    encoded_data = np.asarray(encoded_data)
    sil_scores = None if sil_scores is None else np.asarray(sil_scores)
    best = {}
    for nid, indices in node_samples.items():
        if not indices:
            continue
        indices = _prefer_original_indices(indices, labels, allow_augmented=allow_augmented)
        if not indices:
            continue
        if sil_scores is not None:
            best[nid] = max(indices, key=lambda idx: sil_scores[idx])
        else:
            node = gsom.nodes[nid]
            # fallback: pick sample with smallest distance to the node's weights
            dists = [float(np.sum((encoded_data[i] - node.weights) ** 2)) for i in indices]
            best[nid] = indices[int(np.argmin(dists))]
    return best


def _gsom_image_grid(gsom, best, load_image_fn, title, out_path, thumb_px=64,
                     cmap="plasma", apply_core_overlay=False):
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
        if apply_core_overlay:
            img = _compose_core_overlay_rgb(img, cmap=cmap)
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
    dpi = 80
    ax_width_px = fig_w * dpi
    px_per_unit = ax_width_px / max(x_span + 1.6, 1.0)  # pixels per data-unit
    target_px = px_per_unit * 0.88  # thumbnail target size in figure pixels
    zoom = float(np.clip(target_px / float(max(1, thumb_px)), 0.25, 2.5))

    for nx, ny, img in items:
        if apply_core_overlay:
            oi = OffsetImage(img, zoom=zoom, origin="lower")
        else:
            oi = OffsetImage(img, zoom=zoom, cmap=cmap, origin="lower")
        ab = AnnotationBbox(oi, (nx, ny), frameon=True,
                            bboxprops=dict(edgecolor="black", linewidth=0.4),
                            pad=0.0)
        ax.add_artist(ab)

    ax.set_xlim(x_min - 0.8, x_max + 0.8)
    ax.set_ylim(y_min - 0.8, y_max + 0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Node x", fontsize=8)
    ax.set_ylabel("Node y", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.set_title(title, fontsize=10, pad=4)
    ax.grid(True, linewidth=0.2, alpha=0.3)

    plt.tight_layout()
    # Crop extra whitespace and use a very small pad to minimize output size
    fig.savefig(out_path, dpi=int(dpi), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {title} to {out_path}")


def plot_gsom_representative_originals(gsom, preds, encoded_data, filenames, labels, config, sil_scores=None):
    """Original FITS images placed on the GSOM topology grid."""
    best = _gsom_best_per_node(
        gsom,
        preds,
        labels,
        encoded_data,
        sil_scores=sil_scores,
        allow_augmented=getattr(config, "use_simulated_data", False),
    )
    resolved_path_cache = {}

    def load_fn(idx):
        resolved = _resolve_original_fits_path(filenames[idx], config, cache=resolved_path_cache)
        if resolved is None:
            return None
        try:
            with fits.open(resolved) as hdul:
                img = np.nan_to_num(hdul[0].data, nan=0.0)
                img = _apply_label_flip(img, labels[idx] if labels is not None else "")
                if getattr(config, "use_simulated_data", False):
                    img = _normalize_simulated_plot(img)
                return img
        except Exception:
            return None

    _gsom_image_grid(
        gsom, best, load_fn,
        title="GSOM Representative Originals",
        out_path=os.path.join(_ensure_plot_subdir(config, "representative_plots"), "gsom_representative_originals.png"),
        cmap="plasma",
    )


def plot_gsom_representative_processed(gsom, preds, encoded_data, raw_datas, labels, config, sil_scores=None):
    """Preprocessed (segmented) images placed on the GSOM topology grid."""
    best = _gsom_best_per_node(
        gsom,
        preds,
        labels,
        encoded_data,
        sil_scores=sil_scores,
        allow_augmented=getattr(config, "use_simulated_data", False),
    )
    H = config.sqrt_pixel_number

    def load_fn(idx):
        try:
            return raw_datas[idx].reshape(H, H)
        except Exception:
            return None

    apply_overlay = not getattr(config, "use_simulated_data", False)

    def load_fn_scaled(idx):
        img = load_fn(idx)
        if img is None:
            return None
        if getattr(config, "use_simulated_data", False):
            return _normalize_simulated_plot(img)
        return img

    _gsom_image_grid(
        gsom, best, load_fn_scaled,
        title="GSOM Representative Processed",
        out_path=os.path.join(_ensure_plot_subdir(config, "representative_plots"), "gsom_representative_processed.png"),
        apply_core_overlay=apply_overlay,
    )


def plot_gsom_representative_smoothed(gsom, preds, encoded_data, filenames, labels, config, sil_scores=None):
    """Smoothed noise-removed images placed on the GSOM topology grid."""
    if not hasattr(config, "smoothed_image_file_dir"):
        print("Config has no smoothed_image_file_dir; skipping GSOM smoothed grid.")
        return
    best = _gsom_best_per_node(
        gsom,
        preds,
        labels,
        encoded_data,
        sil_scores=sil_scores,
        allow_augmented=getattr(config, "use_simulated_data", False),
    )

    def load_fn(idx):
        sid = _extract_source_id_from_filename(filenames[idx])
        img = _load_smoothed_image(config, sid)
        return _apply_label_flip(img, labels[idx] if labels is not None else "")

    _gsom_image_grid(
        gsom, best, load_fn,
        title="GSOM Representative Smoothed",
        out_path=os.path.join(_ensure_plot_subdir(config, "representative_plots"), "gsom_representative_smoothed.png"),
    )


def plot_gsom_representative_molecular(gsom, preds, encoded_data, filenames, labels, config, sil_scores=None):
    """Molecular line moment-0 maps placed on the GSOM topology grid (one plot per line)."""
    if not hasattr(config, "molecular_line_file_dir"):
        print("Config has no molecular_line_file_dir; skipping GSOM molecular grids.")
        return
    best = _gsom_best_per_node(
        gsom,
        preds,
        labels,
        encoded_data,
        sil_scores=sil_scores,
        allow_augmented=getattr(config, "use_simulated_data", False),
    )

    for line in ("H2CO", "SiO", "DCN"):
        def load_fn(idx, _line=line):
            sid = _extract_source_id_from_filename(filenames[idx])
            img = _load_molecular_line_image(config, sid, _line)
            return _apply_label_flip(img, labels[idx] if labels is not None else "")

        _gsom_image_grid(
            gsom, best, load_fn,
            title=f"GSOM Representative {line}",
            out_path=os.path.join(_ensure_plot_subdir(config, "representative_plots"), f"gsom_representative_{line}.png"),
        )


def plot_gsom_representative_raw_moment0(gsom, preds, encoded_data, filenames, labels, config, sil_scores=None):
    """Plot the raw moment-0 FITS images (before preprocessing)."""
    if not hasattr(config, "raw_moment0_dir"):
        print("Config has no raw_moment0_dir; skipping raw moment-0 GSOM grid.")
        return
    best = _gsom_best_per_node(
        gsom,
        preds,
        labels,
        encoded_data,
        sil_scores=sil_scores,
        allow_augmented=getattr(config, "use_simulated_data", False),
    )

    for line in ("H2CO", "SiO", "DCN"):
        def load_fn(idx, _line=line):
            sid = _extract_source_id_from_filename(filenames[idx])
            img = _load_raw_moment0_map(config, sid, _line)
            return _apply_label_flip(img, labels[idx] if labels is not None else "")

        _gsom_image_grid(
            gsom, best, load_fn,
            title=f"GSOM Representative Raw {line}",
            out_path=os.path.join(_ensure_plot_subdir(config, "representative_plots"), f"gsom_representative_raw_{line}.png"),
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

    X = np.asarray(encoded_data, dtype=np.float64)

    X_valid = X

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
