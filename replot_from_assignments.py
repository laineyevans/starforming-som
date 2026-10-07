import argparse
import ast
import csv
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
import plotting
from run_pipeline import _load_preprocessed_fits, _load_simulated_fits


def _parse_value(value):
    text = value.strip()
    if text == "":
        return ""
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def load_config_from_run_info(run_dir, out_dir):
    config = Config()
    info_path = os.path.join(run_dir, "run_info.txt")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(f"run_info.txt not found in {run_dir}")

    with open(info_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if not key:
                continue
            setattr(config, key, _parse_value(value))

    config.run_dir = out_dir
    return config


def read_assignments(csv_path):
    entries = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cluster = int(row.get("cluster", ""))
            except Exception:
                continue
            try:
                sil = float(row.get("silhouette_score", "nan"))
            except Exception:
                sil = float("nan")
            node_x = row.get("node_x", "")
            node_y = row.get("node_y", "")
            try:
                node_x = int(node_x) if node_x != "" else None
            except Exception:
                node_x = None
            try:
                node_y = int(node_y) if node_y != "" else None
            except Exception:
                node_y = None
            entries.append({
                "cluster": cluster,
                "filename": row.get("filename", ""),
                "label": row.get("label", ""),
                "silhouette": sil,
                "node_x": node_x,
                "node_y": node_y,
            })
    return entries


def _clean_label(label):
    if label is None:
        return ""
    cleaned = str(label)
    for tag in plotting.AUGMENTATION_TAGS:
        cleaned = cleaned.replace(tag, "")
    return cleaned.strip()


def _is_augmented_label(label):
    if not label:
        return False
    return any(tag in label for tag in plotting.AUGMENTATION_TAGS)


def select_top_entries(entries, n, allow_augmented=False):
    if not entries:
        return []
    if allow_augmented:
        ordered = sorted(entries, key=lambda e: e.get("silhouette", -np.inf), reverse=True)
        return ordered[:n]

    # Deduplicate by base label; prefer non-augmented and higher silhouette
    base_map = {}
    ordered = sorted(
        entries,
        key=lambda e: (
            _is_augmented_label(e.get("label")),
            -float(e.get("silhouette", -np.inf)),
        ),
    )
    for entry in ordered:
        base = _clean_label(entry.get("label"))
        key = base if base else entry.get("filename")
        if key not in base_map:
            base_map[key] = entry
    unique = list(base_map.values())
    unique = sorted(unique, key=lambda e: e.get("silhouette", -np.inf), reverse=True)
    return unique[:n]


def build_processed_map(config):
    if getattr(config, "use_simulated_data", False):
        _, labels, filenames, raw_datas, _, _, _ = _load_simulated_fits(
            config.simulated_fits_dir,
            config.sqrt_pixel_number,
        )
    else:
        _, labels, filenames, raw_datas, _, _, _ = _load_preprocessed_fits(
            config.continuum_preprocessed_dir
        )

    processed_map = {}
    for fname, img in zip(filenames, raw_datas):
        if not fname:
            continue
        processed_map[fname] = img
        processed_map[os.path.basename(fname)] = img
    return processed_map


def _format_entry_title(entry, config):
    return plotting._format_image_subheading(
        0,
        [entry.get("silhouette", np.nan)],
        filenames=[entry.get("filename")],
        labels=[entry.get("label")],
        config=config,
    )


def _format_rep_title(entry, neuron_id, config):
    sil = entry.get("silhouette", np.nan)
    try:
        sil_text = f"S={float(sil):.2f}"
    except Exception:
        sil_text = "S=nan"
    return sil_text


def _load_processed(ax, entry, processed_map, config):
    fname = entry.get("filename", "")
    img = processed_map.get(fname)
    if img is None:
        img = processed_map.get(os.path.basename(fname))
    if img is None:
        return False
    img = np.nan_to_num(np.asarray(img), nan=0.0)
    img = plotting._apply_label_flip(img, entry.get("label"))
    if getattr(config, "use_simulated_data", False):
        img = plotting._normalize_simulated_plot(img)
        ax.imshow(img, origin="lower", cmap="plasma")
    else:
        plotting._imshow_processed_with_core_overlay(ax, img, cmap="plasma")
    return True


def _load_original(ax, entry, config):
    from astropy.io import fits

    resolved = plotting._resolve_original_fits_path(entry.get("filename", ""), config)
    if resolved is None:
        return False
    try:
        with fits.open(resolved) as hdul:
            img = np.nan_to_num(hdul[0].data, nan=0.0)
    except Exception:
        return False

    img = plotting._apply_label_flip(img, entry.get("label"))
    if img.ndim > 2:
        img = np.squeeze(img)
    if img.ndim != 2:
        return False

    if getattr(config, "use_simulated_data", False):
        img = plotting._normalize_simulated_plot(img)
        vmin_local = np.nanmin(img)
        vmax_local = np.nanmax(img)
    else:
        vmin_local = np.nanmin(img)
        vmax_local = np.nanmax(img)
        if not np.isfinite(vmin_local) or not np.isfinite(vmax_local) or vmin_local == vmax_local:
            vmin_local, vmax_local = None, None

    ax.imshow(img, cmap="plasma", origin="lower", vmin=vmin_local, vmax=vmax_local)
    return True


def _load_smoothed(ax, entry, config):
    sid = plotting._extract_source_id_from_filename(entry.get("filename", ""))
    img = plotting._load_smoothed_image(config, sid)
    if img is None:
        return False
    img = plotting._apply_label_flip(img, entry.get("label"))

    vmin_local = np.nanmin(img)
    vmax_local = np.nanmax(img)
    if not np.isfinite(vmin_local) or not np.isfinite(vmax_local) or vmin_local == vmax_local:
        vmin_local, vmax_local = None, None

    ax.imshow(img, origin="lower", cmap="plasma", vmin=vmin_local, vmax=vmax_local)
    return True


def _load_molecular(ax, entry, config, line):
    sid = plotting._extract_source_id_from_filename(entry.get("filename", ""))
    img = plotting._load_molecular_line_image(config, sid, line)
    if img is None:
        return False
    img = plotting._apply_label_flip(img, entry.get("label"))
    ax.imshow(img, origin="lower", cmap="plasma", vmin=0.0, vmax=1.0)
    return True


def _plot_per_neuron_grid(entries_by_cluster, out_dir, name_template, title_prefix, loader, config,
                          nrows, ncols, title_font=12, missing_font=10, suptitle_font=16):
    os.makedirs(out_dir, exist_ok=True)
    n_show = nrows * ncols
    allow_augmented = getattr(config, "use_simulated_data", False)

    for neuron_id, entries in sorted(entries_by_cluster.items()):
        selected = select_top_entries(entries, n_show, allow_augmented=allow_augmented)
        if not selected:
            continue

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.5, nrows * 6.5))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")

        for idx, entry in enumerate(selected):
            ax = axes.flat[idx]
            ok = loader(ax, entry)
            if not ok:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=missing_font)
                continue
            ax.set_title(_format_entry_title(entry, config), fontsize=title_font)

        if title_prefix:
            plt.suptitle(f"{title_prefix} {neuron_id}", fontsize=suptitle_font)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_file = os.path.join(out_dir, name_template.format(neuron_id=neuron_id))
        fig.savefig(out_file, dpi=180)
        plt.close(fig)
        print(f"Saved {out_file}")


def _plot_representative_grid(entries_by_cluster, out_dir, out_name, title, loader, config,
                              title_font=12, missing_font=10, suptitle_font=18, coord_font=14):
    os.makedirs(out_dir, exist_ok=True)
    allow_augmented = getattr(config, "use_simulated_data", False)

    best = {}
    for neuron_id, entries in entries_by_cluster.items():
        selected = select_top_entries(entries, 1, allow_augmented=allow_augmented)
        if selected:
            best[neuron_id] = selected[0]

    fig, axes = plotting._make_grid_canvas(config)
    for i, j, ax, neuron_id in plotting._grid_iter_axes(axes, config):
        ax.axis("off")
        entry = best.get(neuron_id)
        if entry is None:
            continue
        ok = loader(ax, entry)
        if not ok:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=missing_font)
            continue
        ax.set_title(_format_rep_title(entry, neuron_id, config), fontsize=title_font)

    # Add coordinate labels (origin bottom-left) without axes lines
    nrows = int(getattr(config, "n_num", 0))
    ncols = int(getattr(config, "m_num", 0))
    for i, j, ax, _ in plotting._grid_iter_axes(axes, config):
        if nrows <= 0 or ncols <= 0:
            continue
        # x labels under each column (bottom row)
        if i == nrows - 1:
            ax.text(
                0.5, -0.08, f"{j}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=coord_font,
            )
        # y labels next to each row (left column), bottom-left origin
        if j == 0:
            y_coord = nrows - 1 - i
            ax.text(
                -0.08, 0.5, f"{y_coord}",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=coord_font,
            )

    plt.suptitle(title, fontsize=suptitle_font)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_gsom_representative_grid(entries, out_dir, out_name, title, loader, config,
                                   title_font=12, missing_font=10, coord_font=14):
    os.makedirs(out_dir, exist_ok=True)
    allow_augmented = getattr(config, "use_simulated_data", False)

    node_map = defaultdict(list)
    for entry in entries:
        x = entry.get("node_x")
        y = entry.get("node_y")
        if x is None or y is None:
            continue
        node_map[(int(x), int(y))].append(entry)

    if not node_map:
        print(f"No GSOM coordinates available for {title}; skipping.")
        return

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from scipy.ndimage import zoom as ndizoom

    items = []
    for (x, y), entries_at in node_map.items():
        selected = select_top_entries(entries_at, 1, allow_augmented=allow_augmented)
        if not selected:
            continue
        entry = selected[0]
        fig_tmp, ax_tmp = plt.subplots(figsize=(2, 2))
        ax_tmp.axis("off")
        ok = loader(ax_tmp, entry)
        plt.close(fig_tmp)
        if not ok:
            continue
        items.append((x, y, entry))

    if not items:
        print(f"No loadable images for {title}; skipping.")
        return

    xs = np.array([it[0] for it in items], dtype=float)
    ys = np.array([it[1] for it in items], dtype=float)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_span = max(1.0, x_max - x_min + 1.0)
    y_span = max(1.0, y_max - y_min + 1.0)

    cell = 0.9
    fig_w = max(7.0, min(50.0, x_span * cell + 2.5))
    fig_h = max(6.0, min(50.0, y_span * cell + 2.5))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw thumbnails at node coordinates
    for x, y, entry in items:
        img_fig, img_ax = plt.subplots(figsize=(2, 2))
        img_ax.axis("off")
        ok = loader(img_ax, entry)
        img_fig.canvas.draw()
        buf = np.frombuffer(img_fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = img_fig.canvas.get_width_height()
        img_arr = buf.reshape(h, w, 3)
        plt.close(img_fig)

        if not ok:
            continue

        if img_arr.shape[0] != 128 or img_arr.shape[1] != 128:
            zy = 128 / max(img_arr.shape[0], 1)
            zx = 128 / max(img_arr.shape[1], 1)
            img_arr = ndizoom(img_arr, (zy, zx, 1), order=1)

        oi = OffsetImage(img_arr, zoom=0.35)
        ab = AnnotationBbox(oi, (x, y), frameon=True,
                            bboxprops=dict(edgecolor="black", linewidth=0.4),
                            pad=0.0)
        ax.add_artist(ab)

    ax.set_xlim(x_min - 0.8, x_max + 0.8)
    ax.set_ylim(y_max + 0.8, y_min - 0.8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=title_font, pad=6)

    # Keep coordinates, remove axis border
    ax.set_xlabel("Node x", fontsize=coord_font)
    ax.set_ylabel("Node y", fontsize=coord_font)
    ax.set_xticks(np.arange(int(np.floor(x_min)), int(np.ceil(x_max)) + 1))
    ax.set_yticks(np.arange(int(np.floor(y_min)), int(np.ceil(y_max)) + 1))
    ax.tick_params(axis="both", which="major", labelsize=coord_font, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recreate representative/per-neuron plots from source_cluster_assignments.csv with larger fonts."
    )
    parser.add_argument("--run-dir", default="/raid/scratch/mevans/run_12-05_16:24")
    parser.add_argument("--assignments", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    assignments_path = args.assignments or os.path.join(run_dir, "source_cluster_assignments.csv")
    out_dir = args.out_dir or os.path.join(run_dir, "replot_from_assignments_bigfont")

    os.makedirs(out_dir, exist_ok=True)

    config = load_config_from_run_info(run_dir, out_dir)
    entries = read_assignments(assignments_path)
    if not entries:
        raise RuntimeError(f"No entries found in {assignments_path}")

    processed_map = build_processed_map(config)

    entries_by_cluster = defaultdict(list)
    for entry in entries:
        entries_by_cluster[int(entry["cluster"])].append(entry)

    per_neuron_dir = plotting._ensure_plot_subdir(config, "per_neuron_plots")
    rep_dir = plotting._ensure_plot_subdir(config, "representative_plots")

    _plot_per_neuron_grid(
        entries_by_cluster,
        out_dir=os.path.join(per_neuron_dir, "neuron_original_examples"),
        name_template="neuron_{neuron_id:03d}_originals.png",
        title_prefix="Top-Silhouette Originals for Neuron",
        loader=lambda ax, e: _load_original(ax, e, config),
        config=config,
        nrows=3,
        ncols=3,
        title_font=28,
        missing_font=22,
        suptitle_font=32,
    )

    _plot_per_neuron_grid(
        entries_by_cluster,
        out_dir=os.path.join(per_neuron_dir, "neuron_processed_examples"),
        name_template="neuron_{neuron_id:03d}_processed.png",
        title_prefix="Top-Silhouette Processed for Neuron",
        loader=lambda ax, e: _load_processed(ax, e, processed_map, config),
        config=config,
        nrows=3,
        ncols=3,
        title_font=28,
        missing_font=22,
        suptitle_font=32,
    )

    if getattr(config, "smoothed_image_file_dir", None):
        _plot_per_neuron_grid(
            entries_by_cluster,
            out_dir=os.path.join(per_neuron_dir, "neuron_smoothed_examples"),
            name_template="neuron_{neuron_id:03d}_smoothed.png",
            title_prefix="Top-Silhouette Smoothed for Neuron",
            loader=lambda ax, e: _load_smoothed(ax, e, config),
            config=config,
            nrows=3,
            ncols=3,
            title_font=28,
            missing_font=22,
            suptitle_font=32,
        )

    if getattr(config, "molecular_line_file_dir", None):
        line_dir = os.path.join(per_neuron_dir, "neuron_molecular_examples")
        for line in ("H2CO", "SiO", "DCN"):
            line_out = os.path.join(line_dir, line)
            os.makedirs(line_out, exist_ok=True)
            _plot_per_neuron_grid(
                entries_by_cluster,
                out_dir=line_out,
                name_template=f"neuron_{{neuron_id:03d}}_{line}.png",
                title_prefix=f"Top-Silhouette {line} for Neuron",
                loader=lambda ax, e, ln=line: _load_molecular(ax, e, config, ln),
                config=config,
                nrows=3,
                ncols=3,
                title_font=28,
                missing_font=22,
                suptitle_font=32,
            )

    _plot_representative_grid(
        entries_by_cluster,
        out_dir=rep_dir,
        out_name="som_representative_grid_originals.png",
        title="Silhouette-Best Original Image per SOM Neuron",
        loader=lambda ax, e: _load_original(ax, e, config),
        config=config,
        title_font=22,
        missing_font=16,
        suptitle_font=26,
        coord_font=26,
    )

    _plot_representative_grid(
        entries_by_cluster,
        out_dir=rep_dir,
        out_name="som_representative_grid_processed.png",
        title="Silhouette-Best Processed Image per SOM Neuron",
        loader=lambda ax, e: _load_processed(ax, e, processed_map, config),
        config=config,
        title_font=22,
        missing_font=16,
        suptitle_font=26,
        coord_font=18,
    )

    if getattr(config, "smoothed_image_file_dir", None):
        _plot_representative_grid(
            entries_by_cluster,
            out_dir=rep_dir,
            out_name="som_representative_grid_smoothed.png",
            title="Silhouette-Best Smoothed Image per SOM Neuron",
            loader=lambda ax, e: _load_smoothed(ax, e, config),
            config=config,
            title_font=22,
            missing_font=16,
            suptitle_font=26,
            coord_font=18,
        )

    if getattr(config, "molecular_line_file_dir", None):
        for line in ("H2CO", "SiO", "DCN"):
            _plot_representative_grid(
                entries_by_cluster,
                out_dir=rep_dir,
                out_name=f"som_representative_grid_{line}.png",
                title=f"Silhouette-Best {line} Map per SOM Neuron",
                loader=lambda ax, e, ln=line: _load_molecular(ax, e, config, ln),
                config=config,
                title_font=22,
                missing_font=16,
                suptitle_font=26,
                coord_font=18,
            )

    # GSOM representative grids from CSV coordinates
    _plot_gsom_representative_grid(
        entries,
        out_dir=rep_dir,
        out_name="gsom_representative_originals.png",
        title="GSOM Representative Originals",
        loader=lambda ax, e: _load_original(ax, e, config),
        config=config,
        title_font=22,
        missing_font=16,
        coord_font=26,
    )

    _plot_gsom_representative_grid(
        entries,
        out_dir=rep_dir,
        out_name="gsom_representative_processed.png",
        title="GSOM Representative Processed",
        loader=lambda ax, e: _load_processed(ax, e, processed_map, config),
        config=config,
        title_font=22,
        missing_font=16,
        coord_font=26,
    )

    if getattr(config, "smoothed_image_file_dir", None):
        _plot_gsom_representative_grid(
            entries,
            out_dir=rep_dir,
            out_name="gsom_representative_smoothed.png",
            title="GSOM Representative Smoothed",
            loader=lambda ax, e: _load_smoothed(ax, e, config),
            config=config,
            title_font=22,
            missing_font=16,
            coord_font=26,
        )

    if getattr(config, "molecular_line_file_dir", None):
        for line in ("H2CO", "SiO", "DCN"):
            _plot_gsom_representative_grid(
                entries,
                out_dir=rep_dir,
                out_name=f"gsom_representative_{line}.png",
                title=f"GSOM Representative {line}",
                loader=lambda ax, e, ln=line: _load_molecular(ax, e, config, ln),
                config=config,
                title_font=22,
                missing_font=16,
                coord_font=26,
            )


if __name__ == "__main__":
    main()
