import os
import csv
import ast
import numpy as np
import matplotlib.pyplot as plt

from config import Config
import plotting
from run_pipeline import _load_preprocessed_fits, _load_simulated_fits

RUN_DIR = "/raid/scratch/mevans/run_30-04_14:42"
ASSIGNMENTS_CSV = os.path.join(RUN_DIR, "source_cluster_assignments.csv")


def _parse_value(value):
    text = value.strip()
    if text == "":
        return ""
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def load_config_from_run_info(run_dir):
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

    config.run_dir = run_dir
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
            entries.append({
                "cluster": cluster,
                "filename": row.get("filename", ""),
                "label": row.get("label", ""),
                "silhouette": sil,
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


def _dedupe_by_original(entries):
    """Keep one entry per base label, preferring non-augmented and higher silhouette."""
    by_base = {}
    for entry in entries:
        base = _clean_label(entry.get("label"))
        key = base if base else entry.get("filename")
        is_aug = _is_augmented_label(entry.get("label"))
        current = by_base.get(key)
        if current is None:
            by_base[key] = entry
            continue
        current_aug = _is_augmented_label(current.get("label"))
        if current_aug and not is_aug:
            by_base[key] = entry
            continue
        if current_aug == is_aug and entry.get("silhouette", -np.inf) > current.get("silhouette", -np.inf):
            by_base[key] = entry
    return list(by_base.values())


def select_second_best(entries, n_clusters):
    reps = {}
    for nid in range(n_clusters):
        cluster_entries = [e for e in entries if e["cluster"] == nid]
        if not cluster_entries:
            continue
        unique = _dedupe_by_original(cluster_entries)
        ordered = sorted(unique, key=lambda e: e.get("silhouette", -np.inf), reverse=True)
        if len(ordered) < 2:
            continue
        reps[nid] = ordered[1]
    return reps


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
        if fname:
            processed_map[fname] = img
            processed_map[os.path.basename(fname)] = img
    return processed_map


def main():
    config = load_config_from_run_info(RUN_DIR)
    entries = read_assignments(ASSIGNMENTS_CSV)

    n_clusters = int(getattr(config, "n_num", 0)) * int(getattr(config, "m_num", 0))
    reps = select_second_best(entries, n_clusters)

    out_dir = os.path.join(RUN_DIR, "representative_plots_second_best")
    os.makedirs(out_dir, exist_ok=True)

    processed_map = build_processed_map(config)

    def _get_processed_image(entry):
        fname = entry.get("filename", "")
        img = processed_map.get(fname)
        if img is None:
            img = processed_map.get(os.path.basename(fname))
        return img

    def load_processed(entry):
        img = _get_processed_image(entry)
        if img is None:
            return None
        img = np.nan_to_num(np.asarray(img), nan=0.0)
        img = plotting._apply_label_flip(img, entry.get("label"))
        if getattr(config, "use_simulated_data", False):
            img = plotting._normalize_simulated_plot(img)
        plotting._imshow_processed_with_core_overlay(plt.gca(), img, cmap="plasma")
        return img

    def load_original(entry):
        resolved = plotting._resolve_original_fits_path(entry["filename"], config)
        if resolved is None:
            return None
        try:
            from astropy.io import fits
            with fits.open(resolved) as hdul:
                img = np.nan_to_num(hdul[0].data, nan=0.0)
        except Exception:
            return None
        img = plotting._apply_label_flip(img, entry.get("label"))
        if img.ndim > 2:
            img = np.squeeze(img)
        if img.ndim != 2:
            return None
        if getattr(config, "use_simulated_data", False):
            img = plotting._normalize_simulated_plot(img)
            vmin_local, vmax_local = 0.0, 1.0
        else:
            vmin_local = np.nanmin(img)
            vmax_local = np.nanmax(img)
            if not np.isfinite(vmin_local) or not np.isfinite(vmax_local) or vmin_local == vmax_local:
                vmin_local, vmax_local = None, None
        plt.gca().imshow(img, cmap="plasma", origin="lower", vmin=vmin_local, vmax=vmax_local)
        return img

    def load_smoothed(entry):
        sid = plotting._extract_source_id_from_filename(entry["filename"])
        img = plotting._load_smoothed_image(config, sid)
        if img is None:
            return None
        img = plotting._apply_label_flip(img, entry.get("label"))
        vmin_local = np.nanmin(img)
        vmax_local = np.nanmax(img)
        if not np.isfinite(vmin_local) or not np.isfinite(vmax_local) or vmin_local == vmax_local:
            vmin_local, vmax_local = None, None
        plt.gca().imshow(img, origin="lower", cmap="plasma", vmin=vmin_local, vmax=vmax_local)
        return img

    def load_molecular(entry, line):
        sid = plotting._extract_source_id_from_filename(entry["filename"])
        img = plotting._load_molecular_line_image(config, sid, line)
        if img is None:
            return None
        img = plotting._apply_label_flip(img, entry.get("label"))
        plt.gca().imshow(img, origin="lower", cmap="plasma", vmin=0.0, vmax=1.0)
        return img

    def load_raw_moment0(entry, line):
        sid = plotting._extract_source_id_from_filename(entry["filename"])
        img = plotting._load_raw_moment0_map(config, sid, line)
        if img is None:
            return None
        img = plotting._apply_label_flip(img, entry.get("label"))
        plt.gca().imshow(img, origin="lower", cmap="plasma")
        return img

    def plot_grid_with_title(title, out_name, loader):
        fig, axes = plotting._make_grid_canvas(config)
        for i, j, ax, neuron_id in plotting._grid_iter_axes(axes, config):
            ax.axis("off")
            entry = reps.get(neuron_id)
            if entry is None:
                continue
            plt.sca(ax)
            img = loader(entry)
            if img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
                continue
            ax.set_title(
                plotting._format_image_subheading(
                    0,
                    [entry.get("silhouette", np.nan)],
                    filenames=[entry.get("filename")],
                    labels=[entry.get("label")],
                    config=config,
                ),
                fontsize=8,
            )
        plt.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {title} to {out_path}")

    plot_grid_with_title(
        "Second-Best Processed Image per SOM Neuron",
        "som_representative_grid_processed_second.png",
        load_processed,
    )

    plot_grid_with_title(
        "Second-Best Original Image per SOM Neuron",
        "som_representative_grid_originals_second.png",
        load_original,
    )

    if getattr(config, "smoothed_image_file_dir", None):
        plot_grid_with_title(
            "Second-Best Smoothed Image per SOM Neuron",
            "som_representative_grid_smoothed_second.png",
            load_smoothed,
        )

    if getattr(config, "molecular_line_file_dir", None):
        for line in ("H2CO", "SiO", "DCN"):
            plot_grid_with_title(
                f"Second-Best {line} Map per SOM Neuron",
                f"som_representative_grid_{line}_second.png",
                lambda entry, ln=line: load_molecular(entry, ln),
            )

    if getattr(config, "raw_moment0_dir", None):
        for line in ("H2CO", "SiO", "DCN"):
            plot_grid_with_title(
                f"Second-Best Raw {line} Map per SOM Neuron",
                f"som_representative_grid_raw_{line}_second.png",
                lambda entry, ln=line: load_raw_moment0(entry, ln),
            )


if __name__ == "__main__":
    main()
