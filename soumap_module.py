"""Pipeline adapter for the local SOUMAP implementation."""

from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import warnings
from dataclasses import asdict
from importlib import metadata

import numpy as np
import scipy.sparse as sp


SOURCE_COLUMNS = [
    "source_id", "flip_type", "cluster", "node_x", "node_y",
    "filename", "label", "silhouette_score",
]
EXTRA_SOURCE_COLUMNS = [
    "proto_id", "embed_x", "embed_y", "qe", "stage", "algorithm", "seed",
]
PROTO_COLUMNS = [
    "proto_id", "embed_x", "embed_y", "n_sources", "mean_qe",
    "mean_silhouette", "dominant_stage", "dominant_frac", "frac_Q",
    "frac_P", "frac_Y", "frac_H", "frac_na", "representative_source_id",
]
STAGES = ("Q", "P", "Y", "H", "na")


def _require_soumap():
    names = ("CtrlParams", "SOUMAP", "SOMParams", "UMAPParams")
    try:
        package = importlib.import_module("soumap")
        if all(hasattr(package, name) for name in names):
            return tuple(getattr(package, name) for name in names[1:]) + (package.CtrlParams,)
    except ImportError:
        package = None

    local_package = os.path.join(
        os.path.dirname(__file__), "soumap", "src", "soumap"
    )
    init_path = os.path.join(local_package, "__init__.py")
    if os.path.isfile(init_path):
        spec = importlib.util.spec_from_file_location(
            "soumap",
            init_path,
            submodule_search_locations=[local_package],
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules["soumap"] = module
            try:
                spec.loader.exec_module(module)
                if all(hasattr(module, name) for name in names):
                    return tuple(getattr(module, name) for name in names[1:]) + (module.CtrlParams,)
            except ImportError as exc:
                raise ImportError(
                    "SOUMAP was found locally, but one of its dependencies "
                    "could not be imported. Install with "
                    "'python -m pip install -r requirements-soumap.txt'."
                ) from exc

    raise ImportError(
        "SOUMAP is not importable. Install the local package with "
        "'python -m pip install -r requirements-soumap.txt'."
    )


def _array_config(config, prefix):
    return {key[len(prefix):]: value for key, value in vars(config).items()
            if key.startswith(prefix)}


def train_soumap(encoded_data, labels, config):
    X = np.asarray(encoded_data)
    if X.ndim != 2:
        raise ValueError(f"SOUMAP input must be 2-D, got shape {X.shape}")
    if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
        raise ValueError("SOUMAP input must be finite numeric encoded_data")
    if X.shape[0] < 2:
        raise ValueError("SOUMAP requires at least two samples")

    SOUMAP, SOMParams, UMAPParams, CtrlParams = _require_soumap()
    M = getattr(config, "soumap_prototypes", None)
    M = int(M if M is not None else config.n_num * config.m_num)
    if M < 2 or M > X.shape[0]:
        raise ValueError(f"SOUMAP prototypes M={M} must be in [2, {X.shape[0]}]")

    som_values = _array_config(config, "soumap_som_")
    # The current unpinned gtsom dependency renamed the source repo's
    # CONN/CONN_STK values. Preserve the soumap defaults at the pipeline
    # boundary while translating them to the installed API.
    som_values["proto_topo"] = {
        "CONN": "CONN_GD",
        "CONN_STK": "STK_CADJ",
    }.get(som_values["proto_topo"], som_values["proto_topo"])
    umap_values = _array_config(config, "soumap_umap_")
    ctrl_values = _array_config(config, "soumap_ctrl_")
    ctrl_values["random_state"] = getattr(config, "soumap_random_state", None)
    model = SOUMAP(
        M=M,
        som_params=SOMParams(**som_values),
        umap_params=UMAPParams(**umap_values),
        ctrl_params=CtrlParams(**ctrl_values),
    )
    # Pipeline labels are filenames, not categorical class labels. Passing
    # them into GTSOM triggers optional colour-table generation and is not
    # needed for SOUMAP training.
    model.build(X)
    model.fit(X)

    W = np.asarray(model.W, dtype=float)
    coords = np.asarray(model.coords, dtype=float)
    distances = np.linalg.norm(X[:, None, :] - W[None, :, :], axis=2)
    bmu = np.argmin(distances, axis=1).astype(np.int64)
    sample_coords = coords[bmu]
    qe = distances[np.arange(len(X)), bmu]
    second = np.argsort(distances, axis=1)[:, 1]
    cadj = np.zeros((M, M), dtype=np.int64)
    for left, right in zip(bmu, second):
        if left == right:
            continue
        cadj[left, right] += 1
        cadj[right, left] += 1
    return {
        "model": model, "M": M, "W": W, "coords": coords,
        "distances": distances, "bmu": bmu, "sample_coords": sample_coords,
        "qe": qe, "cadj": cadj,
    }


def run_mpec(result, config):
    """Fit optional MPEC clustering on a fitted SOUMAP result."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from soumap import MPEClustering

    mpec = MPEClustering(
        walktrap_n_steps=getattr(config, "mpec_walktrap_n_steps", "auto"),
        kernel_w_cadj=getattr(config, "mpec_kernel_w_cadj", 1.0),
        kernel_w_high_d=getattr(config, "mpec_kernel_w_high_d", 1.0),
        kernel_w_low_d=getattr(config, "mpec_kernel_w_low_d", 1.0),
        kernel_power=getattr(config, "mpec_kernel_power", 2.0),
        kernel_min_value=getattr(config, "mpec_kernel_min_value", 0.01),
        kernel_min_nhbs=getattr(config, "mpec_kernel_min_nhbs", 3),
        kernel_support=getattr(config, "mpec_kernel_support", "dense"),
        walktrap_max_steps=getattr(config, "mpec_walktrap_max_steps", 15),
        remove_empty_protos=getattr(config, "mpec_remove_empty_protos", True),
        verbose=getattr(config, "mpec_verbose", True),
    )
    W = np.asarray(result["W"])
    Y = np.asarray(result["coords"])
    CADJ = sp.csr_matrix(result["cadj"])
    mpec.fit(W, Y, CADJ)

    output_dir = getattr(config, "mpec_output_dir", None) or os.path.join(
        config.run_dir, "mpec_plots"
    )
    os.makedirs(output_dir, exist_ok=True)
    mpec.save(os.path.join(config.run_dir, "mpec.h5"))
    np.save(os.path.join(config.run_dir, "mpec_labels.npy"), mpec.labels_)
    np.save(os.path.join(config.run_dir, "mpec_active_indices.npy"), mpec.active_indices_)

    which = getattr(config, "mpec_plot_which", "all")
    if isinstance(which, str) and which != "all":
        which = [name.strip() for name in which.split(",") if name.strip()]
    mpec.plot_diagnostics(
        embed_coords=Y,
        which=which,
        save_dir=output_dir,
        save_format=getattr(config, "mpec_plot_format", "png"),
        figsize=(6, 5),
        show_graph=getattr(config, "mpec_show_graph", False),
    )
    print(
        f"MPEC complete: {len(mpec.active_labels_)} active prototypes, "
        f"{np.sum(mpec.labels_ < 0)} empty, n_steps={mpec.n_steps_}."
    )
    return mpec


def _source_id(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    for marker in ("_cont_", "_continuum", "_smoothed", "_moment0"):
        if marker in stem:
            return stem.split(marker, 1)[0]
    return stem


def _stage(label):
    for stage in STAGES:
        if stage == label or f" {stage}" in str(label):
            return stage
    return "na"


def _flip_type(label):
    text = str(label)
    if "[hvflip]" in text:
        return "hvflip"
    if "[hflip]" in text:
        return "hflip"
    if "[vflip]" in text:
        return "vflip"
    return "original"


def _catalogue_stages(filenames, labels):
    """Return catalogue stages in sample order using the shared lookup rules."""
    try:
        import save_results

        number_labels, _, classifications = save_results.read_cluster_info()
        catalogue = dict(zip(number_labels, classifications))
        stages = []
        unmatched = 0
        for filename, label in zip(filenames, labels):
            source_id = save_results.extract_id_from_filename(filename)
            if source_id in catalogue:
                stages.append(catalogue[source_id])
            else:
                unmatched += 1
                stages.append("na")
    except (FileNotFoundError, OSError):
        warnings.warn(
            "table1.dat was not found; falling back to stage tokens in labels.",
            RuntimeWarning,
        )
        stages = [_stage(label) for label in labels]
        unmatched = len(stages)

    counts = {stage: stages.count(stage) for stage in STAGES}
    print(
        "SOUMAP stage lookup: "
        + ", ".join(f"{stage}={counts[stage]}" for stage in STAGES)
        + f", unmatched={unmatched}"
    )
    if stages and counts["na"] / len(stages) > 0.9:
        warnings.warn(
            f"SOUMAP stage lookup produced {counts['na']}/{len(stages)} "
            "na stages (>90%).",
            RuntimeWarning,
        )
    return stages


def write_outputs(result, filenames, labels, sil_scores, config):
    bmu = result["bmu"]
    coords = result["coords"]
    qe = result["qe"]
    M = result["M"]
    seed = getattr(config, "soumap_random_state", None)
    stages = _catalogue_stages(filenames, labels)
    source_path = os.path.join(config.run_dir, "source_cluster_assignments.csv")
    with open(source_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOURCE_COLUMNS + EXTRA_SOURCE_COLUMNS)
        for i, (filename, label) in enumerate(zip(filenames, labels)):
            p = int(bmu[i])
            writer.writerow([
                _source_id(filename), _flip_type(label), p, coords[p, 0], coords[p, 1],
                os.path.basename(filename), label, float(sil_scores[i]),
                p, coords[p, 0], coords[p, 1], qe[i], stages[i], "soumap", seed,
            ])

    proto_path = os.path.join(config.run_dir, "soumap_prototypes.csv")
    with open(proto_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PROTO_COLUMNS)
        for p in range(M):
            indices = np.flatnonzero(bmu == p)
            counts = {stage: sum(stages[i] == stage for i in indices) for stage in STAGES}
            dominant = max(STAGES, key=lambda stage: counts[stage])
            representative = ""
            if len(indices):
                representative = _source_id(filenames[indices[np.argmax(np.asarray(sil_scores)[indices])]])
            total = max(len(indices), 1)
            writer.writerow([
                p, coords[p, 0], coords[p, 1], len(indices),
                float(np.mean(qe[indices])) if len(indices) else 0.0,
                float(np.mean(np.asarray(sil_scores)[indices])) if len(indices) else 0.0,
                dominant, counts[dominant] / total,
                *(counts[stage] / total for stage in STAGES), representative,
            ])

    _plot_prototype_stages(result["coords"], bmu, stages, config)

    np.save(os.path.join(config.run_dir, "soumap_W.npy"), result["W"])
    np.save(os.path.join(config.run_dir, "soumap_coords.npy"), result["coords"])
    np.save(os.path.join(config.run_dir, "soumap_cadj.npy"), result["cadj"])
    np.save(os.path.join(config.run_dir, "soumap_sample_bmu.npy"), result["bmu"])

    info = {
        "algorithm": "soumap", "M": M, "seed": seed,
        "run_dir": config.run_dir,
        "premade_autoencoder_results_dir": getattr(config, "premade_autoencoder_results_dir", None),
        "premade_encoded_already_scaled": getattr(config, "premade_encoded_already_scaled", None),
        "ae_bottleneck_dim": getattr(config, "ae_bottleneck_dim", None),
        "ae_use_bottleneck": getattr(config, "ae_use_bottleneck", None),
        "n_samples": len(bmu), "latent_dim": result["W"].shape[1],
        "final_quantization_error": float(np.mean(qe)),
        "soumap": {key: value for key, value in vars(config).items() if key.startswith("soumap_")},
        "soumap_params": {
            "som": asdict(result["model"].som_params),
            "umap": asdict(result["model"].umap_params),
            "ctrl": asdict(result["model"].ctrl_params),
        },
        "versions": {},
        "cadj_route": "sample closest/second-closest prototype counts",
        "neighbour_overlap": "not computed; SOUMAP grid neighbours are undefined",
    }
    for package in ("numpy", "scipy", "scikit-learn", "umap-learn", "soumap", "vqlp", "gtsom"):
        try:
            info["versions"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info["versions"][package] = None
    info["dependency_git_commits"] = {}
    for package in ("vqlp", "gtsom"):
        try:
            direct_url = metadata.distribution(package).read_text("direct_url.json")
            direct_url = json.loads(direct_url or "{}")
            info["dependency_git_commits"][package] = (
                direct_url.get("vcs_info", {}).get("commit_id")
            )
        except (metadata.PackageNotFoundError, json.JSONDecodeError):
            info["dependency_git_commits"][package] = None
    try:
        info["soumap_git_commit"] = subprocess.check_output(
            ["git", "-C", os.path.join(os.path.dirname(__file__), "soumap"),
             "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        info["soumap_git_commit"] = None
    with open(os.path.join(config.run_dir, "soumap_run_info.json"), "w") as handle:
        json.dump(info, handle, indent=2, default=str)
    return source_path, proto_path


def _plot_prototype_stages(coords, bmu, stages, config):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    dominant = []
    hit_counts = []
    for proto_id in range(len(coords)):
        indices = np.flatnonzero(bmu == proto_id)
        counts = [sum(stages[i] == stage for i in indices) for stage in STAGES]
        dominant.append(STAGES[int(np.argmax(counts))] if indices.size else "na")
        hit_counts.append(len(indices))
    hit_counts = np.asarray(hit_counts, dtype=float)
    sizes = _prototype_marker_sizes(hit_counts)
    max_hits = max(float(hit_counts.max()), 1.0)
    plasma = plt.get_cmap("plasma")
    colors = {
        "Q": plasma(0.15), "P": plasma(0.35), "Y": plasma(0.55),
        "H": plasma(0.75), "na": plasma(0.95),
    }
    fig, ax = plt.subplots(figsize=(7, 6))
    for stage in STAGES:
        selected = [i for i, value in enumerate(dominant) if value == stage]
        if selected:
            ax.scatter(coords[selected, 0], coords[selected, 1],
                       s=sizes[selected], c=[colors[stage]], alpha=0.8,
                       edgecolors="0.2", linewidths=0.5)
    if getattr(config, "soumap_label_prototypes", False):
        for proto_id, (x, y) in enumerate(coords):
            ax.annotate(str(proto_id), (x, y), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=7)
    ax.set_xlabel("Embedding x")
    ax.set_ylabel("Embedding y")
    ax.set_title("SOUMAP prototype evolutionary stage")
    ax.set_aspect("equal", adjustable="box")
    x_range = max(float(np.ptp(coords[:, 0])), 1.0)
    y_range = max(float(np.ptp(coords[:, 1])), 1.0)
    ax.margins(x=0.08 * max(1.0, 1.0 / x_range), y=0.08 * max(1.0, 1.0 / y_range))
    stage_handles = [
        Patch(facecolor=colors[stage], edgecolor="0.2", label=stage)
        for stage in STAGES
    ]
    stage_legend = ax.legend(handles=stage_handles, title="Stage",
                             loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.add_artist(stage_legend)
    size_values = [1, max(1, int(round(max_hits / 2))), int(max_hits)]
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="0.2",
               markerfacecolor="0.6", markersize=np.sqrt(30.0 + 570.0 * value / max_hits),
               label=f"{value} hits")
        for value in size_values
    ]
    ax.legend(handles=size_handles, title="Prototype hits",
              loc="lower left", bbox_to_anchor=(1.02, 0))
    fig.tight_layout()
    path = os.path.join(config.run_dir, "soumap_prototype_stages.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _prototype_marker_sizes(hit_counts, s_min=30.0, s_max=600.0):
    hit_counts = np.asarray(hit_counts, dtype=float)
    if hit_counts.size == 0:
        return hit_counts
    max_hits = max(float(hit_counts.max()), 1.0)
    return s_min + (s_max - s_min) * hit_counts / max_hits
