"""CPU-safe embedding and SOM diagnostics."""

import csv
import copy
import os
import re
import traceback
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except ImportError:
    umap = None

try:
    from skimage.measure import label as connected_components
except ImportError:
    connected_components = None


def _extract_source_id_from_filename(path):
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    for marker in ("_cont_", "_continuum_preproc", "_continuum", "_smoothed", "_moment0"):
        if marker in stem:
            return stem.split(marker, 1)[0]
    return stem


def _flip_type(label):
    text = str(label).lower()
    if "[hvflip]" in text:
        return "hvflip"
    if "[hflip]" in text:
        return "hflip"
    if "[vflip]" in text:
        return "vflip"
    if "[rot" in text:
        return "rot"
    return "original"


def _save_scatter(path, embedding, values, title, cmap="viridis", categorical=False,
                  labels=None, center_zero=False):
    rng = np.random.default_rng(42)
    order = rng.permutation(len(embedding))
    fig, ax = plt.subplots(figsize=(7, 6))
    if categorical:
        categories = list(dict.fromkeys(values))
        colors = plt.get_cmap("tab10")(np.arange(len(categories)) % 10)
        lookup = dict(zip(categories, colors))
        for category in categories:
            mask = np.asarray([v == category for v in values])
            selected = order[mask[order]]
            ax.scatter(embedding[selected, 0], embedding[selected, 1], s=6,
                       alpha=0.6, rasterized=True, color=lookup[category],
                       label=str(category))
        ax.legend(markerscale=2, fontsize=8)
    else:
        norm = TwoSlopeNorm(vmin=np.min(values), vcenter=0, vmax=np.max(values)) if center_zero and np.min(values) < 0 < np.max(values) else None
        scatter = ax.scatter(embedding[order, 0], embedding[order, 1],
                             c=np.asarray(values)[order], cmap=cmap, norm=norm,
                             s=6, alpha=0.6, rasterized=True)
        fig.colorbar(scatter, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _image_properties(raw_datas):
    result = {"mask_area_fraction": [], "connected_components": [],
              "centroid_offset": [], "elongation": []}
    for raw in raw_datas:
        image = np.asarray(raw)
        if image.ndim == 3:
            image = image[..., 0]
        image = np.asarray(image, dtype=float)
        mask = image > 0
        result["mask_area_fraction"].append(float(mask.mean()))
        result["connected_components"].append(
            int(connected_components(mask, connectivity=2).max()) if connected_components else 0
        )
        yy, xx = np.indices(image.shape)
        weights = np.maximum(image, 0)
        total = weights.sum()
        cy, cx = (np.mean(yy), np.mean(xx)) if total == 0 else (
            np.sum(yy * weights) / total, np.sum(xx * weights) / total)
        center = np.array([(image.shape[0] - 1) / 2, (image.shape[1] - 1) / 2])
        result["centroid_offset"].append(float(np.linalg.norm([cy, cx] - center)))
        coords = np.column_stack((yy.ravel(), xx.ravel()))
        if total == 0:
            ratio = 1.0
        else:
            cov = np.cov(coords, rowvar=False, aweights=weights.ravel())
            eig = np.maximum(np.linalg.eigvalsh(np.atleast_2d(cov)), 1e-12)
            ratio = float(np.sqrt(eig[-1] / eig[0]))
        result["elongation"].append(max(1.0, ratio))
    return {key: np.asarray(value) for key, value in result.items()}


def _nearest_purity(X, preds, source_ids, k, exclude_sources=False):
    n = len(X)
    if n <= 1:
        return np.zeros(n), np.zeros(n)
    nn = NearestNeighbors(n_neighbors=min(n, k + 1), algorithm="auto").fit(X)
    indices = nn.kneighbors(return_distance=False)
    purity = np.full(n, np.nan)
    for i, neighbours in enumerate(indices):
        selected = [j for j in neighbours if j != i and
                    (not exclude_sources or source_ids[j] != source_ids[i])][:k]
        if selected:
            purity[i] = np.mean(np.asarray(preds)[selected] == preds[i])
    by_cell = np.zeros(int(np.max(preds)) + 1 if len(preds) else 0)
    for cell in range(len(by_cell)):
        values = purity[np.asarray(preds) == cell]
        by_cell[cell] = np.nanmean(values) if np.any(~np.isnan(values)) else np.nan
    return purity, by_cell


def _som_checks(encoded_data, preds, som):
    distances = som.transform(encoded_data)
    order = np.argsort(distances, axis=1)[:, :2]
    adjacent4 = adjacent8 = 0
    for best, second in order:
        a = np.asarray(som._locations[best])
        b = np.asarray(som._locations[second])
        delta = np.abs(a - b)
        adjacent4 += int(np.sum(delta) == 1)
        adjacent8 += int(np.max(delta) <= 1 and np.sum(delta) > 0)
    weights = np.asarray(som.weights)
    adjacent, other = [], []
    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            delta = np.abs(np.asarray(som._locations[i]) - som._locations[j])
            (adjacent if np.sum(delta) == 1 else other).append(np.linalg.norm(weights[i] - weights[j]))
    return {
        "quantization_error": float(np.mean(np.min(distances, axis=1))),
        "topographic_error_4": 1 - adjacent4 / len(encoded_data),
        "topographic_error_8": 1 - adjacent8 / len(encoded_data),
        "mean_adjacent_prototype_distance": float(np.mean(adjacent)) if adjacent else 0.0,
        "mean_nonadjacent_prototype_distance": float(np.mean(other)) if other else 0.0,
    }


def _flip_highlights(path, embedding, source_ids, flips):
    groups = defaultdict(list)
    for index, source in enumerate(source_ids):
        groups[source].append(index)
    available = np.asarray(sorted(groups))
    selected_sources = np.random.default_rng(42).choice(
        available, size=min(8, len(available)), replace=False
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=6, alpha=.15,
               color="lightgrey", rasterized=True)
    colors = plt.get_cmap("tab10")(np.arange(len(selected_sources)))
    for color, source in zip(colors, selected_sources):
        indices = groups[source]
        ax.plot(embedding[indices, 0], embedding[indices, 1], color=color,
                linewidth=.8, alpha=.8)
        ax.scatter(embedding[indices, 0], embedding[indices, 1], s=18,
                   color=color, label=str(source))
    if len(selected_sources):
        ax.legend(fontsize=7)
    ax.set_title("Flip-copy invariance highlights")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _som_multiples(path, embedding, preds, som):
    fig, axes = plt.subplots(som.m, som.n, figsize=(12, 12), squeeze=False)
    for cell, axis in enumerate(axes.flat):
        axis.scatter(embedding[:, 0], embedding[:, 1], s=2, color="lightgrey",
                     alpha=.35, rasterized=True)
        mask = preds == cell
        axis.scatter(embedding[mask, 0], embedding[mask, 1], s=3,
                     color="tab:blue", alpha=.7, rasterized=True)
        axis.set_title(str(cell), fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _fit_tsne(X, perplexity):
    kwargs = dict(n_components=2, init="pca", perplexity=perplexity,
                  learning_rate="auto", random_state=42)
    try:
        return TSNE(max_iter=2000, **kwargs).fit_transform(X)
    except TypeError:
        return TSNE(n_iter=2000, **kwargs).fit_transform(X)


def run_embedding_diagnostics(encoded_data, preds, labels, filenames, raw_datas,
                              sil_samples, config, som=None, fov_by_filename=None,
                              stage_by_source=None):
    np.random.seed(42)
    out_dir = os.path.join(config.run_dir, "embedding_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    X = np.asarray(encoded_data, dtype=float)
    preds = np.asarray(preds)
    labels = list(labels)
    filenames = list(filenames)
    source_ids = np.asarray([_extract_source_id_from_filename(f) for f in filenames])
    flips = np.asarray([_flip_type(label) for label in labels])
    pca_components = min(50, X.shape[0], X.shape[1])
    X_pca = PCA(n_components=pca_components, random_state=42).fit_transform(X)
    max_points = int(getattr(config, "diag_max_points", 6000))
    if len(X) > max_points:
        sources = np.array(sorted(set(source_ids)))
        chosen = np.random.default_rng(42).choice(sources, size=max(1, int(max_points * len(sources) / len(X))), replace=False)
        subset = np.flatnonzero(np.isin(source_ids, chosen))
    else:
        subset = np.arange(len(X))
    X_sub = X_pca[subset]
    summary = [
        "Embedding diagnostics",
        f"N={len(X)}, dimensions={X.shape[1]}, PCA components={pca_components}",
        f"PCA cumulative explained variance={PCA(n_components=pca_components, random_state=42).fit(X).explained_variance_ratio_.sum():.8f}",
        f"Embedding/trustworthiness points={len(subset)}, diag_max_points={max_points}",
        "Evolutionary-stage parser: not used; COMBINEDclassification_info.txt contains cluster class counts, not source stages.",
        "",
    ]
    if som is None:
        summary.append("Grid-based checks skipped: no fixed-grid SOM was supplied (GSOM/non-grid path).")
    embeddings = {}
    if umap is not None:
        for neighbors, min_dist in ((15, .07), (30, .07), (50, .07), (30, .3)):
            key = f"umap_n{neighbors}_d{min_dist}"
            embedding = umap.UMAP(n_neighbors=neighbors, min_dist=min_dist,
                                  metric="euclidean", random_state=42).fit_transform(X_sub)
            embeddings[key] = embedding
            np.save(os.path.join(out_dir, key + ".npy"), embedding)
    else:
        summary.append("UMAP unavailable; skipped UMAP sweep.")
    for perplexity in (15, 30, 50, 100, 200):
        if perplexity >= len(X_sub) / 3:
            continue
        key = f"tsne_p{perplexity}"
        embedding = _fit_tsne(X_sub, perplexity)
        embeddings[key] = embedding
        np.save(os.path.join(out_dir, key + ".npy"), embedding)
    for key, embedding in embeddings.items():
        values = preds[subset]
        if key in ("umap_n30_d0.07", "tsne_p50"):
            _save_scatter(os.path.join(out_dir, key + "_flip.png"), embedding, flips[subset], key + " flip type", categorical=True)
            _save_scatter(os.path.join(out_dir, key + "_silhouette.png"), embedding, np.asarray(sil_samples)[subset], key + " silhouette", center_zero=True)
            props = _image_properties(np.asarray(raw_datas)[subset])
            for prop, values_prop in props.items():
                _save_scatter(os.path.join(out_dir, key + "_" + prop + ".png"), embedding, values_prop, key + " " + prop)
            _flip_highlights(os.path.join(out_dir, key + "_flip_highlights.png"),
                             embedding, source_ids[subset], flips[subset])
            if fov_by_filename:
                fov_values = [fov_by_filename.get(name, np.nan) for name in np.asarray(filenames)[subset]]
                _save_scatter(os.path.join(out_dir, key + "_fov.png"), embedding, np.asarray(fov_values, dtype=float), key + " FOV")
            if stage_by_source:
                stages = [stage_by_source.get(source, "unknown") for source in source_ids[subset]]
                _save_scatter(os.path.join(out_dir, key + "_stage.png"), embedding, stages, key + " stage", categorical=True)
        if som is not None:
            rows = preds[subset] // som.n
            _save_scatter(os.path.join(out_dir, key + "_som_row.png"), embedding, rows, key + " SOM row")
            _save_scatter(os.path.join(out_dir, key + "_som_column.png"), embedding, preds[subset] % som.n, key + " SOM column")
            if key in ("umap_n30_d0.07", "tsne_p50"):
                _som_multiples(os.path.join(out_dir, key + "_som_multiples.png"),
                               embedding, preds[subset], som)
    summary.append("Trustworthiness (best settings marked by highest mean):")
    trust_rows = []
    for key, embedding in embeddings.items():
        scores = [trustworthiness(X_sub, embedding, n_neighbors=k) for k in (10, 30) if len(X_sub) > k + 1]
        if len(scores) == 2:
            trust_rows.append((key, scores[0], scores[1]))
    best_by_method = {}
    for method in ("umap", "tsne"):
        candidates = [row for row in trust_rows if row[0].startswith(method)]
        if candidates:
            best_by_method[method] = max(candidates, key=lambda row: np.mean(row[1:]))
    for row in trust_rows:
        method = row[0].split("_", 1)[0]
        marker = " [BEST]" if best_by_method.get(method) == row else ""
        summary.append(f"  {row[0]}: T@10={row[1]:.6f}, T@30={row[2]:.6f}{marker}")
    purity_metrics = {}
    for space_name, space in (("PCA", X_pca), ("encoded", X)):
        summary.append(f"kNN purity in {space_name} space:")
        purity_metrics[space_name] = {}
        for k in (5, 10, 20):
            raw, raw_cell = _nearest_purity(space, preds, source_ids, k, False)
            filtered, filtered_cell = _nearest_purity(space, preds, source_ids, k, True)
            baseline = float(np.sum([(preds == c).mean() ** 2 for c in np.unique(preds)]))
            summary.append(f"  k={k}: with_same_source={np.nanmean(raw):.6f}, excluding_same_source={np.nanmean(filtered):.6f}, baseline={baseline:.6f}")
            purity_metrics[space_name][k] = (raw, filtered, raw_cell, filtered_cell, baseline)
    groups = defaultdict(list)
    for i, source in enumerate(source_ids):
        groups[source].append(i)
    complete = [indices for indices in groups.values() if set(flips[indices]) >= {"original", "hflip", "vflip", "hvflip"}]
    agreement = [len(set(preds[indices])) == 1 for indices in complete]
    at_most_two = [len(set(preds[indices])) <= 2 for indices in complete]
    summary.append(f"Flip invariance: complete_sources={len(complete)}, all_same_cell={np.mean(agreement) if agreement else 0:.6f}, at_most_two_cells={np.mean(at_most_two) if at_most_two else 0:.6f}")
    if som is not None:
        for key, value in _som_checks(X, preds, som).items():
            summary.append(f"{key}={value:.8f}")
    if getattr(config, "diag_seed_stability", False):
        import som_module
        stability_preds = []
        for seed in range(5):
            np.random.seed(42 + seed)
            trial_config = copy.copy(config)
            trial_config.train_in_blocks_flag = True
            trial_config.diag_seed_stability = False
            trial_config.som_random_state = 42 + seed
            trial_som, trial_preds, _, _, _ = som_module.train_som(
                X, labels, trial_config
            )
            stability_preds.append(trial_preds)
        ari_values = [
            adjusted_rand_score(stability_preds[i], stability_preds[j])
            for i in range(len(stability_preds))
            for j in range(i + 1, len(stability_preds))
        ]
        summary.append(
            f"Seed stability ARI: mean={np.mean(ari_values):.6f}, "
            f"min={np.min(ari_values):.6f}, pairwise_n={len(ari_values)}"
        )
    props = _image_properties(np.asarray(raw_datas))
    with open(os.path.join(out_dir, "cell_summary.csv"), "w", newline="") as handle:
        fields = ["cell", "row", "col", "n_samples", "n_unique_sources", "fraction_flipped_copies",
                  "mean_silhouette", "mean_distance_to_prototype", "knn_purity_k10"] + list(props)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        distances = som.transform(X) if som is not None else np.full((len(X), int(preds.max()) + 1), np.nan)
        purity, _ = _nearest_purity(X_pca, preds, source_ids, 10, True)
        for cell in sorted(set(preds)):
            mask = preds == cell
            row = {"cell": int(cell), "row": int(cell // som.n) if som is not None else "",
                   "col": int(cell % som.n) if som is not None else "",
                   "n_samples": int(mask.sum()), "n_unique_sources": len(set(source_ids[mask])),
                   "fraction_flipped_copies": float(np.mean(flips[mask] != "original")),
                   "mean_silhouette": float(np.mean(np.asarray(sil_samples)[mask])),
                   "mean_distance_to_prototype": float(np.mean(distances[mask, cell])),
                   "knn_purity_k10": float(np.nanmean(purity[mask]))}
            row.update({name: float(np.mean(value[mask])) for name, value in props.items()})
            writer.writerow(row)
    text = "\n".join(summary) + "\n"
    with open(os.path.join(out_dir, "summary.txt"), "w") as handle:
        handle.write(text)
    print(text, end="")
    return {"output_dir": out_dir, "embeddings": embeddings, "purity": purity_metrics,
            "flip_all_same_cell": float(np.mean(agreement)) if agreement else 0.0,
            "flip_at_most_two_cells": float(np.mean(at_most_two)) if at_most_two else 0.0}
