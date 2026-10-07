import csv
import os
from collections import Counter
import numpy as np
import collections
from typing import Dict, List, Optional, Tuple

input_files = ["/raid/scratch/mevans/run_30-04_14:25/source_cluster_assignments.csv",
               "/raid/scratch/mevans/run_30-04_14:35/source_cluster_assignments.csv"] #need list of two files
gsom_is_used = False 

def _read_cluster_csv(path: str) -> Dict[str, int]:
    """Read a source-to-cluster mapping from one CSV file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    mapping: Dict[str, int] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        has_source_id = "source_id" in reader.fieldnames
        has_filename = "filename" in reader.fieldnames
        has_cluster = "cluster" in reader.fieldnames

        if not has_cluster:
            raise ValueError(f"CSV missing required 'cluster' column: {path}")

        for row in reader:
            raw_cluster = row.get("cluster", "")
            try:
                cluster = int(float(raw_cluster))
            except (TypeError, ValueError):
                continue

            source_id = row.get("source_id", "").strip() if has_source_id else ""
            if not source_id and has_filename:
                filename = row.get("filename", "").strip()
                source_id = os.path.splitext(os.path.basename(filename))[0]
            if not source_id:
                source_id = os.path.splitext(os.path.basename(path))[0]

            mapping[source_id] = cluster

    if not mapping:
        raise ValueError(f"No valid rows found in {path}")

    return mapping

def compare_clusters(sorted_og_counts, sorted_comp_counts, original_mapping, comparison_mapping):
    original_clusters = []
    comparison_clusters = []
    percentage_matches = []
    for og_cluster, og_count in sorted_og_counts.items():
        for comp_cluster, comp_count in sorted_comp_counts.items():
            if (comp_count-1000) <= og_count <= (comp_count+1000):
                og_files = [k for k, v in original_mapping.items() if v == og_cluster]
                comp_files = [k for k, v in comparison_mapping.items() if v == comp_cluster]

                matching_files = set(og_files)&set(comp_files)
                if len(matching_files) != 0:
                    #print(f'og cluster {og_cluster} size {og_count},comp cluster {comp_cluster} size {comp_count}, matching files {len(matching_files)}')
                    percent_match = len(matching_files) / og_count * 100
                    print(f'clusters: {og_cluster}, {comp_cluster} - percent match = {round(percent_match,2)}%')
                    original_clusters.append(og_cluster)
                    comparison_clusters.append(comp_cluster)
                    percentage_matches.append(percent_match)

    return original_clusters, comparison_clusters, percentage_matches

def find_match(original_clusters, comparison_clusters, percentage_matches):
    used_values = []
    cluster_to_best_match = {}
    for idx, og_cluster in enumerate(original_clusters):
        if any(used_value == og_cluster for used_value in used_values) == False:
            used_values.append(og_cluster)
            all_og_cluster_indices = [i for i, x in enumerate(original_clusters) if x == og_cluster]

            relevant_og_cluster = [original_clusters[i] for i in all_og_cluster_indices]
            relevant_comp_cluster = [comparison_clusters[i] for i in all_og_cluster_indices]
            relevant_percent_match = [percentage_matches[i] for i in all_og_cluster_indices]
                
            highest_percent_idx = np.argmax(relevant_percent_match)
            matching_cluster = relevant_comp_cluster[highest_percent_idx]
            print(f'cluster {og_cluster} best matches with cluster {matching_cluster} (with {round(np.max(relevant_percent_match),2)}%)')
            cluster_to_best_match.update({og_cluster:matching_cluster})
    return dict(cluster_to_best_match)

def find_neighbours(clusters, max_clusters, length):
    cluster_to_neighbours = {}
    used_values = []
    for cluster in range(max_clusters+1):
        if any(used_value == cluster for used_value in used_values) == False:
            used_values.append(cluster)
            neighbour_clusters = [cluster-length, cluster-1, cluster+1, cluster+length]
            neighbour_clusters = np.array(neighbour_clusters)
            neighbour_clusters = neighbour_clusters[(neighbour_clusters >= 0) & (neighbour_clusters <= max_clusters)]
            cluster_to_neighbours.update({cluster:neighbour_clusters})
    return cluster_to_neighbours

def find_neighbour_match(original_clusters, comparison_clusters, percentage_matches, cluster_to_best_match, cluster_to_neighbours):
    used_values = []
    percentages_list = []
    for idx, og_cluster in enumerate(original_clusters):
        if any(used_value == og_cluster for used_value in used_values) == False:
            used_values.append(og_cluster)
            matching_cluster = cluster_to_best_match.get(og_cluster)
            matching_neighbours = cluster_to_neighbours.get(matching_cluster)

            all_og_cluster_indices = [i for i, x in enumerate(original_clusters) if x == og_cluster]
            relevant_og_cluster = [original_clusters[i] for i in all_og_cluster_indices]
            relevant_comp_cluster = [comparison_clusters[i] for i in all_og_cluster_indices]
            relevant_percent_match = [percentage_matches[i] for i in all_og_cluster_indices]

            common_clusters = {}
            for ind, cluster in enumerate(relevant_comp_cluster):
                if cluster in matching_neighbours:
                    common_clusters.update({cluster:ind}) 
            percentages = []
            percentages.append(np.max(relevant_percent_match))
            for cluster, ind in common_clusters.items():
                percentages.append(relevant_percent_match[ind])
            neighbour_percent = sum(percentages)
            percentages_list.append(neighbour_percent)
            print(f'Cluster {og_cluster} best matches with cluster {matching_cluster}, including neighbours - {round((neighbour_percent),2)}%.')
    print(f'Mean of percentages: {np.mean(percentages_list)}')
    return

def run_cluster_diagnostics(
    encoded_data,
    preds,
    som_transform,
    config,
    gsom_node_coords=None,
):
    """Compute silhouette samples and a proxy score from SOM/GSOM assignments.

    Returns
    -------
    tuple
        (sil_proxy, sil_samples)
    """
    encoded = np.asarray(encoded_data)
    preds = np.asarray(preds)
    dists = np.asarray(som_transform)

    # Silhouette samples
    try:
        from sklearn.metrics import silhouette_samples

        unique_clusters = np.unique(preds)
        if unique_clusters.size < 2 or unique_clusters.size >= len(preds):
            sil_samples = np.zeros(len(preds), dtype=float)
        else:
            sil_samples = silhouette_samples(encoded, preds)
    except Exception:
        sil_samples = np.zeros(len(preds), dtype=float)

    # Proxy: relative separation between best and second-best distances
    if dists.ndim != 2 or dists.shape[0] != len(preds):
        sil_proxy = np.zeros(len(preds), dtype=float)
    else:
        if dists.shape[1] < 2:
            sil_proxy = np.zeros(len(preds), dtype=float)
        else:
            idx = np.argpartition(dists, 1, axis=1)[:, :2]
            row = np.arange(dists.shape[0])[:, None]
            best_two = dists[row, idx]
            best_two.sort(axis=1)
            best = best_two[:, 0]
            second = best_two[:, 1]
            denom = np.where(second > 0, second, 1e-12)
            sil_proxy = (second - best) / denom

    return sil_proxy, sil_samples


def run_cluster_comparison(
    input_files: Optional[List[str]] = None,
    gsom_is_used: bool = False,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Run cluster comparison diagnostics on two assignment CSV files.

    Returns
    -------
    tuple
        (cluster_to_best_match, original_cluster_counts, comparison_cluster_counts)
    """
    files = input_files or []
    if len(files) != 2:
        raise ValueError("input_files must contain exactly two CSV paths")

    original_mapping = _read_cluster_csv(files[0])
    comparison_mapping = _read_cluster_csv(files[1])

    og_max_cluster_number = max(original_mapping.values())
    comp_max_cluster_number = max(comparison_mapping.values())
    original_cluster_counts = {}
    comparison_cluster_counts = {}
    for i in range(og_max_cluster_number + 1):
        original_count = sum(1 for v in original_mapping.values() if v == i)
        original_cluster_counts.update({i: original_count})
    for i in range(comp_max_cluster_number + 1):
        comparison_count = sum(1 for v in comparison_mapping.values() if v == i)
        comparison_cluster_counts.update({i: comparison_count})

    sorted_og_counts = dict(sorted(original_cluster_counts.items(), key=lambda item: item[1]))
    sorted_comp_counts = dict(sorted(comparison_cluster_counts.items(), key=lambda item: item[1]))

    original_clusters, comparison_clusters, percentage_matches = compare_clusters(
        sorted_og_counts,
        sorted_comp_counts,
        original_mapping,
        comparison_mapping,
    )

    cluster_to_best_match = find_match(original_clusters, comparison_clusters, percentage_matches)

    if gsom_is_used is False and original_clusters:
        max_clusters = int(np.max(original_clusters))
        number_of_clusters = max_clusters + 1
        length = int(np.sqrt(number_of_clusters))
        edge_values = list(range(length))
        for i in range(2, length):
            edge_values.append(i * length)
            edge_values.append(((i + 1) * length - 1))
        cluster_to_neighbours = find_neighbours(original_clusters, max_clusters, length)
        find_neighbour_match(
            original_clusters,
            comparison_clusters,
            percentage_matches,
            cluster_to_best_match,
            cluster_to_neighbours,
        )

    return cluster_to_best_match, original_cluster_counts, comparison_cluster_counts


if __name__ == "__main__":
    run_cluster_comparison(input_files=input_files, gsom_is_used=gsom_is_used)



