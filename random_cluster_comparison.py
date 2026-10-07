import csv
import os
import numpy as np
from itertools import permutations
import glob

ROOT_DIR = "/raid/scratch/mevans/random_cluster_baselines/random_compare_gsom_07-05_12:32"
gsom_is_used = True


def _collect_input_files(root_dir):
    pattern = os.path.join(root_dir, "**", "source_cluster_assignments.csv")
    return sorted(glob.glob(pattern, recursive=True))


def _read_cluster_csv(path):
    """Read a source-to-cluster mapping and cluster coordinates from one CSV file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    mapping = {}
    cluster_coords = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        has_source_id = "source_id" in reader.fieldnames
        has_filename = "filename" in reader.fieldnames
        has_cluster = "cluster" in reader.fieldnames
        has_node_x = "node_x" in reader.fieldnames
        has_node_y = "node_y" in reader.fieldnames

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

            if has_node_x and has_node_y:
                try:
                    node_x = int(float(row.get("node_x", "")))
                    node_y = int(float(row.get("node_y", "")))
                    cluster_coords.setdefault(cluster, (node_x, node_y))
                except (TypeError, ValueError):
                    pass

    if not mapping:
        raise ValueError(f"No valid rows found in {path}")

    return mapping, cluster_coords


def compare_clusters(sorted_og_counts, sorted_comp_counts, original_mapping, comparison_mapping):
    original_clusters = []
    comparison_clusters = []
    percentage_matches = []
    for og_cluster, og_count in sorted_og_counts.items():
        for comp_cluster, comp_count in sorted_comp_counts.items():
            if (comp_count - 1000) <= og_count <= (comp_count + 1000):
                og_files = [k for k, v in original_mapping.items() if v == og_cluster]
                comp_files = [k for k, v in comparison_mapping.items() if v == comp_cluster]

                matching_files = set(og_files) & set(comp_files)
                if len(matching_files) != 0:
                    percent_match = len(matching_files) / og_count * 100
                    original_clusters.append(og_cluster)
                    comparison_clusters.append(comp_cluster)
                    percentage_matches.append(percent_match)

    return original_clusters, comparison_clusters, percentage_matches


def find_match(original_clusters, comparison_clusters, percentage_matches):
    used_values = []
    cluster_to_best_match = {}
    for idx, og_cluster in enumerate(original_clusters):
        if any(used_value == og_cluster for used_value in used_values) is False:
            used_values.append(og_cluster)
            all_og_cluster_indices = [i for i, x in enumerate(original_clusters) if x == og_cluster]

            relevant_comp_cluster = [comparison_clusters[i] for i in all_og_cluster_indices]
            relevant_percent_match = [percentage_matches[i] for i in all_og_cluster_indices]

            highest_percent_idx = int(np.argmax(relevant_percent_match))
            matching_cluster = relevant_comp_cluster[highest_percent_idx]
            cluster_to_best_match.update({og_cluster: matching_cluster})
    return dict(cluster_to_best_match)


def find_neighbours(clusters, max_clusters, length):
    cluster_to_neighbours = {}
    used_values = []
    for cluster in range(max_clusters + 1):
        if any(used_value == cluster for used_value in used_values) is False:
            used_values.append(cluster)
            neighbour_clusters = [cluster - length, cluster - 1, cluster + 1, cluster + length]
            neighbour_clusters = np.array(neighbour_clusters)
            neighbour_clusters = neighbour_clusters[(neighbour_clusters >= 0) & (neighbour_clusters <= max_clusters)]
            cluster_to_neighbours.update({cluster: neighbour_clusters})
    return cluster_to_neighbours


def find_neighbours_gsom(cluster_coords):
    cluster_to_neighbours = {}
    coord_to_cluster = {coord: cluster for cluster, coord in cluster_coords.items()}

    for cluster, (x, y) in cluster_coords.items():
        neighbour_coords = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        neighbour_clusters = [coord_to_cluster[c] for c in neighbour_coords if c in coord_to_cluster]
        cluster_to_neighbours[cluster] = np.array(neighbour_clusters)

    return cluster_to_neighbours


def find_neighbour_match(original_clusters, comparison_clusters, percentage_matches, cluster_to_best_match, cluster_to_neighbours):
    used_values = []
    percentages_list = []
    for idx, og_cluster in enumerate(original_clusters):
        if any(used_value == og_cluster for used_value in used_values) is False:
            used_values.append(og_cluster)
            matching_cluster = cluster_to_best_match.get(og_cluster)
            matching_neighbours = cluster_to_neighbours.get(matching_cluster)
            if matching_neighbours is None:
                matching_neighbours = []

            all_og_cluster_indices = [i for i, x in enumerate(original_clusters) if x == og_cluster]
            relevant_comp_cluster = [comparison_clusters[i] for i in all_og_cluster_indices]
            relevant_percent_match = [percentage_matches[i] for i in all_og_cluster_indices]

            common_clusters = {}
            for ind, cluster in enumerate(relevant_comp_cluster):
                if cluster in matching_neighbours:
                    common_clusters.update({cluster: ind})
            percentages = []
            percentages.append(np.max(relevant_percent_match))
            for cluster, ind in common_clusters.items():
                percentages.append(relevant_percent_match[ind])
            neighbour_percent = sum(percentages)
            percentages_list.append(neighbour_percent)
    return percentages_list


def main():
    input_files = _collect_input_files(ROOT_DIR)
    if not input_files:
        raise FileNotFoundError(f"No source_cluster_assignments.csv found under {ROOT_DIR}")

    all_direct_percentages = []
    all_neighbour_percentages = []
    direct_pair_means = []
    neighbour_pair_means = []
    pair_count = 0

    for original_path, comparison_path in permutations(input_files, 2):
        if original_path == comparison_path:
            continue
        pair_count += 1
        original_mapping, original_coords = _read_cluster_csv(original_path)
        comparison_mapping, comparison_coords = _read_cluster_csv(comparison_path)

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
            sorted_og_counts, sorted_comp_counts, original_mapping, comparison_mapping
        )

        best_direct_by_original = {}
        for og_cluster, percent in zip(original_clusters, percentage_matches):
            if og_cluster not in best_direct_by_original or percent > best_direct_by_original[og_cluster]:
                best_direct_by_original[og_cluster] = percent

        if len(best_direct_by_original) > 0:
            all_direct_percentages.extend(best_direct_by_original.values())
            direct_pair_mean = float(np.mean(list(best_direct_by_original.values())))
        else:
            direct_pair_mean = np.nan

        if np.isfinite(direct_pair_mean):
            direct_pair_means.append(direct_pair_mean)

        cluster_to_best_match = find_match(original_clusters, comparison_clusters, percentage_matches)

        if gsom_is_used:
            cluster_to_neighbours = find_neighbours_gsom(comparison_coords)
        else:
            max_clusters = np.max(original_clusters)
            number_of_clusters = max_clusters + 1
            length = int(np.sqrt(number_of_clusters))
            cluster_to_neighbours = find_neighbours(original_clusters, max_clusters, length)
        pair_percentages = find_neighbour_match(
            original_clusters, comparison_clusters, percentage_matches, cluster_to_best_match, cluster_to_neighbours
        )

        if pair_percentages is not None:
            all_neighbour_percentages.extend(pair_percentages)
            neighbour_pair_mean = float(np.mean(pair_percentages)) if len(pair_percentages) > 0 else np.nan
        else:
            neighbour_pair_mean = np.nan

        if np.isfinite(neighbour_pair_mean):
            neighbour_pair_means.append(neighbour_pair_mean)

        print(
            f"pair direct mean: {original_path} <-> {comparison_path} = "
            f"{direct_pair_mean:.2f}" if np.isfinite(direct_pair_mean) else
            f"pair direct mean: {original_path} <-> {comparison_path} = nan"
        )
        print(
            f"pair neighbour mean: {original_path} <-> {comparison_path} = "
            f"{neighbour_pair_mean:.2f}" if np.isfinite(neighbour_pair_mean) else
            f"pair neighbour mean: {original_path} <-> {comparison_path} = nan"
        )

    if len(direct_pair_means) > 0:
        print(f"mean direct overlap mean: {float(np.mean(direct_pair_means)):.2f}")
        print(f"mean direct overlap std: {float(np.std(direct_pair_means)):.2f}")
        print(f"mean direct overlap min: {float(np.min(direct_pair_means)):.2f}")
        print(f"mean direct overlap max: {float(np.max(direct_pair_means)):.2f}")

    if len(neighbour_pair_means) > 0:
        print(f"neighbour overlap mean: {float(np.mean(neighbour_pair_means)):.2f}")
        print(f"neighbour overlap std: {float(np.std(neighbour_pair_means)):.2f}")
        print(f"neighbour overlap min: {float(np.min(neighbour_pair_means)):.2f}")
        print(f"neighbour overlap max: {float(np.max(neighbour_pair_means)):.2f}")

    print(f"number of pairs: {pair_count}")


if __name__ == "__main__":
    main()