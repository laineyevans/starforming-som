import csv
import os
import numpy as np

OUTPUT_ROOT = "/raid/scratch/mevans/random_cluster_baselines/random_compare_gsom_07-05_12:32"
TEMPLATE_CSV = "/raid/scratch/mevans/random_cluster_baselines/random_compare_05-05_12:32/run_01/source_cluster_assignments.csv"
N_RUNS = 100
BASE_GRID_SIZE = 5
EXTRA_NODES = 10
RANDOM_SEED = 42


def _load_template_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Template CSV has no header: {path}")
        rows = [row for row in reader]
        if not rows:
            raise ValueError(f"Template CSV has no data: {path}")
        return reader.fieldnames, rows


def _build_gsom_nodes(rng, base_size=5, extra_nodes=10):
    nodes = {(x, y) for x in range(base_size) for y in range(base_size)}
    for _ in range(extra_nodes):
        candidates = set()
        for (x, y) in nodes:
            candidates.update({(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)})
        candidates = sorted(candidates - nodes)
        if not candidates:
            break
        pick = candidates[int(rng.integers(0, len(candidates)))]
        nodes.add(pick)
    return sorted(nodes, key=lambda c: (c[0], c[1]))


def _ensure_fields(fieldnames):
    required = [
        "source_id",
        "flip_type",
        "cluster",
        "node_x",
        "node_y",
        "filename",
        "label",
        "silhouette_score",
    ]
    for key in required:
        if key not in fieldnames:
            fieldnames.append(key)
    return fieldnames


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    fieldnames, template_rows = _load_template_rows(TEMPLATE_CSV)
    fieldnames = _ensure_fields(list(fieldnames))

    for run_idx in range(1, N_RUNS + 1):
        run_name = f"run_{run_idx:02d}"
        run_dir = os.path.join(OUTPUT_ROOT, run_name)
        os.makedirs(run_dir, exist_ok=True)
        out_csv = os.path.join(run_dir, "source_cluster_assignments.csv")

        rng = np.random.default_rng(RANDOM_SEED + run_idx)
        nodes = _build_gsom_nodes(rng, base_size=BASE_GRID_SIZE, extra_nodes=EXTRA_NODES)
        if len(nodes) < BASE_GRID_SIZE * BASE_GRID_SIZE + EXTRA_NODES:
            raise RuntimeError("Failed to generate the requested number of GSOM nodes.")

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in template_rows:
                cluster_id = int(rng.integers(0, len(nodes)))
                node_x, node_y = nodes[cluster_id]

                out_row = dict(row)
                out_row["cluster"] = cluster_id
                out_row["node_x"] = node_x
                out_row["node_y"] = node_y
                if "silhouette_score" in fieldnames:
                    out_row["silhouette_score"] = "0.0"

                writer.writerow(out_row)


if __name__ == "__main__":
    main()
