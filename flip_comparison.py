import csv
import os
from collections import defaultdict, Counter

csv_path = "/raid/scratch/mevans/run_12-05_16:24/source_cluster_assignments.csv"
run_dir = os.path.dirname(csv_path)
out_path = os.path.join(run_dir, "flip_grouping_summary.txt")

# Store clusters for each source
# {source_id: {flip_type: cluster}}
source_clusters = defaultdict(dict)

with open(csv_path, newline="") as f:
    reader = csv.reader(f)

    for row in reader:
        if not row or row[0] == "source_id":
            continue
        source_id = row[0].strip()
        flip_type = row[1].strip()
        cluster = row[2].strip()

        source_clusters[source_id][flip_type] = cluster

# Count how many copies (original + flips) are grouped together
match_counts = []
flip_total_counts = []

for source_id, flips in source_clusters.items():
    copy_clusters = [
        flips[flip_type]
        for flip_type in ["original", "hflip", "vflip", "hvflip"]
        if flip_type in flips
    ]

    if not copy_clusters:
        continue

    counts = Counter(copy_clusters)
    max_group = max(counts.values())
    grouped = max_group if max_group >= 2 else 0
    match_counts.append(grouped)
    flip_total_counts.append(len(copy_clusters))

# Count frequencies
summary = Counter(match_counts)

# Print and save results
lines = []
for n_matches in sorted(summary.keys(), reverse=True):
    lines.append(f"{summary[n_matches]} sources with {n_matches} copies grouped together")

for line in lines:
    print(line)

with open(out_path, "w") as f:
    for line in lines:
        f.write(line + "\n")