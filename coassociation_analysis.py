"""Compute pairwise co-association stability across multiple clustering CSVs.

Edit the input directory list near the top of the file, then run the script.

It reports:
    - mean pairwise co-association percentage across all comparable source pairs
    - weighted co-association percentage across all pair/run comparisons
    - the fraction of pairs that are always together, never together, or mixed

    The input CSVs are expected to contain a `source_id` column and a `cluster`
    column. If `flip_type` is present, it is appended to the source key so
    augmented samples remain distinct. If `source_id` is missing, the script
    falls back to the filename stem.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence


# Edit these directories as needed. The script will scan each one recursively
# for .csv files and include every CSV it finds.
RUN_DIRS = [
    "/raid/scratch/mevans",
]

PAIRWISE_OUTPUT_CSV = None  # e.g. "/raid/scratch/mevans/pairwise_summary.csv"
MIN_RUNS_PER_PAIR = 2


@dataclass(frozen=True)
class PairwiseCoassociation:
    source_a: str
    source_b: str
    runs_compared: int
    same_cluster_count: int

    @property
    def coassociation_pct(self) -> float:
        if self.runs_compared == 0:
            return float("nan")
        return 100.0 * self.same_cluster_count / self.runs_compared


def _read_cluster_csv(path: str) -> Dict[str, int]:
    """Read a source-to-cluster mapping from one CSV file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    source_to_cluster: Dict[str, int] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        has_source_id = "source_id" in reader.fieldnames
        has_filename = "filename" in reader.fieldnames
        has_cluster = "cluster" in reader.fieldnames
        has_flip_type = "flip_type" in reader.fieldnames

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

            if has_flip_type:
                flip_type = row.get("flip_type", "").strip()
                if flip_type and flip_type.lower() != "original":
                    source_id = f"{source_id}__{flip_type}"

            source_to_cluster[source_id] = cluster

    if not source_to_cluster:
        raise ValueError(f"No valid rows found in {path}")

    return source_to_cluster


def _collect_csvs(input_dirs: Sequence[str]) -> List[str]:
    csv_files: List[str] = []
    for directory in input_dirs:
        if not directory:
            continue
        if not os.path.isdir(directory):
            continue
        for root, _, files in os.walk(directory):
            for name in files:
                if name.lower().endswith(".csv"):
                    csv_files.append(os.path.join(root, name))

    unique: List[str] = []
    seen = set()
    for path in sorted(csv_files):
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(norm)
    return unique


def compute_pairwise_stats(
    runs_source_to_cluster: Sequence[Dict[str, int]],
    min_runs_per_pair: int = 2,
) -> List[PairwiseCoassociation]:
    """Compute pairwise co-association across runs.

    For each pair of sources, only runs where both sources are present are used.
    """
    all_sources = sorted(set().union(*(run.keys() for run in runs_source_to_cluster)))
    pairwise_stats: List[PairwiseCoassociation] = []

    for src_a, src_b in combinations(all_sources, 2):
        runs_compared = 0
        same_cluster_count = 0
        for run in runs_source_to_cluster:
            if src_a not in run or src_b not in run:
                continue
            runs_compared += 1
            if run[src_a] == run[src_b]:
                same_cluster_count += 1

        if runs_compared >= min_runs_per_pair:
            pairwise_stats.append(PairwiseCoassociation(
                src_a,
                src_b,
                runs_compared,
                same_cluster_count,
            ))

    return pairwise_stats


def _format_pct(value: float) -> str:
    if value != value:  # NaN check
        return "nan"
    return f"{value:.2f}%"


def write_pairwise_csv(path: str, stats: Sequence[PairwiseCoassociation]) -> None:
    """Write per-pair stability details to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_a", "source_b", "comparisons", "same_cluster", "coassociation_pct"])
        for stat in stats:
            writer.writerow([
                stat.source_a,
                stat.source_b,
                stat.runs_compared,
                stat.same_cluster_count,
                f"{stat.coassociation_pct:.6f}",
            ])


def main() -> int:
    run_csv_files = _collect_csvs(RUN_DIRS)
    if len(run_csv_files) < 2:
        raise RuntimeError("Please add at least two input directories in RUN_DIRS, each containing CSV files.")

    runs_source_to_cluster = [_read_cluster_csv(path) for path in run_csv_files]
    pairwise_stats = compute_pairwise_stats(
        runs_source_to_cluster,
        min_runs_per_pair=MIN_RUNS_PER_PAIR,
    )

    if not pairwise_stats:
        raise RuntimeError("No comparable source pairs found. Try lowering MIN_RUNS_PER_PAIR.")

    total_pairs = len(pairwise_stats)
    mean_pair_agreement_pct = sum(stat.coassociation_pct for stat in pairwise_stats) / total_pairs
    total_same_cluster = sum(stat.same_cluster_count for stat in pairwise_stats)
    total_comparisons = sum(stat.runs_compared for stat in pairwise_stats)
    overall_agreement_pct = 100.0 * total_same_cluster / total_comparisons if total_comparisons else float("nan")

    pairs_always_together = sum(1 for stat in pairwise_stats if stat.same_cluster_count == stat.runs_compared)
    pairs_never_together = sum(1 for stat in pairwise_stats if stat.same_cluster_count == 0)
    pairs_mixed = total_pairs - pairs_always_together - pairs_never_together

    pairs_always_together_pct = 100.0 * pairs_always_together / total_pairs
    pairs_never_together_pct = 100.0 * pairs_never_together / total_pairs
    pairs_mixed_pct = 100.0 * pairs_mixed / total_pairs

    print("Co-association summary")
    print(f"  Runs compared: {len(run_csv_files)}")
    print(f"  Source pairs analyzed: {total_pairs}")
    print(f"  Average pair agreement: {_format_pct(mean_pair_agreement_pct)}")
    print(f"  Overall pair agreement: {_format_pct(overall_agreement_pct)}")
    print(f"  Pairs always together: {pairs_always_together} ({pairs_always_together_pct:.2f}%)")
    print(f"  Pairs never together: {pairs_never_together} ({pairs_never_together_pct:.2f}%)")
    print(f"  Mixed pairs: {pairs_mixed} ({pairs_mixed_pct:.2f}%)")

    if PAIRWISE_OUTPUT_CSV:
        write_pairwise_csv(PAIRWISE_OUTPUT_CSV, pairwise_stats)
        print(f"Wrote pairwise summary to {PAIRWISE_OUTPUT_CSV}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())