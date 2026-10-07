import os
import re
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# USER SETTINGS
# -------------------------------

RUN_DIR = "/raid/scratch/mevans/5.5.26/run_05-05_15:40"
RESULTS_FILE = "source_cluster_assignments.csv"

OUTFILE = "silhouette_histogram.png"
BINS = 15

# -------------------------------
# PARSER
# -------------------------------

def parse_silhouette_scores(path):
    """
    Extract per-image silhouette scores from source_cluster_assignments.csv.
    """
    sil_values = []

    with open(path, "r", newline="") as f:
        header = f.readline().strip().split(",")
        if "silhouette_score" not in header:
            raise ValueError("CSV missing silhouette_score column")
        idx = header.index("silhouette_score")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) <= idx:
                continue
            try:
                sil_values.append(float(parts[idx]))
            except ValueError:
                continue

    return np.array(sil_values, dtype=float)

# -------------------------------
# PLOT
# -------------------------------

def plot_silhouette_histogram(sil):
    sil = sil[np.isfinite(sil)]

    mean = np.mean(sil)
    median = np.median(sil)

    bin_width = 0.033333333333
    bins = np.arange(
        np.floor(sil.min() / bin_width) * bin_width,
        np.ceil(sil.max() / bin_width) * bin_width + bin_width,
        bin_width
    )

    plt.figure(figsize=(9, 6))

    plt.hist(
        sil,
        bins=bins,
        density=True,
        alpha=0.85,
        color="rebeccapurple",
        edgecolor="black",
        linewidth=0.8
    )

    plt.axvline(mean, linestyle="--", linewidth=2, label=f"Mean = {mean:.3f}", color="orange")
    #plt.axvline(median, linestyle=":", linewidth=2, label=f"Median = {median:.3f}")

    plt.xlabel("Silhouette score", fontsize=18)
    plt.ylabel("Probability density", fontsize=18)
    plt.tick_params(axis="both", labelsize=14)
    plt.legend(fontsize=14, frameon=False)

    plt.tight_layout()
    out_path = os.path.join(RUN_DIR, OUTFILE)
    plt.savefig(out_path, dpi=200)
    plt.close()



# -------------------------------
# MAIN
# -------------------------------

def main():
    path = os.path.join(RUN_DIR, RESULTS_FILE)
    sil = parse_silhouette_scores(path)
    sil = sil[np.isfinite(sil)]

    sil_min = sil.min()
    sil_max = sil.max()

    print(f"Silhouette score range: [{sil_min:.3f}, {sil_max:.3f}]")

    print(f"Silhouette samples: {len(sil)}")
    print(f"Mean silhouette: {np.mean(sil):.3f}")
    print(f"Median silhouette: {np.median(sil):.3f}")

    plot_silhouette_histogram(sil)

if __name__ == "__main__":
    main()
