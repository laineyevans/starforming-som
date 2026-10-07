import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

# -------------------------------
# USER SETTINGS
# -------------------------------

RUN_DIR = "/raid/scratch/mevans/run_15-12_17:55"                     # directory containing the txt files
RESULTS_FILE = "COMBINEDconvae_result.txt"
CORE_FILE = "COMBINEDcore_info.txt"

N_ROWS = 6
N_COLS = 6
CLUSTER_NO = N_ROWS * N_COLS

OUTFILE = "pie_chart.png"

# -------------------------------
# FILE PARSERS
# -------------------------------

def parse_convae_results(path):
    """
    Extract image names and cluster assignments from
    COMBINEDconvae_result.txt
    """
    names = []
    clusters = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # format:
            # name, class \t [clusters] \t ...
            parts = line.split("\t")
            name = parts[0].split(",")[0].strip()
            cl = list(map(int, re.findall(r"\d+", parts[1])))

            names.append(name)
            clusters.append(cl)

    return names, np.array(clusters, dtype=int)


def parse_core_info(path):
    """
    Reads COMBINEDcore_info.txt.
    Missing names imply zero cores.
    """
    core_map = {}

    if not os.path.exists(path):
        return core_map

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, count, *_ = [x.strip() for x in line.split(",")]
            core_map[name] = int(float(count))

    return core_map

# -------------------------------
# TALLY LOGIC (unchanged physics)
# -------------------------------

def make_cluster_tally(noc, clusters):
    noc_options = list(range(0, np.max(noc) + 1))
    tally_dict = {}

    for c in range(CLUSTER_NO):
        tally = np.zeros(len(noc_options))
        for i in range(len(noc)):
            for cl in clusters[i]:
                if cl == c:
                    tally[noc_options.index(noc[i])] += 1
        tally_dict[c] = tally

    return tally_dict, noc_options

# -------------------------------
# PIE CHART PLOTTING
# -------------------------------

def core_pie_charts(results, category_names):
    fig, axes = plt.subplots(
    N_ROWS, N_COLS,
    figsize=(7 * N_COLS, 7 * N_ROWS)   # was 5 * N
)

    axes = np.atleast_2d(axes)

    category_names = np.asarray(category_names, dtype=float)
    vmin, vmax = category_names.min(), category_names.max()

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps["plasma"]

    cluster_sizes = {k: np.sum(v) for k, v in results.items()}
    max_cluster_size = max(cluster_sizes.values()) if cluster_sizes else 1.0

    for k in range(CLUSTER_NO):
        i = k // N_COLS
        j = k % N_COLS
        ax = axes[N_ROWS - 1 - i, j]

        counts = results.get(k)
        if counts is None or np.sum(counts) == 0:
            ax.axis("off")
            continue

        radius = 1.25 * np.sqrt(np.sum(counts) / max_cluster_size)

        colors = cmap(norm(category_names))

        ax.pie(counts, colors=colors, radius=radius)
        ax.set_title(f"Cluster {k}", fontsize=50)


    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm, ax=axes, orientation="horizontal",
        fraction=0.05, pad=0.08
    )
    cbar.set_label("Number of cores", fontsize=60)
    cbar.ax.tick_params(labelsize=50)

    cbar.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.savefig(OUTFILE, dpi=150)
    plt.close(fig)
def print_max_core_images_per_neuron(names, clusters, noc):
    """
    For each neuron, print image ID(s) with the maximum number of cores.
    """
    for c in range(CLUSTER_NO):
        # indices of images assigned to this neuron
        idx = [i for i in range(len(names)) if c in clusters[i]]
        if not idx:
            continue

        core_counts = noc[idx]
        max_cores = core_counts.max()

        # images achieving the maximum
        max_idx = [idx[i] for i in range(len(idx)) if core_counts[i] == max_cores]

        for i in max_idx:
            print(f"Neuron {c:02d} | Image ID: {names[i]} | Cores: {noc[i]}")

# -------------------------------
# MAIN
# -------------------------------

def main():
    names, clusters = parse_convae_results(
        os.path.join(RUN_DIR, RESULTS_FILE)
    )
    core_map = parse_core_info(
        os.path.join(RUN_DIR, CORE_FILE)
    )

    noc = np.array([core_map.get(n, 0) for n in names], dtype=int)

    tally_dict, noc_options = make_cluster_tally(noc, clusters)
    core_pie_charts(tally_dict, noc_options)

    print_max_core_images_per_neuron(names, clusters, noc)


if __name__ == "__main__":
    main()
