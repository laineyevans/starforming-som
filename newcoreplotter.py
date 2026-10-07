import os
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ---------------------------------
# USER SETTINGS
# ---------------------------------

DATA_DIR = "/raid/scratch/mevans/newdata"
RESULTS_FILE = "COMBINEDconvae_result.txt"
CORE_FILE = "COMBINEDcore_info.txt"

N_ROWS = 6
N_COLS = 6
N_NEURONS = N_ROWS * N_COLS

OUTFILE = "representative_neurons_max_cores.png"

# ---------------------------------
# PARSERS
# ---------------------------------

def parse_convae_results(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            name = parts[0].split(",")[0].strip()
            clusters = list(map(int, re.findall(r"\d+", parts[1])))
            records.append((name, clusters))
    return records


def parse_core_info(path):
    core_map = {}
    with open(path, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                core_map[parts[0]] = int(float(parts[1]))
    return core_map


def fits_path_from_id(image_id):
    return os.path.join(
        DATA_DIR,
        f"{image_id}_cont_7MTM2TM1_jointdeconv.image.fits"
    )

# ---------------------------------
# SELECTION LOGIC
# ---------------------------------

def highest_core_image_per_neuron(results, core_map):
    candidates = {n: [] for n in range(N_NEURONS)}

    for name, clusters in results:
        noc = core_map.get(name, 0)
        for c in clusters:
            candidates[c].append((name, noc))

    best = {}
    for n in range(N_NEURONS):
        # sort by core count, descending
        sorted_cands = sorted(candidates[n], key=lambda x: x[1], reverse=True)

        chosen = None
        for name, noc in sorted_cands:
            path = fits_path_from_id(name)
            if os.path.exists(path):
                chosen = (name, noc)
                break

        best[n] = chosen if chosen is not None else (None, None)

    return best


# ---------------------------------
# PLOTTING
# ---------------------------------

def plot_neuron_grid(best_images):
    fig, axes = plt.subplots(
        N_ROWS, N_COLS,
        figsize=(4 * N_COLS, 4 * N_ROWS)
    )
    axes = np.atleast_2d(axes)

    for n in range(N_NEURONS):
        i = n // N_COLS
        j = n % N_COLS
        ax = axes[N_ROWS - 1 - i, j]

        image_id, noc = best_images[n]
        if image_id is None:
            ax.axis("off")
            continue

        path = fits_path_from_id(image_id)
        if not os.path.exists(path):
            ax.axis("off")
            continue

        with fits.open(path, memmap=True) as hdul:
            data = hdul[0].data

        ax.imshow(data, origin="lower", cmap="inferno")
        ax.set_title(f"N{n} | cores={noc}", fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTFILE, dpi=200)
    plt.close(fig)

# ---------------------------------
# MAIN
# ---------------------------------

def main():
    results = parse_convae_results(RESULTS_FILE)
    core_map = parse_core_info(CORE_FILE)

    best = highest_core_image_per_neuron(results, core_map)
    plot_neuron_grid(best)


if __name__ == "__main__":
    main()
