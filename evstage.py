import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
def text_color(stage):
    return "white" if stage == "Q" else "black"

# --------------------------------------------------
# USER INPUTS (FILL THESE)
# --------------------------------------------------

DATA_DIR = "/raid/scratch/mevans/run_15-12_17:55/"
TXT_FILE = "COMBINEDclassification_info.txt"


SOM_NX = 6
SOM_NY = 6

NEURON_TO_PLOT = 35

OUTPUT_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# PARSE TXT FILE
# --------------------------------------------------

clusters = {}

with open(os.path.join(DATA_DIR, TXT_FILE), "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()

    # cluster header
    m = re.match(r"cluster (\d+): Q:(\d+) P:(\d+) Y:(\d+) H:(\d+) na:(\d+)", line)
    if m:
        idx = int(m.group(1))
        obs = np.array(list(map(int, m.groups()[1:])))

        # expected line
        exp_line = lines[i + 1]
        exp = np.array(list(map(float, re.findall(r"[\d\.]+", exp_line))))

        clusters[idx] = {
            "observed": obs,
            "expected": exp,
        }

        i += 2
    else:
        i += 1

# --------------------------------------------------
# NEURON GRID: DOMINANT EVOLUTIONARY STAGE (NA SPLIT)
# --------------------------------------------------

stages = ["Q", "P", "Y", "H", "na"]

# plasma-like colours sampled from the plasma colormap
plasma = plt.get_cmap("plasma")
stage_colors = {
    "Q": plasma(0.15),
    "P": plasma(0.35),
    "Y": plasma(0.55),
    "H": plasma(0.75),
    "na": plasma(0.95),
}

plt.figure(figsize=(6, 6))
ax = plt.gca()

from matplotlib.patches import Polygon, Rectangle

for idx, d in clusters.items():
    y = idx // SOM_NX
    x = idx % SOM_NX

    obs = d["observed"].astype(float)
    fracs = obs / obs.sum()

    order = np.argsort(fracs)[::-1]
    primary = stages[order[0]]
    secondary = stages[order[1]]
    f1 = fracs[order[0]]
    f2 = fracs[order[1]]

    x0, x1 = x - 0.5, x + 0.5
    y0, y1 = y - 0.5, y + 0.5

    if primary == "na":
        # top triangle (na): (x0,y1)-(x1,y1)-(x1,y0)
        tri_top = Polygon([(x0, y1), (x1, y1), (x1, y0)], closed=True,
                          facecolor=stage_colors["na"], edgecolor="none")
        ax.add_patch(tri_top)

        # bottom triangle (secondary): (x0,y1)-(x0,y0)-(x1,y0)
        tri_bot = Polygon([(x0, y1), (x0, y0), (x1, y0)], closed=True,
                          facecolor=stage_colors[secondary], edgecolor="none")
        ax.add_patch(tri_bot)

        # text at triangle centroids
        ax.text(x + 1/6, y + 1/6, f"na\n{f1:.2f}",
                ha="center", va="center", fontsize=10, color="black", clip_on=True)
        ax.text(
            x - 1/6, y - 1/6,
            f"{secondary}\n{f2:.2f}",
            ha="center", va="center",
            fontsize=10, color=text_color(secondary), clip_on=True
        )


    else:
        # full square for dominant physical stage
        ax.add_patch(Rectangle((x0, y0), 1, 1, facecolor=stage_colors[primary], edgecolor="none"))
        ax.text(x, y, f"{primary}\n{f1:.2f}",
                ha="center", va="center", fontsize=11, color="black", clip_on=True)

# grid and axes
ax.set_xlim(-0.5, SOM_NX - 0.5)
ax.set_ylim(-0.5, SOM_NY - 0.5)
ax.set_xticks(range(SOM_NX))
ax.set_yticks(range(SOM_NY))
ax.set_xlabel("SOM x", fontsize=16)
ax.set_ylabel("SOM y", fontsize=16)
ax.set_aspect("equal")

# legend
handles = [
    plt.Line2D([0], [0], marker="s", linestyle="", color=stage_colors[s], label=s)
    for s in ["Q", "P", "Y", "H", "na"]
]
ax.legend(
    handles=handles,
    title="Stage",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=16,
    title_fontsize=18,
    handlelength=1.6,
    handleheight=1.6
)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "neuron_dominant_stage_plasma_split.png"), dpi=200)
plt.close()

# --------------------------------------------------
# PER-NEURON OBSERVED VS EXPECTED PLOT HAS BEEN REMOVED
# --------------------------------------------------