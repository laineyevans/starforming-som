import os
import numpy as np
from astropy.io import fits

data_dir = "/raid/scratch/mevans/Almagal_fits_continuum_maps/7mtm2tm1"

shapes = []
rms_vals = []
flux_min = []
flux_max = []

for fname in os.listdir(data_dir):
    if not fname.endswith(".fits"):
        continue

    path = os.path.join(data_dir, fname)

    with fits.open(path, memmap=True) as hdul:
        data = np.nan_to_num(hdul[0].data)
        header = hdul[0].header

        shapes.append(data.shape)


# pixel dimensions
nx = [s[1] for s in shapes]
ny = [s[0] for s in shapes]

nx_min, nx_max = min(nx), max(nx)
ny_min, ny_max = min(ny), max(ny)



print("Pixel dimensions:")
print(f"nx range: {nx_min} – {nx_max}")
print(f"ny range: {ny_min} – {ny_max}\n")

