import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from astropy.io import fits
from scipy import ndimage
from skimage.morphology import closing
from skimage.measure import label
from multiprocessing import Pool, cpu_count


# ============================================================
# Segmentation utilities
# ============================================================

def island_segmentation(data, threshold, minimum_island_size):
    mask = closing(data > threshold)
    labels_map, n_labels = label(mask, return_num=True, connectivity=2)

    islands = []
    for i in range(1, n_labels + 1):
        region = (labels_map == i)
        if np.sum(region) >= minimum_island_size:
            islands.append(region)
    return islands


def rescale_to_fixed_size(data, target_N):
    zoom = target_N / data.shape[0]
    return ndimage.zoom(data, zoom, order=1)


def process_single_image(data_raw, threshold, noise_multiplier,
                         min_island_start, min_island_end, min_island_step):

    data = np.nan_to_num(data_raw.copy(), nan=0.0).astype(np.float32)

    background = data[data < np.median(data)]
    if background.size == 0:
        background = data.flatten()
    noise_est = np.std(background)

    data[data <= noise_multiplier * noise_est] = 0.0
    max_val = np.max(data)
    if max_val == 0 or not np.isfinite(max_val):
        return None

    data = data / max_val

    min_size = min_island_start
    while min_size >= min_island_end:
        islands = island_segmentation(data, threshold, min_size)
        if len(islands) > 0:
            break
        min_size -= min_island_step

    if len(islands) == 0:
        return None

    mask = np.zeros_like(data, dtype=bool)
    for isl in islands:
        mask |= isl

    return data * mask


# ============================================================
# Autoencoder
# ============================================================

class ConvAE(nn.Module):
    def __init__(self, input_h, input_w, bottleneck_dim=None):
        super().__init__()

        if bottleneck_dim is not None and bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be > 0 when bottleneck is enabled")

        self.use_bottleneck = bottleneck_dim is not None
        self.input_h = input_h
        self.input_w = input_w

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 12, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 16, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.enc_h = input_h // 16
        self.enc_w = input_w // 16
        flat_dim = 16 * self.enc_h * self.enc_w

        if self.use_bottleneck:
            self.fc_enc = nn.Linear(flat_dim, bottleneck_dim)
            self.fc_dec = nn.Linear(bottleneck_dim, flat_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, 4, stride=2, padding=1),
        )

    def encode(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        if not self.use_bottleneck:
            return z
        return self.fc_enc(z)

    def decode(self, z):
        if self.use_bottleneck:
            z = self.fc_dec(z)
        z = z.view(z.size(0), 16, self.enc_h, self.enc_w)
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


# ============================================================
# Plotting
# ============================================================

def plot_all_reconstructions(original, recons_dict, outdir):
    epochs = sorted(recons_dict.keys())
    n = len(epochs) + 1

    vmin, vmax = original.min(), original.max()

    plt.figure(figsize=(4 * n, 4), constrained_layout=True)

    plt.subplot(1, n, 1)
    plt.imshow(original, cmap="plasma", origin="lower", vmin=vmin, vmax=vmax)
    plt.title("Original")
    plt.axis("off")

    for i, ep in enumerate(epochs):
        plt.subplot(1, n, i + 2)
        plt.imshow(recons_dict[ep], cmap="plasma", origin="lower",
                   vmin=vmin, vmax=vmax)
        plt.title(f"Epoch {ep}")
        plt.axis("off")


    plt.savefig(os.path.join(outdir, "all_recon_subplots.png"), dpi=150)
    plt.close()


# ============================================================
# Parallel preprocessing wrapper
# ============================================================

def preprocess_wrapper(args):
    fname, threshold, noise_multiplier, min_island_start, min_island_end, min_island_step, target_size = args
    try:
        with fits.open(fname) as hd:
            raw = hd[0].data

        masked = process_single_image(
            raw,
            threshold,
            noise_multiplier,
            min_island_start,
            min_island_end,
            min_island_step,
        )
        if masked is None:
            return None, None

        img_rescaled = rescale_to_fixed_size(masked, target_size)
        return fname, img_rescaled
    except Exception:
        return None, None


# ============================================================
# Main
# ============================================================

def main():
    target_name = "101377_cont_7MTM2TM1_jointdeconv.image.fits"
    data_dir = "/raid/scratch/mevans/newdata"
    outdir = "/raid/scratch/mevans/ae_example_outputs"
    os.makedirs(outdir, exist_ok=True)

    all_files = [os.path.join(data_dir, f)
                 for f in os.listdir(data_dir) if f.endswith(".fits")]

    torch.manual_seed(0)
    np.random.seed(0)

    threshold = 0.02
    noise_multiplier = 1.0
    min_island_start = 150
    min_island_end = 120
    min_island_step = 10
    target_size = 304

    args_list = [
        (
            f,
            threshold,
            noise_multiplier,
            min_island_start,
            min_island_end,
            min_island_step,
            target_size,
        )
        for f in all_files
    ]

    processed_images = []
    target_image_proc = None
    print("Starting parallel preprocessing on", len(all_files), "files using", cpu_count(), "cores.")

    with Pool(cpu_count()) as p:
        results = p.map(preprocess_wrapper, args_list)
    print("Parallel preprocessing complete.")
    print("Successfully processed", len(processed_images), "images.")

    for fname, img_rescaled in results:
        if fname is None or img_rescaled is None:
            continue
        processed_images.append(img_rescaled)
        if os.path.basename(fname) == target_name:
            target_image_proc = img_rescaled.copy()

    if len(processed_images) == 0:
        raise RuntimeError("No images were successfully processed.")
    if target_image_proc is None:
        raise RuntimeError("Target image was not found or could not be processed.")

    imgs_np = np.stack(processed_images).astype(np.float32)
    imgs_t = torch.from_numpy(imgs_np).unsqueeze(1)  # [N, 1, H, W]

    dataset = TensorDataset(imgs_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae_use_bottleneck = True
    ae_bottleneck_dim = 1256
    ae_learning_rate = 1e-3
    ae_epochs = 300

    model = ConvAE(
        target_size,
        target_size,
        bottleneck_dim=ae_bottleneck_dim if ae_use_bottleneck else None
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=ae_learning_rate)
    loss_fn = nn.MSELoss()

    plot_epochs = {25, 100, 200, 300}
    recons_to_plot = []
    recons_dict = {}
    qe = []

    target_t = torch.tensor(target_image_proc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    for epoch in range(1, ae_epochs + 1):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()

        qe.append(loss.item())
        print(f"Epoch {epoch}/{ae_epochs}  Loss={loss.item():.6f}")

        if epoch in plot_epochs:
            model.eval()
            with torch.no_grad():
                out = model(target_t)[0, 0].cpu().numpy()
            recons_dict[epoch] = out

    plot_all_reconstructions(target_image_proc, recons_dict, outdir)

    plt.figure()
    plt.plot(qe, marker=".")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ae_training_curve.png"), dpi=150)
    plt.close()

    print("Done. Outputs saved in:", outdir)


if __name__ == "__main__":
    main()
