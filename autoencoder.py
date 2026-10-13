import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from multiprocessing import cpu_count


# ============================================================
#  CONVOLUTIONAL AUTOENCODER
# ============================================================

class ConvAE(nn.Module):
    def __init__(self, input_h, input_w, bottleneck_dim=None, in_channels=1):
        super().__init__()

        self.use_bottleneck = bottleneck_dim is not None
        self.input_h = input_h
        self.input_w = input_w
        self.in_channels = in_channels
        self.latent_channels = 32
        self.latent_h = 8
        self.latent_w = 8

        # Encoder (stride=2 downsamples)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 12, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, self.latent_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.latent_h, self.latent_w)),
        )

        flat_dim = self.latent_channels * self.latent_h * self.latent_w

        if self.use_bottleneck:
            self.fc_enc = nn.Linear(flat_dim, bottleneck_dim)
            self.fc_dec = nn.Linear(bottleneck_dim, flat_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_channels, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 12, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, in_channels, 4, stride=2, padding=1),
        )


    def encode(self, x):
        z = self.encoder(x)
        if not self.use_bottleneck:
            return z.view(z.size(0), -1)
        return self.fc_enc(z.view(z.size(0), -1))

    def decode(self, z):
        if self.use_bottleneck:
            z = self.fc_dec(z)
        z = z.view(z.size(0), self.latent_channels, self.latent_h, self.latent_w)
        return self.decoder(z)

    def forward(self, x):
        recon = self.decode(self.encode(x))
        _, _, H, W = x.shape
        return F.interpolate(recon, size=(H, W), mode="bilinear", align_corners=False)

    


# ============================================================
#  TRAINING + ENCODING
# ============================================================

def train_autoencoder(dataset, config):
    """
    Train ConvAE and return:
        encoded_vectors : numpy array
        model           : trained autoencoder
        ae_re           : list of per-epoch reconstruction losses
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(
        dataset,
        batch_size=config.ae_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # infer image size and channels from one sample
    sample = dataset[0][0]  # Tensor with shape (C,H,W) or (H,W)
    if sample.dim() == 2:
        in_channels = 1
        H, W = sample.shape
    elif sample.dim() == 3:
        in_channels, H, W = sample.shape
    else:
        raise ValueError(f"Unexpected sample shape {sample.shape}; expected (H,W) or (C,H,W)")

    model = ConvAE(
        H,
        W,
        bottleneck_dim=config.ae_bottleneck_dim if config.ae_use_bottleneck else None,
        in_channels=in_channels,
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=config.ae_learning_rate)
    loss_fn = nn.MSELoss()

    # initialise QE list INSIDE the function
    ae_re = []

    # training loop
    for epoch in range(config.ae_epochs):
        model.train()
        epoch_loss = 0.0

        for (batch,) in loader:
            batch = batch.to(device)

            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        ae_re.append(epoch_loss)

        print(f"AE epoch {epoch+1}/{config.ae_epochs}, loss={epoch_loss:.6f}")

    # encode all samples
    model.eval()
    full_loader = DataLoader(dataset, batch_size=config.ae_batch_size, shuffle=False)

    encoded_list = []
    with torch.no_grad():
        for (batch,) in full_loader:
            batch = batch.to(device)
            if config.ae_use_bottleneck:
                z = model.encode(batch)
            else:
                conv = model.encoder(batch)
                z = conv.view(conv.size(0), -1)
            encoded_list.append(z.cpu().numpy())

    encoded_vectors = np.vstack(encoded_list)
    print(f"Encoded feature shape: {encoded_vectors.shape}")

    return encoded_vectors, model, ae_re
