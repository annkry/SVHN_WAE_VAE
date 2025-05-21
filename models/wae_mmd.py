import torch
import torch.nn as nn
import torch.nn.functional as F

class WAE_MMD(nn.Module):
    def __init__(self, latent_dim=32, kernel_bandwidth=2.0):
        super(WAE_MMD, self).__init__()
        self.latent_dim = latent_dim
        self.kernel_bandwidth = kernel_bandwidth

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim)
        )

        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        z = self.decoder_input(z).view(-1, 256, 4, 4)
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon

    def mmd_loss(self, z, prior_z):
        def kernel(x, y):
            C = 2 * self.latent_dim * self.kernel_bandwidth
            return torch.exp(-torch.sum((x.unsqueeze(1) - y.unsqueeze(0))**2, dim=2) / C)

        xx_kernel = kernel(z, z)
        yy_kernel = kernel(prior_z, prior_z)
        xy_kernel = kernel(z, prior_z)
        return xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean()