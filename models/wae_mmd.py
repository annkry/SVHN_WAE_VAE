"""
    Defines the Wasserstein Autoencoder with Maximum Mean Discrepancy (WAE-MMD).
    Implements encoder, decoder, forward pass, and MMD loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class WAE_MMD(nn.Module):
    def __init__(self, latent_dim=32, kernel_bandwidth=2.0):
        """
            Initialize WAE-MMD model with encoder, decoder, and MMD parameters.

            Args:
                latent_dim (int): Dimensionality of latent space.
                kernel_bandwidth (float): Bandwidth for RBF kernel used in MMD.
        """
        super(WAE_MMD, self).__init__()
        self.latent_dim = latent_dim
        self.kernel_bandwidth = kernel_bandwidth

        # encoder: convolutional network mapping input to latent space
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

        # decoder: deconvolutional network mapping latent space to image
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
        """Encode input image to latent vector."""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent vector to reconstructed image."""
        z = self.decoder_input(z).view(-1, 256, 4, 4)
        return self.decoder(z)

    def forward(self, x):
        """Forward pass through encoder and decoder."""
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon

    def mmd_loss(self, z, prior_z):
        """
            Compute Maximum Mean Discrepancy (MMD) loss between encoded and prior samples.

            Args:
                z (Tensor): Encoded latent vectors.
                prior_z (Tensor): Samples from prior distribution.

            Returns:
                Tensor: MMD loss value.
        """
        def kernel(x, y):
            C = 2 * self.latent_dim * self.kernel_bandwidth
            return torch.exp(-torch.sum((x.unsqueeze(1) - y.unsqueeze(0))**2, dim=2) / C)

        xx_kernel = kernel(z, z)
        yy_kernel = kernel(prior_z, prior_z)
        xy_kernel = kernel(z, prior_z)
        return xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean()