"""
    Defines the Variational Autoencoder (VAE) model.
    Includes encoder, decoder, reparameterization trick, and forward pass.
"""

import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        """
            Initialize VAE with encoder, decoder, and latent variables.

            Args:
                latent_dim (int): Dimensionality of the latent space.
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # encoder: convolutional network to extract features from image
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )

        # latent vectors mu and log variance
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # decoder: reconstruct image from latent space
        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        """Encode input image into mean and log variance of latent space."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
            Sample from latent distribution using reparameterization trick.

            Args:
                mu (Tensor): Mean of latent distribution.
                logvar (Tensor): Log variance of latent distribution.

            Returns:
                Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent vector into reconstructed image."""
        x = self.decoder_input(z)
        x = x.view(-1, 128, 4, 4)
        return self.decoder(x)

    def forward(self, x):
        """
            Forward pass through encoder, reparameterization, and decoder.

            Args:
                x (Tensor): Input image batch.

            Returns:
                Tuple[Tensor, Tensor, Tensor]: Reconstructed image, mean, log variance.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar