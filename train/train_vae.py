"""
    Training script for Variational Autoencoder (VAE) on the SVHN dataset.
    Includes training loop and VAE loss computation.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def vae_loss(recon_x, x, mu, logvar):
    """
        Compute the VAE loss as the sum of reconstruction loss (MSE) and KL divergence.
        
        Args:
            recon_x (Tensor): Reconstructed input.
            x (Tensor): Original input.
            mu (Tensor): Mean of latent distribution.
            logvar (Tensor): Log variance of latent distribution.

        Returns:
            Tensor: Total VAE loss.
    """

    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

def train_vae(model, device, batch_size=100, epochs=100, lr=1e-3):
    """
        Train a Variational Autoencoder (VAE) on the SVHN dataset.

        Args:
            model (nn.Module): VAE model instance.
            device (torch.device): Device to run training on.
            batch_size (int): Training batch size.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
    """

    transform = transforms.ToTensor()
    train_set = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = vae_loss(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader.dataset):.4f}")

    # save the trained model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/vae_svhn.pth')