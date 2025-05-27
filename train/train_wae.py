"""
    Training script for Wasserstein Autoencoder (WAE-MMD) on the SVHN dataset.
    Includes training loop and custom MMD-based regularization.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def train_wae(model, device, batch_size=128, epochs=100, lr=1e-3):
    """
        Train a Wasserstein Autoencoder with MMD loss on the SVHN dataset.

        Args:
            model (nn.Module): WAE model instance.
            device (torch.device): Device to run training on.
            batch_size (int): Training batch size.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
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

            # encode and reconstruct
            z, recon = model(data)

            # generate prior samples
            prior_z = torch.randn_like(z).to(device)

            # compute losses
            recon_loss = F.mse_loss(recon, data)
            mmd = model.mmd_loss(z, prior_z)
            loss = recon_loss + mmd

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # save the trained model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/wae_svhn.pth')