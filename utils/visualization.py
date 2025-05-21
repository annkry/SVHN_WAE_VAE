import torch
from torchvision.utils import save_image, make_grid
import os

def save_reconstruction(data, recon, filename, nrow=10):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    assert len(data) == len(recon), "Mismatch in number of originals and reconstructions"
    
    rows = []
    for i in range(0, len(data), nrow):
        orig_chunk = data[i:i+nrow]
        recon_chunk = recon[i:i+nrow]
        
        orig_row = make_grid(orig_chunk, nrow=nrow, padding=0)
        recon_row = make_grid(recon_chunk, nrow=nrow, padding=0)
        
        rows.extend([orig_row, recon_row])

    final_grid = torch.cat(rows, dim=1)
    save_image(final_grid, filename, normalize=True, padding=0)

def save_interpolations(model, z_start, z_end, filename, steps=8):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    interpolations = []
    for i in range(len(z_start)):
        z = torch.stack([(1 - alpha) * z_start[i] + alpha * z_end[i] for alpha in torch.linspace(0, 1, steps=8)])
        imgs = model.decode(z).cpu()
        interpolations.append(imgs)
    interpolations = torch.cat(interpolations)
    save_image(interpolations, filename, nrow=steps, normalize=True, padding=0)

def save_random_samples(model, latent_dim, filename, num_samples=30):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    z = torch.randn(num_samples, latent_dim).to(next(model.parameters()).device)
    imgs = model.decode(z).cpu()
    save_image(imgs, filename, nrow=5, normalize=True, padding=0)