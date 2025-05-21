import torch
import torch.nn.functional as F
import numpy as np
from pytorch_fid import fid_score
import os
from torchvision import transforms
from utils.visualization import save_reconstruction, save_interpolations, save_random_samples
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def compute_sharpness_generated(model, latent_dim, num_images=1000, device="cuda"):
    model.eval().to(device)
    laplace_kernel = torch.tensor([[0, 1, 0], 
                                   [1, -4, 1], 
                                   [0, 1, 0]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sharpness_scores = []

    with torch.no_grad():
        for _ in range((num_images // 32) + 1):
            batch_size = min(32, num_images - len(sharpness_scores))
            z = torch.randn(batch_size, latent_dim).to(device)
            imgs = model.decode(z)
            if imgs.shape[1] == 3:
                imgs = imgs.mean(dim=1, keepdim=True)
            edges = F.conv2d(imgs, laplace_kernel, padding=1)
            var = torch.var(edges, dim=[1, 2, 3])
            sharpness_scores.extend(var.cpu().numpy())
            if len(sharpness_scores) >= num_images:
                break
    return np.mean(sharpness_scores[:num_images])

def save_images(tensor_list, path_list, transform=None):
    os.makedirs(os.path.dirname(path_list[0]), exist_ok=True)
    for tensor, path in zip(tensor_list, path_list):
        img = transform(tensor) if transform else transforms.ToPILImage()(tensor)
        img.save(path)

def compute_fid(real_dir, fake_dir, device="cuda"):
    return fid_score.calculate_fid_given_paths([real_dir, fake_dir], batch_size=32, device=device, dims=2048)

def evaluate_fid(model, dataloader, batch_size, latent_dim, model_name, device="cuda"):
    model.eval()
    real_dir = f"results/fid/{model_name}/real"
    fake_dir = f"results/fid/{model_name}/fake"
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Save real images
    real_images = []
    for batch, _ in dataloader:
        real_images.append(batch)
        if len(real_images) * batch.size(0) >= 5000:
            break
    real_images = torch.cat(real_images, dim=0)[:5000].cpu()
    to_pil = transforms.ToPILImage()
    resize = transforms.Resize((299, 299))
    for i, img in enumerate(real_images):
        to_pil(resize(img)).save(os.path.join(real_dir, f"real_{i}.png"))

    # Save generated images
    generated = []
    with torch.no_grad():
        while len(generated) * batch_size < 5000:
            z = torch.randn(batch_size, latent_dim).to(device)
            out = model.decode(z).cpu()
            generated.append(out)
    generated = torch.cat(generated, dim=0)[:5000]
    for i, img in enumerate(generated):
        to_pil(resize(img)).save(os.path.join(fake_dir, f"fake_{i}.png"))

    return compute_fid(real_dir, fake_dir, device)

def load_data(normalize=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if normalize else lambda x: x
    ])
    test_set = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    return DataLoader(test_set, batch_size=100, shuffle=False)

def evaluate_model(model, model_type, batch_size, latent_dim, device):
    model.eval()
    dataloader = load_data(normalize=(model_type == "wae"))
    data, _ = next(iter(dataloader))
    data = data.to(device)

    # Reconstructions
    if model_type == "vae":
        recon, _, _ = model(data)
    else:
        _, recon = model(data)
    save_reconstruction(data[:30].cpu(), recon[:30].cpu(), f"results/reconstructions/{model_type}.png")

    # Interpolations
    if model_type == "vae":
        z_start, _ = model.encode(data[:6])
        z_end, _ = model.encode(data[6:12])
    else:
        z_start = model.encode(data[:6])
        z_end = model.encode(data[6:12])
    save_interpolations(model, z_start, z_end, f"results/interpolations/{model_type}.png")

    # Generation of random samples
    save_random_samples(model, latent_dim, f"results/samples/{model_type}.png")

    # Sharpness
    sharpness = compute_sharpness_generated(model, latent_dim, num_images=1000, device=device)

    # FID
    fid = evaluate_fid(model, dataloader, batch_size, latent_dim, model_type, device)
    
    print(f"Sharpness ({model_type}): {sharpness:.4f}")
    print(f"FID score ({model_type}): {fid:.4f}")