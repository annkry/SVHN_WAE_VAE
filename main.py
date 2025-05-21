import argparse
import torch
from models.vae import VAE
from models.wae_mmd import WAE_MMD
from train.train_vae import train_vae
from train.train_wae import train_wae
from utils.evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Run VAE or WAE-MMD on SVHN")
    parser.add_argument('--model', choices=['vae', 'wae'], required=True, help="Choose model to run")
    parser.add_argument('--train', action='store_true', help="Train the model")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate the model")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 32

    if args.model == 'vae':
        model = VAE(latent_dim=latent_dim).to(device)
        if args.train:
            train_vae(model, device)
        if args.evaluate:
            model.load_state_dict(torch.load('checkpoints/vae_svhn.pth', map_location=device, weights_only=True))
            evaluate_model(model, 'vae', 100, latent_dim, device)

    elif args.model == 'wae':
        model = WAE_MMD(latent_dim=latent_dim).to(device)
        if args.train:
            train_wae(model, device)
        if args.evaluate:
            model.load_state_dict(torch.load('checkpoints/wae_svhn.pth', map_location=device, weights_only=True))
            evaluate_model(model, 'wae', 128, latent_dim, device)

if __name__ == '__main__':
    main()