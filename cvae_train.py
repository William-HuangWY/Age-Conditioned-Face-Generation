import os, torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import matplotlib.pyplot as plt
from cvae_model import ConditionalVariationAutoEncoder
from face_age_dataset import FaceAgeDataset
from loss_function import ELBO_loss, identity_loss, age_loss

# pip install insightface
from insightface.model_zoo import get_model

# git clone https://github.com/siriusdemon/pytorch-DEX.git
# cd pytorch-DEX
# pip install .
import dex
from dex.models import Age


if __name__ == "__main__":
    os.makedirs("saved_models", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    # dataset = FaceAgeDataset(target_size=256, padding=True) # dataset_256_padding
    dataset = FaceAgeDataset(target_size=256, padding=False) # dataset_256_resize
    # dataset = FaceAgeDataset(target_size=128, padding=False) # dataset_128_resize
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    model = ConditionalVariationAutoEncoder(dataset.target_size, latent_dim=512, condition_dim=dataset.condition_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    weights = { 'beta': 0.1, 'identity': 1.0, 'age': 0.001 }
    
    # identity network
    identity_net = get_model('arcface_r100_v1')
    identity_net = identity_net.to(device)
    identity_net.eval()
    for p in identity_net.parameters(): p.requires_grad = False
    
    # age estimator
    age_net = Age().to(device)
    age_net.load_state_dict(torch.load(os.path.join(os.path.dirname(dex.__file__), "pth", "age_sd.pth"), map_location=device))
    age_net.eval()
    for p in age_net.parameters(): p.requires_grad = False
    
    
    num_epochs = 150 # 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        weights['beta'] = min(0.1, (epoch + 1) / 50 * 0.1) # KL weight warm-up
        
        for batch_idx, (imgs, conds, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            conds = conds.to(device)
            
            optimizer.zero_grad()
            x_hat, mu, log_var = model(imgs, conds)
            
            recon_loss, kl_loss = ELBO_loss(x_hat, imgs, mu, log_var)
            loss = recon_loss + weights['beta'] * kl_loss
            loss += weights['identity'] * (identity_loss(imgs, x_hat, identity_net) if identity_net is not None else 0)
            if epoch > 40: loss += weights['age'] * (age_loss(x_hat, conds, age_net) if age_net is not None else 0) # start age loss after some epochs to stabilize training
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping to prevent explosion
            optimizer.step()
            
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {total_loss / len(dataloader):.4f}")
        print("Recon:", recon_loss.item(), "KL:", kl_loss.item())
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"saved_models/cvae_epoch_{epoch+1}.pth")
    