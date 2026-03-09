import os, torch
import torch.optim as optim
from torch.utils.data import DataLoader

from cvae_model import ConditionalVariationAutoEncoder
from face_age_dataset import FaceAgeDataset
from loss_function import ELBO_loss, identity_loss, age_loss


if __name__ == "__main__":
    os.makedirs("saved_models", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    dataset = FaceAgeDataset(target_size=256, padding=True, condition_dim=5) # dataset_256_padding
    # dataset = FaceAgeDataset(target_size=256, padding=False, condition_dim=5) # dataset_256_resize
    # dataset = FaceAgeDataset(target_size=128, padding=False, condition_dim=5) # dataset_128_resize
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    model = ConditionalVariationAutoEncoder(256, latent_dim=64, condition_dim=dataset.condition_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    weights = { 'beta': 1.0, 'identity': 1.0, 'age': 1.0 }
    
    identity_net = None
    age_net = None
    
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for imgs, conds, _ in dataloader:
            imgs = imgs.to(device)
            conds = conds.to(device)
            
            optimizer.zero_grad()
            x_hat, mu, log_var = model(imgs, conds)
            
            recon_loss, kl_loss = ELBO_loss(x_hat, imgs, mu, log_var)
            loss = recon_loss + weights['beta'] * kl_loss
            loss += weights['identity'] * (identity_loss(imgs, x_hat, identity_net) if identity_net is not None else 0)
            loss += weights['age'] * (age_loss(x_hat, conds.argmax(dim=1), age_net) if age_net is not None else 0)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {total_loss / len(dataloader):.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"saved_models/cvae_epoch_{epoch+1}.pth")
    