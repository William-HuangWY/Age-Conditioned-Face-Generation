import torch.optim as optim
import loss_function 
device = "cuda" if torch.cuda.is_available() else "cpu"

image_size = 128
latent_dim = 64
condition_dim = 5
vae_loss = loss_function.vae_loss
model = ConditionalVariationAutoEncoder(image_size, latent_dim, condition_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    
    for imgs, conds in dataloader:
        imgs = imgs.to(device)
        conds = conds.to(device)
        
        optimizer.zero_grad()
        x_hat, mu, log_var = model(imgs, conds)
        loss, recon_loss, kl_loss = vae_loss(x_hat, imgs, mu, log_var)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Loss: {total_loss/len(dataset):.4f} | "
          f"Recon: {total_recon/len(dataset):.4f} | "
          f"KL: {total_kl/len(dataset):.4f}")
    
    # 每幾個 epoch 儲存模型
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"cvae_epoch{epoch+1}.pth")