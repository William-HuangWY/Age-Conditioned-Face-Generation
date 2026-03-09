import random, torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from cvae_model import ConditionalVariationAutoEncoder
from face_age_dataset import FaceAgeDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = FaceAgeDataset(target_size=256, padding=True, condition_dim=5) # dataset_256_padding
model = ConditionalVariationAutoEncoder(
    256,
    latent_dim=64,
    condition_dim=dataset.condition_dim
).to(device)

model.load_state_dict(torch.load("saved_models/cvae_epoch_10.pth", map_location=device))
model.eval()

indices = random.sample(range(len(dataset)), 8)
imgs, conds, _ = zip(*[dataset[i] for i in indices])

imgs = torch.stack(imgs).to(device)
conds = torch.stack(conds).to(device)

with torch.no_grad():
    x_hat, _, _ = model(imgs, conds)

imgs = imgs.cpu()
recons = x_hat.cpu()

grid = vutils.make_grid(torch.cat([imgs, recons], dim=0), nrow=8)

plt.figure(figsize=(12,4))
plt.title("Top: Original | Bottom: Reconstruction")
plt.imshow(grid.permute(1,2,0))
plt.axis("off")
plt.show()