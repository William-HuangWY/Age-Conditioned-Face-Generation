import random, torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from cvae_model import ConditionalVariationAutoEncoder
from face_age_dataset import FaceAgeDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
# dataset = FaceAgeDataset(target_size=256, padding=True) # dataset_256_padding
# dataset = FaceAgeDataset(target_size=256, padding=False) # dataset_256_resize
dataset = FaceAgeDataset(target_size=128, padding=False) # dataset_128_resize
model = ConditionalVariationAutoEncoder(dataset.target_size, latent_dim=256, condition_dim=dataset.condition_dim).to(device)

model.load_state_dict(torch.load("saved_models/cvae_epoch_100.pth", map_location=device))
model.eval()

# sample young images from the dataset
samples, num_sample = [], 8
indices = list(range(len(dataset)))
random.shuffle(indices)
for i in indices:
    img, cond, age = dataset[i]
    bucket = cond.argmax().item()
    if bucket < 4: samples.append((img, bucket))
    if len(samples) == num_sample: break

imgs = torch.stack([s[0] for s in samples]).to(device)
src_buckets = torch.tensor([s[1] for s in samples])
print("Source buckets:", src_buckets.tolist())
target_buckets = torch.clamp(src_buckets + 3, max=dataset.condition_dim - 1)
print("Target buckets:", target_buckets.tolist())

conds = torch.zeros(num_sample, dataset.condition_dim)
conds[torch.arange(num_sample), target_buckets] = 1
conds = conds.to(device)

with torch.no_grad():
    x_hat, _, _ = model(imgs, conds)
    
imgs = imgs.cpu()
recons = x_hat.cpu()
grid = vutils.make_grid(torch.cat([imgs, recons], dim=0), nrow=num_sample)
plt.figure(figsize=(10,4))
plt.title("Top: Original | Bottom: Age +3 Bucket")
plt.imshow(grid.permute(1,2,0).clamp(0,1))
plt.axis("off")
plt.show()