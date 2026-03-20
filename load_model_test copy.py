import random, torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from cvae_model import ConditionalVariationAutoEncoder
from face_age_dataset import FaceAgeDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
# dataset = FaceAgeDataset(target_size=256, padding=True) # dataset_256_padding
# dataset = FaceAgeDataset(target_size=256, padding=False) # dataset_256_resize
dataset = FaceAgeDataset(target_size=256, padding=False) # dataset_128_resize
model = ConditionalVariationAutoEncoder(dataset.target_size, latent_dim=512, condition_dim=dataset.condition_dim).to(device)

from PIL import Image

img = Image.open("/Users/yingying330/Downloads/Will_280pix.png").convert("RGB")
img = dataset._process_image(img)
img = img.unsqueeze(0).to(device)
# sample young images from the dataset
# samples, num_sample = [], 8
# indices = list(range(len(dataset)))
# random.shuffle(indices)
# for i in indices:
#     img, cond, age = dataset[i]
#     bucket = cond.argmax().item()
#     if bucket < 4: samples.append((img, bucket))
#     if len(samples) == num_sample: break

# imgs = torch.stack([s[0] for s in samples]).to(device)
# src_buckets = torch.tensor([s[1] for s in samples])
# print("Source buckets:", src_buckets.tolist())
# target_buckets = torch.full_like(src_buckets, 6) # torch.clamp(src_buckets + 3, max=dataset.condition_dim - 1)
# print("Target buckets:", target_buckets.tolist())

# conds = torch.zeros(num_sample, dataset.condition_dim)
# conds[torch.arange(num_sample), target_buckets] = 1
# conds = conds.to(device)
import os
import re
model_dir = "/Users/yingying330/Desktop/face_aging/model_save"

model_names = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
def get_epoch(model_name):
    m = re.search(r'epoch_(\d+)', model_name)
    return int(m.group(1)) if m else 0

model_names = sorted(model_names, key=get_epoch)

cond = torch.zeros(1, dataset.condition_dim)
cond[0,6] = 1
cond = cond.to(device)
results=[]
for model_name in model_names:
    model.load_state_dict(torch.load(f"{model_dir}/{model_name}", map_location=device))
    model.eval()
    with torch.no_grad():
    # x_hat, _, _ = model(imgs, conds)

        x_hat, _, _ = model(img, cond)
    results.append(x_hat.cpu())
imgs = img.cpu()
all_imgs = [imgs] + results
all_imgs = torch.cat(all_imgs, dim=0)
recons = x_hat.cpu()
grid = vutils.make_grid(all_imgs, nrow=len(all_imgs))
plt.figure(figsize=(12,4))
plt.title("Original + Different Model Outputs")
plt.imshow(grid.permute(1,2,0).clamp(0,1))
plt.axis("off")
all_names = ['Original']
for name in model_names:
    name_no_ext = name.replace(".pth","")
    epo = str(get_epoch(name))
    all_names.append("epoch:" + epo)
n = len(all_imgs)

plt.figure(figsize=(3*n,4))

for i in range(n):
    plt.subplot(1, n, i+1)
    img = all_imgs[i].permute(1,2,0).clamp(0,1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(all_names[i], fontsize=12)

plt.show()
