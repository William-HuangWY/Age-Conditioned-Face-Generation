import torch
import os
from PIL import Image

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# class VariationAutoEncoder(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super(VariationAutoEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, latent_dim * 2)  # Output mean and log variance
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, input_dim),
#             nn.Sigmoid()  # Assuming input is normalized between 0 and 1
#         )
#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#     def forward(self, x):
#         encoded = self.encoder(x)
#         mu, log_var = encoded.chunk(2, dim=-1)  # Split into mean and log variance
#         z = self.reparameterize(mu, log_var)
#         decoded = self.decoder(z)
#         return decoded, mu, log_var

class FaceAgeDataset(Dataset):
    """
    
    datasets/face_age/
        0/
        1/
        2/
        ...
        110/
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir, image_size=128, condition_dim=5, transform=None):
        self.root_dir = root_dir
        self.image_size = image_size
        self.condition_dim = condition_dim
        self.samples = []

        # 如果外面沒有傳 transform，就使用預設 resize 128 + ToTensor
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),  # 這裡就是 resize 128
                transforms.ToTensor(),                        # 轉成 [0,1]
            ])
        else:
            self.transform = transform

        # scan dataset
        for age_folder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, age_folder)

            if not os.path.isdir(folder_path):
                continue

            try:
                age = int(age_folder)
            except ValueError:
                continue

            for fname in os.listdir(folder_path):
                ext = os.path.splitext(fname.lower())[1]
                if ext not in self.IMG_EXTS:
                    continue

                img_path = os.path.join(folder_path, fname)
                self.samples.append((img_path, age))

    def __len__(self):
        return len(self.samples)

    def age_to_bucket(self, age):
        """
        split age into 8 buckets
        
        """
        if 12<= age <=18:
            return 0
        elif 19 <= age <= 25:
            return 1
        elif 26 <= age <= 31:
            return 2
        elif 32 <= age <= 39:
            return 3
        elif 40 <= age <= 53:
            return 4
        elif 54 <= age <= 67:
            return 5
        elif 68 <= age <= 80:
            return 6
        else:
            return 7

    def bucket_to_onehot(self, bucket_idx):
        condition = torch.zeros(self.condition_dim, dtype=torch.float32)
        condition[bucket_idx] = 1.0
        return condition

    def __getitem__(self, idx):
        img_path, age = self.samples[idx]

        # 讀圖片
        image = Image.open(img_path).convert("RGB")

        # resize 128 + tensor
        image = self.transform(image)

        # age -> bucket -> one-hot
        bucket_idx = self.age_to_bucket(age)
        condition = self.bucket_to_onehot(bucket_idx)

        return image, condition, age



class ConditionalVariationAutoEncoder(nn.Module):
    '''
    image x + condition c
        ↓
    Encoder CNN
        ↓
    μ(x, c), logσ²(x, c)
        ↓
    sample z
        ↓
    z + condition c
        ↓
    Decoder CNN
        ↓
    reconstruct image x̂
    '''
    def __init__(self, input_dim, latent_dim, condition_dim=5): # 5 buckets for 5 age groups
        super().__init__()
        self.img_size = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # (B, 32, img_size/2, img_size/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (B, 64, img_size/4, img_size/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (B, 128, img_size/8, img_size/8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # (B, 256, img_size/16, img_size/16)
            nn.ReLU(),
        )
        
        # compute flatten size after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_dim, input_dim)
            h = self.encoder_conv(dummy)
            self.feature_shape = h.shape[1:]  # (C, H, W)
            self.flatten_dim = h.view(1, -1).shape[1] # (1, 256*H*W)
        # C, H, W = self.feature_shape
            
        # Latent space
        self.fc_mu = nn.Linear(self.flatten_dim + self.condition_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim + self.condition_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim + self.condition_dim, self.flatten_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # (B, 128, img_size/8, img_size/8)
            nn.BatchNorm2d(128), # stabilize training
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (B, 64, img_size/4, img_size/4)
            nn.BatchNorm2d(64), # stabilize training
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # (B, 32, img_size/2, img_size/2)
            nn.BatchNorm2d(32), # stabilize training
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), # (B, 3, img_size, img_size)
            nn.Sigmoid() # normalized between 0 and 1
        )

    def reparameterize(self, mu, log_var):
        '''Reparameterization trick to sample from N(mu, var) from N(0,1).'''
        std = torch.exp(0.5 * log_var) # (B, latent_dim)
        eps = torch.randn_like(std) # (B, latent_dim)
        return mu + eps * std

    def encode(self, x, c: torch.Tensor):
        # q_phi(z|x, c)
        h = self.encoder_conv(x)
        h = torch.flatten(h, start_dim=1) # (B, flatten_dim)
        h = torch.cat([h, c], dim=1) # (B, flatten_dim + condition_dim)
        mu = self.fc_mu(h) # (B, latent_dim)
        log_var = self.fc_log_var(h) # (B, latent_dim)
        return mu, log_var

    def decode(self, z, c: torch.Tensor):
        # p_theta(x|z, c)
        z = torch.cat([z, c], dim=1) # (B, latent_dim + condition_dim)
        h = self.decoder_fc(z) # (B, flatten_dim)
        h = h.view(z.size(0), *self.feature_shape) # (B, C, H, W)
        return self.decoder_conv(h)

    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z, c)
        return x_hat, mu, log_var


if __name__ == "__main__":
    image_size = 128 # or 256x256
    latent_dim, condition_dim = 64, 5
    dataset = FaceAgeDataset(
        root_dir="datasets/face_age",
        image_size=image_size,
        condition_dim=condition_dim
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )
    cvae = ConditionalVariationAutoEncoder(image_size, latent_dim, condition_dim)
    
    dummy_image = torch.randn(2, 3, image_size, image_size) # (B, C, H, W)
    dummy_condition = torch.randn(2, condition_dim) # (B, condition_dim)
    reconstructed, mu, log_var = cvae(dummy_image, dummy_condition)
    images, conditions, ages = next(iter(dataloader))
    print("Input image shape:", images.shape)          # (B, 3, 128, 128)
    print("Condition shape:", conditions.shape)        # (B, 5)
    print("Ages:", ages)                               # (B,)
    print("Reconstructed shape:", reconstructed.shape) # (B, 3, image_size, image_size)
    print("Mu shape:", mu.shape) # (B, latent_dim)
    print("Log Var shape:", log_var.shape) # (B, latent_dim)
    
