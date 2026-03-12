import torch
from torch import nn
from torch.nn import functional as F

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
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), # (B, 64, img_size/2, img_size/2)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (B, 128, img_size/4, img_size/4)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # (B, 256, img_size/8, img_size/8)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # (B, 512, img_size/16, img_size/16)
            nn.BatchNorm2d(512),
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
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # (B, 512, img_size/8, img_size/8)
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1), # (B, 256, img_size/8, img_size/8)
            nn.BatchNorm2d(256), # stabilize training
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # (B, 256, img_size/4, img_size/4)
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), # (B, 128, img_size/4, img_size/4)
            nn.BatchNorm2d(128), # stabilize training
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # (B, 128, img_size/2, img_size/2)
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), # (B, 64, img_size/2, img_size/2)
            nn.BatchNorm2d(64), # stabilize training
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # (B, 64, img_size, img_size)
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), # (B, 32, img_size, img_size)
            nn.BatchNorm2d(32), # stabilize training
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1), # (B, 16, img_size, img_size)
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1), # (B, 3, img_size, img_size)
            nn.Sigmoid() # normalized between 0 and 1
        )

    def reparameterize(self, mu, log_var):
        '''Reparameterization trick to sample from N(mu, var) from N(0,1).'''
        log_var = torch.clamp(log_var, -10, 10) # prevent overflow
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
    cvae = ConditionalVariationAutoEncoder(image_size, latent_dim, condition_dim)
    
    dummy_image = torch.randn(2, 3, image_size, image_size) # (B, C, H, W)
    dummy_condition = torch.randn(2, condition_dim) # (B, condition_dim)
    reconstructed, mu, log_var = cvae(dummy_image, dummy_condition)
    print("Reconstructed shape:", reconstructed.shape) # (B, 3, image_size, image_size)
    print("Mu shape:", mu.shape) # (B, latent_dim)
    print("Log Var shape:", log_var.shape) # (B, latent_dim)
    
