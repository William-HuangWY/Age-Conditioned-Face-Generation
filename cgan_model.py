import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, condition_dim=5, img_size=128):
        super().__init__()

        self.init_size = img_size // 16
        self.l1 = nn.Linear(latent_dim + condition_dim, 256 * self.init_size * self.init_size)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        out = self.l1(x)
        out = out.view(out.size(0), 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, condition_dim=5, img_size=128):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3 + condition_dim, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Linear(256 * (img_size // 16) * (img_size // 16), 1)

    def forward(self, img, c):

        B, _, H, W = img.shape

        # broadcast condition
        c_map = c.view(B, -1, 1, 1).expand(B, -1, H, W)

        x = torch.cat([img, c_map], dim=1)

        out = self.conv(x)
        out = out.view(B, -1)

        validity = self.fc(out)

        return validity