import torch.optim as optim
import torch.nn as nn

from cgan_model import Generator, Discriminator
G = Generator()
D = Discriminator()

criterion = nn.BCEWithLogitsLoss()

g_optimizer = optim.Adam(G.parameters(), lr=2e-4)
d_optimizer = optim.Adam(D.parameters(), lr=2e-4)