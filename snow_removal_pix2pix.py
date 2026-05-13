# Snow Removal Using Pix2Pix (Simplified Version)
# Run this in Google Colab

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 100
batch_size = 8
image_size = 256
train_dir = '/content/snow_pix2pix/train'
save_model_path = '/content/pix2pix_snow.pth'

# Dataset Class
class Pix2PixDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted([os.path.join(folder, f) for f in os.listdir(folder)])
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        w = img.width // 2
        input_img = img.crop((0, 0, w, img.height))
        target_img = img.crop((w, 0, img.width, img.height))
        return self.transform(input_img), self.transform(target_img)

# Generator
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

# Load Data
dataset = Pix2PixDataset(train_dir)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Models
G = UNetGenerator().to(device)
D = PatchDiscriminator().to(device)

criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()

opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Training Loop
for epoch in range(epochs):
    for input_img, target_img in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
        input_img = input_img.to(device)
        target_img = target_img.to(device)

        # Generate fake image
        fake_img = G(input_img)

        # Train Discriminator
        D_real = D(input_img, target_img)
        D_fake = D(input_img, fake_img.detach())

        real_label = torch.ones_like(D_real).to(device)
        fake_label = torch.zeros_like(D_fake).to(device)

        loss_D = criterion_GAN(D_real, real_label) + criterion_GAN(D_fake, fake_label)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator
        D_fake = D(input_img, fake_img)
        loss_G_GAN = criterion_GAN(D_fake, real_label)
        loss_G_L1 = criterion_L1(fake_img, target_img)
        loss_G = loss_G_GAN + 100 * loss_G_L1

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    if (epoch + 1) % 10 == 0:
        save_image((fake_img + 1) / 2.0, f'/content/sample_epoch_{epoch+1}.png')

# Save Model
torch.save(G.state_dict(), save_model_path)
print(f"Training complete. Model saved to {save_model_path}")
