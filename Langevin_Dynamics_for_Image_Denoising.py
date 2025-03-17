import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Load MNIST dataset (noisy version)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x))]
)
dataset = MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# Simple CNN for Score-Based Noise Estimation
class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                64, 1, kernel_size=3, padding=1
            ),  # Output 1 channel (score estimate)
        )

    def forward(self, x):
        return self.conv(x)


# Train DSM Model
def train_dsm(score_net, dataloader, noise_std=0.1, lr=1e-3, epochs=5):
    optimizer = optim.Adam(score_net.parameters(), lr=lr)

    for epoch in range(epochs):
        for x, _ in dataloader:
            x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            noise = noise_std * torch.randn_like(x)
            noisy_x = x + noise
            score_target = -noise / noise_std**2  # True score function

            loss = ((score_net(noisy_x) - score_target) ** 2).mean()  # DSM loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: DSM Loss = {loss.item():.4f}")


score_model = ScoreNet().cuda()
train_dsm(score_model, dataloader)


# Langevin Dynamics for Denoising
def langevin_denoising(score_net, noisy_img, steps=50, step_size=0.02):
    x = (
        noisy_img.clone()
        .detach()
        .to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    )

    for _ in range(steps):
        noise = torch.randn_like(x)
        score = score_net(x).detach()
        x = x + step_size * score + torch.sqrt(2 * step_size) * noise  # Langevin update

    return x.cpu().detach()


# Test with a noisy image
noisy_image, _ = dataset[0]
noisy_image = noisy_image.unsqueeze(0).cuda()
denoised_image = langevin_denoising(score_model, noisy_image)

# Plot results
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(noisy_image.cpu().squeeze(), cmap="gray")
plt.title("Noisy Image")

plt.subplot(1, 2, 2)
plt.imshow(denoised_image.cpu().squeeze(), cmap="gray")
plt.title("Denoised Image (Langevin)")
plt.show()
