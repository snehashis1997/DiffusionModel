import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np


# --------------------------
# 1. Define Noise Scheduler
# --------------------------
class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t, noise):
        """Adds noise to an image at time step t."""
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t]).view(
            -1, 1, 1, 1
        )
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise


# --------------------------
# 2. Define Simple UNet Model for Denoising
# --------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x


# --------------------------
# 3. Training the DDPM Model
# --------------------------
scheduler = NoiseScheduler()
model = UNet().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10

for epoch in range(num_epochs):
    for i, (x, _) in enumerate(dataloader):
        x = x.cuda()
        t = torch.randint(0, scheduler.timesteps, (x.shape[0],)).cuda()
        noise = torch.randn_like(x).cuda()
        x_noisy = scheduler.add_noise(x, t, noise)

        predicted_noise = model(x_noisy)  # Predict noise
        loss = torch.mean((predicted_noise - noise) ** 2)  # MSE loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item()}"
            )

print("DDPM Training Completed! 🎉")


# --------------------------
# 4. Sampling New Images from the Model
# --------------------------
def sample_ddpm(model, scheduler, img_size=(3, 32, 32), num_steps=1000):
    """Generate images using DDPM reverse process."""
    x = torch.randn(1, *img_size).cuda()
    with torch.no_grad():
        for t in reversed(range(num_steps)):
            noise_pred = model(x)
            if t > 0:
                x = (x - noise_pred) / torch.sqrt(
                    scheduler.alphas[t]
                )  # Reverse process
            else:
                x = x - noise_pred
    return x


# Generate an image
generated_image = sample_ddpm(model, scheduler).cpu().squeeze().permute(1, 2, 0).numpy()
plt.imshow(np.clip(generated_image, 0, 1))
plt.title("Generated Image from DDPM")
plt.show()
