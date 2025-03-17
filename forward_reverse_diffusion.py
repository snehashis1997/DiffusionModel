import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load a sample image
from torchvision import transforms
from PIL import Image

# Load and preprocess an image
image_path = "sample.jpg"  # Replace with your image
img = Image.open(image_path).convert("L")  # Convert to grayscale
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
x0 = transform(img).unsqueeze(0)  # Shape: (1, 1, 64, 64)


# Define diffusion process
def forward_process(x0, timesteps=50, beta=0.02):
    noise_levels = torch.linspace(0, 1, timesteps)
    noisy_images = [
        (1 - noise_levels[i]) * x0 + noise_levels[i] * torch.randn_like(x0) * beta
        for i in range(timesteps)
    ]
    return noisy_images


# Define reverse denoising process (simple averaging)
def reverse_process(noisy_images, timesteps=50):
    denoised_images = []
    x_t = noisy_images[-1]
    for i in range(timesteps - 1, -1, -1):
        x_t = x_t * 0.9 + noisy_images[i] * 0.1  # Simple smoothing step
        denoised_images.append(x_t)
    return denoised_images[::-1]  # Reverse to match time order


# Generate images
timesteps = 50
noisy_images = forward_process(x0, timesteps=timesteps)
denoised_images = reverse_process(noisy_images, timesteps=timesteps)

# Plot animation
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].set_title("Noising")
ax[1].set_title("Denoising")

img1 = ax[0].imshow(noisy_images[0].squeeze().numpy(), cmap="gray")
img2 = ax[1].imshow(denoised_images[0].squeeze().numpy(), cmap="gray")


def update(frame):
    img1.set_array(noisy_images[frame].squeeze().numpy())
    img2.set_array(denoised_images[frame].squeeze().numpy())
    return img1, img2


ani = animation.FuncAnimation(fig, update, frames=timesteps, interval=100)
plt.show()
