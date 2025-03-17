import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import data, img_as_float
from skimage.transform import resize

# Load and preprocess an example image (grayscale)
image = img_as_float(data.camera())  # Example grayscale image
image = resize(image, (64, 64))  # Resize for better visualization

# Define diffusion parameters
num_steps = 1000  # Total diffusion steps
betas = np.linspace(0.0001, 0.02, num_steps)  # Noise schedule

# Generate noisy images (forward diffusion)
noisy_images = [image]
for beta in betas:
    noise = np.random.normal(0, 1, image.shape)
    new_image = np.sqrt(1 - beta) * noisy_images[-1] + np.sqrt(beta) * noise
    noisy_images.append(new_image)

# Reverse Diffusion: Denoising Process
reconstructed_images = [noisy_images[-1]]  # Start from the most noisy image
for beta in reversed(betas):
    noise = np.random.normal(0, 1, image.shape)
    new_image = (reconstructed_images[-1] - np.sqrt(beta) * noise) / np.sqrt(1 - beta)
    reconstructed_images.append(new_image)

# Create animation
fig, ax = plt.subplots(figsize=(5, 5))
img_display = ax.imshow(reconstructed_images[0], cmap="gray")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Reverse Diffusion Process (Denoising)")


def update(frame):
    img_display.set_data(reconstructed_images[frame])
    ax.set_title(f"Step {frame}/{num_steps}")
    return [img_display]


ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=100)

plt.show()
