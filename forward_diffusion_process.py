import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import data, img_as_float
from skimage.transform import resize

# Load and preprocess an example image (grayscale)
image = img_as_float(data.camera())  # Example grayscale image
image = resize(image, (64, 64))  # Resize for better visualization

# Define diffusion parameters
num_steps = 5000  # Total diffusion steps
betas = np.linspace(0.0001, 0.02, num_steps)  # Noise schedule

# Generate noisy images over time
noisy_images = [image]  # Start with the original image
for beta in betas:
    noise = np.random.normal(0, 1, image.shape)
    new_image = np.sqrt(1 - beta) * noisy_images[-1] + np.sqrt(beta) * noise
    noisy_images.append(new_image)

# Create animation
fig, ax = plt.subplots(figsize=(5, 5))
img_display = ax.imshow(noisy_images[0], cmap="gray")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Forward Diffusion Process")


def update(frame):
    img_display.set_data(noisy_images[frame])
    ax.set_title(f"Step {frame}/{num_steps}")
    return [img_display]


ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=100)

plt.show()
