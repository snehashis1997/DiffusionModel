import torch
import numpy as np
import matplotlib.pyplot as plt


# Define the SDE parameters
def f(x, t):  # Drift term (mean function)
    return -0.5 * x


def g(t):  # Diffusion term (noise)
    return torch.sqrt(torch.tensor(0.1))


# Euler-Maruyama solver for SDE
def euler_maruyama(x0, t0, T, dt):
    x = x0
    t = t0
    steps = int((T - t0) / dt)

    for _ in range(steps):
        dw = torch.randn_like(x) * np.sqrt(dt)  # Brownian motion
        x = x + f(x, t) * dt + g(t) * dw
        t += dt
    return x


# Example: Generate an image from noise
x0 = torch.randn(1, 3, 64, 64)  # Random noise (batch, channels, height, width)
generated_image = euler_maruyama(x0, 0, 1, 0.01)

# Convert to NumPy and normalize to [0,1]
image_np = generated_image.squeeze(0).detach().cpu().numpy()  # Remove batch dim
image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())  # Normalize

# Plot the generated image
plt.figure(figsize=(4, 4))
plt.imshow(np.transpose(image_np, (1, 2, 0)))  # Convert (C, H, W) → (H, W, C)
plt.axis("off")
plt.title("Generated Image using SDE")
plt.show()
