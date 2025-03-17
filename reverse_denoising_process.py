import numpy as np
import matplotlib.pyplot as plt
import torch

# Define parameters
T = 100  # Total diffusion steps
beta_t = np.linspace(0.0001, 0.02, T)  # Noise variance schedule
sqrt_one_minus_beta = np.sqrt(1 - beta_t)

# Generate a simple 2D dataset (e.g., a Gaussian blob)
x0 = np.random.randn(100, 2)  # Original data
noisy_x = [x0]

# Forward diffusion: Add noise over time
for t in range(T):
    noise = np.sqrt(beta_t[t]) * np.random.randn(*x0.shape)
    x_t = sqrt_one_minus_beta[t] * noisy_x[-1] + noise
    noisy_x.append(x_t)

# Reverse diffusion: Approximate denoising process
reverse_x = [noisy_x[-1]]  # Start from the most noisy data
for t in reversed(range(T)):
    noise = np.sqrt(beta_t[t]) * np.random.randn(*x0.shape)
    reverse_x.append((reverse_x[-1] - noise) / sqrt_one_minus_beta[t])

# Plot forward & reverse process
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(*zip(*noisy_x[::20]), alpha=0.5, label="Noisy Data")
ax[0].set_title("Forward Diffusion (Adding Noise)")

ax[1].scatter(*zip(*reverse_x[::-20]), alpha=0.5, label="Denoised Data")
ax[1].set_title("Reverse Diffusion (Removing Noise)")

plt.legend()
plt.show()
