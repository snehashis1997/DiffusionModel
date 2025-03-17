import torch
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
T = 1000  # Time steps
beta_min, beta_max = 0.1, 20  # Noise schedule (continuous)
timesteps = torch.linspace(0, 1, T)  # Normalized time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define beta(t) schedule (continuous)
def beta_t(t):
    return beta_min + (beta_max - beta_min) * t


# Forward diffusion process (Continuous SDE)
def forward_sde(x0, dt=1 / T):
    x_t = x0.clone()
    for t in range(T):
        noise = torch.randn_like(x_t) * torch.sqrt(dt)
        x_t = (
            x_t
            - 0.5 * beta_t(timesteps[t]) * x_t * dt
            + torch.sqrt(beta_t(timesteps[t])) * noise
        )
    return x_t


# Reverse process (Euler-Maruyama)
def reverse_sde(x_T, dt=1 / T):
    x_t = x_T.clone()
    for t in reversed(range(T)):
        noise = torch.randn_like(x_t) * torch.sqrt(dt)
        score = -x_t  # Approximate gradient of log-likelihood
        x_t = (
            x_t
            + 0.5 * beta_t(timesteps[t]) * score * dt
            + torch.sqrt(beta_t(timesteps[t])) * noise
        )
    return x_t


# Discrete Markov Chain (DDPM-like)
def ddpm_denoising(x_T):
    x_t = x_T.clone()
    for t in reversed(range(T)):
        noise = torch.randn_like(x_t)
        score = -x_t  # Approximate gradient
        x_t = (
            x_t
            + beta_t(timesteps[t]) * score
            + torch.sqrt(beta_t(timesteps[t])) * noise
        )
    return x_t


# Generate samples
x0 = torch.randn(100, 2).to(device)  # 2D Gaussian data
x_noisy_sde = forward_sde(x0).cpu().numpy()
x_noisy_ddpm = forward_sde(x0).cpu().numpy()
x_denoised_sde = reverse_sde(torch.tensor(x_noisy_sde)).cpu().numpy()
x_denoised_ddpm = ddpm_denoising(torch.tensor(x_noisy_ddpm)).cpu().numpy()

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].scatter(x0[:, 0].cpu(), x0[:, 1].cpu(), color="blue", label="Original Data")
axes[0].set_title("Original Data")

axes[1].scatter(
    x_noisy_sde[:, 0], x_noisy_sde[:, 1], color="red", label="Noisy Data (SDE)"
)
axes[1].set_title("Noisy Samples (SDE)")

axes[2].scatter(
    x_denoised_sde[:, 0], x_denoised_sde[:, 1], color="green", label="Denoised (SDE)"
)
axes[2].scatter(
    x_denoised_ddpm[:, 0],
    x_denoised_ddpm[:, 1],
    color="orange",
    label="Denoised (DDPM)",
)
axes[2].set_title("Denoised Samples (SDE vs. DDPM)")

plt.legend()
plt.show()
