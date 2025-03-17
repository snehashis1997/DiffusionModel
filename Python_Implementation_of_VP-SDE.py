import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint, sdeint


# Define Variance-Preserving SDE
class VPSDE(nn.Module):
    def __init__(self, beta_min=0.1, beta_max=20.0):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        """Linear noise schedule."""
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def drift(self, t, x):
        """Drift coefficient dx/dt"""
        return -0.5 * self.beta(t) * x

    def diffusion(self, t, x):
        """Diffusion coefficient"""
        return torch.sqrt(self.beta(t))


# Define the Score Network (Approximating Gradient of Log Probability)
class ScoreNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),  # Predicting noise
        )

    def forward(self, x, t):
        return self.fc(x)


# Solve Reverse Process Using Stochastic and ODE Methods
def generate_image_vpsde(score_model, solver="euler", num_steps=50):
    sde = VPSDE()

    # Initial noise
    x_T = torch.randn(1, 3, 32, 32)  # 32x32 image

    # Time discretization
    t_eval = torch.linspace(1, 0, num_steps)

    # ODE solver (Probability Flow)
    class ReverseODE(nn.Module):
        def forward(self, t, x):
            score = score_model(x, t)
            return sde.drift(t, x) - sde.diffusion(t, x) ** 2 * score

    ode_func = ReverseODE()
    generated_images = odeint(ode_func, x_T, t_eval, method=solver)

    return generated_images


# Example Usage
score_model = ScoreNetwork(input_dim=3 * 32 * 32)  # Dummy model
generated_img = generate_image_vpsde(score_model, solver="rk4")

print("Generated Image Shape:", generated_img[-1].shape)


# Plot the sampling trajectories
def plot_trajectories(images):
    plt.figure(figsize=(12, 6))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i].squeeze().detach().cpu().numpy(), cmap="gray")
        plt.axis("off")
    plt.show()


# Generate images using different solvers
euler_samples = generate_image_vpsde(score_model, solver="euler")
rk4_samples = generate_image_vpsde(score_model, solver="rk4")

# Plot the results
plot_trajectories([euler_samples[-1], rk4_samples[-1]])
