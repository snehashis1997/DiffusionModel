import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SimpleScoreNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x_t, t):
        t = t.view(-1, 1)  # Ensure t is a column vector
        return self.net(torch.cat([x_t, t], dim=1))


def forward_diffusion(x_0, t, beta_schedule):
    """
    Adds Gaussian noise to x_0 according to a predefined variance schedule.
    """
    beta_t = beta_schedule[t].view(-1, 1)  # Noise variance at step t
    alpha_t = torch.prod(1 - beta_schedule[: t + 1])  # Cumulative product

    noise = torch.randn_like(x_0)  # Sample noise
    x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise  # Diffusion step

    return x_t, noise  # Return noisy sample and actual noise


def loss_function(model, x_0, t, beta_schedule):
    """
    Computes the denoising score matching loss with importance weighting.
    """
    x_t, noise = forward_diffusion(x_0, t, beta_schedule)
    predicted_noise = model(x_t, t)

    # Importance weight (higher for early steps)
    weight = 1 / (beta_schedule[t] + 1e-5)

    # Loss with reweighting
    loss = (weight * (predicted_noise - noise) ** 2).mean()
    return loss


# Initialize model, optimizer, and noise schedule
input_dim = 2  # Example: 2D toy data
model = SimpleScoreNet(input_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define a linear beta schedule (variance)
T = 1000
beta_schedule = torch.linspace(0.0001, 0.02, T)  # Linear schedule

# Training loop
for epoch in range(100):  # Number of training epochs
    t = torch.randint(0, T, (1,))  # Random timestep
    x_0 = torch.randn(64, input_dim)  # Sample batch

    loss = loss_function(model, x_0, t, beta_schedule)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
