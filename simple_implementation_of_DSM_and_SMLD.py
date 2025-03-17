import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Generate toy 2D Gaussian dataset
def sample_data(n_samples=1000):
    mean = torch.tensor([0.0, 0.0])
    cov = torch.tensor([[1.0, 0.8], [0.8, 1.0]])  # Correlated Gaussian
    return torch.distributions.MultivariateNormal(mean, cov).sample((n_samples,))


# Define Score Network (MLP)
class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


# DSM Training: Learn to estimate score
def train_dsm(score_net, data, noise_std=0.1, lr=1e-3, epochs=2000):
    optimizer = optim.Adam(score_net.parameters(), lr=lr)

    for epoch in range(epochs):
        noisy_data = data + noise_std * torch.randn_like(data)  # Add noise
        score_target = -(noisy_data - data) / noise_std**2  # True score
        loss = ((score_net(noisy_data) - score_target) ** 2).mean()  # DSM loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: DSM Loss = {loss.item():.4f}")


# SMLD: Langevin Dynamics Sampling
def langevin_dynamics(score_net, n_samples=500, steps=100, step_size=0.01):
    x = torch.randn(n_samples, 2)  # Start from Gaussian noise
    for _ in range(steps):
        noise = torch.randn_like(x)
        score = score_net(x).detach()
        x = x + step_size * score + torch.sqrt(2 * step_size) * noise
    return x


# Train DSM model
torch.manual_seed(0)
data = sample_data()
score_model = ScoreNet()
train_dsm(score_model, data)

# Sample from trained score function using Langevin Dynamics
generated_samples = langevin_dynamics(score_model).detach().numpy()

# Plot results
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.3, label="Real Data")
plt.scatter(
    generated_samples[:, 0],
    generated_samples[:, 1],
    alpha=0.3,
    label="Generated (SMLD)",
    color="red",
)
plt.legend()
plt.title("Denoising Score Matching (DSM) + Langevin Dynamics (SMLD)")
plt.show()
