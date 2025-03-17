import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Define a simple affine coupling layer for RealNVP
class AffineCoupling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, input_dim)
        )
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, input_dim)
        )

    def forward(self, x, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)  # Split input
        scale = self.scale_net(x1)
        translate = self.translate_net(x1)

        if reverse:  # Inverse transformation
            x2 = (x2 - translate) * torch.exp(-scale)
        else:  # Forward transformation
            x2 = x2 * torch.exp(scale) + translate

        return torch.cat([x1, x2], dim=1)


# Define a simple RealNVP Model
class RealNVP(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [AffineCoupling(input_dim) for _ in range(num_layers)]
        )

    def forward(self, x):
        log_det = 0
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer(z, reverse=True)
        return z


# Generate some synthetic data (e.g., 2D spirals)
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=1000, noise=0.05)
X = torch.tensor(X, dtype=torch.float32)

# Train RealNVP model
model = RealNVP(input_dim=2, num_layers=4)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()
    transformed = model(X)
    loss = -torch.mean(
        torch.sum(-0.5 * transformed**2, dim=1)
    )  # Negative Log Likelihood
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

# Generate new samples
z = torch.randn(1000, 2)
generated_samples = model.inverse(z).detach().numpy()

# Plot results
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title("Original Data")
plt.subplot(1, 2, 2)
plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5, color="red")
plt.title("Generated Samples (RealNVP)")
plt.show()
