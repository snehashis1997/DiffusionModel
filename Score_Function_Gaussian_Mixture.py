import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Define Gaussian mixture centers
centers = torch.tensor([[-2, -2], [2, 2], [-2, 2], [2, -2]])


# Compute score function (gradient of log-density)
def score_function(x):
    score = torch.zeros_like(x)
    for center in centers:
        score += (
            -(x - center)
            / 0.5
            * torch.exp(-torch.sum((x - center) ** 2, dim=1, keepdim=True) / 1.0)
        )
    return score


# Generate grid points
x = torch.linspace(-3, 3, 30)
y = torch.linspace(-3, 3, 30)
X, Y = torch.meshgrid(x, y, indexing="ij")
points = torch.stack([X.flatten(), Y.flatten()], dim=1)

# Compute scores
scores = score_function(points)
U, V = scores[:, 0].reshape(X.shape), scores[:, 1].reshape(Y.shape)

# Plot
plt.figure(figsize=(6, 6))
sns.kdeplot(x=points[:, 0], y=points[:, 1], fill=True, cmap="Blues", alpha=0.5)
plt.quiver(X, Y, U, V, color="red", scale=30)
plt.title("Score Function of Gaussian Mixture")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
