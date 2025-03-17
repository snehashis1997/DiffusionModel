import torch


# Define a simple 2D Gaussian potential function
def potential_energy(x):
    return torch.sum(x**2, dim=1) / 2  # U(x) = ||x||^2 / 2


# Compute the score (negative gradient of potential)
def score_function(x):
    return -x  # ∇ log p(x) = -x for a standard Gaussian


# Langevin Dynamics Sampling
def langevin_sampling(num_samples=1000, step_size=0.1, num_steps=50):
    x = torch.randn(num_samples, 2)  # Initialize from Gaussian noise
    for _ in range(num_steps):
        noise = torch.randn_like(x)  # Brownian motion term
        x = x + step_size * score_function(x) + (2 * step_size) ** 0.5 * noise
    return x


# Generate samples
samples = langevin_sampling()
print(samples[:5])  # Print first few samples
