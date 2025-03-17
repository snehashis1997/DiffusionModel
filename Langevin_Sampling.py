# import numpy as np
# import matplotlib.pyplot as plt
# import torch

# # Langevin Sampling Parameters
# eta = 0.1  # Step size
# steps = 100  # Number of iterations

# # Define a simple score function (gradient of log probability)
# def score_function(x):
#     return -x  # Assume data is centered around 0 (Gaussian-like)

# # Generate initial noisy image
# np.random.seed(42)
# x = np.random.randn(28, 28)  # Random noise (as an example)

# # Langevin Sampling Process
# images = [x]
# for t in range(steps):
#     noise = np.sqrt(2 * eta) * np.random.randn(*x.shape)
#     x = x + eta * score_function(x) + noise  # Langevin update
#     if t % 20 == 0:  # Save some intermediate steps for visualization
#         images.append(x)

# # Plot evolution of the image during Langevin Sampling
# fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
# for i, img in enumerate(images):
#     axes[i].imshow(img, cmap='gray')
#     axes[i].axis('off')
#     axes[i].set_title(f"Step {i * 20}")

# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Define a 2D energy function (Gaussian Mixture Model)
def potential_energy(x):
    mu1, mu2 = np.array([1, 1]), np.array([-1, -1])  # Two Gaussian centers
    sigma = np.eye(2) * 0.5  # Shared covariance
    return -(
        multivariate_normal.pdf(x, mean=mu1, cov=sigma)
        + multivariate_normal.pdf(x, mean=mu2, cov=sigma)
    )


# Compute the gradient (force) of the potential energy
def grad_potential(x):
    mu1, mu2 = np.array([1, 1]), np.array([-1, -1])
    sigma_inv = np.linalg.inv(np.eye(2) * 0.5)  # Inverse covariance
    grad1 = (x - mu1) * multivariate_normal.pdf(x, mean=mu1, cov=np.eye(2) * 0.5)
    grad2 = (x - mu2) * multivariate_normal.pdf(x, mean=mu2, cov=np.eye(2) * 0.5)
    return -(grad1 + grad2)


# Langevin dynamics parameters
eta = 0.01  # Noise level
dt = 0.1  # Step size
steps = 1000  # Number of steps
x = np.array([0.0, 0.0])  # Initial position
trajectory = [x]

# Simulate Langevin dynamics
for _ in range(steps):
    noise = np.sqrt(2 * eta * dt) * np.random.randn(2)
    x = x - grad_potential(x) * dt + noise
    trajectory.append(x)

trajectory = np.array(trajectory)

# Plot trajectory
plt.figure(figsize=(6, 6))
plt.scatter(trajectory[:, 0], trajectory[:, 1], s=1, alpha=0.5, label="Langevin Path")
plt.scatter([1, -1], [1, -1], color="red", marker="x", label="Gaussian Centers")
plt.title("Langevin Dynamics Simulation")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid()
plt.show()
