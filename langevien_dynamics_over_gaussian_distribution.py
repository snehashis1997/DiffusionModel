import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Langevin Dynamics Parameters
num_samples = 100  # Number of particles
num_steps = 50  # Number of iterations
step_size = 0.1  # Step size for updates

# Initialize particles from standard Gaussian
x = torch.randn(num_samples, 2)


# Function for Score (Negative Gradient of Log Density)
def score_function(x):
    return -x  # ∇ log p(x) = -x for a standard Gaussian


# Store trajectories for visualization
trajectories = np.zeros((num_steps, num_samples, 2))
trajectories[0] = x.numpy()

# Perform Langevin Sampling
for t in range(1, num_steps):
    noise = torch.randn_like(x)  # Brownian motion noise
    x = x + step_size * score_function(x) + (2 * step_size) ** 0.5 * noise
    trajectories[t] = x.numpy()

# Visualization: Animate Langevin Dynamics in 2D
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_title("Langevin Dynamics in 2D")
scat = ax.scatter([], [], c="blue", alpha=0.6)


# Animation Function
def update(frame):
    scat.set_offsets(trajectories[frame])  # Update particle positions
    return (scat,)


ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=100)
plt.show()
