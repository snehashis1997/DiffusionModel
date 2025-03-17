import numpy as np


# Define true and approximate distributions
def p_xz(x, z):
    return np.exp(-0.5 * (x - z) ** 2)  # Gaussian likelihood


def q_zx(z, x):
    return np.exp(-0.5 * (z - x) ** 2)  # Approximate posterior


def p_z(z):
    return np.exp(-0.5 * z**2)  # Prior


# Importance sampling estimate of ELBO
def importance_sampling_elbo(x, num_samples=1000):
    z_samples = np.random.normal(loc=x, scale=1.0, size=num_samples)
    weights = p_xz(x, z_samples) * p_z(z_samples) / q_zx(z_samples, x)
    elbo = np.mean(np.log(weights))
    return elbo


x_sample = 1.0  # Example input
elbo_estimate = importance_sampling_elbo(x_sample)
print(f"Estimated ELBO: {elbo_estimate}")
