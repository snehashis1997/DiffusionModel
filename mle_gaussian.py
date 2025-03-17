import numpy as np
from scipy.stats import norm

# Generate synthetic data from a Normal distribution
np.random.seed(42)
data = np.random.normal(loc=5.0, scale=2.0, size=1000)  # True mean=5, std=2

# Maximum Likelihood Estimation (MLE)
mu_mle = np.mean(data)  # MLE estimate for mean
sigma_mle = np.std(data, ddof=0)  # MLE estimate for std (population estimate)

# Likelihood function for a Gaussian
likelihood = np.prod(norm.pdf(data, loc=mu_mle, scale=sigma_mle))

print(f"Estimated μ: {mu_mle:.2f}, Estimated σ: {sigma_mle:.2f}")
print(
    f"Likelihood: {likelihood:.4e}"
)  # Very small because of product of many probabilities
