import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate synthetic data (true mean = 5, std = 2)
np.random.seed(42)
data = np.random.normal(loc=5, scale=2, size=1000)

# MLE estimates
mle_mean = np.mean(data)  # MLE estimate for mean
mle_std = np.std(data, ddof=0)  # MLE estimate for std (using N instead of N-1)

print(f"MLE Estimated Mean: {mle_mean:.3f}")
print(f"MLE Estimated Std: {mle_std:.3f}")

# Plot histogram and MLE estimated distribution
x = np.linspace(min(data), max(data), 1000)
pdf = norm.pdf(x, mle_mean, mle_std)

plt.hist(data, bins=30, density=True, alpha=0.6, color="b", label="Histogram of Data")
plt.plot(x, pdf, "r", label="MLE Gaussian Fit")
plt.xlabel("X values")
plt.ylabel("Density")
plt.title("MLE Gaussian Estimation")
plt.legend()
plt.show()
