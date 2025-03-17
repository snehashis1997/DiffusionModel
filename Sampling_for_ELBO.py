import torch
import torch.nn.functional as F


def log_prob_normal(x, mean, log_std):
    """Log probability of a normal distribution"""
    return (
        -0.5 * ((x - mean) / log_std.exp()) ** 2
        - log_std
        - 0.5 * torch.log(2 * torch.pi)
    )


# Example: Estimating ELBO using Importance Sampling
def estimate_elbo(
    x, prior_mean, prior_log_std, variational_mean, variational_log_std, num_samples=10
):
    z_samples = variational_mean + variational_log_std.exp() * torch.randn(
        (num_samples, *variational_mean.shape)
    )

    # Compute log probabilities
    log_p_x_given_z = log_prob_normal(
        x, z_samples, torch.tensor(1.0)
    )  # Likelihood term
    log_p_z = log_prob_normal(z_samples, prior_mean, prior_log_std)  # Prior
    log_q_z_given_x = log_prob_normal(
        z_samples, variational_mean, variational_log_std
    )  # Variational posterior

    # Importance weighting
    weights = log_p_x_given_z + log_p_z - log_q_z_given_x  # Log importance weights
    elbo = torch.mean(weights)  # Monte Carlo estimate of ELBO

    return elbo


# Example inputs
x = torch.tensor(2.0)  # Observed data
prior_mean = torch.tensor(0.0)
prior_log_std = torch.tensor(0.0)
variational_mean = torch.tensor(1.0)
variational_log_std = torch.tensor(-0.5)

# Compute ELBO estimate
elbo_estimate = estimate_elbo(
    x, prior_mean, prior_log_std, variational_mean, variational_log_std
)
print(f"Estimated ELBO: {elbo_estimate:.4f}")
