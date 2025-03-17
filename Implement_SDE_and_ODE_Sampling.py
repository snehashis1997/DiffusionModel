import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchdiffeq import odeint, sdeint


# Define Variance-Preserving SDE
class VPSDE(nn.Module):
    def __init__(self, beta_min=0.1, beta_max=20.0):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        """Linear noise schedule"""
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def drift(self, t, x):
        """Drift coefficient for reverse process"""
        return -0.5 * self.beta(t) * x

    def diffusion(self, t, x):
        """Diffusion coefficient"""
        return torch.sqrt(self.beta(t))


# Define the Score Network (approximates ∇ log p(x))
class ScoreNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, input_dim)
        )

    def forward(self, x, t):
        return self.fc(x)


# Solve Reverse Process Using SDE and ODE
def sample_images(score_model, solver="euler", num_steps=50, method="SDE"):
    sde = VPSDE()

    # Initial noise
    x_T = torch.randn(1, 3, 32, 32)  # 32x32 image

    # Time discretization
    t_eval = torch.linspace(1, 0, num_steps)

    if method == "SDE":
        # Reverse SDE
        class ReverseSDE(nn.Module):
            def f(self, t, x):
                return sde.drift(t, x) - sde.diffusion(t, x) ** 2 * score_model(x, t)

            def g(self, t, x):
                return sde.diffusion(t, x)

        sde_func = ReverseSDE()
        sampled_images = sdeint(sde_func, x_T, t_eval, method=solver)

    else:
        # Probability Flow ODE
        class ReverseODE(nn.Module):
            def forward(self, t, x):
                return sde.drift(t, x) - sde.diffusion(t, x) ** 2 * score_model(x, t)

        ode_func = ReverseODE()
        sampled_images = odeint(ode_func, x_T, t_eval, method=solver)

    return sampled_images


# Instantiate Model
score_model = ScoreNetwork(input_dim=3 * 32 * 32)

# Sample Images using both methods
sde_images = sample_images(score_model, solver="euler", method="SDE")
ode_images = sample_images(score_model, solver="rk4", method="ODE")


# Visualization
def plot_comparison(sde_imgs, ode_imgs):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(sde_imgs[-1].squeeze().detach().cpu().numpy(), cmap="gray")
    plt.title("SDE Sampling (Diverse)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(ode_imgs[-1].squeeze().detach().cpu().numpy(), cmap="gray")
    plt.title("ODE Sampling (Fast, Less Diverse)")
    plt.axis("off")

    plt.show()


plot_comparison(sde_images, ode_images)
