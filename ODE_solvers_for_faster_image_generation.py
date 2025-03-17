import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


# Define the Score-Based Model (U-Net or any pretrained model)
class SimpleScoreNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),  # Predicting noise
        )

    def forward(self, x, t):
        return self.fc(x)


# Define the Probability Flow ODE function
class ProbabilityFlowODE(nn.Module):
    def __init__(self, score_model):
        super().__init__()
        self.score_model = score_model

    def forward(self, t, x):
        """dx/dt = -0.5 * score(x, t)"""
        score = self.score_model(x, t)
        return -0.5 * score


# Generate an image using Euler and RK4 solvers
def generate_image_ode(score_model, solver="rk4", num_steps=20):
    # Define the ODE function
    ode_func = ProbabilityFlowODE(score_model)

    # Initial Gaussian noise
    x_T = torch.randn(1, 3, 32, 32)  # Assuming 32x32 image

    # Time discretization
    t_eval = torch.linspace(1, 0, num_steps)  # From T=1 to T=0

    # Solve ODE (Euler or RK4)
    if solver == "euler":
        generated_images = odeint(ode_func, x_T, t_eval, method="euler")
    elif solver == "rk4":
        generated_images = odeint(ode_func, x_T, t_eval, method="rk4")

    return generated_images[-1]  # Return final generated image


# Example Usage
score_model = SimpleScoreNetwork(input_dim=3 * 32 * 32)  # Dummy model
generated_img = generate_image_ode(score_model, solver="rk4")
print("Generated Image Shape:", generated_img.shape)
