import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm


# Define the SDE (Simple Ornstein-Uhlenbeck for illustration)
class DiffusionSDE:
    def __init__(self, beta_min=0.1, beta_max=20, T=1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def drift(self, x, t):
        return -0.5 * self.beta(t) * x

    def diffusion(self, t):
        return torch.sqrt(self.beta(t))


# Euler-Maruyama solver for sampling
def euler_maruyama_sampler(model, sde, num_steps=100, img_size=(1, 28, 28)):
    dt = sde.T / num_steps
    x = torch.randn((1, *img_size)).to(
        next(model.parameters()).device
    )  # Start from noise

    for i in tqdm(range(num_steps)):
        t = torch.tensor(1 - i / num_steps).float().to(x.device)
        drift = sde.drift(x, t)
        diffusion = sde.diffusion(t)
        noise = torch.randn_like(x)

        # Euler-Maruyama step
        x = x + drift * dt + diffusion * torch.sqrt(dt) * noise

    return x


# Predictor-Corrector (PC) Sampling
def predictor_corrector_sampler(
    model, sde, num_steps=100, img_size=(1, 28, 28), corrector_steps=1
):
    dt = sde.T / num_steps
    x = torch.randn((1, *img_size)).to(
        next(model.parameters()).device
    )  # Start from noise

    for i in tqdm(range(num_steps)):
        t = torch.tensor(1 - i / num_steps).float().to(x.device)

        # Predictor (Euler step)
        drift = sde.drift(x, t)
        diffusion = sde.diffusion(t)
        noise = torch.randn_like(x)
        x = x + drift * dt + diffusion * torch.sqrt(dt) * noise

        # Corrector (Langevin MCMC)
        for _ in range(corrector_steps):
            grad = model(x, t)  # Learned score function
            x = (
                x + 0.1 * grad + torch.randn_like(x) * 0.01
            )  # Small noise step for refinement

    return x


# Load your trained diffusion model
# model = MyTrainedDiffusionModel()
sde = DiffusionSDE()

# Generate samples
samples = euler_maruyama_sampler(model, sde).detach().cpu().numpy()

# Visualize
plt.imshow(samples.squeeze(), cmap="gray")
plt.title("Generated Image")
plt.axis("off")
plt.show()
