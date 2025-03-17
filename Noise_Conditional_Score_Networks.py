import torch
import torch.nn as nn
import torch.optim as optim


# Define a Score Network conditioned on noise scale σ
class NCSN(nn.Module):
    def __init__(self, input_dim):
        super(NCSN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 128),  # Extra input for σ
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),  # Output same dimension as input
        )

    def forward(self, x, sigma):
        sigma = sigma.view(-1, 1).expand_as(x)  # Broadcast σ to match input size
        x_sigma = torch.cat([x, sigma], dim=1)  # Concatenate x and σ
        return self.net(x_sigma)


# Generate synthetic data (e.g., 2D Gaussian mixture)
def sample_data(num_samples=1000):
    centers = torch.tensor([[-2, -2], [2, 2], [-2, 2], [2, -2]])
    data = centers[torch.randint(0, 4, (num_samples,))]
    noise = torch.randn_like(data) * 0.1
    return data + noise


# Define the NCSN loss function
def ncsn_loss(score_net, x, sigma):
    noise = torch.randn_like(x) * sigma
    x_noisy = x + noise
    target_score = -noise / (sigma**2)  # True score function of Gaussian noise
    predicted_score = score_net(x_noisy, sigma)
    return torch.mean((predicted_score - target_score) ** 2)


# Training loop
device = "cuda" if torch.cuda.is_available() else "cpu"
score_net = NCSN(input_dim=2).to(device)
optimizer = optim.Adam(score_net.parameters(), lr=1e-3)

# Different noise levels
sigma_levels = torch.tensor([0.01, 0.1, 0.5, 1.0])

# Train for 1000 iterations
for step in range(1000):
    x = sample_data(128).to(device)
    sigma = sigma_levels[torch.randint(0, len(sigma_levels), (128,))].to(
        device
    )  # Randomly sample σ
    loss = ncsn_loss(score_net, x, sigma)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, NCSN Loss: {loss.item()}")
