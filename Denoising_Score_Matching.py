import torch
import torch.nn as nn
import torch.optim as optim


# Define a simple neural network to estimate the score function
class ScoreNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ScoreNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),  # Output same dimension as input
        )

    def forward(self, x):
        return self.net(x)


# Generate synthetic data (e.g., 2D Gaussian mixture)
def sample_data(num_samples=1000):
    centers = torch.tensor([[-2, -2], [2, 2], [-2, 2], [2, -2]])
    data = centers[torch.randint(0, 4, (num_samples,))]
    noise = torch.randn_like(data) * 0.1
    return data + noise


# Define the DSM loss
def dsm_loss(score_net, x, sigma=0.1):
    noise = torch.randn_like(x) * sigma
    x_noisy = x + noise
    target_score = -noise / (sigma**2)  # True score function of Gaussian noise
    predicted_score = score_net(x_noisy)
    return torch.mean((predicted_score - target_score) ** 2)


# Training loop
device = "cuda" if torch.cuda.is_available() else "cpu"
score_net = ScoreNetwork(input_dim=2).to(device)
optimizer = optim.Adam(score_net.parameters(), lr=1e-3)

# Train for 1000 iterations
for step in range(1000):
    x = sample_data(128).to(device)
    loss = dsm_loss(score_net, x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, DSM Loss: {loss.item()}")
