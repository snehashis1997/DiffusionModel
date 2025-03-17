import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Define Encoder (q(z|x))
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean of q(z|x)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance of q(z|x)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar  # Return mean and log variance


# Reparameterization Trick (Samples from q(z|x))
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)  # Random noise
    return mu + eps * std  # Sampled latent variable


# Define Decoder (p(x|z))
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))  # Output probability for reconstruction


# Define VAE (Combining Encoder and Decoder)
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)  # Sample latent representation
        recon_x = self.decoder(z)  # Reconstruct x
        return recon_x, mu, logvar


# ELBO Loss (Reconstruction Loss + KL Divergence)
def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")  # Data likelihood
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
    return recon_loss + kl_loss


# Training VAE
input_dim = 28 * 28  # Example: MNIST image size
hidden_dim = 400
latent_dim = 20
vae = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Dummy data (replace with real dataset)
x = torch.randn(64, input_dim)  # 64 samples, each of size 28x28
recon_x, mu, logvar = vae(x)

# Compute loss and backpropagate
loss = loss_function(recon_x, x, mu, logvar)
loss.backward()
optimizer.step()

print("VAE trained for one step!")
