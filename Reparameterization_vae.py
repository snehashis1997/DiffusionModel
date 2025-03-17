import torch
import torch.nn.functional as F


class VAE(torch.nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = torch.nn.Linear(784, 128)
        self.mu_layer = torch.nn.Linear(128, latent_dim)
        self.logvar_layer = torch.nn.Linear(128, latent_dim)
        self.decoder = torch.nn.Linear(latent_dim, 784)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Convert log-variance to standard deviation
        eps = torch.randn_like(std)  # Sample from N(0,1)
        return mu + std * eps  # Reparameterized sample

    def forward(self, x):
        x = F.relu(self.encoder(x))
        mu, logvar = self.mu_layer(x), self.logvar_layer(x)
        z = self.reparameterize(mu, logvar)  # Sampling with backpropagation support
        x_reconstructed = torch.sigmoid(self.decoder(z))
        return x_reconstructed, mu, logvar
