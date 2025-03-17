import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# --------------------------
# 1. ActNorm (Activation Normalization)
# --------------------------
class ActNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x, reverse=False):
        if reverse:
            return (x - self.bias) / self.scale
        else:
            return x * self.scale + self.bias


# --------------------------
# 2. Invertible 1×1 Convolution
# --------------------------
class Inv1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        Q = torch.linalg.qr(torch.randn(num_channels, num_channels))[
            0
        ]  # Random orthogonal matrix
        self.W = nn.Parameter(Q)

    def forward(self, x, reverse=False):
        B, C, H, W = x.shape
        W = self.W.view(C, C, 1, 1)
        if reverse:
            W_inv = torch.inverse(self.W).view(C, C, 1, 1)
            return F.conv2d(x, W_inv)
        else:
            return F.conv2d(x, W)


# --------------------------
# 3. Affine Coupling Layer
# --------------------------
class AffineCoupling(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels // 2, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        shift, scale = self.net(x1).chunk(2, dim=1)
        scale = torch.sigmoid(scale + 2)  # Ensure scale is positive
        if reverse:
            x2 = (x2 - shift) / scale
        else:
            x2 = x2 * scale + shift
        return torch.cat([x1, x2], dim=1)


# --------------------------
# 4. Glow Model (Stacking Multiple Blocks)
# --------------------------
class Glow(nn.Module):
    def __init__(self, num_channels, num_flows):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(ActNorm(num_channels))
            self.flows.append(Inv1x1Conv(num_channels))
            self.flows.append(AffineCoupling(num_channels))

    def forward(self, x, reverse=False):
        if reverse:  # Sampling (Inverse Flow)
            for flow in reversed(self.flows):
                x = flow(x, reverse=True)
        else:  # Training (Normal Flow)
            for flow in self.flows:
                x = flow(x)
        return x


# --------------------------
# 5. Example Usage
# --------------------------
if __name__ == "__main__":
    # Input image (batch size 1, 3 channels, 32x32 resolution)
    x = torch.randn(1, 3, 32, 32)

    # Create Glow model with 3 flows
    glow = Glow(num_channels=3, num_flows=3)

    # Forward pass (encode image to latent space)
    z = glow(x)

    # Reverse pass (decode latent space back to image)
    x_reconstructed = glow(z, reverse=True)

    # Show input and reconstructed images
    x_np = x.squeeze().permute(1, 2, 0).detach().numpy()
    x_recon_np = x_reconstructed.squeeze().permute(1, 2, 0).detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(np.clip(x_np, 0, 1))
    axes[0].set_title("Original Image")
    axes[1].imshow(np.clip(x_recon_np, 0, 1))
    axes[1].set_title("Reconstructed Image")
    plt.show()

    def sample_glow(glow, num_samples=4, img_size=(3, 32, 32)):
        """Generate new images from the Glow model."""
        z = torch.randn(num_samples, *img_size)  # Sample from Gaussian prior
        with torch.no_grad():
            samples = glow(z, reverse=True)  # Reverse flow to generate images
        return samples

    # Example usage:
    num_samples = 4
    glow_model = Glow(num_channels=3, num_flows=3)  # Assume Glow is pre-trained
    samples = sample_glow(glow_model, num_samples)

    # Visualize generated images
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i, img in enumerate(samples):
        img = img.permute(1, 2, 0).detach().numpy()
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].axis("off")
    plt.show()
