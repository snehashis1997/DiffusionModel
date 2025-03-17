import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


class FlowModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t):
        t = t.expand_as(x)
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


# Generate a simple 3D sphere dataset
def sample_sphere(n=1024):
    phi = np.random.uniform(0, np.pi, n)
    theta = np.random.uniform(0, 2 * np.pi, n)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=-1)


# Convert to PyTorch tensor
target_cloud = torch.tensor(sample_sphere(1024), dtype=torch.float32)

# Initialize Flow Model
flow_model = FlowModel().cuda()
optimizer = optim.Adam(flow_model.parameters(), lr=1e-3)

# Training loop
for epoch in range(2000):
    noise = torch.randn_like(target_cloud).cuda()
    t = torch.rand((target_cloud.shape[0], 1)).cuda()
    xt = (1 - t) * noise + t * target_cloud.cuda()

    pred_v = flow_model(xt, t)
    target_v = target_cloud.cuda() - noise  # True vector field

    loss = ((pred_v - target_v) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


def generate_samples(flow_model, steps=100):
    x = torch.randn((1024, 3)).cuda()  # Start from noise
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0], 1), i * dt).cuda()
        x = x + dt * flow_model(x, t)
    return x.cpu().detach().numpy()


generated_points = generate_samples(flow_model)

# Visualize in Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(generated_points)
o3d.visualization.draw_geometries([pcd])
