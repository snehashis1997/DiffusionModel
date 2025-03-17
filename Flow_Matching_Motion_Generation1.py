import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import smplx
import matplotlib.pyplot as plt
import open3d as o3d

# Load SMPL model (using SMPL-X for better articulation)
smpl_model = smplx.create(
    model_path="path_to_smpl_model", model_type="smpl", gender="male"
)

# Randomly sample some real motion sequences (pose parameters)
pose_data = torch.randn(1000, 72)  # 1000 motion frames, 72D SMPL pose params
pose_data = pose_data.cuda()


class MotionFlowMatching(nn.Module):
    def __init__(self, input_dim=72, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t):
        t = t.expand_as(x)  # Expand time variable to match input shape
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


# Initialize the model and optimizer
motion_flow = MotionFlowMatching().cuda()
optimizer = optim.Adam(motion_flow.parameters(), lr=1e-3)

# Training loop
for epoch in range(2000):
    noise = torch.randn_like(pose_data).cuda()
    t = torch.rand((pose_data.shape[0], 1)).cuda()
    xt = (1 - t) * noise + t * pose_data  # Interpolating between noise and real motion

    pred_v = motion_flow(xt, t)
    target_v = pose_data - noise  # True motion trajectory

    loss = ((pred_v - target_v) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


def generate_motion(motion_flow, steps=100):
    x = torch.randn((100, 72)).cuda()  # Start from random noise
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0], 1), i * dt).cuda()
        x = x + dt * motion_flow(x, t)  # Integrate motion ODE
    return x.cpu().detach().numpy()


generated_motion = generate_motion(motion_flow)


def visualize_smpl_motion(pose_sequence, smpl_model):
    for i in range(len(pose_sequence)):
        output = smpl_model(
            body_pose=torch.tensor(pose_sequence[i, 3:], dtype=torch.float32).unsqueeze(
                0
            )
        )
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(output.vertices[0].detach().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(smpl_model.faces)
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.visualization.draw_geometries([mesh])


# Visualize the generated motion
visualize_smpl_motion(generated_motion, smpl_model)
