import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Constants
g = 9.81  # gravity
L1, L2 = 1.0, 1.0  # lengths of pendulums
m1, m2 = 1.0, 1.0  # masses


# Equations of motion
def double_pendulum(t, y):
    θ1, z1, θ2, z2 = y
    Δθ = θ2 - θ1

    denominator1 = (m1 + m2) * L1 - m2 * L1 * np.cos(Δθ) ** 2
    denominator2 = (L2 / L1) * denominator1

    dθ1_dt = z1
    dz1_dt = (
        m2 * g * np.sin(θ2) * np.cos(Δθ)
        - m2 * np.sin(Δθ) * (L1 * z1**2 * np.cos(Δθ) + L2 * z2**2)
        - (m1 + m2) * g * np.sin(θ1)
    ) / denominator1
    dθ2_dt = z2
    dz2_dt = (
        (m1 + m2)
        * (L1 * z1**2 * np.sin(Δθ) - g * np.sin(θ2) + g * np.sin(θ1) * np.cos(Δθ))
        + m2 * L2 * z2**2 * np.sin(Δθ) * np.cos(Δθ)
    ) / denominator2

    return [dθ1_dt, dz1_dt, dθ2_dt, dz2_dt]


# Initial conditions (angle in radians, angular velocity)
y0 = [np.pi / 2, 0, np.pi / 2, 0]  # Start with both pendulums at 90 degrees
t_span = [0, 10]  # Simulation time
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Time points for evaluation

# Solve the differential equations
sol = solve_ivp(double_pendulum, t_span, y0, t_eval=t_eval, method="RK45")

# Extract angles
θ1, θ2 = sol.y[0], sol.y[2]

# Convert angles to x, y positions
x1 = L1 * np.sin(θ1)
y1 = -L1 * np.cos(θ1)
x2 = x1 + L2 * np.sin(θ2)
y2 = y1 - L2 * np.cos(θ2)

# Animation setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_title("Chaotic Double Pendulum")

(line,) = ax.plot([], [], "o-", lw=2)
(trace,) = ax.plot([], [], "r-", lw=1, alpha=0.5)
trail_x, trail_y = [], []


def animate(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    trail_x.append(x2[i])
    trail_y.append(y2[i])
    trace.set_data(trail_x[-50:], trail_y[-50:])  # Keep last 50 points for trail
    return line, trace


ani = animation.FuncAnimation(fig, animate, frames=len(t_eval), interval=20, blit=True)

attractor_path = "Chaotic_Double_Pendulum.gif"
ani.save(attractor_path, writer="pillow", fps=30)
plt.show()