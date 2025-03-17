import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Lorenz system parameters
sigma = 10
rho = 28
beta = 8 / 3


# Define the Lorenz system
def lorenz(t, state):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]


# Time range
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 5000)

# Initial conditions
initial_conditions = [1.0, 1.0, 1.0]

# Solve the ODE
sol = solve_ivp(lorenz, t_span, initial_conditions, t_eval=t_eval)
x, y, z = sol.y

# Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection="3d")
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_zlim(min(z), max(z))
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Animated Lorenz Attractor")

# Initialize the trajectory line and point
(line,) = ax.plot([], [], [], lw=1.5, color="b")
(point,) = ax.plot([], [], [], "ro", markersize=4)


# Animation update function
def update(num):
    line.set_data(x[:num], y[:num])
    line.set_3d_properties(z[:num])
    point.set_data(x[num], y[num])
    point.set_3d_properties(z[num])
    return line, point


# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(x), interval=10, blit=False)

# Show animation
plt.show()
