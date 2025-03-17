import numpy as np
import matplotlib.pyplot as plt
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

# Initial conditions (small perturbations lead to different outcomes)
initial_conditions = [1.0, 1.0, 1.0]

# Solve the ODE
sol = solve_ivp(lorenz, t_span, initial_conditions, t_eval=t_eval)

# Extract solutions
x, y, z = sol.y

# Plot the Lorenz attractor
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection="3d")
ax.plot(x, y, z, color="b", lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")
plt.show()
