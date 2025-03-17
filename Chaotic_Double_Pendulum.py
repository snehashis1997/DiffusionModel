import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Parameters
L1, L2 = 1.0, 1.0   # Lengths of pendulums (m)
m1, m2 = 1.0, 1.0   # Masses (kg)
g = 9.81            # Gravity (m/s²)

# Equations of motion for the double pendulum
def double_pendulum(t, state):
    θ1, ω1, θ2, ω2 = state
    
    delta = θ2 - θ1
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
    den2 = (L2 / L1) * den1
    
    dθ1 = ω1
    dω1 = ((m2 * L1 * ω1 ** 2 * np.sin(delta) * np.cos(delta) +
            m2 * g * np.sin(θ2) * np.cos(delta) +
            m2 * L2 * ω2 ** 2 * np.sin(delta) -
            (m1 + m2) * g * np.sin(θ1)) / den1)
    
    dθ2 = ω2
    dω2 = ((-L1 / L2) * ω1 ** 2 * np.sin(delta) * np.cos(delta) +
           (m1 + m2) * g * np.sin(θ1) * np.cos(delta) -
           (m1 + m2) * L2 * ω2 ** 2 * np.sin(delta) -
           (m1 + m2) * g * np.sin(θ2)) / den2
    
    return [dθ1, dω1, dθ2, dω2]

# Time span and initial conditions
t_span = (0, 50)
t_eval = np.linspace(0, 50, 5000)  # High resolution
initial_state = [np.pi / 2, 0, np.pi / 2, 0]  # (θ₁, ω₁, θ₂, ω₂)

# Solve the system
sol = solve_ivp(double_pendulum, t_span, initial_state, t_eval=t_eval, method='RK45')
θ1, ω1, θ2, ω2 = sol.y

# Poincaré section: Find points where ω₁ crosses zero
crossings = np.where((ω1[:-1] > 0) & (ω1[1:] < 0))[0]  # Detect downward zero crossings
θ1_poincare = θ1[crossings]
ω2_poincare = ω2[crossings]

# Setup animation figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(min(θ1_poincare) - 0.1, max(θ1_poincare) + 0.1)
ax.set_ylim(min(ω2_poincare) - 0.1, max(ω2_poincare) + 0.1)
ax.set_xlabel("θ₁ (Angle of First Pendulum)")
ax.set_ylabel("ω₂ (Angular Velocity of Second Pendulum)")
ax.set_title("Poincaré Map Animation of a Chaotic Double Pendulum")
ax.grid()

scatter, = ax.plot([], [], 'ro', markersize=3)  # Red dots

# Animation function
def update(frame):
    scatter.set_data(θ1_poincare[:frame], ω2_poincare[:frame])
    return scatter,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(θ1_poincare), interval=10, blit=False)
ani.save('chaotic_pendulam_simulation.gif', writer='pillow', fps=30)
# Show animation
plt.show()
