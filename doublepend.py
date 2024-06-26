import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# constant vs
g = 9.81  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
m1 = 1.0  # mass of pendulum 1 in kg
m2 = 1.0  # mass of pendulum 2 in kg

# Initial conditions: [theta1, z1, theta2, z2]
# where theta1 and theta2 are the angles (in radians) and z1 and z2 are the angular velocities (in radians/s)
y0 = [np.pi / 2, 0, np.pi / 2, 0]

# span
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

def deriv(t, y):
    theta1, z1, theta2, z2 = y
    delta = theta2 - theta1

    denominator1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) * np.cos(delta)
    denominator2 = (L2 / L1) * denominator1

    dydx = np.zeros_like(y)
    dydx[0] = z1
    dydx[1] = ((m2 * L1 * z1 * z1 * np.sin(delta) * np.cos(delta)
                + m2 * g * np.sin(theta2) * np.cos(delta)
                + m2 * L2 * z2 * z2 * np.sin(delta)
                - (m1 + m2) * g * np.sin(theta1))
               / denominator1)
    dydx[2] = z2
    dydx[3] = ((-m2 * L2 * z2 * z2 * np.sin(delta) * np.cos(delta)
                + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
                - (m1 + m2) * L1 * z1 * z1 * np.sin(delta)
                - (m1 + m2) * g * np.sin(theta2))
               / denominator2)

    return dydx

# system of differential equations
solution = solve_ivp(deriv, t_span, y0, t_eval=t_eval, vectorized=True)

# collecting the angles
theta1 = solution.y[0]
theta2 = solution.y[2]

# the xy coordinates of the pendulums
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')

# grid lines and axis numbers
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# dark gray
line, = ax.plot([], [], 'o-', lw=2, color='darkgray')

# the elapsed time
time_template = 'Time = %.1f s'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def update(frame):
    thisx = [0, x1[frame], x2[frame]]
    thisy = [0, y1[frame], y2[frame]]
    line.set_data(thisx, thisy)
    time_text.set_text(time_template % t_eval[frame])
    return line, time_text

ani = FuncAnimation(fig, update, frames=len(t_eval),
                    init_func=init, blit=True)

plt.show()
