import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Constants
# Constants
GRID = {'X': 100, 'Y': 100}
TIMESTEPS = 1000
CS_SQUARED = 1/3  # Lattice speed of sound squared
OMEGA = 1.0       # Relaxation parameter (tunable for fluid viscosity)

# Lattice weights and directions (D2Q9)
# [4/9] - Weight for resting state
# [1/9] - Weight for cardinal directions
# [1/36] - Weight for diagonal directions
WEIGHTS = np.array([4/9] + [1/9]*4 + [1/36]*4)
DISCRETE_VELOCITY_VECTORS = np.array([
    [-1,  1], [ 0,  1], [ 1,  1],
    [-1,  0], [ 0,  0], [ 1,  0],
    [-1, -1], [ 0, -1], [ 1, -1]
])

# Lambda for equilibrium distribution
f_eq = lambda w, rho, u, c: w * rho * (
    1 + (np.sum(c * u, axis=-1) / CS_SQUARED) + 
    (np.sum(c * u, axis=-1)**2 / (2 * CS_SQUARED**2)) - 
    (np.sum(u * u, axis=-1) / (2 * CS_SQUARED))
)


# Lambda for updating f_i (collision step) using relaxation parameter omega.
f_new = lambda f, f_eq: (1 - OMEGA) * f + OMEGA * f_eq

# Perform the collision step by updating distribution functions 
# for each lattice direction using the equilibrium distribution function. 
# The collision step updates the distribution functions based on the local density and velocity values.
#
def collision_step(f, rho, u):
    for i, c in enumerate(DISCRETE_VELOCITY_VECTORS):
        f_eq_i = f_eq(WEIGHTS[i], rho, u, c)
        f[:, :, i] = f_new(f[:, :, i], f_eq_i)
    return f


# Perform the streaming step by shifting the distribution functions 
# along their respective discrete velocity vectors.
# This simulates the movement of particles through the lattice.
# The streaming step updates the distribution functions by moving them to new positions based on their velocities.
#
def streaming_step(f):
    f_streamed = np.zeros_like(f)
    for i, c in enumerate(DISCRETE_VELOCITY_VECTORS):
        f_streamed[:,:,i] = np.roll(
            np.roll(f[:,:,i], c[0], axis = 0), c[1], axis=1
        )
    return f_streamed

# Initialize 
# f- grid
# rho - density
# u - velocities 
# 
f = np.zeros((GRID['X'], GRID['Y'], len(DISCRETE_VELOCITY_VECTORS))) 
rho = np.ones((GRID['X'], GRID['Y']))
u = np.zeros((GRID['X'], GRID['Y'], 2))


# For simulation:
# Introducing a disturbance at the center of the grid
center_x, center_y = GRID['X'] // 2, GRID['Y'] // 2
disturbance_radius = 5
for x in range(center_x - disturbance_radius, center_x + disturbance_radius):
    for y in range(center_y - disturbance_radius, center_y + disturbance_radius):
        if 0 <= x < GRID['X'] and 0 <= y < GRID['Y']:
            rho[x, y] += 1  # Increase density in the disturbed region



plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
density_plot = ax.imshow(rho, cmap='viridis', origin='lower')

# Simulation
for t in range(TIMESTEPS):
    logging.info(f"Timestep {t+1}/{TIMESTEPS}")
    f = collision_step(f, rho, u)
    f = streaming_step(f)
    
    # Update density and velocity fields
    rho = np.sum(f, axis=-1)  # Compute density from f
    
    # Compute velocity explicitly
    u = np.zeros((GRID['X'], GRID['Y'], 2))  # Reset velocity field
    for i, c in enumerate(DISCRETE_VELOCITY_VECTORS):
        u[..., 0] += f[..., i] * c[0]
        u[..., 1] += f[..., i] * c[1]
    u /= rho[..., None]  # Normalize by density

    # Update plot
    density_plot.set_data(rho)
    plt.title(f"Timestep {t+1}")
    plt.pause(0.01)

plt.ioff()  # Turn off interactive mode
plt.show()






