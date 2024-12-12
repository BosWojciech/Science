import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

# Constants
GRID_WIDTH = 250
GRID_HEIGHT = 250
TIMESTEPS = 1000
RELAXATION_TIME = 0.53
PLOT_FREQUENCY = 2

# Lattice properties
NUM_DIRECTIONS = 9
LATTICE_WEIGHTS = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
LATTICE_VELOCITIES_X = np.array([0,  1,  0, -1,  0,  1, -1, -1,  1])
LATTICE_VELOCITIES_Y = np.array([0,  0,  1,  0, -1,  1,  1, -1, -1])
#
#
# [-1, 1][0, 1][1, 1]
# [-1, 0][0, 0][1, 0]
# [-1,-1][0,-1][1,-1]
#

# Initialize the distribution function
distribution_func = np.ones(
    (GRID_HEIGHT, GRID_WIDTH, NUM_DIRECTIONS)
) + 0.01 * np.random.rand(GRID_HEIGHT, GRID_WIDTH, NUM_DIRECTIONS)

# Add disturbance at the center
CENTER_X, CENTER_Y = GRID_WIDTH // 2, GRID_HEIGHT // 2
DISTURBANCE_RADIUS = 10

for y in range(GRID_HEIGHT):
    for x in range(GRID_WIDTH):
        if np.sqrt((x - CENTER_X) ** 2 + (y - CENTER_Y) ** 2) < DISTURBANCE_RADIUS:
            distribution_func[y, x, :] += 1

# Equilibrium distribution function calculation
calculate_equilibrium = (
    lambda rho, ux, uy, cx, cy, w: rho
    * w
    * (
        1
        + 3 * (cx * ux + cy * uy)
        + 9 * (cx * ux + cy * uy) ** 2 / 2
        - 3 * (ux**2 + uy**2) / 2
    )
)

# Main simulation loop
plt.ion()
fig, ax = plt.subplots()

logger.info("Starting simulation...")

for timestep in range(TIMESTEPS):
    # Periodic boundary conditions
    # distribution_func[:, -1, [6, 7, 8]] = distribution_func[:, -2, [6, 7, 8]]
    # distribution_func[:, 0, [2, 3, 4]] = distribution_func[:, 1, [2, 3, 4]]

    # Streaming step
    for i, (cx, cy) in enumerate(zip(LATTICE_VELOCITIES_X, LATTICE_VELOCITIES_Y)):
        distribution_func[:, :, i] = np.roll(distribution_func[:, :, i], cx, axis=1)
        distribution_func[:, :, i] = np.roll(distribution_func[:, :, i], cy, axis=0)

    # Calculate macroscopic variables
    rho = np.sum(distribution_func, axis=2)
    ux = np.sum(distribution_func * LATTICE_VELOCITIES_X, axis=2) / rho
    uy = np.sum(distribution_func * LATTICE_VELOCITIES_Y, axis=2) / rho

    # Collision step
    equilibrium_func = np.ones_like(distribution_func)
    for i, (cx, cy, w) in enumerate(
        zip(LATTICE_VELOCITIES_X, LATTICE_VELOCITIES_Y, LATTICE_WEIGHTS)
    ):
        equilibrium_func[:, :, i] = calculate_equilibrium(rho, ux, uy, cx, cy, w)

    distribution_func -= (1 / RELAXATION_TIME) * (distribution_func - equilibrium_func)

    # Logging
    if timestep % (TIMESTEPS // 10) == 0:
        logger.info(f"Timestep {timestep}/{TIMESTEPS} complete")

    # Visualization
    if timestep % PLOT_FREQUENCY == 0:
        speed = np.sqrt(ux**2 + uy**2)
        ax.clear()
        ax.imshow(speed, cmap="viridis", origin="lower")
        ax.set_title(f"Timestep {timestep}")
        plt.pause(0.01)

plt.ioff()
plt.show()

logger.info("Simulation complete.")
