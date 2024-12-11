import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Constants
GRID = { 'X': 100, 'Y': 100 }
TAU = 0.6
TIMESTEPS = 1000

# Lattice weights and directions (D2Q9)
WEIGHTS = np.array([4/9] + [1/9]*4 + [1/36]*4)
DIRECTIONS = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
