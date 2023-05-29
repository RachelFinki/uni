import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#MAYA IS THE QWEEN
def simulate_electrons(num_electrons, radius, tau, num_steps):
    # Initialize electron positions and velocities randomly within the sphere
    phi = np.random.uniform(0, 2 * np.pi, size=num_electrons)
    costheta = np.random.uniform(-1, 1, size=num_electrons)
    theta = np.arccos(costheta)

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    positions = np.column_stack((x, y, z))
    velocities = np.zeros_like(positions)
    # Simulate electron motion for the given number of steps
    for step in range(num_steps):
        # Calculate the electric field on each electron due to other electrons
        electric_fields = np.zeros_like(positions)
        for i in range(num_electrons):
            r = np.linalg.norm(positions - positions[i], axis=1)
            mask = np.arange(num_electrons) != i  # Exclude self-interaction
            epsilon = 1e-8  # Small epsilon value to avoid division by zero
            electric_fields[i] = np.sum((positions - positions[i]) * mask[:, np.newaxis] / (r[:, np.newaxis]**3 + epsilon), axis=0)

        # Update electron velocities based on the electric fields
        velocities += tau * electric_fields

        # Update electron positions based on the velocities
        positions += tau * velocities

        # Enforce the constraint of electrons being trapped inside the sphere
        distances = np.linalg.norm(positions, axis=1)
        mask = distances > radius
        positions[mask] *= radius / distances[mask][:, np.newaxis]

    return positions

# Parameters
num_electrons = 200
radius = 1.0
tau = 1e-3
num_steps = 10000

# Simulate electron motion
equilibrium_positions = simulate_electrons(num_electrons, radius, tau, num_steps)

# Plot initial and equilibrium positions in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*equilibrium_positions.T, s=5, c='mediumturquoise')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Final Electron Positions')
plt.show()

def rand_sphere_points_2(n, r):
    """
    Generate n evenly distributed random points within a 3D sphere with radius r.
    Returns a numpy array of shape (n, 3).
    """
    points = np.zeros((n, 3))
    for i in range(n):
        # Generate a random point inside a unit cube
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        # Check if the point is inside the sphere
        while x**2 + y**2 + z**2 > 1:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(-1, 1)
        # Scale the point to the desired radius
        scale = r * np.power(np.random.uniform(0, 1), 1/3) / np.sqrt(x**2 + y**2 + z**2)
        points[i, 0] = scale * x
        points[i, 1] = scale * y
        points[i, 2] = scale * z
    return points

num_electrons = 200
radius = 1.0
tau = 1e-3
num_steps = 10000

# Simulate electron motion
equilibrium_positions = rand_sphere_points_2(num_electrons, radius)

# Plot initial and equilibrium positions in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*equilibrium_positions.T, s=5, c='deeppink')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Initial Electron Positions')
plt.show()
