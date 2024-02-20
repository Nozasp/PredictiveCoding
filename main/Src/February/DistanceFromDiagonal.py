import numpy as np

N = 10
# Create a 30x30 matrix
matrix = np.random.rand(N, N)

# Compute the distance of each cell from the diagonal
distances = np.abs(np.arange(N)[np.newaxis, :] - np.arange(N)[:, np.newaxis])

print(distances)

