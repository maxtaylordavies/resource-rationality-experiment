import json
import time

import GPy
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import to_range, avg_pool_2d

SIDE_LENGTH = 8
NUM_BINS = 4


def gp_covariance_matrix(var=1.0, scale=1.0):
    xx, yy = jnp.mgrid[0:SIDE_LENGTH, 0:SIDE_LENGTH]
    X = jnp.vstack((xx.flatten(), yy.flatten())).T
    k = GPy.kern.RBF(input_dim=2, variance=var, lengthscale=scale)  # define kernel
    return k.K(X)  # compute covariance matrix


seed = int(time.time())
rng_key = random.PRNGKey(seed)
print(f"using seed {seed}")

# generate spatially correlated utility heatmap
K = gp_covariance_matrix(scale=1.5)
u = random.multivariate_normal(rng_key, jnp.zeros(SIDE_LENGTH**2), K)
u = to_range(u, 0, 1).reshape((SIDE_LENGTH, SIDE_LENGTH))

fig, axs = plt.subplots(1, 4)

for i in tqdm(range(4)):
    patch_size = int(2**i)
    _u = to_range(avg_pool_2d(u, patch_size), 0, 1)
    _u = jnp.digitize(_u, jnp.array([0.25, 0.5, 0.75]))
    axs[i].imshow(_u, cmap="viridis")
    axs[i].axis("off")
    with open(f"../heatmaps/1/{patch_size}.txt", "w") as f:
        json.dump(_u.tolist(), f)

plt.show()
