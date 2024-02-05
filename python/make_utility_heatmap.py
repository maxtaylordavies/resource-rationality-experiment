import json
import time

import GPy
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import to_range, digitize, avg_pool_2d

SIDE_LENGTH = 8
NUM_BINS = 4
NUM_ROUNDS = 3
PLOT = False


def gp_covariance_matrix(var=1.0, scale=1.0):
    xx, yy = jnp.mgrid[0:SIDE_LENGTH, 0:SIDE_LENGTH]
    X = jnp.vstack((xx.flatten(), yy.flatten())).T
    k = GPy.kern.RBF(input_dim=2, variance=var, lengthscale=scale)  # define kernel
    return k.K(X)  # compute covariance matrix


# set seed
seed = int(time.time())
rng_key = random.PRNGKey(seed)
print(f"using seed {seed}")

# generate covariance matrix
K = gp_covariance_matrix(scale=1.5)

# sample utility functions
for round, key in enumerate(random.split(rng_key, NUM_ROUNDS)):
    u = random.multivariate_normal(key, jnp.zeros(SIDE_LENGTH**2), K)
    u = to_range(u, 0, 1).reshape((SIDE_LENGTH, SIDE_LENGTH))

    axs = []
    if PLOT:
        fig, axs = plt.subplots(1, 4)

    for i in tqdm(range(4), desc=f"Round {round + 1}"):
        patch_size = int(2**i)
        _u = avg_pool_2d(u, patch_size)
        _u = digitize(_u, NUM_BINS)
        with open(f"../heatmaps/{round + 1}/{patch_size}.txt", "w") as f:
            json.dump(_u.tolist(), f)

        if PLOT:
            axs[i].imshow(_u, cmap="viridis")
            axs[i].set_title(f"patch size {patch_size}")

    if PLOT:
        plt.show()
