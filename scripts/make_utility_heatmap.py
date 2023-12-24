import GPy
import jax.numpy as jnp
from jax import random
import json
import time

SIDE_LENGTH = 3
NUM_BINS = 4


def gp_covariance_matrix(var=1, scale=1):
    xx, yy = jnp.mgrid[0:SIDE_LENGTH, 0:SIDE_LENGTH]
    X = jnp.vstack((xx.flatten(), yy.flatten())).T
    k = GPy.kern.RBF(input_dim=2, variance=var, lengthscale=scale)  # define kernel
    return k.K(X)  # compute covariance matrix


rng_key = random.PRNGKey(int(time.time()))

# generate spatially correlated utility heatmap
K = gp_covariance_matrix()
v = random.multivariate_normal(rng_key, jnp.zeros(SIDE_LENGTH**2), K)

# normalize to [0, 1] then bin into NUM_BINS bins
v = ((v - v.min()) / (v.max() - v.min())).reshape((SIDE_LENGTH, SIDE_LENGTH))
v = jnp.digitize(v, jnp.array([0.25, 0.5, 0.75]))

# save to text file
with open("../heatmaps/1.txt", "w") as f:
    json.dump(v.tolist(), f)
