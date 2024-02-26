import json
import time
import os

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import to_range, digitize, avg_pool_2d, gp_covariance_matrix_2d

SIDE_LENGTH = 8
NUM_BINS = 10
NUM_ROUNDS = 10
PLOT = True

length_scales = {"rough": 0.5, "smooth": 2.0}

# set seed
seed = int(time.time())
rng_key = random.PRNGKey(seed)
print(f"using seed {seed}")

for name, scale in length_scales.items():
    # generate covariance matrix
    K = gp_covariance_matrix_2d(SIDE_LENGTH, scale=scale)

    # sample utility functions
    for round, key in tqdm(
        enumerate(random.split(rng_key, NUM_ROUNDS)), desc=f"scale {scale}"
    ):
        u = random.multivariate_normal(key, jnp.zeros(SIDE_LENGTH**2), K)
        u = to_range(u, 0, 1).reshape((SIDE_LENGTH, SIDE_LENGTH))

        # create folder if it doesn't exist
        dir = f"../heatmaps/{name}/{round + 1}"
        os.makedirs(dir, exist_ok=True)

        fig, axs = plt.subplots(1, 4)
        for i in range(4):
            patch_size = int(2**i)
            _u = avg_pool_2d(u, patch_size)
            _u = digitize(_u, NUM_BINS)
            with open(f"{dir}/{patch_size}.txt", "w") as f:
                json.dump(_u.tolist(), f)

            if PLOT:
                axs[i].imshow(_u, cmap="viridis")
                axs[i].set_title(f"patch size {patch_size}")

        if PLOT:
            fig.savefig(f"{dir}/heatmap.png")
