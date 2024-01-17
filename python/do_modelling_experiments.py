import time

import jax.numpy as jnp
from jax import random
from jax.config import config
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.utils import avg_pool_1d
from src.modelling import (
    expected_return_optimal,
    expected_return_approx,
    sample_utility_function,
    simulate_choices,
    simulate_predictions,
    choose,
)


config.update("jax_enable_x64", True)
sns.set_theme()


def state_agg_experiment(n_seeds: int, betas, n_options=32):
    results = []
    pool_sizes = [2**i for i in range(int(jnp.log2(n_options)) + 1)]

    for seed in tqdm(range(n_seeds)):
        for beta in betas:
            rng_key = random.PRNGKey(seed)
            u = sample_utility_function(
                rng_key, n_options, lengthscale=2, min_val=0, max_val=1
            )
            pairs, choices = simulate_choices(rng_key, u, beta, n_trials=1000)

            for ps in pool_sizes:
                u_hat = avg_pool_1d(u, ps)
                results.append(
                    {
                        "beta": beta,
                        "pool_size": ps,
                        "accuracy": expected_return_approx(u, u_hat, beta),
                        "type": "predicted",
                    }
                )
                results.append(
                    {
                        "beta": beta,
                        "pool_size": ps,
                        "accuracy": simulate_predictions(pairs, choices, u_hat),
                        "type": "empirical",
                    }
                )

    return pd.DataFrame(results)


num_seeds = 5
betas = [0.01, 0.2, 0.5, 1.0]
results = state_agg_experiment(num_seeds, betas, n_options=512)

fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(
    data=results,
    x="pool_size",
    y="accuracy",
    hue="beta",
    style="type",
    markers=True,
    dashes=True,
    ax=ax,
    palette=sns.color_palette("husl", n_colors=len(betas)),
)
ax.set_xscale("log", base=2)
ax.set(
    title="Predicted vs empirical accuracy using state-aggregated representations",
    xlabel="Patch size",
    ylabel="Choice prediction accuracy",
    ylim=(0.45, 1.05),
)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig("../figures/state_agg_experiment.pdf", bbox_inches="tight")

# rng_key = random.PRNGKey(int(time.time()))
# n_trials, beta = 10000, 0.1
# keys = random.split(rng_key, n_trials)
# u = sample_utility_function(rng_key, 128, lengthscale=2)
# score = 0

# for i in tqdm(range(n_trials)):
#     pair = random.choice(rng_key, jnp.arange(len(u)), shape=(2,), replace=False)
#     choice = choose(keys[i], pair[0], pair[1], u, beta)
#     if u[pair[0]] > u[pair[1]]:
#         score += int(choice == pair[0])
#     else:
#         score += int(choice == pair[1])

# print(score / n_trials)
