from itertools import product
import time

import jax.numpy as jnp
from jax import random
from jax.config import config
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.utils import (
    avg_pool_1d,
    aggregated_covariance_matrix,
    gaussian_entropy,
    save_figure,
)
from src.modelling import (
    expected_acc_approx,
    sample_covariance_matrix,
    sample_utility_function,
    simulate_choices,
    simulate_predictions,
)

pd.options.mode.chained_assignment = None
config.update("jax_enable_x64", True)
sns.set_theme()


def state_agg_experiment(n_seeds: int, betas: List[float], n_options=128):
    results = []
    pool_sizes = [int(2**i) for i in range(int(jnp.log2(n_options)) + 1)]
    cov = sample_covariance_matrix(n_options, lengthscale=1)

    for _ in tqdm(range(n_seeds)):
        seed = int(time.time())
        rng_key = random.PRNGKey(seed)
        u = sample_utility_function(rng_key, cov, min_val=-1, max_val=1)
        for ps in pool_sizes:
            u_hat = avg_pool_1d(u, ps)
            cost = gaussian_entropy(aggregated_covariance_matrix(cov, ps))
            for beta in betas:
                # pairs, choices = simulate_choices(rng_key, u, beta, n_trials=1000)
                results.append(
                    {
                        "beta": beta,
                        "pool_size": ps,
                        "accuracy": expected_acc_approx(u, u_hat, beta),
                        "cost": float(cost),
                        "type": "predicted",
                        "seed": seed,
                    }
                )
                # results.append(
                #     {
                #         "beta": beta,
                #         "pool_size": ps,
                #         "accuracy": simulate_predictions(pairs, choices, u_hat),
                #         "cost": float(cost),
                #         "type": "empirical",
                #         "seed": seed,
                #     }
                # )

    results = pd.DataFrame(results)
    results["cost"] = results["cost"] / results["cost"].max()
    results["cost"][results["cost"] < 0.0] = 0.0
    return results


num_seeds = 100
lambdas = [0.1, 0.5, 1.0, 5.0]
betas = [0.01, 0.1, 1.0, 5.0]
results = state_agg_experiment(num_seeds, betas, n_options=512)

# accuracy and efficient frontier plots
palette = sns.color_palette("husl", n_colors=len(betas))
# fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
# sns.lineplot(
#     data=results,
#     x="pool_size",
#     y="accuracy",
#     hue="beta",
#     style="type",
#     markers=True,
#     dashes=True,
#     ax=axs[0],
#     palette=palette,
# )
# axs[0].set_xscale("log", base=2)
# axs[0].set(
#     title="Theoretical vs empirical accuracy using state-aggregated representations",
#     xlabel="Patch size",
#     ylabel="Choice prediction accuracy",
#     ylim=(0.45, 1.05),
# )
# handles, labels = axs[0].get_legend_handles_labels()
# labels[0] = "$\\beta$"
# axs[0].legend(handles=handles, labels=labels)

results = results[results["type"] == "predicted"]
# sns.lineplot(
#     data=results,
#     x="cost",
#     y="accuracy",
#     hue="beta",
#     ax=axs[1],
#     palette=palette,
#     legend=False,
# )
# axs[1].set(
#     title="Accuracy-cost efficient frontier for state-aggregated representations",
#     xlabel="Cost",
# )
# save_figure(fig, "state_agg_accuracy")

# # cost-adjusted value plot
# fig, axs = plt.subplots(1, len(lambdas), sharex=True, figsize=(16, 4))
# for i, l in enumerate(lambdas):
#     results["value"] = results["accuracy"] - 0.5 - (l * results["cost"])
#     sns.lineplot(
#         data=results,
#         x="pool_size",
#         y="value",
#         hue="beta",
#         markers=True,
#         ax=axs[i],
#         palette=palette,
#         legend=i == 0,
#     )
#     axs[i].set_xscale("log", base=2)
#     axs[i].set(
#         title=f"$\lambda$ = {l}",
#         xlabel="Patch size",
#         ylabel="Value",
#     )
# handles, labels = axs[0].get_legend_handles_labels()
# axs[0].legend(handles=handles, labels=labels, title="$\\beta$")
# fig.suptitle("Value of state-aggregated representations")
# save_figure(fig, "state_agg_value")

# optimal patch size plot
results_, lambdas = [], jnp.linspace(0, 20, 501)
for l, beta, seed in tqdm(list(product(lambdas, betas, results["seed"].unique()))):
    tmp = results[(results["beta"] == beta) & (results["seed"] == seed)]
    tmp["value"] = tmp["accuracy"] - 0.5 - (l * tmp["cost"])
    max_value, max_ps = (
        tmp["value"].max(),
        tmp["pool_size"].iloc[tmp["value"].argmax()],
    )
    results_.append(
        {
            "beta": beta,
            "lambda": float(l),
            "value": float(max_value),
            "pool_size": int(max_ps),
        }
    )

fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(
    data=pd.DataFrame(results_),
    x="lambda",
    y="pool_size",
    hue="beta",
    ax=ax,
    palette=palette,
)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title="$\\beta$")
ax.set_yscale("log", base=2)
ax.set(
    title="Optimal patch size for state-aggregated representations",
    xlabel="$\lambda$",
    ylabel="Patch size",
)
save_figure(fig, "state_agg_optimal_patch_size")
