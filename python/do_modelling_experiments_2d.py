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
    avg_pool_2d,
    gp_covariance_matrix_2d,
    aggregated_covariance_matrix,
    gaussian_entropy,
    save_figure,
    to_range,
)
from src.modelling import (
    expected_acc_approx_2d,
    sample_utility_function_2d,
    simulate_choices_2d,
    simulate_predictions_2d,
)

pd.options.mode.chained_assignment = None
config.update("jax_enable_x64", True)
sns.set_theme()


def state_agg_experiment(
    n_seeds: int, betas: List[float], lengthscales: List[float], patch_sizes: List[int]
):
    seeds = [int(time.time()) + i for i in range(n_seeds)]
    side_length, results = max(patch_sizes), []

    for ls in lengthscales:
        cov = gp_covariance_matrix_2d(side_length, scale=ls)
        for seed in tqdm(seeds):
            key = random.PRNGKey(seed)
            u = sample_utility_function_2d(key, cov, min_val=0, max_val=1)

            costs = jnp.array(
                [
                    gaussian_entropy(aggregated_covariance_matrix(cov, ps))
                    for ps in patch_sizes
                ]
            )
            costs = to_range(costs, 0, 1)

            for ps_idx, ps in enumerate(patch_sizes):
                u_hat = avg_pool_2d(u, ps, same_shape=True)
                for beta in betas:
                    pairs, choices = simulate_choices_2d(key, u, beta, n_trials=100)
                    results.append(
                        {
                            "beta": beta,
                            "patch_size": ps,
                            "accuracy": expected_acc_approx_2d(u, u_hat, beta),
                            "cost": float(costs[ps_idx]),
                            "type": "theoretical",
                            "seed": seed,
                            "scale": ls,
                        }
                    )
                    results.append(
                        {
                            "beta": beta,
                            "patch_size": ps,
                            "accuracy": simulate_predictions_2d(pairs, choices, u_hat),
                            "cost": float(costs[ps_idx]),
                            "type": "simulated",
                            "seed": seed,
                            "scale": ls,
                        }
                    )

    results = pd.DataFrame(results)
    # results["cost"] = results["cost"] - results["cost"].min()
    # results["cost"] = results["cost"] / results["cost"].max()
    return results


num_seeds = 50
side_length = 32
lambdas = [0.1, 0.5, 1.0]
betas = [0.01, 0.1, 1.0, 5.0]
lengthscales = [0.5, 0.75, 1.0]

patch_sizes = [2**i for i in range(int(jnp.log2(side_length + 1) + 1))]
results = state_agg_experiment(num_seeds, betas, lengthscales, patch_sizes)
beta_df = results[results["scale"] == 0.75]
scale_df = results[results["beta"] == 0.01]

beta_palette = sns.color_palette(["#35B886", "#3B8B83", "#405F81", "#46327E"])
scale_palette = sns.color_palette(["#DF517E", "#E56E51", "#EB8A23"])

# accuracy plot for varying beta and lengthscale
fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
sns.lineplot(
    data=beta_df,
    x="patch_size",
    y="accuracy",
    hue="beta",
    style="type",
    markers=True,
    dashes=True,
    linewidth=2,
    ax=axs[0],
    palette=beta_palette,
)
sns.lineplot(
    data=scale_df,
    x="patch_size",
    y="accuracy",
    hue="scale",
    style="type",
    markers=True,
    dashes=True,
    linewidth=2,
    ax=axs[1],
    palette=scale_palette,
)
for ax in axs:
    ax.set_xscale("log", base=2)
    ax.set(
        xlabel="Patch size",
        xticks=patch_sizes,
        xticklabels=[str(ps) for ps in patch_sizes],
        ylabel="Choice prediction accuracy",
        ylim=(0.45, 1.05),
    )
handles, labels = axs[0].get_legend_handles_labels()
labels[0] = "$\\beta$"
axs[0].legend(handles=handles, labels=labels)
handles, labels = axs[1].get_legend_handles_labels()
labels[0] = "lengthscale"
axs[1].legend(handles=handles, labels=labels)
fig.suptitle("Choice prediction accuracy from state-aggregated representations")
save_figure(fig, "state_agg_accuracy")

# cost-adjusted value plot for varying beta and lengthscale
beta_df = beta_df[beta_df["type"] == "theoretical"]
fig, axs = plt.subplots(2, len(lambdas), sharex=True, figsize=(12, 6))
for i, l in enumerate(lambdas):
    beta_df["value"] = beta_df["accuracy"] - 0.5 - (l * beta_df["cost"])
    sns.lineplot(
        data=beta_df,
        x="patch_size",
        y="value",
        hue="beta",
        markers=True,
        linewidth=2,
        ax=axs[0, i],
        palette=beta_palette,
        legend=i == 0,
    )
    axs[0, i].set_xscale("log", base=2)
    axs[0, i].set(
        title=f"$\lambda$ = {l}",
        xlabel="Patch size",
        xticks=patch_sizes,
        xticklabels=[str(ps) for ps in patch_sizes],
        ylabel="Value",
    )
handles, labels = axs[0, 0].get_legend_handles_labels()
axs[0, 0].legend(handles=handles, labels=labels, title="$\\beta$")

scale_df = scale_df[scale_df["type"] == "theoretical"]
for i, l in enumerate(lambdas):
    scale_df["value"] = scale_df["accuracy"] - 0.5 - (l * scale_df["cost"])
    sns.lineplot(
        data=scale_df,
        x="patch_size",
        y="value",
        hue="scale",
        markers=True,
        linewidth=2,
        ax=axs[1, i],
        palette=scale_palette,
        legend=i == 0,
    )
    axs[1, i].set_xscale("log", base=2)
    axs[1, i].set(
        xlabel="Patch size",
        xticks=patch_sizes,
        xticklabels=[str(ps) for ps in patch_sizes],
        ylabel="Value",
    )
handles, labels = axs[1, 0].get_legend_handles_labels()
axs[1, 0].legend(handles=handles, labels=labels, title="lengthscale")
fig.suptitle("Cost-adjusted value of state-aggregated representations")
save_figure(fig, "state_agg_value")

# optimal patch size plots for varying beta and lengthscale
fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax in axs:
    ax.set_yscale("log", base=2)
    ax.set(
        xlabel="$\lambda$",
        yticks=patch_sizes,
        yticklabels=[str(ps) for ps in patch_sizes],
    )
axs[0].set(ylabel="Optimal patch size")

beta_df_, lambdas = [], jnp.linspace(0, 5, 501)
for l, beta, seed in tqdm(list(product(lambdas, betas, beta_df["seed"].unique()))):
    tmp = beta_df[(beta_df["beta"] == beta) & (beta_df["seed"] == seed)]
    tmp["value"] = tmp["accuracy"] - 0.5 - (l * tmp["cost"])
    max_value, max_ps = (
        tmp["value"].max(),
        tmp["patch_size"].iloc[tmp["value"].argmax()],
    )
    beta_df_.append(
        {
            "beta": beta,
            "lambda": float(l),
            "value": float(max_value),
            "patch_size": int(max_ps),
        }
    )
sns.lineplot(
    data=pd.DataFrame(beta_df_),
    x="lambda",
    y="patch_size",
    hue="beta",
    linewidth=2,
    ax=axs[0],
    palette=beta_palette,
)
handles, labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles=handles, labels=labels, title="$\\beta$")

scale_df_, lambdas = [], jnp.linspace(0, 5, 501)
for l, ls, seed in tqdm(
    list(product(lambdas, lengthscales, scale_df["seed"].unique()))
):
    tmp = scale_df[(scale_df["scale"] == ls) & (scale_df["seed"] == seed)]
    tmp["value"] = tmp["accuracy"] - 0.5 - (l * tmp["cost"])
    max_value, max_ps = (
        tmp["value"].max(),
        tmp["patch_size"].iloc[tmp["value"].argmax()],
    )
    scale_df_.append(
        {
            "scale": ls,
            "lambda": float(l),
            "value": float(max_value),
            "patch_size": int(max_ps),
        }
    )
sns.lineplot(
    data=pd.DataFrame(scale_df_),
    x="lambda",
    y="patch_size",
    hue="scale",
    linewidth=2,
    ax=axs[1],
    palette=scale_palette,
)
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles=handles, labels=labels, title="lengthscale")
fig.suptitle("Optimal patch size for state-aggregated representations")
save_figure(fig, "state_agg_optimal_patch_size")
