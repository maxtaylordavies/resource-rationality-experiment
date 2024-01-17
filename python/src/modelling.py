from collections import Counter

import GPy
import jax.numpy as jnp
from jax import random

from .utils import to_range, boltzmann_1d


def sample_utility_function(
    rng_key: random.KeyArray,
    n_options: int,
    lengthscale=1,
    min_val=-1,
    max_val=1,
) -> jnp.ndarray:
    kernel = GPy.kern.RBF(1, variance=1, lengthscale=lengthscale)
    X = jnp.arange(n_options)[:, None]
    K = kernel.K(X, X)
    u = random.multivariate_normal(rng_key, jnp.zeros(n_options), K)
    return to_range(u, min_val, max_val)


def expected_delta(u: jnp.ndarray) -> float:
    deltas = jnp.abs(u[:, None] - u[None, :])
    return float(jnp.sum(deltas) / (len(u) * (len(u) - 1)))


def expected_delta_sign_product(u: jnp.ndarray, u_hat: jnp.ndarray) -> float:
    u_deltas = u[:, None] - u[None, :]
    u_hat_deltas = u_hat[:, None] - u_hat[None, :]
    sign_product = jnp.sign(jnp.multiply(u_deltas, u_hat_deltas))
    return float(jnp.sum(sign_product) / (len(u) * (len(u) - 1)))


def expected_return_optimal(u: jnp.ndarray, beta: float) -> float:
    e_delta = expected_delta(u)
    return float(1 / (jnp.exp(-e_delta / beta) + 1))


def expected_return_approx(u: jnp.ndarray, u_hat: jnp.ndarray, beta: float) -> float:
    e_delta = expected_delta(u)
    e_delta_sign_prod = expected_delta_sign_product(u, u_hat)
    tmp = float(e_delta_sign_prod * jnp.tanh(e_delta / (2 * beta)))
    return 0.5 + (tmp / 2)


def choose(
    rng_key: random.KeyArray, i: int, j: int, u: jnp.ndarray, beta: float
) -> int:
    p = boltzmann_1d(jnp.array([u[i], u[j]]), beta)
    return int(random.choice(rng_key, jnp.array([i, j]), p=p))


def simulate_choices(
    rng_key: random.KeyArray,
    u: jnp.ndarray,
    beta: float,
    n_trials=10000,
) -> float:
    pairs = random.choice(
        rng_key, jnp.arange(len(u)), shape=(n_trials, 2), replace=True
    )
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]  # remove pairs of same state

    keys = random.split(rng_key, len(pairs))
    choices = jnp.array(
        [choose(key, i, j, u, beta) for key, (i, j) in zip(keys, pairs)]
    )
    return pairs, choices


def simulate_predictions(
    pairs: jnp.ndarray,
    choices: jnp.ndarray,
    u_hat: jnp.ndarray,
) -> float:
    tmp = jnp.argmax(u_hat[pairs], axis=1)
    pred = pairs[jnp.arange(len(pairs)), tmp]
    return float(jnp.mean(pred == choices))
    # score = 0
    # for i in range(len(pairs)):
    #     if u_hat[pairs[i, 0]] > u_hat[pairs[i, 1]]:
    #         score += int(choices[i] == pairs[i, 0])
    #     else:
    #         score += int(choices[i] == pairs[i, 1])
    # return score / len(pairs)
