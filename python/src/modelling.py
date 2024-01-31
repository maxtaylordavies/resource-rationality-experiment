from collections import Counter

import GPy
import jax.numpy as jnp
from jax import random

from .utils import to_range


def sample_covariance_matrix(
    n_options: int,
    variance=1,
    lengthscale=1,
) -> jnp.ndarray:
    kernel = GPy.kern.RBF(1, variance=variance, lengthscale=lengthscale)
    X = jnp.arange(n_options)[:, None]
    return kernel.K(X, X)


def sample_utility_function(
    rng_key: random.KeyArray,
    cov: jnp.ndarray,
    min_val=-1,
    max_val=1,
) -> jnp.ndarray:
    n_options = cov.shape[0]
    u = random.multivariate_normal(rng_key, jnp.zeros(n_options), cov)
    return to_range(u, min_val, max_val)


def expected_acc_optimal(u: jnp.ndarray, beta: float) -> float:
    deltas = jnp.abs(u[:, None] - u[None, :])
    tmp = 1 / (1 + jnp.exp(-deltas / beta))
    return float(jnp.mean(tmp))


def expected_acc_approx(u: jnp.ndarray, u_hat: jnp.ndarray, beta: float) -> float:
    u_deltas = u[:, None] - u[None, :]
    u_hat_deltas = u_hat[:, None] - u_hat[None, :]
    sign_prod = jnp.sign(jnp.multiply(u_deltas, u_hat_deltas))
    tmp = jnp.multiply(sign_prod, jnp.tanh(jnp.abs(u_deltas) / (2 * beta)))
    return 0.5 + (float(jnp.mean(tmp)) / 2)


def choose(
    rng_key: random.KeyArray, i: int, j: int, u: jnp.ndarray, beta: float
) -> int:
    p_choose_i = 1 / (1 + jnp.exp((u[j] - u[i]) / beta))
    return i if random.uniform(rng_key) < p_choose_i else j


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
