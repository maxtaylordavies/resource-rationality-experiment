from collections import Counter

import GPy
import jax.numpy as jnp
from jax import random

from .utils import to_range


def sample_utility_function_1d(
    rng_key: random.KeyArray,
    cov: jnp.ndarray,
    min_val=-1,
    max_val=1,
) -> jnp.ndarray:
    n_options = cov.shape[0]
    u = random.multivariate_normal(rng_key, jnp.zeros(n_options), cov)
    return to_range(u, min_val, max_val)


def sample_utility_function_2d(
    rng_key: random.KeyArray,
    cov: jnp.ndarray,
    min_val=-1,
    max_val=1,
) -> jnp.ndarray:
    side_length = int(jnp.sqrt(cov.shape[0]))
    u = random.multivariate_normal(rng_key, jnp.zeros(side_length**2), cov)
    return to_range(u.reshape(side_length, side_length), min_val, max_val)


def expected_acc_optimal(u: jnp.ndarray, beta: float) -> float:
    deltas = jnp.abs(u[:, None] - u[None, :])
    tmp = 1 / (1 + jnp.exp(-deltas / beta))
    return float(jnp.mean(tmp))


def expected_acc_approx_1d(u: jnp.ndarray, u_hat: jnp.ndarray, beta: float) -> float:
    u_deltas = u[:, None] - u[None, :]
    print(f"u_deltas.shape: {u_deltas.shape}")
    u_hat_deltas = u_hat[:, None] - u_hat[None, :]
    print(f"u_hat_deltas.shape: {u_hat_deltas.shape}")
    sign_prod = jnp.sign(jnp.multiply(u_deltas, u_hat_deltas))
    tmp = jnp.multiply(sign_prod, jnp.tanh(jnp.abs(u_deltas) / (2 * beta)))
    return 0.5 + (float(jnp.mean(tmp)) / 2)


def expected_acc_approx_2d(u: jnp.ndarray, u_hat: jnp.ndarray, beta: float) -> float:
    def deltas(X: jnp.ndarray) -> jnp.ndarray:
        X_ = X[:, :, jnp.newaxis, jnp.newaxis]
        return X_ - X_.T

    u_deltas, u_hat_deltas = deltas(u), deltas(u_hat)
    sign_prod = jnp.sign(jnp.multiply(u_deltas, u_hat_deltas))

    tmp = jnp.multiply(sign_prod, jnp.tanh(jnp.abs(u_deltas) / (2 * beta)))
    return 0.5 + (float(jnp.mean(tmp)) / 2)


def choose_1d(rng_key: random.KeyArray, i, j, u: jnp.ndarray, beta: float) -> int:
    p_choose_i = 1 / (1 + jnp.exp((u[j] - u[i]) / beta))
    return i if random.uniform(rng_key) < p_choose_i else j


def choose_2d(
    rng_key: random.KeyArray, r1, c1, r2, c2, u: jnp.ndarray, beta: float
) -> tuple[int, int]:
    p_choose_1 = 1 / (1 + jnp.exp((u[r2, c2] - u[r1, c1]) / beta))
    tmp = random.uniform(rng_key)
    return (r1, c1) if random.uniform(rng_key) <= p_choose_1 else (r2, c2)


def simulate_choices_1d(
    rng_key: random.KeyArray,
    u: jnp.ndarray,
    beta: float,
    n_trials=10000,
):
    pairs = random.choice(
        rng_key, jnp.arange(len(u)), shape=(n_trials, 2), replace=True
    )
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]  # remove pairs of same state

    keys = random.split(rng_key, len(pairs))
    choices = jnp.array(
        [choose_1d(key, i, j, u, beta) for key, (i, j) in zip(keys, pairs)]
    )
    return pairs, choices


def simulate_choices_2d(
    rng_key: random.KeyArray,
    u: jnp.ndarray,
    beta: float,
    n_trials=10000,
):
    pairs = random.choice(
        rng_key, jnp.arange(u.shape[0]), shape=(n_trials, 2, 2), replace=True
    )

    choices = jnp.array(
        [
            choose_2d(key, r1, c1, r2, c2, u, beta)
            for key, ((r1, c1), (r2, c2)) in zip(
                random.split(rng_key, len(pairs)), pairs
            )
        ]
    )
    return pairs, choices


def simulate_predictions_1d(
    pairs: jnp.ndarray,
    choices: jnp.ndarray,
    u_hat: jnp.ndarray,
) -> float:
    tmp = jnp.argmax(u_hat[pairs], axis=1)
    pred = pairs[jnp.arange(len(pairs)), tmp]
    return float(jnp.mean(pred == choices))


# def simulate_predictions_2d(
#     pairs: jnp.ndarray,
#     choices: jnp.ndarray,
#     u_hat: jnp.ndarray,
# ) -> float:
#     # pairs has shape (n_trials, 2, 2)
#     # choices has shape (n_trials,)
#     # u_hat has shape (n_options, n_options)
#     tmp = jnp.argmax(u_hat[pairs[:, 0], pairs[:, 1]], axis=1)
#     pred = pairs[jnp.arange(len(pairs)), tmp]

#     return float(jnp.mean(pred == choices))


def simulate_predictions_2d(
    pairs: jnp.ndarray,
    choices: jnp.ndarray,
    u_hat: jnp.ndarray,
) -> float:
    correct = 0

    for i in range(len(pairs)):
        r1, c1 = pairs[i, 0]
        r2, c2 = pairs[i, 1]
        vs = [u_hat[r1, c1], u_hat[r2, c2]]
        if vs[0] > vs[1]:
            correct += int(jnp.allclose(choices[i], pairs[i, 0]))
        elif vs[1] > vs[0]:
            correct += int(jnp.allclose(choices[i], pairs[i, 1]))
        else:
            idx = random.choice(random.PRNGKey(i), jnp.array([0, 1]))
            correct += int(jnp.array_equal(choices[i], pairs[i, idx]))

    return correct / len(pairs)
