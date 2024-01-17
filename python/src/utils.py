import jax.numpy as jnp


def to_range(x: jnp.ndarray, min_val: float, max_val: float) -> jnp.ndarray:
    x = (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))
    return (max_val - min_val) * x + min_val


def avg_pool_1d(x: jnp.ndarray, pool_size: int) -> jnp.ndarray:
    if pool_size == 1:
        return x
    tmp = jnp.mean(x.reshape(-1, pool_size), axis=1)
    return jnp.repeat(tmp, pool_size)


def boltzmann_1d(x: jnp.ndarray, beta: float) -> jnp.ndarray:
    p = jnp.exp(x / beta)
    return p / jnp.sum(p)
