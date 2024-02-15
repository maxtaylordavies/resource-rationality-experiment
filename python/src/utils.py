import GPy
import jax.numpy as jnp
from tqdm import tqdm


def to_range(x: jnp.ndarray, min_val: float, max_val: float) -> jnp.ndarray:
    delta = jnp.max(x) - jnp.min(x)
    if delta == 0:
        return x
    x = (x - jnp.min(x)) / (delta)
    return (max_val - min_val) * x + min_val


def digitize(x: jnp.ndarray, num_bins: int) -> jnp.ndarray:
    bins = jnp.linspace(0, 1, num_bins + 1)[1:-1]
    return jnp.digitize(to_range(x, 0, 1), bins)


def avg_pool_1d(x: jnp.ndarray, pool_size: int) -> jnp.ndarray:
    if pool_size == 1:
        return x
    tmp = jnp.mean(x.reshape(-1, pool_size), axis=1)
    return to_range(jnp.repeat(tmp, pool_size), jnp.min(x), jnp.max(x))


def avg_pool_2d(x: jnp.ndarray, pool_size: int) -> jnp.ndarray:
    if pool_size == 1:
        return x

    r, c = x.shape
    r, c = r // pool_size, c // pool_size
    tmp = jnp.mean(x.reshape(r, pool_size, c, pool_size), axis=(1, 3))
    return tmp
    # return jnp.repeat(jnp.repeat(tmp, pool_size, axis=0), pool_size, axis=1)


def gp_covariance_matrix_1d(
    n_options: int,
    variance=1,
    lengthscale=1,
) -> jnp.ndarray:
    kernel = GPy.kern.RBF(1, variance=variance, lengthscale=lengthscale)
    X = jnp.arange(n_options)[:, None]
    return kernel.K(X, X)


def gp_covariance_matrix_2d(side_length: int, var=1.0, scale=1.0):
    xx, yy = jnp.mgrid[0:side_length, 0:side_length]
    X = jnp.vstack((xx.flatten(), yy.flatten())).T
    k = GPy.kern.RBF(input_dim=2, variance=var, lengthscale=scale)  # define kernel
    return k.K(X)  # compute covariance matrix


def aggregated_covariance_matrix(K, patch_size):
    # K is a covariance matrix of shape (n_options, n_options)
    # for a 1D GP with RBF kernel. we want to compute the covariance
    # matrix that would correspond to the outputs of the GP after
    # applying average pooling with patch_size.
    if patch_size == 1:
        return K
    n_patches = K.shape[0] // patch_size
    K_patches = K.reshape((n_patches, patch_size, n_patches, patch_size))
    return jnp.mean(K_patches, axis=(1, 3))


def gaussian_entropy(cov):
    sign, nat_log_abs_det = jnp.linalg.slogdet(cov)
    log_det = nat_log_abs_det / jnp.log(2)

    if sign < 0:
        raise ValueError("Covariance matrix is not positive definite")

    tmp = log_det + (cov.shape[0] * jnp.log2(2 * jnp.pi * jnp.e))
    return tmp / 2


def save_figure(fig, name):
    fig.tight_layout()
    for fmt in ["pdf", "svg"]:
        fig.savefig(f"../figures/{name}.{fmt}")
