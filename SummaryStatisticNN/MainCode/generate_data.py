from __future__ import annotations
import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial

tfd = tfp.distributions


# Function which takes a data set X and output variables y to create a summary statistic

def default_summary_fn(X, y) -> tf.Tensor:
    """Compute summary statistics from (X, y)."""
    batch_size, n_per_sample, no_params = X.shape
    mean_x = tf.reduce_mean(X, axis=1)  # (B, D)
    mean_y = tf.reduce_mean(y, axis=1)  # (B, 1)

    xc = X - mean_x[:, None, :]
    yc = y - mean_y[:, None, :]

    cov_x = tf.matmul(xc, xc, transpose_a=True) / n_per_sample  # (B, D, D)
    cov_xy = tf.matmul(xc, yc, transpose_a=True) / n_per_sample  # (B, D, 1)

    return tf.concat(
        [mean_x,
         mean_y,
         tf.reshape(cov_x, [-1, no_params * no_params]),
         tf.reshape(cov_xy, [-1, no_params])],
        axis=-1
    )


# Unspecific code that reduces repeatability must be combined with a distribution

def _simulate_core(theta_sampler,
                   summary_fn,
                   *,
                   no_params: int,
                   batch_size: int,
                   n_per_sample: int,
                   noise_std: float) -> tuple[tf.Tensor, tf.Tensor]:
    """
    θ ~ supplied sampler   (B, D, 1)
    X ~ U(0,1)             (B, n, D)
    Returns (summary, θ)   (B, summary vector length), (B, D, 1)
    """
    theta = theta_sampler([batch_size, no_params, 1])
    X = tf.random.uniform([batch_size, n_per_sample, no_params])
    noise = noise_std * tf.random.normal([batch_size, n_per_sample, 1])
    y = tf.matmul(X, theta) + noise

    summary = summary_fn(X, y)  # Use passed-in summary logic
    return summary, theta


# Distribution functions, just follow same style to reproduce new

def uniform_theta(theta_range: tuple[float, float]):
    return lambda shape: tf.random.uniform(shape, *theta_range)


def laplace_theta(loc: float, scale: float):
    dist = tfd.Laplace(loc, scale)
    return dist.sample


def spike_theta(loc: float, scale: float, zero_prob: float):
    dist = tfd.Normal(loc, scale)

    def _sampler(shape):
        θ = dist.sample(shape)
        mask = tf.cast(tf.random.uniform(shape) > zero_prob, θ.dtype)
        return θ * mask

    return _sampler


# Functions to create the full simulator for each corresponding distribution

def make_uniform_simulator(theta_range, *, summary_fn=default_summary_fn, **kwargs):
    return partial(_simulate_core, uniform_theta(theta_range), summary_fn, **kwargs)


def make_laplace_simulator(loc, scale, *, summary_fn=default_summary_fn, **kwargs):
    return partial(_simulate_core, laplace_theta(loc, scale), summary_fn, **kwargs)


def make_spike_simulator(loc, scale, zero_prob, *, summary_fn=default_summary_fn, **kwargs):
    return partial(_simulate_core, spike_theta(loc, scale, zero_prob), summary_fn, **kwargs)


# Function to create the data set, must pass in the desired data simulator

def make_dataset(n_examples: int,
                 simulator,  # any callable returning (summary, θ)
                 batch_size: int) -> tf.data.Dataset:
    """Emit un-batched (summary, θ) pairs."""
    steps = n_examples // batch_size
    ds = tf.data.Dataset.range(steps) \
        .map(lambda _: simulator(), num_parallel_calls=tf.data.AUTOTUNE) \
        .unbatch() \
        .prefetch(tf.data.AUTOTUNE)
    return ds


"""Example usage

# Common hyper-params
common_sim_cfg = dict(
    no_params     = 5,
    n_per_sample  = 100,
    noise_std     = 0.2,
    summary_fn  = my_custom_summary
)

uniform_sim  = make_uniform_simulator(theta_range=(-1.0, 1.0), **common_sim_cfg)
laplace_sim  = make_laplace_simulator(loc=0.0, scale=1.0, **common_sim_cfg)
spike_sim    = make_spike_simulator(loc=5.0, scale=1.0, zero_prob=0.3, **common_sim_cfg)

uniform_ds = make_dataset(100_000, uniform_sim,  common_sim_cfg['batch_size'])
laplace_ds = make_dataset(100_000, laplace_sim,  common_sim_cfg['batch_size'])
spike_ds   = make_dataset(100_000, spike_sim,    common_sim_cfg['batch_size'])


Note:

To add a new data generator
1.  Define the distribution (or sampler) function.
2.  Wrap it with a `make_*_simulator`-style helper, passing your `summary_fn`
    if you need a different set of summary statistics.
"""
