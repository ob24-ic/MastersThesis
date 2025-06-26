from __future__ import annotations
import numpy as np
import tensorflow as tf
from functools import partial


# Generic relative error code must be combined with a batch generator function

def _avg_rel_error_core(simulate_batch,
                        *,
                        model: tf.keras.Model,
                        batchsize: int,
                        no_tests: int,
                        mc_samples: int) -> float:
    """
    Generic Monte-Carlo dropout relative-error evaluator.
    `simulate_batch` must accept `batch_size=` and return (summaries, θ_true).
    """
    # Fresh synthetic test batch
    summaries, theta_true = simulate_batch(batch_size=no_tests)      # (N, ·), (N, D, 1)
    theta_true_np = np.squeeze(theta_true.numpy(), axis=-1)          # (N, D)

    # Monte-Carlo passes
    preds = [
        model.predict(summaries, batch_size=batchsize, verbose=0)
        for _ in range(mc_samples)
    ]
    theta_pred_mean = np.mean(np.stack(preds, axis=0), axis=0)       # (N, D)

    # Relative ℓ₂ error
    eps   = 1e-8
    diff  = theta_pred_mean - theta_true_np
    numer = np.linalg.norm(diff, axis=1)
    denom = np.linalg.norm(theta_true_np, axis=1) + eps
    return np.mean(numer / denom)


def make_avg_rel_error_fn(simulator,
                  *,
                  batchsize: int,
                  no_tests: int,
                  mc_samples: int):
    """Return a function that only needs `model` to evaluate error."""
    return partial(_avg_rel_error_core,
                   simulate_batch=simulator,
                   batchsize=batchsize,
                   no_tests=no_tests,
                   mc_samples=mc_samples)



""" Wrap with the simulators from MainCode/generate_data for example:

common_sim_cfg = dict(
    no_params     = 5,
    n_per_sample  = 100,
    noise_std     = 0.2
)

Create simulator functions
uniform_sim = make_uniform_simulator(theta_range=(-1., 1.), **common_sim_cfg)
laplace_sim = make_laplace_simulator(loc=0., scale=1., **common_sim_cfg)
spike_sim   = make_spike_simulator(loc=5., scale=1., zero_prob=0.3, **common_sim_cfg)


Wrap error evaluators

uniform_error = make_avg_rel_error_fn(
    uniform_sim,
    batchsize  = 256,
    no_tests   = 2_000,
    mc_samples = 30
)

laplace_error = make_avg_rel_error_fn(
    laplace_sim,
    batchsize  = 256,
    no_tests   = 2_000,
    mc_samples = 30
)

spike_error = make_avg_rel_error_fn(
    spike_sim,
    batchsize  = 256,
    no_tests   = 2_000,
    mc_samples = 30
)

model = ...  # trained tf.keras.Model

print(f"Uniform test error : {uniform_error(model=model):.4f}")
print(f"Laplace test error : {laplace_error(model=model):.4f}")
print(f"Spike-and-slab err.: {spike_error(model=model):.4f}")
"""