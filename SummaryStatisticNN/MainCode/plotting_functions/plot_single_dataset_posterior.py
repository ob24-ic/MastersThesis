import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from functools import partial
from typing import Callable, Tuple

Tensor = tf.Tensor


# Generic single data set plot code must be combined with a batch generator function

def _plot_posteriors_core(simulate_batch: Callable[..., Tuple[Tensor, ...]],
                          *,
                          model: tf.keras.Model,
                          no_params: int,
                          mc_samples: int,
                          bins: int = 30,
                          show_mc_stats: bool = True,
                          xlim: Tuple[float, float] = (-5., 10.)) -> None:
    """
    1. Draw one synthetic data set with `simulate_batch(batch_size=1)`.
       The callable must return at least (summaries, θ_true, ...).
    2. Do `mc_samples` Monte-Carlo dropout forward passes.
    3. Plot per-dimension histograms, plus optional MC mean/std indicators.
    """

    # Simulate 1 dataset
    summaries, theta_true, *_ = simulate_batch(batch_size=1)
    theta_true = theta_true.numpy().squeeze(-1).flatten()          # (D,)

    # Monte-Carlo passes
    preds = np.stack(
        [model.predict(summaries, verbose=0)[0] for _ in range(mc_samples)],
        axis=0
    )                                                              # (K, D)
    mc_mean = preds.mean(axis=0)
    mc_std  = preds.std (axis=0)

    # Figure layout
    n_cols  = 5
    n_rows  = int(np.ceil(no_params / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3, n_rows * 3.2),
        squeeze=False
    )

    xmin, xmax = xlim

    # loop over each θ
    for j, ax in enumerate(axes.flat):
        if j >= no_params:
            ax.axis("off")
            continue

        # histogram of MC-dropout draws
        ax.hist(preds[:, j],
                bins=bins,
                color="lightsteelblue",
                edgecolor="k",
                alpha=0.75,
                density=True)

        # true θ
        ax.axvline(theta_true[j], color="crimson",
                   linewidth=2, label=r"$\theta_{\text{true}}$")

        if show_mc_stats:
            # mean (dashed) and ±1σ band
            ax.axvline(mc_mean[j], color="k", linestyle="--",
                       linewidth=2, label="MC mean")
            ax.axvspan(mc_mean[j] - mc_std[j],
                       mc_mean[j] + mc_std[j],
                       color="grey", alpha=0.25, label="±1 σ")

        # aesthetics
        ax.set_xlim(xmin, xmax)
        ax.set_title(f"θ{j + 1}")
        if j % n_cols == 0:
            ax.set_ylabel("density")
        ax.grid(alpha=.3, linestyle="--")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.show()

    return None



# Wrapper so that only have to pass in the model in which we want ot return the results for

def make_posterior_plotter(simulate_batch, *,
                           no_params: int,
                           **engine_defaults):
    """
    Pre-bind the simulator and common hyper-params, returning a function that
    only needs the trained Keras `model`
    """
    return partial(_plot_posteriors_core,
                   simulate_batch,
                   no_params=no_params,
                   **engine_defaults)

# ─────────────────────────────────────────────────────────────────────────────
""" Example usage:

common_sim_cfg = dict(
    no_params     = 5,
    n_per_sample  = 100,
    noise_std     = 0.2
)

uniform_sim = make_uniform_simulator(theta_range=(-1., 1.), **common_sim_cfg)
laplace_sim  = make_laplace_simulator(loc=0.0, scale=1.0, **common_sim_cfg)
spike_sim   = make_spike_simulator(loc=5., scale=1., zero_prob=0.3, **common_sim_cfg)


uniform_plot = make_posterior_plotter(
    simulate_batch=uniform_sim,
    no_params=5,
    mc_samples=100,
    bins=40
)

laplace_plot = make_posterior_plotter(
    simulate_batch=laplace_sim,   
    no_params=5,
    mc_samples=100,
    bins=40,
    show_mc_stats=True                   
)

spike_plot = make_posterior_plotter(
    simulate_batch=spike_sim,   
    no_params=5,
    mc_samples=100,
    bins=40,
    show_mc_stats=True                   
)

Call them whenever you need a plot:
uniform_plot(model=my_trained_net)
laplace_plot(model=my_trained_net)
spike_plot  (model=my_trained_net)
"""
