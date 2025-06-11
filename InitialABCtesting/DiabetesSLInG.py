import time
import scipy
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from scipy.stats import laplace, uniform, norm, expon
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def make_distance(xobs: np.ndarray):
    """Return a distance function bound to xobs."""
    iqr = np.subtract(*np.percentile(xobs, [75, 25]))

    def distance(xsim: np.ndarray) -> float:
        rmsd = np.sqrt(np.mean((xobs - xsim) ** 2))
        return rmsd / iqr

    return distance


def linear_SLInG(features, xobs, n_iters = 10000, delta_min = 0.01, delta_max=10, grace_ratio=0.1, burn_period_ratio=0.2, sigma_initial = 1, lambda_rate=1, logging=False):
    """
    Linear SLInG based on algorithm from notes
    :param features:
    :param xobs:
    :param n_iters:
    :param delta_abc:
    :param grace_ratio:
    :param burn_period_ratio:
    :param sigma_initial:
    :return:
    """
    # Determine number of iters for grace period and burn in phase
    grace_period = int(grace_ratio * n_iters)
    burn_period = int(burn_period_ratio * n_iters)

    # Initialise a standard normal distribution for our kernel
    std_normal = norm(loc=0, scale=1)

    # Get number of parameters to search for
    K = features.shape[1]

    # Initialise epsilon and theta and sigmas (squared)
    epsilons = np.zeros((n_iters, K),dtype=np.float64)
    thetas   = np.zeros((n_iters, K),dtype=np.float64)
    epsilons[0] = np.ones(K,dtype=float) #uniform(loc=0, scale=1).rvs(size=K)
    thetas[0] = laplace(loc=0, scale=epsilons[0]).rvs(size=K)
    sigmas = sigma_initial * np.ones(K,dtype=float)

    # Generate model
    xsim = (features @ thetas[0]).reshape(-1,1)

    # Initialise distance function
    distance = make_distance(xobs)

    # Compute dist the initial distance
    dist = distance(xsim)

    delta_abc = max(delta_min, dist)

    # We will store the number of accepted chains
    accepted = 0

    # Linearly decrease delta_abc to the min value within the burn period
    increment = (delta_abc - delta_min) / burn_period

    # main loop
    for i in range(1, n_iters):
        # Update the delta_abc if in the burn period
        if i < burn_period:
            delta_abc = max(delta_min, delta_abc - increment)

        # copy forward previous values by default
        eps_current = epsilons[i-1].copy()
        theta_current = thetas[i-1].copy()

        # shuffle order of updates - we dont alter the order that theyre stored just that theyre accessed
        perm = np.random.permutation(K)

        # Only update epsilons if after the grace period
        if i > grace_period:
            for k in perm:
                # Need to sample epsilon k from the conditional posterior - we do so from the exponential distribution
                eps_prop = np.random.exponential(scale=1.0)

                # We will make use of the log-pdf for numerical stability
                inv_eps_prop = 1.0 / eps_prop
                inv_eps_old = 1.0 / eps_current[k]

                # Laplace log-pdf:  −log(2ε) − |θ|/ε
                log_laplace_new = -np.log(2 * eps_prop) - np.abs(theta_current[k]) * inv_eps_prop
                log_laplace_old = -np.log(2 * eps_current[k]) - np.abs(theta_current[k]) * inv_eps_old

                # Exponential log-pdf:  −log(β) − ε/β  here β = 1/rate = 1.0
                log_expon_new = -eps_prop
                log_expon_old = -eps_current[k]

                log_num = log_laplace_new + log_expon_new
                log_den = log_laplace_old + log_expon_old

                # Form the log‐ratio and clip it - for numerical stability
                log_alpha = log_num - log_den
                log_alpha = np.clip(log_alpha, -1000, 1000)

                # Back to normal space
                alpha_eps = np.exp(log_alpha)
                alpha_eps = min(1.0, alpha_eps)

                # Simulate random number to determine whether we accept otherwise remains same
                if np.random.rand() < alpha_eps:
                    # Prevent epsilon from getting too small
                    eps_current[k] = max(1e-5,eps_prop)


        # Now we loop over the theta's
        for k in perm:
            if i > burn_period / 2:
                # compute the empirical variance of previous theta_k’s - will only use the past 100 values
                past = thetas[max(0,i-100):i, k]
                var = np.var(past)
                # If 0 then dont update
                if var > 1e-8:
                    sigmas[k] = np.sqrt(var)

            # Sample the proposed theta from the (normal) proposal distribution, q, centered at prev theta, variance simga
            theta_prop_k = np.random.normal(loc=theta_current[k], scale=sigmas[k])
            # reject auto if its outside -1000 1000
            if np.abs(theta_prop_k) > 1000:
                thetas[i, k] = thetas[i - 1, k]
                continue

            theta_prop = theta_current.copy()
            theta_prop[k] = theta_prop_k

            # Incremental model update - for efficiency
            delta = theta_prop_k - theta_current[k]
            xsim_prop = xsim + delta * features[:, k:k + 1]

            # Compute distance
            dist_prop = distance(xsim_prop)

            # Compute the components of the accept probability

            # First the Kernel Ratio
            # the exponent difference:
            log_k = std_normal.logpdf(dist_prop / delta_abc) - std_normal.logpdf(dist / delta_abc)

            # clip it into a safe range so exp() never overflows/underflows completely
            log_k = np.clip(log_k, -700, +700)
            kernel_ratio = np.exp(log_k)

            # Next prior
            inv_eps = 1.0 / eps_current[k]
            log_prior_new = -np.log(2 * eps_current[k]) - np.abs(theta_prop_k) * inv_eps
            log_prior_old = -np.log(2 * eps_current[k]) - np.abs(theta_current[k]) * inv_eps
            prior_ratio = np.exp(log_prior_new - log_prior_old)

            # Can ignore proposal ratio q due to symmetry

            # Calculate alpha ensuring doesnt exceed 1
            alpha = min(1, kernel_ratio * prior_ratio)
            if logging:
                print(f"k = {k} alpha = {alpha}")
            # Draw random number, if greater than alpha then we accept. Equivalent to a bernoulli trial.
            if np.random.random() < alpha:
                thetas[i, k] = theta_prop_k
                xsim = xsim_prop.copy()
                dist = dist_prop
                theta_current = theta_prop.copy()
                accepted += 1
            else:
                thetas[i, k] = thetas[i-1, k]
        if i%100 == 0:
            print(f"iteration {i} accepted % : {accepted / i}")
        epsilons[i] = eps_current
    return thetas

def trace_plots(chain):
    var_names = diabetes.feature_names
    n_iters, K = chain.shape

    fig, axes = plt.subplots(K, 1, figsize=(12, 2 * K), sharex=True)
    for k in range(K):
        axes[k].plot(chain[:, k], lw=0.5)
        axes[k].set_ylabel(var_names[k])
    axes[-1].set_xlabel("Iteration")
    fig.suptitle("Trace plots of SLInG chains", y=1.02)
    plt.tight_layout()
    plt.show()


def run_and_plot_sling(
    features: np.ndarray,
    y_obs: np.ndarray,
    *,
    delta_list=(0.015, 0.02, 0.035, 0.045),
    n_iters=10_000,
    burn_in=0.20,
    grace_ratio=0.10,
    burn_ratio=0.20,
    seed=0,
):
    """
    Always-standardise-X version.
    Runs `linear_SLInG` for each δ_min and plots the error-bar forest.

    Parameters
    ----------
    features : ndarray (N × K)  – raw predictors (not yet scaled)
    y_obs    : ndarray (N × 1)  – response
    Other args: see docstring in earlier version.
    """
    rng = np.random.default_rng(seed)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)   # each column: mean 0, sd 1
    sd_x     = scaler.scale_                   # original σ_x  (length-K)


    colours  = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    markers  = ["^", ">", "<", "v"]
    offsets  = np.linspace(-0.25, 0.25, len(delta_list))

    var_names = diabetes.feature_names         # K = 10 here
    y_pos     = np.arange(len(var_names))

    plt.figure(figsize=(7, 4))
    summary = {}

    # loop over δ
    for δ, dx, col, mk in zip(delta_list, offsets, colours, markers):
        chain = linear_SLInG(
            X_scaled, y_obs,
            n_iters=n_iters,
            delta_min=δ,
            grace_ratio=grace_ratio,
            burn_period_ratio=burn_ratio,
            logging=False,
        )

        chain = chain[int(burn_in * len(chain)):]        # drop burn-in
        chain_std = chain / sd_x               # rescale for plot

        med  = np.median(chain_std, axis=0)
        low  = np.percentile(chain_std,  2.5, axis=0)
        high = np.percentile(chain_std, 97.5, axis=0)
        xerr = np.vstack([med - low, high - med])

        summary[δ] = {"median": med, "low": low, "high": high}

        plt.errorbar(
            med, y_pos + dx,
            xerr=xerr,
            fmt=mk, color=col,
            capsize=4, markersize=5, linestyle="none",
            label=fr"$\delta_{{ABC}} = {δ}$",
        )

    # styling
    plt.axvline(0, color="k", lw=1, ls="--")
    plt.yticks(y_pos, var_names)
    plt.xlabel("Standardized coefficients")
    plt.ylim(-1, len(var_names))
    plt.legend(loc="upper right", frameon=False)
    plt.title("SLInG posterior for diabetes data")
    plt.tight_layout()
    plt.show()

    return summary


if __name__ == "__main__":
    diabetes = load_diabetes()
    X_raw    = diabetes.data
    y_raw    = diabetes.target.reshape(-1, 1)

    # run trace plot on one δ for illustration
    chain = linear_SLInG(StandardScaler().fit_transform(X_raw), y_raw, delta_min=0.015)
    # discard burn in period
    chain = chain[int(0.2 * len(chain)):]
    trace_plots(chain)

    # reproduce the multi-δ figure
    run_and_plot_sling(X_raw, y_raw, n_iters=50000)




