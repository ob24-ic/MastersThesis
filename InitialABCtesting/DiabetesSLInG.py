import time
import scipy
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from scipy.stats import laplace, uniform, norm, expon
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def make_distance(xobs: np.ndarray):
    """
    Build a fast distance function *pre-bound* to the observed data.

    The returned callable computes

    .. math::

        d(x_{\text{obs}}, x_{\text{sim}}) \;=\;
        \frac{\sqrt{\frac{1}{N}\sum_i (x_{\text{obs},i}-x_{\text{sim},i})^2}}
             {\operatorname{IQR}(x_{\text{obs}})}

    i.e. the root-mean-squared deviation (RMSD) between observed and
    simulated vectors, scaled by the interquartile range of the observed
    data.  Because the IQR is fixed, it is computed **once** and captured
    in the closure, making every subsequent distance call O(*N*) instead
    of O(*N* + sort).

    Parameters
    ----------
    xobs : ndarray, shape (N,) or (N, 1)
        Observed response vector.  Can be 1-D or a column vector.

    Returns
    -------
    distance : callable
        A single-argument function ``distance(xsim)`` that returns a
        float—RMSD/IQR—for any simulated vector of the same length.

    Examples
    --------
    >>> y_obs = np.array([1.2, 0.7, 3.4, 2.2])
    >>> dist  = make_distance(y_obs)
    >>> y_sim = np.array([1.0, 0.9, 3.3, 2.0])
    >>> dist(y_sim)
    """
    iqr = np.subtract(*np.percentile(xobs, [75, 25]))

    def distance(xsim: np.ndarray) -> float:
        rmsd = np.sqrt(np.mean((xobs - xsim) ** 2))
        return rmsd / iqr

    return distance



def linear_SLInG(
    features: np.ndarray,
    xobs: np.ndarray,
    n_iters: int = 10_000,
    delta_min: float = 0.01,
    delta_max: float = 10.0,
    grace_ratio: float = 0.10,
    burn_period_ratio: float = 0.20,
    sigma_initial: float = 1.0,
    lambda_rate: float = 1.0,
    logging: bool = False,
) -> np.ndarray:
    """
    Run the **Linear SLInG** (Simulation‐Based Likelihood-free Inference with
    Gibbs updates) algorithm for a standard linear forward model *x = X θ*.

    The sampler alternates between

    1. **ε–updates** (component-wise draws from an exponential proposal
       followed by MH acceptance) and
    2. **θ–updates** (random-walk Metropolis with distance-based ABC kernel),

    while slowly annealing the ABC tolerance *δ<sub>ABC</sub>* from its
    initial RMSD/IQR down to ``delta_min``.

    Parameters
    ----------
    features : (N, K) ndarray
        Design matrix ``X``.  Columns should be centred; scaling to unit
        variance is recommended for numerical stability.
    xobs : (N, 1) or (N,) ndarray
        Observed response vector ``y``.
    n_iters : int, default ``10_000``
        Total number of MCMC iterations.
    delta_min : float, default ``0.01``
        Lower bound to which the ABC tolerance *δ<sub>ABC</sub>* is
        annealed during burn-in.
    delta_max : float, default ``10``
        *Currently unused*: kept for API compatibility with earlier drafts.
    grace_ratio : float in (0, 1), default ``0.10``
        Fraction of iterations during which ε<sub>k</sub> are **not**
        updated (a “grace period” that lets θ adapt first).
    burn_period_ratio : float in (0, 1), default ``0.20``
        Length of the *δ*-annealing schedule as a fraction of
        ``n_iters``.
    sigma_initial : float, default ``1.0``
        Initial proposal standard deviation for every θ<sub>k</sub>.  After
        half the burn-in the code adaptively replaces this with the
        empirical σ of the last 100 θ-draws.
    lambda_rate : float, default ``1.0``
        Rate parameter of the exponential proposal for ε.  (Kept
        explicit so you can experiment, but the body hard-codes β = 1.)
    logging : bool, default ``False``
        If *True*, prints the per-parameter acceptance probability each
        time a θ move is proposed and a summary every 100 iterations.

    Returns
    -------
    thetas : ndarray, shape (n_iters, K)
        The full MCMC chain of regression-coefficient draws (including
        burn-in and grace period).  Discard the first
        ``burn_in = burn_period_ratio × n_iters`` rows if you want the
        post-burn sample only.

    Notes
    -----
    *Distance metric*
        ``make_distance(xobs)`` pre-computes the IQR of the observed data
        so each RMSD evaluation is O(*N*).  The kernel is a standard
        normal PDF on the scaled distance.

    *Incremental forward model*
        Because only one θ<sub>k</sub> changes in each inner loop, the
        candidate simulation ``xsim_prop`` is updated with a cheap
        rank-1 operation::

            xsim_prop = xsim + (θ*_k − θ_k) · X[:, k]

        avoiding a fresh ``X @ θ`` multiplication.

    *Numerical safeguards*
        The code clips log-ratios to ±1000 (ε-step) and ±700 (kernel step)
        to prevent overflow/underflow in ``np.exp``.

    Examples
    --------
    >>> diabetes = load_diabetes()
    >>> X        = StandardScaler().fit_transform(diabetes.data)
    >>> y        = diabetes.target.reshape(-1, 1)
    >>> chain    = linear_SLInG(X, y, n_iters=20_000, delta_min=0.015)
    >>> chain    = chain[int(0.2 * len(chain)):]   # drop 20 % burn-in
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
    """
    Quick diagnostic trace plot for a SLInG (or any MCMC) coefficient matrix.

    Parameters
    ----------
    chain : ndarray, shape (n_iters, K)
        The sampled parameter matrix.  Each column corresponds to a single
        regression coefficient; each row is one MCMC iteration **after any
        burn-in you wish to discard**.

    Notes
    -----
    • The helper grabs the diabetes-dataset variable names from the global
      ``diabetes`` object, so call it only *after* you have executed
      ``diabetes = load_diabetes()``.
    • Line width is set to 0.5 for speed and clarity on long chains.
    • The y-axis is individually labelled for every coefficient; the x-axis
      (iteration index) is shared.

    Example
    -------
    >>> chain = linear_SLInG(X_scaled, y, n_iters=15_000, delta_min=0.02)
    >>> chain = chain[int(0.2 * len(chain)):]   # drop 20 % burn-in
    >>> trace_plots(chain)
    """
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
    n_iters: int = 10_000,
    burn_in: float = 0.20,
    grace_ratio: float = 0.10,
    burn_ratio: float = 0.20,
    seed: int = 0,
):
    """
    Fit and visualise a family of SLInG posterior chains for the
    scikit-learn diabetes data (or any linear-regression design matrix).

    The helper does three things:

    1. **Standardises** every predictor column (mean 0, variance 1) before
       sampling – this usually speeds up convergence.
    2. Runs :pyfunc:`linear_SLInG` once for each value in ``delta_list``.
    3. Converts the raw θ-draws back to the “paper” scale
       by **dividing** each coefficient by the pre-scaling standard deviation
       of its predictor, then plots a forest of error bars.

    Parameters
    ----------
    features : (N, K) ndarray
        Raw design matrix ``X``. Columns are centred and scaled internally;
        the original values are only used to compute each column’s
        standard deviation ``sd_x`` for the final rescaling step.
    y_obs : (N, 1) or (N,) ndarray
        Response vector ``y`` in its original units. It is *not* scaled.
    delta_list : sequence of float, default ``(0.015, 0.02, 0.035, 0.045)``
        The ℓ₂ thresholds δₐᵦ꜀ (one per coloured chain) to compare.
    n_iters : int, default ``10 000``
        Number of MCMC iterations in each SLInG run.
    burn_in : float in (0, 1), default ``0.20``
        Fraction of samples discarded as burn-in before summarising.
    grace_ratio : float in (0, 1), default ``0.10``
        Initial fraction of iterations during which ε-adaptation is *not*
        attempted (mirrors the algorithm’s “grace period”).
    burn_ratio : float in (0, 1), default ``0.20``
        Fraction of iterations over which δₐᵦ꜀ is tapered from its initial
        value down to ``delta_min`` (the “burn-in schedule” in SLInG).
    seed : int, default ``0``
        Seed passed to NumPy’s random generator for reproducible chains.

    Returns
    -------
    summary : dict[float, dict[str, ndarray]]
        A nested dictionary keyed by each δ.  For every threshold the inner
        dict contains

        * ``"median"`` – posterior medians (shape ``(K,)``)
        * ``"low"``    – 2.5 % posterior quantiles
        * ``"high"``   – 97.5 % posterior quantiles

        All arrays are on the **rescaled** (paper) axis.

    Notes
    -----
    *Predictor rescaling*:
    After sampling on the unit-variance design matrix ``X_scaled``, every
    coefficient sample ``θ_j`` is divided by the original column standard
    deviation ``sd_x[j]``.  That matches the convention used in the
    original SLInG paper, where coefficients lie in the hundreds.

    *Plot*:
    The function draws a horizontal error-bar plot (“forest plot”) with a
    tiny vertical jitter so the four δ levels do not occlude each other.
    Colours and marker shapes follow the figure in the paper
    (blue ▲, orange ▶, green ◀, red ▼).

    Examples
    --------
    >>> diabetes = load_diabetes()
    >>> run_and_plot_sling(diabetes.data, diabetes.target[:, None],
    ...                    n_iters=50_000, seed=42)
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




