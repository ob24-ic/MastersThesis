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


def simulate_model(features, thetas):
    """returns xsim as a column vector"""
    return (features @ thetas).reshape(-1,1)

def linear_SLInG(features, xobs, n_iters = 10000, delta_abc = 0.01, delta_min = 0.01, delta_max=10, grace_period_ratio=0.1, burn_period_ratio=0.2, sigma_initial = 1, lambda_rate=1, logging=False):
    """
    Linear SLInG based on algorithm from notes
    :param features:
    :param xobs:
    :param n_iters:
    :param delta_abc:
    :param grace_period_ratio:
    :param burn_period_ratio:
    :param sigma_initial:
    :return:
    """
    # Determine number of iters for grace period and burn in phase
    grace_period = int(grace_period_ratio * n_iters)
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
    xsim = simulate_model(features, thetas[0])

    # Initialise distance function
    distance = make_distance(xobs)

    # Compute dist the initial distance
    dist = distance(xsim)

    delta_abc = max(delta_min, dist)

    accepted = 0
    increment = (delta_abc - delta_min) / burn_period
    # main loop
    for i in range(1, n_iters):
        if i < burn_period:
            delta_abc = max(delta_min, delta_abc - increment)
        if logging:
            print("Iteration %d" % i)
        # copy forward previous values by default
        epsi = epsilons[i-1].copy()
        thetai = thetas[i-1].copy()

        # shuffle order of updates - we dont alter the order that theyre stored just that theyre accessed
        perm = np.random.permutation(K)

        # Only update epsilons if after the grace period
        if i > grace_period:
            for k in perm:
                # Need to sample epsilon k from the conditional posterior
                # propose ε* ~ Exponential(rate=λ)  (independent MH)
                rate = 1.0
                eps_prop = expon(scale=1 / rate).rvs()

                # acceptance ratio for ε_k:
                # 1) compute log‐numerator and log‐denominator
                log_num = (
                        laplace.logpdf(thetai[k], scale=eps_prop)
                        + expon.logpdf(eps_prop, scale=1 / rate)
                )
                log_den = (
                        laplace.logpdf(thetai[k], scale=epsi[k])
                        + expon.logpdf(epsi[k], scale=1 / rate)
                )

                # 2) form the log‐ratio and clip it
                log_alpha = log_num - log_den
                log_alpha = np.clip(log_alpha, -1000, 1000)

                # 3) back to normal space
                alpha_eps = np.exp(log_alpha)
                alpha_eps = min(1.0, alpha_eps)

                if np.random.rand() < alpha_eps:
                    epsi[k] = max(1e-5,eps_prop)
                # else keep old epsi[k]

        # loop over each parameter again in the random order
        for k in perm:
            if i > burn_period / 2:
                # compute the empirical variance of previous theta_k’s
                past = thetas[max(0,i-100):i, k]
                var = np.var(past)
                # If 0 then dont update
                if var > 1e-8:
                    sigmas[k] = np.sqrt(var)

            # Sample the proposed theta from the (normal) proposal distribution, q, centered at prev theta, variance simga
            q = norm(loc=thetai[k], scale=sigmas[k])
            theta_prop_k = q.rvs()
            # reject auto if its outside -1000 1000
            if np.abs(theta_prop_k) > 1000:
                thetas[i, k] = thetas[i - 1, k]
                continue

            theta_prop = thetai.copy()
            theta_prop[k] = theta_prop_k

            # Simulate the model
            xsim_prop = simulate_model(features, theta_prop)

            # Compute distance
            dist_prop = distance(xsim_prop)

            # Compute the components of the accept probability

            # First the Kernel Ratio
            # the exponent difference:
            log_k = std_normal.logpdf(dist_prop / delta_abc) \
                    - std_normal.logpdf(dist / delta_abc)

            # clip it into a safe range so exp() never overflows/underflows completely
            log_k = np.clip(log_k, -700, +700)

            kernel_ratio = np.exp(log_k)
            # Next prior
            prior_ratio = laplace(loc=0, scale=epsi[k]).pdf(theta_prop_k) / laplace(loc=0, scale=epsi[k]).pdf(thetai[k])

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
                thetai = theta_prop.copy()
                accepted += 1
            else:
                thetas[i, k] = thetas[i-1, k]
        if i%100 == 0:
            print(f"iteration {i} accepted % : {accepted / i}")
        epsilons[i] = epsi
    return thetas



if __name__ == '__main__':
    # Load the dataset
    diabetes = load_diabetes()
    target = np.array(diabetes.target).reshape(-1, 1)
    features_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    features_matrix = np.array(diabetes.data)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_matrix)

    # Run the sling algorithm
    t1 = time.time()
    chain = linear_SLInG(features_scaled, target, delta_min=0.015)
    print("time taken:", time.time() - t1)

    # Plot

    # First we drop first 20%
    chain = chain[int(0.2*len(chain)):]

    # Get desired stats for the plot
    median = np.median(chain, axis=0)
    low = np.percentile(chain, 2.5, 0)
    high = np.percentile(chain, 97.5, 0)

    # standard deviations of original features
    # sds = features_matrix.std(axis=0, ddof=1)

    var_names = diabetes.feature_names
    K = len(var_names)
    y_pos = np.arange(K)

    plt.figure(figsize=(6, 4))

    # sd_x = features_matrix.std(axis=0, ddof=1)  # length-K
    # sd_y = target.std(ddof=1)  # scalar

    # median_s = median * sd_x / sd_y
    # low_s = low * sd_x / sd_y
    # high_s = high * sd_x / sd_y

    xerr = np.vstack([median - low,
                      high - median])

    plt.errorbar(median, y_pos,
                 xerr=xerr,
                 label=f'δₐᵦ꜀={0.015}',
                 capsize=4, markersize=5, linestyle='none')

    # vertical line at zero
    plt.axvline(0, color='k', lw=1, linestyle='--')

    plt.yticks(y_pos, var_names)
    plt.xlabel("Standardized coefficients")
    plt.ylim(-1, K)
    plt.legend(loc='upper right', frameon=False)
    plt.title("SLInG posterior for diabetes data")
    plt.tight_layout()
    plt.show()

    import numpy as np
    import matplotlib.pyplot as plt

    # assume
    #   chain      # your numpy array of shape (n_iters, K)
    #   var_names  # list of length K with your parameter names

    n_iters, K = chain.shape

    # 1) Trace plots
    fig, axes = plt.subplots(K, 1, figsize=(12, 2 * K), sharex=True)
    for k in range(K):
        axes[k].plot(chain[:, k], lw=0.5)
        axes[k].set_ylabel(var_names[k])
    axes[-1].set_xlabel("Iteration")
    fig.suptitle("Trace plots of SLInG chains", y=1.02)
    plt.tight_layout()
    plt.show()


    # 2) A simple ESS estimate via integrated autocorrelation time
    def autocorr(x):
        """Return autocorrelation of 1D array x."""
        x = x - np.mean(x)
        n = len(x)
        f = np.fft.fft(x, n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[:n].real
        return acf / acf[0]


    def integrated_autocorr_time(x, max_lag=None):
        ac = autocorr(x)
        if max_lag is None:
            max_lag = len(x) // 2
        # sum until the first negative drop (Geyer’s rule-of-thumb)
        positive = ac[1:max_lag][ac[1:max_lag] > 0]
        return 1 + 2 * np.sum(positive)


    ess = np.empty(K)
    for k in range(K):
        tau = integrated_autocorr_time(chain[:, k])
        ess[k] = n_iters / tau

    print("Parameter   ESS   (out of {:,} samples)".format(n_iters))
    for name, e in zip(var_names, ess):
        print(f"{name:>5s}    {e:7.1f}")
