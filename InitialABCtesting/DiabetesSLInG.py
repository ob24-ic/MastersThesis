import scipy
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from scipy.stats import laplace, uniform, norm, expon
import matplotlib.pyplot as plt

def distance(xobs, xsim):
    """Function to calculate the distance between observations and simulated observations given by RMSD / IQR"""
    # Get number of observations
    J = xobs.shape[0]

    # Compute the root mean squared distance between observed and simulated
    rmsd = np.sqrt(np.sum((xobs - xsim) ** 2) / J)

    # Calculate the IQR
    xsim_sorted = np.sort(xobs,axis=0)
    IQR = xsim_sorted[int(3*J/4),0] - xsim_sorted[int(J/4),0]
    return rmsd / IQR

def simulate_model(features, thetas):
    """returns xsim as a column vector"""
    return (features @ thetas).reshape(-1, 1)

def linear_SLInG(features, xobs, n_iters = 500, delta_abc = 0.01, delta_min = 0.01, delta_max=10, grace_period_ratio=0.05, burn_period_ratio=0.1, sigma_initial = 1, lambda_rate=1, logging=False):
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
    epsilons = np.zeros((n_iters, K))
    thetas   = np.zeros((n_iters, K))
    epsilons[0] = np.ones(K) #uniform(loc=0, scale=1).rvs(size=K)
    thetas[0] = laplace(loc=0, scale=epsilons[0]).rvs(size=K)
    sigmas = sigma_initial * np.ones(K)

    # Generate model
    xsim = simulate_model(features, thetas[0])

    # Compute dist the initial distance
    dist = distance(xobs, xsim)

    # delta_abc = max(delta_min, dist / delta_max)

    # main loop
    for i in range(1, n_iters):
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
                #   posterior ∝ Laplace(θ_k; scale=ε) * Expon(ε; rate)
                num = (laplace(scale=eps_prop).pdf(thetai[k]) *
                       rate * np.exp(-rate * eps_prop))
                den = (laplace(scale=epsi[k]).pdf(thetai[k]) *
                       rate * np.exp(-rate * epsi[k]))
                alpha_eps = min(1, num / den)

                if np.random.rand() < alpha_eps:
                    epsi[k] = eps_prop
                # else keep old epsi[k]

        # loop over each parameter again in the random order
        for k in perm:
            if i > burn_period / 2:
                # compute the empirical variance of previous theta_k’s
                past = thetas[:i, k]
                var = np.var(past)

                # If 0 then dont update
                if var > 1e-8:
                    sigmas[k] = var

            # Sample the proposed theta from the (normal) proposal distribution, q, centered at prev theta, variance simga
            q = norm(loc=thetai[k], scale=sigmas[k])
            theta_prop_k = q.rvs()
            theta_prop = thetai.copy()
            theta_prop[k] = theta_prop_k

            # Simulate the model
            xsim_prop = simulate_model(features, theta_prop)

            # Compute distance
            dist_prop = distance(xobs, xsim_prop)

            # Compute the components of the accept probability

            # First the Kernel Ratio
            kernel_ratio = std_normal.pdf(dist_prop/delta_abc) / std_normal.pdf(dist/delta_abc)

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
            else:
                thetas[i, k] = thetas[i-1, k]

    return thetas



if __name__ == '__main__':
    # Load the dataset
    diabetes = load_diabetes()
    target = np.array(diabetes.target).reshape(-1, 1)
    features_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    features_matrix = np.array(diabetes.data)

    # Run the sling algorithm
    theta_final = linear_SLInG(features_matrix, target, n_iters=10)

    # Plot
    # 1) sample for each δ
    all_summaries = []
    for δ in [0.015, 0.02, 0.035, 0.045]:
        chain = linear_SLInG(features_matrix, target,  delta_abc = δ)
        post = chain[int(0.2 * len(chain)):]  # drop 20%
        means = post.mean(0)
        low = np.percentile(post, 2.5, 0)
        high = np.percentile(post, 97.5, 0)
        all_summaries.append((δ, means, low, high))

    # 2) standard deviations of original features
    sds = features_matrix.std(axis=0, ddof=1)

    # 3) plotting

    var_names = diabetes.feature_names  # length K
    K = len(var_names)
    y_pos = np.arange(K)

    plt.figure(figsize=(6, 4))
    colors = ['C0', 'C1', 'C2', 'C3']
    markers = ['^', 's', 'o', 'd']

    for (δ, means, low, hi), c, m in zip(all_summaries, colors, markers):
        # standardize if needed:
        means_s = means * sds
        low_s = low * sds
        hi_s = hi * sds

        # plot interval bars
        plt.errorbar(means_s, y_pos,
                     xerr=[means_s - low_s, hi_s - means_s],
                     fmt=m, color=c, label=f'δₐᵦ꜀={δ}',
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
