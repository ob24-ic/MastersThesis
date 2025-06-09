import scipy
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
from scipy.stats import laplace, uniform, norm, expon

def distance(xobs, xsim):
    """Function to calculate the distance between observations and simulated observations given by RMSD / IQR"""
    # Get number of observations
    J = xobs.shape[0]

    # Compute the root mean squared distance between observed and simulated
    rmsd = np.sqrt(np.sum((xobs - xsim) ** 2) / J)

    # Calculate the IQR
    xobs_sorted = np.sort(xobs,axis=0)
    IQR = xobs_sorted[int(3*J/4),0] - xobs_sorted[int(J/4),0]
    return rmsd / IQR

def simulate_model(features, thetas):
    """returns xsim as a column vector"""
    return (features @ thetas).reshape(-1, 1)

def linear_SLInG(features, xobs, n_iters = 10, delta_abc = 0.01, delta_min = 0.01, delta_max=10, grace_period_ratio=0.05, burn_period_ratio=0.1, sigma_initial = 1, lambda_rate=1, logging=False):
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
    epsilons[0] = uniform(loc=0, scale=1).rvs(size=K)
    thetas[0] = laplace(loc=0, scale=epsilons[0]).rvs(size=K)
    sigmas = sigma_initial * np.ones(K)

    # Generate model
    xsim = simulate_model(features, thetas[0])

    # Compute dist the initial distance
    dist = distance(xobs, xsim)

    delta_abc = max(delta_min, dist / delta_max)

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
    theta_final = linear_SLInG(features_matrix, target, n_iters=1000)

    # Plot

