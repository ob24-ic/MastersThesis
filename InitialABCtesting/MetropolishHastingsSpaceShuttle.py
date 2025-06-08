import numpy as np
from scipy.stats import norm, expon, gaussian_kde
from numpy import exp, log
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def compute_likelihood(alpha, beta):
    """Function compute the likelihood from the data"""
    # Calculate log likelihood - for stability
    log_likelihood = 0.0
    for xi, yi in zip(x, y):
        eta = alpha + beta * xi
        pi = 1 / (1 + exp(-eta))
        # add log of Bernoulli
        log_likelihood += yi * log(pi) + (1 - yi) * log(1 - pi)

    # return the actual likelihood
    return exp(log_likelihood)

def compute_posterior(alpha, beta, b):
    """Function compute the (un-normalised) posterior"""
    # Calculate the prior
    prior = 1 / b * exp(alpha) * exp(-exp(alpha) / b)

    # Now get our likelihood
    likelihood = compute_likelihood(alpha, beta)

    posterior = prior * likelihood
    return posterior

def accept_probability(alpha, alpha_prop, beta, beta_prop, b):
    """Function compute the accept probability from the proposed alpha, beta"""

    # compute the target density ratio
    fy = compute_posterior(alpha_prop, beta_prop, b)
    fx = compute_posterior(alpha, beta, b)

    target_ratio = fy / fx

    # compute the proposal ratio
    q_rev_alpha = expon(scale=b).pdf(np.exp(alpha)) * np.exp(alpha)
    q_fwd_alpha = expon(scale=b).pdf(np.exp(alpha_prop)) * np.exp(alpha_prop)

    q_rev_beta = norm(loc=-0.2322, scale=np.sqrt(0.1082)).pdf(beta)
    q_fwd_beta = norm(loc=-0.2322, scale=np.sqrt(0.1082)).pdf(beta_prop)

    proposal_ratio = (q_rev_alpha * q_rev_beta) / (q_fwd_alpha * q_fwd_beta)

    # cant have a prob of over 1 so we take min with 1
    return min(1, target_ratio * proposal_ratio)


def run_MH(n_iters, alpha0=None, beta0=None, logging=False):
    """
    Runs the Metropolis-Hastings algorithm on this problem for a specified number of iterations.
    Can also specifiy the initial guess alpha and beta.
    """

    # first we need to find b we fit a standard logistic regression from sklearn and use the intercept M MLE as our b
    # Dont use any regularisation
    model = LogisticRegression(fit_intercept=True, penalty=None, solver='lbfgs', max_iter=1000)
    model.fit(np.reshape(x, (-1, 1)), y)

    # Extract MLE which will use as our initial values and compute b
    alpha_hat = model.intercept_[0]
    beta_hat = model.coef_[0, 0]
    b = np.exp(alpha_hat)

    if logging:
        print(f"found alpha MLE = {alpha_hat}, beta MLE = {beta_hat}, b = {b}")

    # if specify alpha and beta use those initial values else use MLE
    if alpha0 is not None:
        alpha = alpha0
    else:
        alpha = alpha_hat
    if beta0 is not None:
        beta = beta0
    else:
        beta = beta_hat

    # Define our proposal distributions which we will draw from
    alpha_proposal_distribution = expon(scale=b)
    beta_proposal_distribution = norm(-0.2322,0.1082)

    # Store the samples
    samples = [(alpha, beta)]
    for _ in range(n_iters):
        alpha_prop = log(alpha_proposal_distribution.rvs())
        beta_prop = beta_proposal_distribution.rvs()

        # Call our probability function
        accept_prob = accept_probability(alpha, alpha_prop, beta, beta_prop, b)

        # This changes with the correct probability
        if np.random.random() < accept_prob:
            alpha = alpha_prop
            beta = beta_prop

        samples.append((alpha, beta))

    return np.array(samples).T


def generate_plots(chain):
    """ For a given chain we generate the plots similar to those in the lecture notes
    The chain should be a numpy array of 2 x niters
    """

    # ensure it’s a NumPy array
    chain = np.asarray(chain)

    # check it’s 2-D
    if chain.ndim != 2:
        raise ValueError(f"`chain` must be 2-D, got ndim={chain.ndim}")

    # check the first dimension is size 2
    if chain.shape[0] != 2:
        raise ValueError(f"`chain` must have shape (2, n_iters), got {chain.shape}")

    # unpack
    alpha_samps = chain[0, :]
    beta_samps = chain[1, :]

    # Joint scatter + KDE contours
    plt.figure(figsize=(5, 5))
    plt.scatter(alpha_samps, beta_samps, s=3, alpha=0.3, color='black')

    # build a kernel density estimate over the 2D points
    xy = np.vstack([alpha_samps, beta_samps])
    kde = gaussian_kde(xy)
    a_min, a_max = alpha_samps.min(), alpha_samps.max()
    b_min, b_max = beta_samps.min(), beta_samps.max()
    A, B = np.meshgrid(np.linspace(a_min, a_max, 100),
                       np.linspace(b_min, b_max, 100))
    Z = kde(np.vstack([A.ravel(), B.ravel()])).reshape(A.shape)

    plt.contour(A, B, Z, colors='blue', linewidths=1)
    plt.xlabel('Intercept α')
    plt.ylabel('Slope β')
    plt.title('Joint posterior + KDE contours')
    plt.tight_layout()
    plt.show()

    # Marginals + running-mean traces
    burn = len(alpha_samps) // 10  # discard first 10% as burn-in
    a_post = alpha_samps[burn:]
    b_post = beta_samps[burn:]

    # compute running means
    rm_alpha = np.cumsum(a_post) / np.arange(1, len(a_post) + 1)
    rm_beta = np.cumsum(b_post) / np.arange(1, len(b_post) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6),
                             gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})

    # histogram of alpha
    axes[0, 0].hist(a_post, bins=30, density=True)
    axes[0, 0].set_title('Posterior of α')
    axes[0, 0].set_xlabel('α')
    axes[0, 0].set_ylabel('Density')

    # running mean of alpha
    axes[0, 1].plot(rm_alpha)
    axes[0, 1].set_title('Running mean of α')
    axes[0, 1].set_xlabel('Iteration (post burn-in)')
    axes[0, 1].set_ylabel('Mean α')

    # histogram of beta
    axes[1, 0].hist(b_post, bins=30, density=True)
    axes[1, 0].set_title('Posterior of β')
    axes[1, 0].set_xlabel('β')
    axes[1, 0].set_ylabel('Density')

    # running mean of beta
    axes[1, 1].plot(rm_beta)
    axes[1, 1].set_title('Running mean of β')
    axes[1, 1].set_xlabel('Iteration (post burn-in)')
    axes[1, 1].set_ylabel('Mean β')

    plt.tight_layout()
    plt.show()

    # Predictive histograms at fixed temperatures
    temps = [65, 45, 31]  # temperatures in °F
    fig, axs = plt.subplots(1, len(temps), figsize=(5 * len(temps), 4))

    for ax, xstar in zip(axs, temps):
        p_star = 1 / (1 + np.exp(-(alpha_samps + beta_samps * xstar)))
        ax.hist(p_star, bins=30, density=True)
        ax.set_title(f'Failure Prob @ {xstar}°F')
        ax.set_xlim(0, 1)
        ax.set_xlabel('p')
        ax.set_ylabel('Density')

    plt.tight_layout()
    plt.show()

    return None





if __name__ == '__main__':
    x = [53, 57, 58, 63, 66, 67, 67, 67, 68, 69, 70, 70, 70, 70, 72, 73, 75, 75, 76, 76, 78, 79, 81]
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    chain = run_MH(n_iters=2000, alpha0=None, beta0=None, logging=True)
    generate_plots(chain)