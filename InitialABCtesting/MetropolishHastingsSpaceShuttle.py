import numpy as np
from scipy.stats import norm, expon
from numpy import exp, log
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

x = [53, 57, 58, 63, 66, 67, 67, 67, 68, 69, 70, 70, 70, 70, 72, 73, 75, 75, 76, 76, 78, 79, 81]
y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]


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

    return np.array(samples)


chain = run_MH(n_iters=2000, alpha0=0, beta0=0, logging=True)

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
ax1.plot(chain[:,0]); ax1.set_ylabel('alpha')
ax2.plot(chain[:,1]); ax2.set_ylabel('beta'); ax2.set_xlabel('iter')
plt.show()



