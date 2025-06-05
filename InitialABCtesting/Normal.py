import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def generate_sample(N, mu, var):
    return np.sort((stats.norm.rvs(size=N,loc=mu,scale=var)))

def abcAlgorithm(true_sample, tol, number_of_samples=100, log=False):
    N = len(true_sample)
    mu_array = stats.uniform.rvs(size=number_of_samples, loc=-1, scale=2)
    sigma_array = stats.uniform.rvs(size=number_of_samples, loc=0, scale=2)
    pairs = zip(mu_array, sigma_array)
    feasible_pairs = []

    for mu, sigma in pairs:
        current_sample = generate_sample(N, mu, sigma)
        error = np.linalg.norm(true_sample - current_sample)/N
        if error < tol:
            feasible_pairs.append((mu, sigma))
            if log:
                print(f"feasible pair found {mu} and {sigma} with error {error}")
        else:
            if log:
                print(f"Rejected {mu} and {sigma} with error {error}")
    return feasible_pairs


if __name__ == '__main__':
    feasible_pairs = abcAlgorithm(generate_sample(10000, 0,1), tol=1e-3, number_of_samples=1000)
    mu_vals, sigma_vals = zip(*feasible_pairs)

    plt.figure()
    plt.hist(mu_vals, bins=20)
    plt.title("Histogram of accepted μ values")
    plt.xlabel("μ");
    plt.ylabel("Frequency")
    plt.show()

    plt.figure()
    plt.hist(sigma_vals, bins=20)
    plt.title("Histogram of accepted σ values")
    plt.xlabel("σ");
    plt.ylabel("Frequency")
    plt.show()

    plt.figure()
    plt.scatter(mu_vals, sigma_vals, alpha=0.6)
    plt.title("Accepted (μ, σ) pairs")
    plt.xlabel("μ");
    plt.ylabel("σ")
    plt.show()
