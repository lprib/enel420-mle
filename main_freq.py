#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
import scipy


def generate_likelihood_fn(x_values, y_values):
    def likelihood_fn(params):
        freq, standard_deviation = params[0], params[1]
        # predicted_y = slope * x_values + y_intercept
        predicted_y = np.sin(2*np.pi*freq*x_values)

        negative_log_likelihood = - \
            np.sum(stats.norm.logpdf(
                y_values, loc=predicted_y, scale=standard_deviation))

        return negative_log_likelihood
    return likelihood_fn


num_samples = 1000
freq = 0.1
x = np.linspace(0, 10, num_samples)
offsets = np.random.normal(loc=0.0, scale=0.2, size=num_samples)
y = np.sin(2*np.pi*freq*x) + offsets

parameter_guess = np.array([0, 0])

optimization_result = optimize.minimize(generate_likelihood_fn(
    x, y), parameter_guess, method="Nelder-Mead", options={"disp": True})

res = optimization_result["x"]
print(res)
# est_y_intercept, est_slope, est_standard_dev = res[0], res[1], res[2]
est_freq, est_std_dev = res[0], res[1]


# est_ys = est_slope*x + est_y_intercept
est_ys = np.sin(2*np.pi*est_freq*x)

plt.scatter(x, y)
plt.plot(x, est_ys, "C1")
plt.show()
