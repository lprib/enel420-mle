#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
import scipy


def generate_likelihood_fn(x_values, y_values):
    def likelihood_fn(params):
        y_intercept, slope, standard_deviation = params[0], params[1], params[2]
        predicted_y = slope * x_values + y_intercept

        negative_log_likelihood = - \
            np.sum(stats.norm.logpdf(
                y_values, loc=predicted_y, scale=standard_deviation))

        return negative_log_likelihood
    return likelihood_fn


num_samples = 1000
slope = 2
x = np.linspace(0, 10, num_samples)
offsets = np.random.normal(loc=0.0, scale=3.0, size=num_samples)
y = slope*x + offsets

parameter_guess = np.array([0, 0, 0])

optimization_result = optimize.minimize(generate_likelihood_fn(
    x, y), parameter_guess, method="Nelder-Mead", options={"disp": True})

res = optimization_result["x"]
est_y_intercept, est_slope, est_standard_dev = res[0], res[1], res[2]


est_ys = est_slope*x + est_y_intercept

plt.scatter(x, y)
plt.plot(x, est_ys, "C1")
plt.show()
