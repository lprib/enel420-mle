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
xs = np.random.normal(loc=2.0, scale=0.5, size=num_samples)
ys = np.random.normal(loc=5.0, scale=1.5, size=num_samples)

plt.axis((0, 10, 0, 10))
plt.scatter(xs, ys)
plt.grid()
plt.show()
