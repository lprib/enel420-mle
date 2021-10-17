#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats, optimize
import scipy


def generate_likelihood_fn(x_values):
    def likelihood_fn(params):
        mean, standard_deviation = params[0], params[1]

        # Negative here so that when opimizer minimizes, it actually finds the maximum
        negative_log_likelihood = - \
            np.sum(stats.norm.logpdf(x_values, loc=mean, scale=standard_deviation))

        return negative_log_likelihood
    return likelihood_fn


num_samples = 1000
xs = np.random.normal(loc=7.0, scale=1.5, size=num_samples)
ys = np.random.normal(loc=5.0, scale=0.2, size=num_samples)

guess_x = np.array([0, 0])
guess_y = np.array([0, 0])

xres = optimize.minimize(generate_likelihood_fn(xs), guess_x, method="Nelder-Mead", options={"disp": True})["x"]
yres = optimize.minimize(generate_likelihood_fn(ys), guess_y, method="Nelder-Mead", options={"disp": True})["x"]
# optimization_result = optimize.minimize(generate_likelihood_fn(
#     x, y), parameter_guess, method="Nelder-Mead", options={"disp": True})

print(xres, yres)

ax = plt.gca()

# draw ellipse at (meanx, meany) with size (stddevx, stddevy)
ellipse = Ellipse(xy=(xres[0], yres[0]), width=xres[1], height=yres[1], lw=5, edgecolor="C1", fc="None")
ax.add_patch(ellipse)

plt.axis((0, 10, 0, 10))
plt.scatter(xs, ys)
plt.grid()
plt.show()
