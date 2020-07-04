import numpy as np
import matplotlib.pyplot as plt
from pd import *
from sampling import *


def func(X): #p(z)
    return np.exp(-.5 * np.sum(X ** 2, axis=-1) / 5.)


sampler = Metropolis(target=func, prop=Gaussian(0, 2), dim=2)
samples = sampler.rvs(100)


X, Y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
X_test = np.array([X, Y]).reshape(2, -1).T
Z = func(X_test).reshape(X.shape)
plt.contour(X, Y, Z, levels=5)
plt.scatter(samples[:, 0], samples[:, 1])
plt.show()
