import numpy as np
import matplotlib.pyplot as plt
from pd import *
from sampling import *


def func(X): #p(z)
    return np.exp(-.5 * np.sum(X ** 2, axis=-1) / 5.)


sampler = Metropolis(target=func, prop=Gaussian(0., 2.))
samples = sampler.rvs(100)


x = np.linspace(-10, 10, 100)[:, None]
y = func(x) / np.sqrt(2 * np.pi * 5.)
plt.plot(x, y, label="probability density function", c='b')
plt.hist(samples, density=1, alpha=0.5, ec='k',
        label="metropolis sample", color='limegreen')
plt.scatter(samples, np.random.normal(scale=0.003, size=len(samples)),
            label="samples", s=5, c='magenta')
plt.xlim(-10, 10)
plt.show()
