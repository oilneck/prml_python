import numpy as np
import matplotlib.pyplot as plt
from pd import *
from sampling import *


func = lambda x : Gamma(alpha=4, beta=5).pdf(x) # p(z)


''' sampling with metropolis algorithm'''
sampler = Metropolis(target=func, prop=Gaussian(0., 1.))
samples = sampler.rvs(300)


x = np.linspace(0, 2.5, 100)[:, None]
plt.plot(x, func(x), label="probability density function", c='b')
plt.hist(samples, density=1, alpha=0.5, ec='k', bins=12,
        label="metropolis sample", color='limegreen')
plt.scatter(samples, np.random.normal(scale=0.003, size=len(samples)),
            label="samples", s=5, c='magenta')
plt.xlim(-.1, 2.6)
plt.show()
