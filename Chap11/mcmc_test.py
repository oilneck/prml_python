import numpy as np
import matplotlib.pyplot as plt
from pd import *
from sampling import *

# target distribution p(z)
func = lambda x : Gamma(alpha=4, beta=5).pdf(x) + Gaussian(2.5, .1).pdf(x) * 1.5


''' sampling with metropolis hastings algorithm'''
sampler = MetropolisHastings(target=func, prop=Gaussian(0., 1.))
samples = sampler.rvs(500)


x = np.linspace(0, 4, 100)[:, None]
plt.plot(x, func(x), label="probability density function", c='r')
plt.hist(samples, density=1, alpha=0.5, ec='k', bins=30,
        label="metropolis sample", color='limegreen')
plt.scatter(samples, np.random.normal(scale=0.003, size=len(samples)) - .02,
            label="samples", s=5, c='b')
plt.xlim(-.1, 3.5)
plt.show()
