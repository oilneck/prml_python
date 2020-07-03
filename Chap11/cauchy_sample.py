import numpy as np
import matplotlib.pyplot as plt
from pd import *
from sampling import *


# target function ~p(z)
func = lambda x : Gamma(alpha=10, beta=1).pdf(x)
# proposal distribution q(z)
proba = Cauchy(x0=10 - 1, gamma=np.sqrt(2 * 10 - 1))

''' Rejection Sampling'''
sampler = RejectionSampling(target=func, prop=proba, offset=1.8)
samples = sampler.rvs(size=300)


x = np.linspace(0, 30, 200)
plt.plot(x, sampler.k * proba.pdf(x), label="proposal", c='r')
plt.plot(x, func(x), label="gamma", c='limegreen')
plt.hist(samples, bins=20, density=True, color='b', ec='k', alpha=.8)
plt.scatter(samples, np.random.normal(scale=.001, size=len(samples)) - .005, s=5, label="samples", c='magenta')
plt.legend(fontsize=15)
plt.xlim(0, 30)
plt.show()
