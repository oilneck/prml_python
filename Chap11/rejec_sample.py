import numpy as np
import matplotlib.pyplot as plt
from pd import *
from sampling import *


# target function ~p(z)
func = lambda x : 1 / (2 + (x + 4) ** 2) + np.exp(-(x - 5) ** 2 / 10)
# proposal distribution q(z)
proba = Gaussian(mu=2., var=25.)

''' Rejection sampling '''
sampler = RejectionSampling(target=func, prop=proba, offset=17)
samples = sampler.rvs(size=100)


x = np.linspace(-12, 17, 200)
plt.plot(x, sampler.k * proba.pdf(x), label=r"$kq(z)$", c='b')
plt.plot(x, func(x), label=r"$\tilde{p}(z)$", c='r')
plt.fill_between(x, func(x), sampler.k * proba.pdf(x), color="gray",alpha=0.4)
plt.hist(samples, density=True, alpha=0.4, color='limegreen')
plt.scatter(samples, np.random.normal(scale=.01, size=len(samples)), s=5, label="samples", c='magenta')
plt.xlim(-12, 17)
plt.legend(bbox_to_anchor=(1.05, 0.), loc='upper left', borderaxespad=0, fontsize=15)
plt.show()
