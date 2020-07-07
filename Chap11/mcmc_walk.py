import numpy as np
import matplotlib.pyplot as plt
from pd import *
from sampling import *
from deepL_module.base import *

# center & covariance
s_max = 10
s_min = 1
mu = np.array([1.5, 1.5]) * s_max / np.sqrt(2)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
cov = H @ np.diag([s_max ** 2, s_min ** 2]) @ H


# target function p(z)
func = lambda x : MultivariateGaussian(mu=mu, cov=cov).pdf(x)

# proposal distribution q(z)
prop = MultivariateGaussian(mu=np.zeros(2), cov=np.eye(2) * s_min)


''' MCMC sampling '''
sampler = MetropolisHastings(target=func, prop=prop, dim=2)
samples = sampler.rvs(120, downsample=1, init_x=10)

plt.plot(*ellipse2D_orbit(mu, cov).T, color='b')
plt.plot(*sampler.path['accept'].T, color='#00ff00')
plt.plot(*sampler.path['reject'].T, color='r')
plt.show()
