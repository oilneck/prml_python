import numpy as np
import matplotlib.pyplot as plt
from pd import *
from sampling import *
from deepL_module.base import *

# center & covariance
mu = np.array([1.5, 1.5])
cov = np.array([[1.125, .875], [.875, 1.125]])

# target function p(z)
func = lambda x : MultivariateGaussian(mu=mu, cov=cov).pdf(x)

# proposal distribution q(z)
prop = MultivariateGaussian(mu=np.zeros(2), cov=np.eye(2) * 0.2)


''' MCMC sampling '''
sampler = MetropolisHastings(target=func, prop=prop, dim=2)
sampler.rvs(100, downsample=1)


plt.plot(*ellipse2D_orbit(mu, cov).T, color='k')
plt.plot(*sampler.path['accept'].T, color='#00ff00')
plt.plot(*sampler.path['reject'].T, color='r')
plt.xlim(0, 3)
plt.ylim(0, 3)
plt.show()
