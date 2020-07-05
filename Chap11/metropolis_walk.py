import numpy as np
import matplotlib.pyplot as plt
from pd import *
from sampling import *
from deepL_module.base import *

# center & covariance
mu = np.array([1.5, 1.5])
cov = np.array([[1.2, .875], [.875, 1.2]])

# target function p(z)
func = lambda x : MultivariateGaussian(mu=mu, cov=cov).pdf(x)


''' Metropolis sampling '''
sampler = Metropolis(target=func, prop=Gaussian(0, .05), dim=2)
sampler.rvs(150, downsample=1, init_x=1.5)


plt.plot(*ellipse2D_orbit(mu, cov).T, color='k')
plt.plot(*sampler.path['accept'].T, color='#00ff00')
plt.plot(*sampler.path['reject'].T, color='r')
plt.xlim(0, 3)
plt.ylim(0, 3)
plt.show()
