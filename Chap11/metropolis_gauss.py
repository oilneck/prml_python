import numpy as np
import matplotlib.pyplot as plt
from deepL_module.base import *
from pd import *
from sampling import *

# center & covariance
mu = np.array([0, 0])
cov = np.array([[3, 1], [1, 3]])

# target function p(z)
func = lambda x :MultivariateGaussian(mu=mu, cov=cov).pdf(x)

# Ellipse orbit
X, Y = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
X_test = np.array([X, Y]).reshape(2, -1).T
Z = func(X_test).reshape(X.shape)


''' Metropolis sampling '''
sampler = Metropolis(target=func, prop=Gaussian(0, 2), dim=2)
samples = sampler.rvs(100, downsample=10)


plt.contour(X, Y, Z, levels=5, colors='k', zorder=3)
plt.scatter(samples[:, 0], samples[:, 1], color='limegreen', s=20)
plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
plt.show()

print("--- population ---\n")
print("mean\n", mu)
print("\ncovariance\n", cov)

print("\n\n--- samples ---\n")
print("mean\n", np.mean(samples, axis=0))
print("\ncovariance\n", np.cov(samples, rowvar=False))
