import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, gamma
from deepL_module.datasets import *
from deepL_module.base import *


class VariationalGaussianMixture(object):

    def __init__(self, n_component=10, alpha0=1.):
        self.n_component = n_component
        self.alpha0 = alpha0
        self.eff_means = {}
        self.eff_covs = {}

    def init_params(self, X):
        self.sample_size, self.ndim = X.shape
        self.alpha0 = np.ones(self.n_component) * self.alpha0
        self.m0 = np.zeros(self.ndim)
        self.W0 = np.eye(self.ndim)
        self.nu0 = self.ndim
        self.beta0 = 1.

        self.component_size = self.sample_size / self.n_component + np.zeros(self.n_component)
        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        indices = np.random.choice(self.sample_size, self.n_component, replace=False)
        self.m = X[indices].T
        self.W = np.tile(self.W0, (self.n_component, 1, 1)).T
        self.nu = self.nu0 + self.component_size

    def get_params(self):
        return self.alpha, self.beta, self.m, self.W, self.nu

    def fit(self, X, iter_max=100):
        self.init_params(X)
        for i in range(iter_max):
            params = np.hstack([array.flatten() for array in self.get_params()])
            r = self.e_like_step(X)
            self.eff_means['step' + str(i+1)] = self.m.T[self.component_size > 1]
            self.eff_covs['step' + str(i+1)] = self.W.transpose(2,0,1)[self.component_size > 1]
            self.m_like_step(X, r)
            if np.allclose(params, np.hstack([array.ravel() for array in self.get_params()])):
                break
        else:
            print("parameters may not have converged")

    def e_like_step(self, X):
        d = X[:, :, None] - self.m
        gauss = np.exp(
            -0.5 * self.ndim / self.beta
            - 0.5 * self.nu * np.sum(
                np.einsum('ijk,njk->nik', self.W, d) * d,
                axis=1)
        )
        pi = np.exp(digamma(self.alpha) - digamma(self.alpha.sum()))
        Lambda = np.exp(digamma(self.nu - np.arange(self.ndim)[:, None]).sum(axis=0) + self.ndim * np.log(2) + np.linalg.slogdet(self.W.T)[1])
        r = pi * np.sqrt(Lambda) * gauss
        r /= np.sum(r, axis=-1, keepdims=True)
        r[np.isnan(r)] = 1. / self.n_component
        return r

    def m_like_step(self, X, r):
        self.component_size = r.sum(axis=0)
        Xm = X.T.dot(r) / self.component_size
        d = X[:, :, None] - Xm
        S = np.einsum('nik,njk->ijk', d, r[:, None, :] * d) / self.component_size
        self.alpha = self.alpha0 + self.component_size
        self.beta = self.beta0 + self.component_size
        self.m = (self.beta0 * self.m0[:, None] + self.component_size * Xm) / self.beta
        d = Xm - self.m0[:, None]
        self.W = np.linalg.inv(
            np.linalg.inv(self.W0)
            + (self.component_size * S).T
            + (self.beta0 * self.component_size * np.einsum('ik,jk->ijk', d, d) / (self.beta0 + self.component_size)).T).T
        self.nu = self.nu0 + self.component_size

    def predict_proba(self, X):
        covs = self.nu * self.W
        precisions = np.linalg.inv(covs.T).T
        d = X[:, :, None] - self.m
        exponents = np.sum(np.einsum('nik,ijk->njk', d, precisions) * d, axis=1)
        gausses = np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(covs.T).T * (2 * np.pi) ** self.ndim)
        gausses *= (self.alpha0 + self.component_size) / (self.n_component * self.alpha0 + self.sample_size)
        return np.sum(gausses, axis=-1)

    def classify(self, X):
        return np.argmax(self.e_like_step(X), 1)

    def student_t(self, X):
        nu = self.nu + 1 - self.ndim
        L = nu * self.beta * self.W / (1 + self.beta)
        d = X[:, :, None] - self.m
        maha_sq = np.sum(np.einsum('nik,ijk->njk', d, L) * d, axis=1)
        return (
            gamma(0.5 * (nu + self.ndim))
            * np.sqrt(np.linalg.det(L.T))
            * (1 + maha_sq / nu) ** (-0.5 * (nu + self.ndim))
            / (gamma(0.5 * nu) * (nu * np.pi) ** (0.5 * self.ndim)))

    def predict_dist(self, X):
        return (self.alpha * self.student_t(X)).sum(axis=-1) / self.alpha.sum()





X = load_faithful(normalize=True)#create_toy_data()

model = VariationalGaussianMixture(n_component=10, alpha0=0.01)
model.fit(X, iter_max=100)
labels = model.classify(X)

x_test, y_test = np.meshgrid(
    np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
X_test = np.array([x_test, y_test]).reshape(2, -1).transpose()
probs = model.predict_dist(X_test)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=cm.get_cmap())
N = len(model.eff_covs)
for mu, cov in zip(model.eff_means['step'+str(N)], model.eff_covs['step'+str(N)]):
    x,y = ellipse2D_orbit(mu, cov).T
    plt.plot(x, y, color='r')
plt.contour(x_test, y_test, probs.reshape(100, 100))
plt.show()
