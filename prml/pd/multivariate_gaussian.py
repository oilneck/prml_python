import numpy as np

class MultivariateGaussian(object):

    def __init__(self, mu:np.ndarray=None, cov:np.ndarray=None):
        self.mu = mu
        self.cov = cov
        self.dim = len(mu)

    def pdf(self, X):
        dev = X - self.mu
        invC = np.linalg.inv(self.cov)
        maha_sq = np.sum(dev * (invC @ dev.T).T, axis=-1)
        gauss = np.exp(-0.5 * maha_sq)
        coef = np.sqrt(np.linalg.det(invC))
        coef /= (2 * np.pi) ** (0.5 * self.dim)
        return coef * gauss

    def draw(self, sample_size=1000):
        return np.random.multivariate_normal(self.mu, self.cov, sample_size)
