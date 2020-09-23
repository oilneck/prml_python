import numpy as np
from scipy.special import digamma, gamma
import scipy

class VariationalRegressor(object):

    def __init__(self, a0:float=1, b0:float=5e+3, c0:float=1, d0:float=1):
        self.a0 = a0
        self.b0 = b0
        self.c0 = c0
        self.d0 = d0


    def calc_params(self):
        X = self.PHI.copy()
        self.a = self.a0 + 0.5 * X.shape[1]
        self.b = self.b0
        self.b += 0.5 * (self.w_ave @ self.w_ave + np.trace(self.w_cov))
        self.c = self.c0 + 0.5 * X.shape[0]
        self.d = 0.5 * np.sum((self.t - X @ self.w_ave) ** 2) + self.d0
        self.d += 0.5 * np.trace(X.T @ X @ self.w_cov)
        return self.a / self.b, self.c / self.d


    def fit(self, PHI:np.ndarray, t:np.ndarray, n_iter:int=100):
        self.PHI = PHI
        self.t = t
        M = np.size(PHI, 1)
        I = np.eye(M)
        self.w_cov = np.random.normal(size=I.shape)
        self.w_ave = np.ones(M)
        for n in range(n_iter):
            self.alpha, self.beta = self.calc_params()
            params = [self.alpha, self.beta]
            self.w_cov = np.linalg.inv(I * self.alpha + self.beta * PHI.T @ PHI)
            self.w_ave = self.beta * self.w_cov @ PHI.T @ t
            if np.allclose(params, self.calc_params()):
                break


    def predict(self, X:np.ndarray, get_std:bool=False):
        mean = np.sum(X * np.array([self.w_ave] * X.shape[0]), axis=1)
        if get_std:
            var = 1 / self.beta + np.sum(X * (self.w_cov @ X.T).T, axis=1)
            std = np.sqrt(var)
            return mean, std
        return mean


    def lower_bound(self):
        N, D = self.PHI.shape
        invS = np.linalg.inv(self.w_cov)
        assert np.all(np.linalg.eigvalsh(invS) > 0)
        U = np.linalg.cholesky(invS)
        KLw = -np.sum(np.log(np.diag(U)))
        KLalpha = -self.a * np.log(self.b)
        KLbeta = -self.c * np.log(self.d)
        const = np.log(gamma(self.a)) - np.log(gamma(self.a0))
        const += np.log(gamma(self.c)) - np.log(gamma(self.c0))
        const += self.a0 * np.log(self.b0) + self.c0 * np.log(self.d0)
        const += 0.5 * (D - N * np.log(2 * np.pi))
        return KLalpha + const + KLw + KLbeta
