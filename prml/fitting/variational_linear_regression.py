import numpy as np
from scipy.special import digamma, gamma

class VariationalRegressor(object):

    def __init__(self, beta:float=0.01, a0:float=1., b0:float=1.):
        self.beta = beta
        self.a0 = a0
        self.b0 = b0


    def calc_resp(self):
        self.a = self.a0 + 0.5 * self.PHI.shape[1]
        self.b = self.b0
        self.b += 0.5 * (self.w_ave @ self.w_ave + np.trace(self.w_cov))
        return self.a / self.b


    def fit(self, PHI:np.ndarray, t:np.ndarray, n_iter:int=100):
        self.PHI = PHI
        M = np.size(PHI, 1)
        I = np.eye(M)
        self.w_cov = np.random.normal(size=I.shape)
        self.w_ave = np.ones(M)
        for _ in range(n_iter):
            resp = self.calc_resp()
            self.w_cov = np.linalg.inv(I * resp + self.beta * PHI.T @ PHI)
            self.w_ave = self.beta * self.w_cov @ PHI.T @ t
            if np.allclose(resp, self.calc_resp()):
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
        U = np.linalg.cholesky(invS)
        KLw = -np.sum(np.log(np.diag(U)))
        KLalpha = -self.a * np.log(self.b)
        const = np.log(gamma(self.a)) - np.log(gamma(self.a0))
        const += self.a0 * np.log(self.b0) + 0.5 * (D - N * np.log(2 * np.pi))
        return KLalpha + const + KLw
