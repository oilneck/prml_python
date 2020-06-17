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


    def lower_bound(self, PHI:np.ndarray, t:np.ndarray):
        N = len(t)
        D = np.size(PHI, 1)




        invS = np.linalg.inv(self.w_cov)
        U = np.linalg.cholesky(invS)
        KLw = -np.sum(np.log(np.diag(U)))
        KLalpha = -self.a * np.log(self.b)
        const = np.log(gamma(self.a)) - np.log(gamma(self.a0))
        const += self.a0 * np.log(self.b0) + 0.5 * (D - N * np.log(2 * np.pi))
        return KLalpha + const + KLw

        # m = self.w_ave.reshape(-1, 1)
        # E_tw = N * np.log(self.beta / (2 * np.pi)) - self.beta * t @ t
        # E_tw += self.beta * m.T @ PHI.T @ t
        # E_tw -= self.beta * np.trace(PHI.T @ PHI @ (self.w_cov + m @ m.T))
        #
        #
        # E_wa = -D * np.log(2 * np.pi) + D * (digamma(self.a) - np.log(self.b))
        # E_wa -= self.calc_resp() * (m.T @ m + np.trace(self.w_cov))
        # E_wa = E_wa.ravel()
        #
        # E_a = - self.b0 * self.calc_resp() +  self.a0 * np.log(self.b0)
        # E_a += (self.a0 - 1) * (digamma(self.a) - np.log(self.b))
        # E_a -= np.log(gamma(self.a))
        #
        #
        #
        # E_w = np.linalg.slogdet(self.w_cov)[1] + D * (1 + np.log(2 * np.pi))
        #
        #
        # Eq_a = np.log(gamma(self.a)) - (self.a - 1) * digamma(self.a)
        # Eq_a += self.a - np.log(self.b)
        #
        # return E_tw + E_wa + E_w + E_a + Eq_a
