import numpy as np
from fitting.bayesian_regression import Bayesian_Regression

class EM_Regressor(Bayesian_Regression):

    def __init__(self, alpha:float=1., beta:float=1.):
        super().__init__(alpha,beta)

    def e_step(self):
        X = np.copy(self.PHI)
        invS = self.alpha * np.eye(np.size(X, 1)) + self.beta * X.T @ X
        S = np.linalg.inv(invS)
        mu = self.beta * S @ X.T @ self.t
        return mu, S

    def m_step(self, mean, cov):# update params
        X = np.copy(self.PHI)
        N, M = X.shape
        self.alpha = M / (np.dot(mean, mean) + np.trace(cov))
        self.beta = N
        self.beta /= np.sum((self.t - X @ mean) ** 2) + np.trace(X.T @ X @ cov)



    def fit(self, PHI:np.ndarray, t:np.ndarray, n_iter:int=100):
        self.PHI = PHI
        self.t = t

        for _ in range(n_iter):
            params = np.copy([self.alpha, self.beta])
            mean, cov = self.e_step()
            self.m_step(mean, cov)
            if np.allclose(params, [self.alpha, self.beta]):
                break

        self.w_mean = mean
        self.w_cov = cov
