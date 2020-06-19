import numpy as np
from deepL_module.base import sigmoid

class VariationalClassifier(object):

    def __init__(self, a0:float=1e-4, b0:float=1e-4):
        self.a0 = a0
        self.b0 = b0


    def calc_resp(self):
        a = self.a0 + 0.5 * self.X.shape[1]
        b = self.b0 + 0.5 * (np.sum(self.w_ave ** 2) + np.trace(self.w_cov))
        return a / b

    def lamb(self, x):
        return 0.5 * (sigmoid(x) - 0.5) / x

    def fit(self, X:np.ndarray, t:np.ndarray, n_iter:int=100):
        self.X = X
        M = np.size(X, 1)
        I = np.eye(M)
        self.w_cov = np.random.normal(size=I.shape)
        self.w_ave = np.ones(M)

        for _ in range(n_iter):
            mmT = np.outer(self.w_ave, self.w_ave)
            xi = np.sum(X @ (mmT + self.w_cov) * X, axis=1)
            resp = self.calc_resp()
            invS = I / resp + 2 * (X.T * self.lamb(xi)) @ X
            self.w_cov = np.linalg.inv(invS)
            self.w_ave = self.w_cov @ X.T @ (t - 0.5)
            if np.allclose(resp, self.calc_resp()):
                break


    def predict(self, X:np.ndarray):
        mu_a = X @ self.w_ave
        var_a = np.sum(X @ self.w_cov * X, axis=1)
        kappa_var = 1 / np.sqrt(1 + np.pi * var_a / 8)
        return sigmoid(mu_a * kappa_var)


    def posterior(self, X_test:np.ndarray):
        W = np.random.multivariate_normal(self.w_ave, self.w_cov)
        return X_test @ W
