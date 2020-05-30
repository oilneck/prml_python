import numpy as np
import copy

class RVM_regression(object):

    def __init__(self, kernel, alpha=1., beta=1.):
        self.kernel = kernel
        self.alpha = alpha
        self.beta = beta

    def vector2col(self, X):
        if X.ndim == 1:
            X = X.reshape(len(X), 1)
        return X

    def fit(self, X:np.ndarray, t:np.ndarray, n_iter:int=1000):
        X = self.vector2col(X)
        self.X_train = X
        self.t_train = t
        PHI = self.kernel(X, X)
        N = len(t)
        self.alpha = [self.alpha] * N
        for _ in range(n_iter):
            old_param = np.r_[self.alpha, self.beta]
            cov_inv = np.diag(self.alpha) + self.beta * PHI.T @ PHI
            cov = np.linalg.inv(cov_inv)
            mean = self.beta * cov @ PHI.T @ t
            gamma = 1. - self.alpha * np.diag(cov)
            self.alpha = gamma / np.square(mean)
            self.alpha = self.alpha.clip(max=1e10)
            self.beta = (N - gamma.sum()) / np.sum( (t - PHI @ mean) ** 2 )
            if np.allclose(old_param, np.r_[self.alpha, self.beta]):
                break

        self.mean = mean
        self.cov = cov
        RV_mask = self.alpha < 1e9
        self.relevance_vector = {'x':X[RV_mask], 't':t[RV_mask]}

    def predict(self, x:np.ndarray, get_std:bool=False):
        X = self.vector2col(x)
        K = self.kernel(X, self.X_train)
        pred_mean = K @ self.mean
        if get_std:
            pred_var = np.sum(K @ self.cov * K, axis=1) + self.beta ** (-1)
            pred_std = np.sqrt(pred_var)
            return pred_mean, pred_std
        return pred_mean
