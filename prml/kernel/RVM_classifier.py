import numpy as np


class RVM_classifier(object):

    def __init__(self, kernel, alpha:float=1.):
        self.kernel = kernel
        self.alpha = alpha

    def vector2mat(self, X):
        if X.ndim == 1:
            X = X.reshape(len(X), 1)
        return X

    def sigmoid(self, a):
        return 0.5 + 0.5 * np.tanh(0.5 * a)

    def update_param(self, w, max_iter=10):
        PHI = self.kernel(self.X_train, self.X_train)
        t = self.t_train
        for _ in range(max_iter):
            y = self.sigmoid(PHI @ w)
            B = np.diag(y * (1 - y))
            gradE = PHI.T @ (y - t) + self.alpha * w
            Hessian = PHI.T @ B @ PHI + np.diag(self.alpha)
            H_inv = np.linalg.inv(Hessian)
            w -= H_inv @ gradE
        return w, H_inv

    def fit(self, X:np.ndarray, t:np.ndarray, n_iter:int=100):
        X = self.vector2mat(X)
        self.X_train = X
        self.t_train = t
        N = len(t)
        PHI = self.kernel(X, X)
        self.alpha = [self.alpha] * N
        w_map = np.zeros(N)
        # updating prameter alpha
        for _ in range(n_iter):
            old_param = np.copy(self.alpha)
            w_map, cov = self.update_param(w_map)
            gamma = 1 - self.alpha * np.diag(cov)
            self.alpha = gamma / np.square(w_map)
            self.alpha = self.alpha.clip(max=1e10)
            if np.allclose(old_param, self.alpha):
                break

        self.w_map = w_map
        self.cov = cov
        RV_mask = self.alpha < 1e9
        self.relevance_vector = {'x':X[RV_mask], 't':t[RV_mask]}


    def predict(self, X:np.ndarray):
        X = self.vector2mat(X)
        PHI = self.kernel(X, self.X_train)
        mu_a = np.dot(PHI, self.w_map)
        var_a = np.sum(PHI * (self.cov @ PHI.T).T, axis=1)
        kappa_var = 1 / np.sqrt(1 + np.pi * var_a / 8)
        return self.sigmoid(mu_a * kappa_var)
