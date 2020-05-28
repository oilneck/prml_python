import numpy as np

class GP_regression(object):

    def __init__(self, kernel, beta=1.):
        self.kernel = kernel
        self.beta = beta

    def _fit(self, X:np.ndarray, t:np.ndarray):
        self.X_train = X
        self.t_train = t
        I = np.eye(len(X))
        Gram = self.kernel(X,X)
        self.cov = Gram + np.reciprocal(self.beta) * I
        self.cov_inv = np.linalg.inv(self.cov)

    def predict(self, x, get_std:bool=False):
        K = self.kernel(x, self.X_train)
        m_x = K @ self.cov_inv @ self.t_train
        if get_std:
            c = np.reciprocal(self.beta) + self.kernel(x,x,False)
            var = c - np.sum(K @ self.cov_inv * K, axis=1)
            return m_x.ravel(), np.sqrt(var).ravel()
        return m_x.ravel()

    def fit(self, x:np.ndarray, t:np.ndarray, lr:float=0.001, n_iter:int=1):
        for _ in range(n_iter):
            params = self.kernel.get_params()
            self._fit(x,t)
            _grads = self.kernel.gradient(x,x).values()
            C = np.copy(self.cov_inv)
            grads = [-np.trace(C @ grad) + t @ C @ grad @ C @ t for grad in _grads]
            self.kernel.update_params(lr * np.array(grads))
            if np.allclose(params, self.kernel.get_params()):
                break
