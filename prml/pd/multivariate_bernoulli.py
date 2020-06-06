import numpy as np

class MultivariateBernoulli(object):

    def __init__(self, n_components:int):
        self.n_cls = n_components


    def log_bern(self, X:np.ndarray):
        np.clip(self.means, 1e-10, 1-1e-10, self.means)
        log_p = X[:,None,:] * np.log(self.means)
        log_q = (1 - X[:,None,:]) * np.log(1 - self.means)
        return np.sum(log_p + log_q, axis=-1)

    def gamma(self, X):
        np.seterr(divide='ignore')
        prob = self.param * np.exp(self.log_bern(X))
        prob /= prob.sum(axis=1)[:,None]
        return prob

    def update_param(self, resp):
        N_k = np.sum(resp, axis=0)
        self.means = resp.T @ self.X / N_k[:,None]
        self.param = N_k / len(self.X)

    def fit(self, X:np.ndarray, n_iter:int=10):
        self.X = X
        self.means = np.random.uniform(.25, .75, size=(self.n_cls, X.shape[1]))
        self.param = np.ones((1, self.n_cls))
        for _ in range(n_iter):
            old_param = np.r_[self.param.ravel(), self.means.ravel()]
            responsibility = self.gamma(X)
            self.update_param(responsibility)
            if np.allclose(old_param, np.r_[self.param.ravel(), self.means.ravel()]):
                break
