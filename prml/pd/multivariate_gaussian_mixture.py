import numpy as np

class MultivariateGaussianMixture(object):

    def __init__(self, n_components:int):
        self.n_cls = n_components
        self.cls_X = {}
        self.centers = {}


    def gauss(self, X):
        dev = X[None,:,:] - self.means[:,None,:]
        vect = np.einsum('kij, knj -> kni', np.linalg.inv(self.covs), dev)
        gauss = np.exp(-0.5 * np.sum(dev * vect, axis=-1)).T
        gauss /= np.sqrt(2 * np.pi * np.linalg.det(self.covs))
        return gauss


    def gamma(self, X):
        prob = self.gauss(X) * self.param
        norm = prob.sum(axis=-1)[:, None]
        return prob / norm


    def update_params(self):
        resp = self.gamma(self.X)
        N_k = np.sum(resp, axis=0).reshape(-1,1)
        self.means = resp.T @ self.X / N_k
        diff = (self.X[:,None,:] - self.means).transpose(1,2,0)
        diff_ = diff.transpose(0,2,1) * np.expand_dims(resp, 0).T
        self.covs = diff @ diff_ / np.expand_dims(N_k, 2)
        self.param = N_k.ravel() / len(self.X)


    def log_likelihood(self, X):
        args = np.sum(self.gauss(X) * self.param, axis=-1)
        log_likeli = np.log(args)
        return log_likeli.sum()


    def fit(self, X:np.ndarray, n_iter:int=100):
        self.X = X
        D = X.shape[1]
        self.means = np.random.normal(size=(self.n_cls, D))
        self.covs =  np.array([np.eye(D)] * self.n_cls)
        self.param = np.ones(self.n_cls)
        for n in range(n_iter):

            old_ll = self.log_likelihood(X)
            self.cls_X['step' + str(n+1)] = self.classify(X)
            self.centers['step' + str(n+1)] = self.means.copy()

            self.update_params()

            if np.allclose(old_ll, self.log_likelihood(X)):
                break


    def joint_prob(self, X):
        return self.param * self.gauss(X)


    def predict(self, X):
        return np.sum(self.joint_prob(X), axis=-1)


    def classify(self, X):
        return np.argmax(self.joint_prob(X), axis=1)
