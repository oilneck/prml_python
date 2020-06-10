import numpy as np
from scipy.special import digamma, gamma


class Variational_MG(object):

    def __init__(self, n_components:int, alpha_0:float=0.01):
        self.n_cls = n_components
        self.alpha_0 = alpha_0

    def init_W(self, X):
        self.X = X
        self.D = X.shape[1]
        self.alpha_0 = np.ones(self.n_cls) * self.alpha_0
        self.beta_0 = 1.
        self.m_0 = np.zeros(self.D)
        self.W_0 = np.eye(self.D)
        self.nu_0 = self.D

        self.N_k = len(X) / self.n_cls + np.zeros(self.n_cls)
        self.alpha = self.alpha_0 + self.N_k
        self.beta = self.beta_0 + self.N_k
        self.nu = self.nu_0 + self.N_k
        self.m = X[np.random.choice(len(X), self.n_cls, replace=False)]
        self.W = np.array([self.W_0] * self.n_cls)

    def get_params(self):
        return [self.alpha, self.beta, self.m, self.W, self.nu]

    def to_flat(self, params):
        return np.hstack([obj.ravel() for obj in params])

    def gauss(self, X):
        dev = X[None,:,:] - self.m[:,None,:]
        map = np.einsum('kij, knj->kni', self.W, dev)
        gauss = np.exp(- 0.5 * self.nu[:,None] * np.sum(dev * map, axis=-1)).T
        gauss *= np.exp(-0.5 * self.D * np.reciprocal(self.beta)[:,None]).T
        return gauss

    def make_resp(self):
        tilde_pi = np.exp(digamma(self.alpha) - digamma(self.alpha.sum()))
        args = digamma(self.nu - np.arange(self.D)[:,None]).sum(axis=0)
        Lam = np.exp(args + self.D * np.log(2) + np.linalg.slogdet(self.W)[1])
        resp = tilde_pi * np.sqrt(Lam) * self.gauss(self.X)
        resp /= np.sum(resp, axis=-1, keepdims=True)
        resp[np.isnan(resp)] = 1 / self.n_cls
        return resp

    def update_params(self, resp):
        self.N_k = resp.sum(axis=0)
        comp_size = self.N_k.reshape(-1, 1)
        bar_x = resp.T @ self.X / comp_size
        diff = (self.X[:,None,:] - bar_x).transpose(1,2,0)
        diff_ = diff.transpose(0,2,1) * np.expand_dims(resp, 0).T
        S = diff @ diff_ / np.expand_dims(comp_size, 2)

        self.alpha = self.alpha_0 + self.N_k
        self.beta = self.beta_0 + self.N_k

        self.m = (self.beta_0 * self.m_0 + comp_size * bar_x)
        self.m *= np.reciprocal(self.beta)[:,None]


        self.nu = self.nu_0 + self.N_k

        coef = self.beta_0 * self.N_k / (self.beta_0 + self.N_k)
        d = (bar_x - self.m_0)[:,:,None]
        d_mat = np.einsum('nij,nkj->nik' ,d ,d)


        self.W = np.linalg.inv(np.linalg.inv(self.W_0) +
                               self.N_k[:, None, None] * S +
                               coef[:, None, None] * d_mat)

    def fit(self, X, n_iter:int=100):
        self.init_W(X)
        for i in range(n_iter):
            old_param = self.to_flat(self.get_params())
            self.update_params(self.make_resp())
            if np.allclose(old_param, self.to_flat(self.get_params())):
                break

    def student_t(self, X):
        st_nu = self.nu + 1 - self.D
        L = st_nu * self.beta * self.W.T / (1 + self.beta)
        d = X[:, :, None] - self.m.T
        maha_sq = np.sum(np.einsum('nik,ijk->njk', d, L) * d, axis=1)
        return (
            gamma(0.5 * (st_nu + self.D))
            * np.sqrt(np.linalg.det(L.T))
            * (1 + maha_sq / st_nu) ** (-0.5 * (st_nu + self.D))
            / (gamma(0.5 * st_nu) * (st_nu * np.pi) ** (0.5 * self.D)))

    def predict(self, X):
        return (self.alpha * self.student_t(X)).sum(axis=-1) / self.alpha.sum()


    def classify(self, X):
        return np.argmax(self.make_resp(), 1)
