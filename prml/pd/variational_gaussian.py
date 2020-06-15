import numpy as np
from scipy.stats import gamma, norm


class VariationalGaussian(object):

    def __init__(self, mu:float=0.5, beta:float=10, a:float=1, b:float=2):
        self.mu0 = mu
        self.beta0 = beta
        self.beta = 0
        self.a0 = a
        self.b0 = b
        self.a = a
        self.b = b
        self.history = {}


    def Gam(self, x, a, b):
        return  gamma.pdf(x, a=a, scale=1 / b)


    def calc_resp(self):
        return self.a / self.b


    def mu_dist(self, resp):
        N = len(self.X)
        Xave = np.mean(self.X)
        self.mu = (self.beta0 * self.mu0 + N * Xave) / (self.beta0 + N)
        self.beta = (self.beta0 + N) * resp
        return (self.mu, self.beta, self.a, self.b)



    def beta_dist(self, resp):
        N = len(self.X)
        Xave = np.mean(self.X)
        self.a = self.a0 + 0.5 * (N + 1)
        self.b = self.b0 + 0.5 * N * (np.var(self.X) + np.reciprocal(N * resp))
        self.b += 0.5 * self.beta0 * (
                                        Xave ** 2 + np.reciprocal(N * resp)
                                        - 2 * self.mu0 * Xave + self.mu0 ** 2
                                     )
        return (self.mu, self.beta, self.a, self.b)


    def fit(self, X:np.ndarray, n_iter:int=10):
        self.X = X
        self.history['muStep1'] = (self.mu0, self.beta0, self.a0, self.b0)
        self.history['tauStep1'] = self.history.get('muStep1')
        for n in range(1, n_iter):

            resp = self.calc_resp()
            old_param = np.array([self.beta, self.b])

            self.history['muStep' + str(n+1)] = self.mu_dist(resp)
            self.history['tauStep' + str(n+1)] = self.beta_dist(resp)

            if np.allclose(np.array([self.beta, self.b]), old_param):
                break


    def predict(self, mu, tau):
        Norm = norm.pdf(mu, self.mu, np.sqrt(1 / self.beta))
        Gamma = self.Gam(tau, a=self.a, b=self.b)
        return Norm * Gamma


    def pdf(self, mu, tau):
        N = len(self.X)
        Xsum = self.beta0 * self.mu0 + np.sum(self.X, axis=-1)
        dev = N + self.beta0
        ave = Xsum / dev
        beta = dev
        a = 1 + 0.5 * dev
        b = self.b0 + 0.5 * self.beta0 * self.mu0 ** 2 + 0.5 * np.sum(self.X ** 2)
        b -= 0.5 * Xsum ** 2 / dev
        tau_N = dev * tau
        return norm.pdf(mu, ave, np.sqrt(1 / tau_N)) * self.Gam(tau, a=a, b=b)
