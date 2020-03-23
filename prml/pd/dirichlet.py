import numpy as np
from scipy.special import gamma

class Dirichlet(object):

    def __init__(self,alpha):
        self.alpha = alpha

    def draw(self,sample_size:int=1000):
        return np.random.dirichlet(self.alpha,sample_size)

    def _check_input(self,mu):
        mu = np.asarray(mu).T
        alpha = np.asarray(self.alpha)
        if np.min(alpha) <= 0:
            raise ValueError("'alpha' must be greater than 0")
        elif alpha.ndim != 1:
            raise ValueError("Parameter vector 'alpha' must be one dimensional")

        if alpha.ndim == 1:
            alpha = alpha[:,None]

        if alpha.shape[0] == mu.shape[0]:
            pass
        elif alpha.shape[0] != mu.shape[0]:
            add_mu = np.array([1 - np.sum(mu,0)])
            if add_mu.ndim == 1:
                mu = np.append(mu, add_mu)
            elif add_mu.ndim == 2:
                mu = np.vstack((mu,add_mu))
            else:
                raise ValueError("input mu dimensional error")
        if np.min(mu) < 0:
            raise ValueError("input value must be"
                            "0 or positive value")
        if np.max(mu) > 1:
            raise ValueError("input value must be"
                            "smaller than or equal to 1")
        return mu.T



    def pdf(self, mu):
        mu = self._check_input(mu)
        alpha = np.asarray(self.alpha)
        coef = gamma(np.sum(alpha)) / np.prod(gamma(alpha))
        return coef * np.prod(mu ** (alpha - 1), axis=-1)
