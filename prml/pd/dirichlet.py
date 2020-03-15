import numpy as np
from scipy.special import gamma

class Dirichlet(object):

    def __init__(self,alpha):
        self.alpha = alpha

    def draw(self,sample_size:int=1000):
        return np.random.dirichlet(self.alpha,sample_size)

    def pdf(self, mu):
        coef = gamma(np.sum(self.alpha)) / np.prod(gamma(self.alpha))
        return coef * np.prod(np.array(mu) ** (np.array(self.alpha) - 1), axis=-1)
