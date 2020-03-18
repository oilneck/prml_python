import numpy as np
from scipy.special import gamma

class Beta(object):

    def __init__(self,alpha:float,beta:float):
        if alpha >= 0:
            self.a = alpha
        if beta >= 0:
            self.b = beta

    def fit(self,n_observe):
        assert (np.asarray(n_observe) >= 0).all(),\
        'the number of observation must be positive'
        if np.asarray(n_observe).size == 1:
            self.a += np.array(n_observe)
        else:
            assert np.asarray(n_observe).size == 2,'too many values in input list'
            [n_ones,n_zeros] = n_observe
            self.a += n_ones
            self.b += n_zeros

    def draw(self,sample_size:int=1000):
        return np.random.beta(self.a, self.b, size=(sample_size,))


    def pdf(self,mu):
        np.seterr(divide='ignore')
        coef = gamma(self.a + self.b) / (gamma(self.a) * gamma(self.b))
        return coef * np.power(mu,self.a - 1) * np.power(1 - mu,self.b - 1)
