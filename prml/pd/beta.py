import numpy as np
from scipy.special import gamma

class Beta(object):

    def __init__(self,alpha:float,beta:float):
        if alpha >= 0:
            self.a = alpha
        if beta >= 0:
            self.b = beta


    def draw(self,sample_size:int=1000):
        return np.random.beta(self.a, self.b, size=(sample_size,))


    def pdf(self,mu):
        np.seterr(divide='ignore')
        coef = gamma(self.a + self.b) / (gamma(self.a) * gamma(self.b))
        return coef * np.power(mu,self.a - 1) * np.power(1 - mu,self.b - 1)
