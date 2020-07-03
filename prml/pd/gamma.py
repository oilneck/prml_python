import numpy as np
import warnings
warnings.simplefilter('ignore', RuntimeWarning)
from scipy.special import gamma
from scipy.stats import norm

class Gamma(object):

    def __init__(self, alpha:float, beta:float):
        '''
        Gam(z|a,b)= b^a * z^(a-1) * exp(-b*z) / gamma(a)
        '''
        if alpha >= 0:
            self.a = alpha
        if beta >= 0:
            self.b = beta

    def norm_gamma(self,mu,x,mu0:float=0,beta:float=2):
        '''p(mu,x)=N(mu|mu_0,(beta * lambda)^(-1)) * Gam(x|a,b)'''
        np.seterr(divide='ignore')
        var = np.reciprocal((beta * x).astype(float))
        return norm.pdf(mu,mu0,var) * self.pdf(x)


    def draw(self, sample_size:int=1000):
        tau = 1 / self.b
        return np.random.gamma(shape=self.a, scale=tau, size=(sample_size,))


    def pdf(self, x):
        np.seterr(divide='ignore')
        coef = np.power(self.b, self.a) / gamma(self.a)
        return coef * np.power(x, self.a - 1) * np.exp(-self.b * x)
