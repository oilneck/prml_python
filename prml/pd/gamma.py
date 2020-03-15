import numpy as np
from scipy.special import gamma
from scipy.stats import norm

class Gamma(object):

    def __init__(self,alpha:float,beta:float):
        if alpha >= 0:
            self.a = alpha
        if beta >= 0:
            self.b = beta

    def norm_gamma(self,mu,x,mu0:float=0,beta:float=2):
        '''p(mu,x)=N(mu|mu_0,(beta * lambda)^(-1)) * Gam(x|a,b)'''
        np.seterr(divide='ignore')
        var = 1 / (beta * x)
        return norm.pdf(mu,mu0,var) * self.pdf(x)


    def draw(self,sample_size:int=1000):
        return np.random.gamma(shape=self.a,scale=1 / self.b,size=(sample_size,))


    def pdf(self,x):
        np.seterr(divide='ignore') 
        coef = np.power(self.b,self.a) / gamma(self.a)
        return coef * np.power(x,self.a - 1) * np.exp(-self.b * x)
