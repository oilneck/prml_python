import numpy as np
from scipy.special import gamma

class Gamma(object):

    def __init__(self,alpha:float,beta:float):
        if alpha >= 0:
            self.a = alpha
        if beta >= 0:
            self.b = beta


    def draw(self,sample_size:int=1000):
        return np.random.gamma(shape=self.a,scale=1 / self.b,size=(sample_size,))


    def pdf(self,x):
        coef = np.power(self.b,self.a) / gamma(self.a)
        return coef * np.power(x,self.a - 1) * np.exp(-self.b * x)
