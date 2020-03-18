import numpy as np
from scipy.special import comb

class Binomial(object):

    def __init__(self,trials:int=1,prob:float=0.5):
        self.N = trials
        self.p = prob

    def draw(self,sample_size:int=1000):
        return np.random.binomial(self.N,self.p,sample_size)

    def pdf(self,m:int):
        coef = comb(self.N, m, exact=True)
        return coef * np.power(self.p, m) * np.power(1 - self.p, self.N - m)
