import numpy as np
from scipy.stats import cauchy

class Cauchy(object):

    def __init__(self, x0:float=0., gamma:float=1.):
        '''
        p(x|x0,gamma) = (1 / pi) * gamma / (gamma^2 + (x - x0)^2)
        '''
        self.loc = x0
        self.scale = gamma


    def draw(self, sample_size:int=1000):
        return cauchy.rvs(loc=self.loc, scale=self.scale, size=sample_size)

    def pdf(self, x):
        z = (x - self.loc) / self.scale
        norm = np.pi * self.scale
        return 1 / (1 + z ** 2) / norm
