import numpy as np
from scipy.special import gamma

class Students_t(object):

    def __init__(self,df:float=1.,mu:float=0.,tau:float=1.):
        self.df = df
        self.mu = mu
        self.tau = tau

    def set_df(self,dof:float):
        assert dof > 0,\
        'degree of freedom must be positive value'
        self.df = dof

    def draw(self,sample_size:int=1000):
        return np.random.standard_t(self.df,size=sample_size)

    def pdf(self,x):
        d = x - self.mu
        Del = self.tau * d ** 2
        coef = np.sqrt(self.tau / (np.pi * self.df)) * gamma(0.5 + 0.5 * self.df) / gamma(0.5 * self.df)
        return coef * (1 + Del / self.df) ** (-0.5 * (self.df + 1))
