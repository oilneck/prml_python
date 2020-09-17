import numpy as np
from scipy.special import gamma, digamma
from deepL_module.base.solver import Solver

class Students_t(object):

    def __init__(self,df:float=1.,mu:float=0.,tau:float=1.):
        self.df = df
        self.mu = mu
        self.tau = tau
        self.solver = Solver()
        self.resp = {'eta':None, 'log_eta':None}


    def set_df(self, dof:float):
        assert dof > 0,\
        'degree of freedom must be positive value'
        self.df = dof


    def draw(self, sample_size:int=1000):
        return np.random.standard_t(self.df,size=sample_size)


    def pdf(self, x):
        self.df = np.clip(self.df,None,a_max=300)
        d = x - self.mu
        Del = self.tau * d ** 2
        coef = np.sqrt(self.tau / (np.pi * self.df)) * gamma(0.5 + 0.5 * self.df) / gamma(0.5 * self.df)
        return coef * (1 + Del / self.df) ** (-0.5 * (self.df + 1))


    def e_step(self, x):
        Eeta = (self.df + 1) / (self.df + self.tau * (x - self.mu) ** 2)
        Elog_eta = digamma((self.df + 1) / 2)
        Elog_eta -= np.log(.5 * (self.df + self.tau * (x - self.mu) ** 2))
        self.resp['eta'], self.resp['log_eta'] = Eeta, Elog_eta


    def m_step(self, x):
        self.mu = (x * self.resp['eta']).sum() / np.sum(self.resp['eta'])
        self.tau = len(x) / np.sum(self.resp['eta'] * (x - self.mu) ** 2)
        f = lambda z: digamma(z) - np.log(z) - 1 - np.mean(self.resp['log_eta']) + np.mean(self.resp['eta'])
        self.df = self.solver.bisect(f, 0, 5) * 2


    def fit(self, x:np.ndarray, n_iter:int=100):

        for _ in range(n_iter):
            params = [self.df, self.mu, self.tau]
            self.e_step(x)
            self.m_step(x)
            if np.allclose(params, [self.df, self.mu, self.tau]):
                break
