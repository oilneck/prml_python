import numpy as np
from pd.gamma import Gamma as pd_gamma

class Gaussian(object):
    '''
    p(x|mu,var)= (1 / sqrt(2*pi*var)) * exp(-0.5 * (x-mu)^2 / var)
    '''

    def __init__(self, mu:float=0., var:float=1.):
        self.mu = mu
        self.var = var
        self.bayes_fixed_var = None
        self.bayes_fixed_mu = None

    def set_mu(self,mu):
        if isinstance(mu,(int,float,np.number)):
            self.mu = np.asarray(mu)
            self.bayes_fixed_var = None
        elif isinstance(mu,(np.ndarray,list)):
            self.mu = np.asarray(mu)
            self.bayes_fixed_var = None
        elif isinstance(mu,type(self)):
            self.mu = mu.mu
            self.var = mu.var
            self.bayes_fixed_var = mu.var
        else:
            self.mu = None

    def set_var(self,variance):
        if isinstance(variance,(int,float,np.number)):
            assert variance > 0 , \
            'Expected value is positive definite but input value [{0}] is negative'.format(variance)
            self.var = np.asarray(np.abs(variance))
            self.bayes_fixed_mu = None
            self.a,self.b = None,None
        elif isinstance(variance,(np.ndarray,list)):
            assert (np.asarray(variance) > 0).all(), \
            'Expected value is positive definite but input value [{0}] has negative value'.format(variance)
            self.var = np.asarray(np.abs(variance))
            self.bayes_fixed_mu = None
            self.a,self.b = None,None
        elif isinstance(variance,pd_gamma):
            self.bayes_fixed_mu = self.mu
            self.a,self.b = variance.a,variance.b
        else:
            self.var = None

    def set_param(self,ave,var):
        self.set_mu(ave)
        self.set_var(var)


    def draw(self, sample_size:int=1000):
        return np.random.normal(
                                loc=self.mu,
                                scale=np.sqrt(self.var),
                                size=(sample_size,)
                                )

    def pdf(self,x:np.ndarray):
        np.seterr(divide='ignore')
        norm_factor = np.reciprocal(np.sqrt(2 * np.pi * self.var))
        z = (x - self.mu) / np.sqrt(self.var)
        return norm_factor * np.exp(-0.5 * z ** 2)

    def var_pdf(self,x:np.ndarray):
        return pd_gamma(self.a,self.b).pdf(x)

    def fit(self,train_x:np.ndarray):
        mu_bayes = self.bayes_fixed_var is not None
        var_bayes = self.bayes_fixed_mu is not None
        if mu_bayes and var_bayes:
            raise NotImplementedError
        elif mu_bayes:
            self.mu_fit(train_x)
        elif var_bayes:
            self.var_fit(train_x)
        else:
            self.ML_fit(train_x)

    def mu_fit(self,train_x:np.ndarray):
        '''Variance known & Average unknown'''
        N = len(train_x)
        mu_ML = np.mean(train_x,0)
        denom = N * self.var + self.bayes_fixed_var
        self.mu = (self.bayes_fixed_var * self.mu + N * self.var * mu_ML) / denom
        self.var = (self.bayes_fixed_var * self.var) / denom

    def var_fit(self,train_x:np.ndarray):
        '''Variance unknown & Average known'''
        N = len(train_x)
        self.a += 0.5 * N
        self.b += 0.5 * np.square(train_x - self.bayes_fixed_mu).sum()

    def ML_fit(self,train_x:np.ndarray):
        '''Variance unknown & Average unknown'''
        N = len(train_x)
        assert N > 1, 'There is only one sample'
        self.mu = np.mean(train_x,axis = 0)
        self.var = np.var(train_x,axis = 0) * N / (N - 1)
