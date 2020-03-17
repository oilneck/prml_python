import numpy as np

class Gauss(object):
    '''
    p(x|mu,var)= (1 / sqrt(2*pi*var)) * exp(-0.5 * (x-mu)^2 / var)
    '''

    def __init__(self,mu:float=0.,var:float=1.):
        self.mu = mu
        self.var = var

    def set_mu(self,mu):
        if isinstance(mu,(int,float,np.number)):
            self.mu = np.asarray(mu)
        elif isinstance(mu,(np.ndarray,list)):
            self.mu = np.asarray(mu)
        else:
            self.mu = None

    def set_var(self,variance):
        if isinstance(variance,(int,float,np.number)):
            assert variance > 0 , \
            'Expected value is positive definite but input value [{0}] is negative'.format(variance)
            self.var = np.asarray(np.abs(variance))
        elif isinstance(variance,(np.ndarray,list)):
            assert (np.asarray(variance) > 0).all(), \
            'Expected value is positive definite but input value [{0}] has negative value'.format(variance)
            self.var = np.asarray(np.abs(variance))
        else:
            self.var = None

    def draw(self,sample_size:int=1000):
        return np.random.normal(loc=self.mu,scale=np.sqrt(self.var),size=(sample_size,))

    def pdf(self,x):
        np.seterr(divide='ignore')
        norm_factor = 1 / np.sqrt(2 * np.pi * self.var)
        z = (x - self.mu) / np.sqrt(self.var)
        return norm_factor * np.exp(-0.5 * z ** 2)
