import numpy as np
from .kernel import Kernel
from deepL_module.nn.optimizers import *

class GaussianKernel(Kernel):

    def __init__(self, alpha ,beta):
        '''
        k(x,y) = alpha * exp(-0.5 * beta * (x-y)^2)
        '''
        self.__alpha = alpha
        self.__beta = beta

    @property
    def alpha(self):
        pass

    @property
    def beta(self):
        pass

    @alpha.getter
    def alpha(self):
        return self.__alpha

    @beta.getter
    def beta(self):
        return self.__beta

    def get_params(self):
        return np.copy((self.alpha,self.beta))

    def set_params(self,alpha,beta):
        self.__alpha = alpha
        self.__beta = beta

    def __call__(self, x, y, create_pairs:bool=True):
        alpha,beta = self.get_params()
        if create_pairs:
            x,y = self.get_pairs(x,y)
        z = np.sum((x - y) ** 2, axis=-1)
        return alpha * np.exp(-0.5 * beta * z)

    def gradient(self, x, y, create_pairs:bool=True):
        if create_pairs:
            x,y = self.get_pairs(x,y)
        alpha,beta = self.get_params()
        z = np.sum((x - y) ** 2, axis=-1)
        diff_a = np.exp(-0.5 * beta * z)
        diff_b = -0.5 * z * diff_a * alpha
        return {'dalpha':diff_a,'dbeta':diff_b}

    def update_params(self, grads):
        updates = self.get_params() + grads
        self.set_params(*updates)
