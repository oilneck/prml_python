import numpy as np
from .kernel import Kernel
from deepL_module.nn.optimizers import *
import copy

class GaussianKernel(Kernel):

    def __init__(self, alpha ,beta, bias=0.):
        '''
        k(x,y) = bias + alpha * exp(-0.5 * beta * (x-y)^2)
        '''
        self.__params = {'alpha':alpha, 'beta':beta, 'bias':bias}


    def get_params(self):
        return list(self.__params.values()).copy()

    def set_params(self, updates):
        for n,key in enumerate(self.__params.keys()):
            self.__params[key] = updates[n]

    def __call__(self, x, y, create_pairs:bool=True):
        [alpha,beta,bias] = self.get_params()
        if create_pairs:
            x,y = self.get_pairs(x,y)
        args = -0.5 * beta * np.sum((x - y) ** 2, axis=-1)
        return alpha * np.exp(args) + bias

    def gradient(self, x, y, create_pairs:bool=True):
        if create_pairs:
            x,y = self.get_pairs(x,y)
        alpha,beta,bias = self.get_params()
        z = np.sum((x - y) ** 2, axis=-1)
        diff_a = np.exp(-0.5 * beta * z)
        diff_b = -0.5 * z * diff_a * alpha
        return [diff_a,diff_b,np.ones(diff_a.shape)]

    def updates(self, grads):
        quantity = self.get_params() + grads
        self.set_params(quantity)
