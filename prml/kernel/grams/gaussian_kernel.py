import numpy as np
from .kernel import Kernel
from deepL_module.nn.optimizers import *
from copy import copy,deepcopy

class GaussianKernel(Kernel):

    def __init__(self, alpha ,beta, bias=1., gamma=0.):
        '''
        k(x,y) = bias + alpha * exp(-0.5 * beta * (x-y)^2) + gamma * x.y
        '''
        self.__params = {'alpha':alpha, 'beta':beta, 'bias':bias, 'gamma':gamma}


    def get_params(self):
        return list(deepcopy(self.__params).values())

    def set_params(self, updates):
        for n,key in enumerate(self.__params.keys()):
            self.__params[key] = updates[n]

    def __call__(self, x, y, create_pairs:bool=True):
        alpha,beta,bias,gamma = self.get_params()
        if create_pairs:
            x,y = self.get_pairs(x,y)
        args = -0.5 * beta * np.sum((x - y) ** 2, axis=-1)
        return alpha * np.exp(args) + bias

    def gradient(self, x, y, create_pairs:bool=True):
        if create_pairs:
            x,y = self.get_pairs(x,y)
        alpha,beta,_,_ = self.get_params()
        z = np.sum((x - y) ** 2, axis=-1)
        diff_a = np.exp(-0.5 * beta * z)
        diff_b = -0.5 * z * diff_a * alpha
        return [diff_a, diff_b, np.ones(diff_a.shape), np.sum(x * y, axis=-1)]

    def updates(self, grads):
        quantity = self.get_params() + grads
        self.set_params(quantity)
