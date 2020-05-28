import numpy as np
from .kernel import Kernel
from deepL_module.nn.optimizers import *
from copy import copy,deepcopy

class PolynomialKernel(Kernel):

    def __init__(self, degree:int=2, bias=1.):
        '''
        k(x,y) = (x.y + bias)^M
        '''
        self.__params = {'degree':degree, 'bias':bias}

    def get_params(self):
        return list(deepcopy(self.__params).values())

    def __call__(self, x, y, create_pairs:bool=True):
        if create_pairs:
            x,y = self.get_pairs(x,y)
        M, const = self.get_params()
        return (np.sum(x * y, axis=-1) + const) ** M
