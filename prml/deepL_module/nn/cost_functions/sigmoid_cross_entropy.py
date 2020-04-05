import numpy as np
from deepL_module.base.functions import *

class Sigmoid_cross_entropy(object):

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def activate(self,x):
        return sigmoid(x)

    def __call__(self,x,t):
        self.t = t
        self.y = self.activate(x).clip(min = 1e-10, max = 1-1e-10)
        self.loss = np.sum(-self.t * np.log(self.y) - (1 - self.t) * np.log(1 - self.y))
        return np.copy(self.loss)

    def delta(self,dout=1):
        return self.y - self.t
