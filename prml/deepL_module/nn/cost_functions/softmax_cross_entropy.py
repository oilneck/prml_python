import numpy as np
from deepL_module.base.functions import *

class Softmax_cross_entropy(object):

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def activate(self,x):
        return softmax(x)

    def __call__(self,x,t):
        self.t = t
        self.y = self.activate(x).clip(min=1e-8)
        batch_size = self.y.shape[0]
        self.loss = -np.sum( self.t * np.log(self.y) ) / batch_size
        return np.copy(self.loss)

    def delta(self,dout=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size
