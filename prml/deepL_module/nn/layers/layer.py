import numpy as np
from .affine import Affine
from deepL_module.base.functions import *

class Linear_Layer(object):

    def forward(self, x):
        out = identity_function(x)
        self.out = out
        return out

    def backward(self,delta):
        dx = delta
        return dx

class Sigmoid_Layer(Affine):

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self,delta):
        dx = self.out * (1.0 - self.out) * delta
        return dx


class Tanh_Layer(object):

    def forward(self, x):
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self,delta):
        dx = (1. - self.out **2) * delta
        return dx
