import numpy as np
from .affine import Affine
from deepL_module.base.functions import *

class Linear_Layer(Affine):

    def __init__(self,W,b):
        super().__init__(W,b)
        self.out = None

    def forward(self, x):
        out = identity_function(self.fp(x))
        self.out = out
        return out

    def backward(self,delta):
        dout = delta
        dx = self.bp(dout)
        return dx

class Sigmoid_Layer(Affine):

    def __init__(self,W,b):
        super().__init__(W,b)
        self.out = None

    def forward(self, x):
        out = sigmoid(self.fp(x))
        self.out = out
        return out

    def backward(self,delta):
        dout = self.out * (1.0 - self.out) * delta
        dx = self.bp(dout)
        return dx


class Tanh_Layer(Affine):

    def __init__(self,W,b):
        super().__init__(W,b)
        self.out = None

    def forward(self, x):
        out = np.tanh(self.fp(x))
        self.out = out
        return out

    def backward(self,delta):
        dout = (1. - self.out ** 2) * delta
        dx = self.bp(dout)
        return dx

class Relu_Layer(Affine):

    def __init__(self,W,b):
        super().__init__(W,b)
        self.out = None

    def forward(self,x):
        out = relu(self.fp(x))
        self.out = out
        return out

    def backward(self,delta):
        dout = (self.out > 0).astype(float)
        dx = self.bp(dout)
        return dx
