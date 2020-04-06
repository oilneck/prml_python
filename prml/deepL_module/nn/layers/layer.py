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
        diff_relu = (self.out > 0).astype(float)
        dout = diff_relu * delta
        dx = self.bp(dout)
        return dx

class Softsign_Layer(Affine):

    def __init__(self,W,b):
        super().__init__(W,b)
        self.activate = None

    def forward(self,x):
        self.activate = self.fp(x)
        _act = np.copy(self.activate)
        out = softsign(_act)
        return out

    def backward(self,delta):
        diff_ = 1 / np.square(1. + np.abs(self.activate))
        dout = diff_ * delta
        dx = self.bp(dout)
        return dx


class Softplus_Layer(Affine):

    def __init__(self,W,b):
        super().__init__(W,b)
        self.activate = None

    def forward(self,x):
        self.activate = self.fp(x)
        _act = np.copy(self.activate)
        out = softplus(_act)
        return out

    def backward(self,delta):
        diff_ = sigmoid(self.activate)
        dout = diff_ * delta
        dx = self.bp(dout)
        return dx

class Elu_Layer(Affine):

    def __init__(self,W,b):
        super().__init__(W,b)
        self.activate = None
        self.alpha = 1.0

    def forward(self,x):
        self.activate = self.fp(x)
        _act = np.copy(self.activate)
        out = elu(_act,self.alpha)
        return out

    def backward(self,delta):
        act = self.activate
        diff_ = np.where(act > 0, 1, self.alpha * np.exp(act))
        dout = diff_ * delta
        dx = self.bp(dout)
        return dx

class Swish_Layer(Affine):

    def __init__(self,W,b):
        super().__init__(W,b)
        self.activate = None
        self.out = None
        self.beta = 1.0

    def forward(self,x):
        self.activate = self.fp(x)
        _act = np.copy(self.activate)
        out = swish(_act,self.beta)
        self.out = out
        return out

    def backward(self,delta):
        act = self.activate
        beta = self.beta
        diff_ = beta * self.out + sigmoid(beta * act) * (1 - beta * self.out)
        dout = diff_ * delta
        dx = self.bp(dout)
        return dx
