import numpy as np
from deepL_module.base.functions import *

class Linear_Layer():

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = identity_function(x)
        self.out = out
        return out

    def backward(self,delta):
        dx = delta
        return dx

class Sigmoid_Layer():

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self,delta):
        dx = self.out * (1.0 - self.out) * delta
        return dx


class Tanh_Layer():

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self,delta):
        dx = (1. - self.out ** 2) * delta
        return dx

class Relu_Layer():

    def __init__(self):
        self.out = None

    def forward(self,x):
        out = relu(x)
        self.out = out
        return out

    def backward(self,delta):
        diff_relu = (self.out > 0).astype(float)
        dx = diff_relu * delta
        return dx

class Softsign_Layer():

    def __init__(self):
        self.activate = None

    def forward(self,x):
        self.activate = x
        _act = np.copy(self.activate)
        out = softsign(_act)
        return out

    def backward(self,delta):
        diff_ = 1 / np.square(1. + np.abs(self.activate))
        dx = diff_ * delta
        return dx


class Softplus_Layer():

    def __init__(self):
        self.activate = None

    def forward(self,x):
        self.activate = x
        _act = np.copy(self.activate)
        out = softplus(_act)
        return out

    def backward(self,delta):
        diff_ = sigmoid(self.activate)
        dx = diff_ * delta
        return dx

class Elu_Layer():

    def __init__(self):
        self.activate = None
        self.alpha = 1.0

    def forward(self,x):
        self.activate = x
        _act = np.copy(self.activate)
        out = elu(_act,self.alpha)
        return out

    def backward(self,delta):
        act = self.activate
        diff_ = np.where(act > 0, 1, self.alpha * np.exp(act))
        dx = diff_ * delta
        return dx

class Swish_Layer():

    def __init__(self):
        self.activate = None
        self.out = None
        self.beta = 1.0

    def forward(self,x):
        self.activate = x
        _act = np.copy(self.activate)
        out = swish(_act,self.beta)
        self.out = out
        return out

    def backward(self,delta):
        act = self.activate
        beta = self.beta
        diff_ = beta * self.out + sigmoid(beta * act) * (1 - beta * self.out)
        dx = diff_ * delta
        return dx
