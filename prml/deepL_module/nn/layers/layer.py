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

    @staticmethod
    def identity(x):
        return Linear_Layer().forward(x)


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

    @staticmethod
    def sigmoid(x):
        return Sigmoid_Layer().forward(x)


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

    @staticmethod
    def tanh(x):
        return Tanh_Layer().forward(x)


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

    @staticmethod
    def relu(x):
        return Relu_Layer().forward(x)


class Softsign_Layer():

    def __init__(self):
        self.activate = None
        self.out = None

    def forward(self,x):
        self.activate = x
        out = softsign(x)
        self.out = out
        return out

    def backward(self,delta):
        diff_ = 1 / np.square(1. + np.abs(self.activate))
        dx = diff_ * delta
        return dx

    @staticmethod
    def softsign(x):
        return Softsign_Layer().forward(x)


class Softplus_Layer():

    def __init__(self):
        self.activate = None
        self.out = None

    def forward(self,x):
        self.activate = x
        out = softplus(x)
        self.out = out
        return out

    def backward(self,delta):
        diff_ = sigmoid(self.activate)
        dx = diff_ * delta
        return dx

    @staticmethod
    def softplus(x):
        return Softplus_Layer().forward(x)


class Elu_Layer():

    def __init__(self):
        self.activate = None
        self.alpha = 1.0
        self.out = None

    def forward(self,x):
        self.activate = x
        out = elu(x, self.alpha)
        self.out = out
        return out

    def backward(self,delta):
        act = self.activate
        diff_ = np.where(act > 0, 1, self.alpha * np.exp(act))
        dx = diff_ * delta
        return dx

    @staticmethod
    def elu(x):
        return Elu_Layer().forward(x)


class Swish_Layer():

    def __init__(self):
        self.activate = None
        self.out = None
        self.beta = 1.0

    def forward(self,x):
        self.activate = x
        out = swish(x, self.beta)
        self.out = out
        return out

    def backward(self,delta):
        act = self.activate
        beta = self.beta
        diff_ = beta * self.out + sigmoid(beta * act) * (1 - beta * self.out)
        dx = diff_ * delta
        return dx

    @staticmethod
    def swish(x):
        return Swish_Layer().forward(x)

class Mish_Layer():

    def __init__(self):
        self.activate = None
        self.out = None

    def omega(self,x):
        return 4 * (x+1) + 4 * np.exp(2*x) + np.exp(3*x) + np.exp(x) * (4*x + 6)

    def delta(self,x):
        return 2 * np.exp(x) + np.exp(2*x) + 2

    def forward(self,x):
        self.activate = x
        out = mish(x)
        self.out = out
        return out

    def backward(self,delta):
        act = self.activate
        diff_ = np.exp(act) * self.omega(act) / np.square(self.delta(act))
        dx = diff_ * delta
        return dx

    @staticmethod
    def mish(x):
        return Mish_Layer().forward(x)
