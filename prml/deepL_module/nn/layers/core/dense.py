import numpy as np
from .activation import *

class Dense():

    def __init__(self, units, input_dim:int=None, activation='linear'):
        self.units = units
        self.input_dim = input_dim
        self.X = None
        self.dW = None
        self.db = None
        self.out = None
        self.act_func = Activation(activation).func


    def set_param(self, W, b):
        self.W = W
        self.b = b
        self.n_param = W.size + b.size


    def forward(self,X):
        self.original_x_shape = X.shape
        X = X.reshape(X.shape[0], -1)
        self.X = X
        affine = np.dot(X,self.W) + self.b
        self.out = self.act_func.forward(affine)
        return self.out


    def backward(self, delta):
        delta = self.act_func.backward(delta)
        dx = np.dot(delta, self.W.T)
        dx = dx.reshape(*self.original_x_shape)
        self.dW = np.dot(self.X.T, delta)
        self.db = np.sum(delta, axis=0)
        return dx
