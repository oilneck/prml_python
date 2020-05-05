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
        self.func = Activation(activation).get()


    def set_param(self, W, b):
        self.W = W
        self.b = b
        self.n_param = W.size + b.size


    def forward(self,X):
        self.X = X
        affine = np.dot(X,self.W) + self.b
        self.out = self.func.forward(affine)
        return self.out


    def backward(self, delta):
        delta = self.func.backward(delta)
        dx = np.dot(delta, self.W.T)
        self.dW = np.dot(self.X.T, delta)
        self.db = np.sum(delta, axis=0)
        return dx
