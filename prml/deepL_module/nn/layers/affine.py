import numpy as np

class Affine(object):

    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.X = None
        self.dW = None
        self.db = None

    def fp(self,X):
        self.X = X
        out = np.dot(X,self.W) + self.b
        return out

    def bp(self,delta):
        dx = np.dot(delta, self.W.T)
        self.dW = np.dot(self.X.T, delta)
        self.db = np.sum(delta, axis=0)
        return dx