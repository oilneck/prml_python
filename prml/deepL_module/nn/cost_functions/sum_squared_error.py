import numpy as np

class Sum_squared_error(object):

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def activate(self,x):
        return x

    def __call__(self,x,t):
        self.t = t
        self.y = self.activate(x)
        self.loss = 0.5 * np.sum( (self.y - self.t) ** 2 )

    def delta(self,dout=1):
        return self.y - self.t
