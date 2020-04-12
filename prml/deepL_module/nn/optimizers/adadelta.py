import numpy as np

class Adadelta(object):

    def __init__(self, lr:float=1.0, rho:float=0.95, epsilon:float=1e-6):
        self.lr = lr
        self.rho = rho
        self.eps = epsilon
        self.msg = None # mean squared gradient
        self.msu = None # mean squared update

    def update(self, params, grads):
        if self.msg is None:
            self.msg = {}
            self.msu = {}
            for key, val in params.items():
                self.msg[key] = np.zeros_like(val)
                self.msu[key] = np.zeros_like(val)

        for key in params.keys():
            eps = self.eps
            self.msg[key] *= self.rho
            self.msg[key] += (1 - self.rho) * np.square(grads[key])
            delta = \
            grads[key] * np.sqrt((eps + self.msu[key]) / (eps + self.msg[key]))
            self.msu[key] *= self.rho
            self.msu[key] += (1. - self.rho) * np.square(delta)
            params[key] -= self.lr * delta
