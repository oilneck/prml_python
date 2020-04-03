import numpy as np

class Momentum(object):

    def __init__(self,lr:float=0.01,momentum:float=0.9):
        self.lr = lr
        self.momentum = momentum
        self.inertia = None

    def update(self, params, grads):
        if self.inertia is None:
            self.inertia = {}
            for key, val in params.items():
                self.inertia[key] = np.zeros_like(val)

        for key in params.keys():
            self.inertia[key] *= self.momentum
            self.inertia[key] -= self.lr * grads[key]
            params[key] += self.inertia[key]
