import numpy as np

class RMSprop(object):

    def __init__(self, lr=0.1, rho = 0.99):
        self.lr = lr
        self.rho = rho
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] *= self.rho
            self.v[key] += (1 - self.rho) * np.square(grads[key])
            params[key] -= self.lr * grads[key] / (np.sqrt(self.v[key]) + 1e-8)
