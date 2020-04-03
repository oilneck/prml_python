import numpy as np

class Adagrad(object):

    def __init__(self, lr:float=0.01,epsilon:float=1e-8):
        self.lr = lr
        self.eps = epsilon
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] += np.square(grads[key])
            params[key] -= self.lr * grads[key] / (np.sqrt(self.v[key]) + self.eps)
