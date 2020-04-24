import numpy as np
from deepL_module.nn.layers import *

class Batch_norm_Layer(Affine):

    def __init__(self, W, b, gamma:float = 1., beta:float = 0.):
        super().__init__(W,b)
        self.dW = np.zeros_like(W)
        self.db = np.zeros_like(b)
        self.gamma = gamma
        self.beta = beta
        self.xc = None
        self.var = None

    def forward(self, x):
        self.n, _ = x.shape
        mu = np.mean(x, axis=0)
        self.xc = x - mu
        self.var = np.var(x, axis=0)

        X_norm = (x - mu) / np.sqrt(self.var + 1e-8)
        out = self.gamma * X_norm + self.beta

        return out

    def backward(self, delta):
        std_inv = 1. / np.sqrt(self.var + 1e-8)

        dX_norm = delta * self.gamma
        dvar = -.5 * np.sum(dX_norm * self.xc, axis=0) * std_inv ** 3
        dmu = np.sum(dX_norm * -std_inv, axis=0) \
            + dvar * np.mean(-2. * self.xc, axis=0)
        dx = (dX_norm * std_inv) + (dvar * 2 * self.xc / self.n) + (dmu / self.n)
        return dx
