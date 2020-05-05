import numpy as np
from .affine import Affine

class Dense(Affine):

    def __init__(self, units):
        self.units = units

    def set_param(self, W, b):
        self.W = W
        self.b = b
        self.n_param = W.size + b.size
