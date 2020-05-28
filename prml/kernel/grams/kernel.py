import numpy as np

class Kernel(object):

    def __init__(self):
        pass

    def _check_dims(self,x):
        if x.ndim == 1:
            x = x[:,None]
        return x

    def get_pairs(self, x, y):
        x,y = self._check_dims(x), self._check_dims(y)
        X = np.repeat(x, len(y), axis=0).reshape(x.shape[0], len(y), x.shape[1])
        Y = np.array([y] * len(x)).reshape(len(x), *y.shape)
        return X,Y
