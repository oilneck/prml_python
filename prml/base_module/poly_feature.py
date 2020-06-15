import numpy as np
class Poly_Feature(object):

    def __init__(self,degree=9):
        self.M = degree


    def transform(self, X:np.ndarray):
        if X.ndim == 1:
            X = X[:, None]
        pows = np.repeat(np.arange(1, self.M + 1), X.shape[1])
        PHI = np.tile(X, self.M) ** pows
        ones_mat = np.ones(X.shape[0]).reshape(-1, 1)
        return np.concatenate([ones_mat, PHI], axis=1)
