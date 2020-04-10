import numpy as np
from deepL_module.base import*

class Binary_accuracy(object):

    def __init__(self):
        pass

    def accuracy(self, y_pred:np.ndarray, y_true:np.ndarray, normalize:bool=True):

        y_pred, y_true = self._check_dim(y_pred, y_true)

        y = np.where(y_pred >= 0.5, 1., 0.)
        t = y_true

        if normalize:
            accuracy = np.sum(y == t) / float(y.shape[0])
        else:
            accuracy = np.sum(y == t)

        return accuracy



    def _check_dim(self, y_pred, y_true):
        y_pred, y_true = np.array([y_pred, y_true])

        if y_pred.ndim == 1:
            y_pred = y_pred[:,None]

        if y_true.ndim == 1:
            y_true = y_true[:,None]

        return y_pred, y_true
