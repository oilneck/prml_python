import numpy as np
from deepL_module.base import*

class Categorical_accuracy(object):

    def __init__(self):
        pass

    def accuracy(self, y_pred:np.ndarray, y_true:np.ndarray, normalize:bool=True):

        y_pred, y_true = self._check_dim(y_pred, y_true)

        y = np.argmax(y_pred, axis=1)
        t = np.zeros_like(y)

        if y_true.ndim != 1 : t = np.argmax(y_true, axis=1)

        if normalize:
            accuracy = np.sum(y == t) / float(y.shape[0])
        else:
            accuracy = np.sum(y == t)

        return accuracy

    def _check_dim(self,y_pred,y_true):
        y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
        if y_pred.ndim == 1:
            y_pred = to_categorical(y_pred)

        if y_true.ndim == 1:
            y_true = to_categorical(y_true)

        return y_pred, y_true
