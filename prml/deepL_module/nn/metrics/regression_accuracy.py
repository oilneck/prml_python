import numpy as np

class Regression_accuracy(object):

    def __init__(self):
        pass

    def accuracy(self, y_pred:np.ndarray, y_true:np.ndarray):

        y_pred, y_true = self._check_dim(y_pred, y_true)

        numerator = np.var(y_true - y_pred, axis = 0)
        denominator = np.var(y_true, axis = 0)

        nonzero_numerator = numerator != 0
        nonzero_denominator = denominator != 0
        valid = nonzero_numerator & nonzero_denominator
        accuracy = np.ones(y_true.shape[1])

        accuracy[valid] = 1. - (numerator[valid] / denominator[valid])


        accuracy[nonzero_numerator & ~nonzero_denominator] = 0.

        return np.asarray(accuracy)

    def _check_dim(self, y_pred, y_true):
        y_pred, y_true = np.array([y_pred, y_true])

        if y_pred.ndim == 1:
            y_pred = y_pred[:,None]

        if y_true.ndim == 1:
            y_true = y_true[:,None]

        return y_pred, y_true
