import numpy as np
import copy

class GP_classifier(object):

    def __init__(self, kernel, fluctuation=0.01):
        self.kernel = kernel
        self.nu = fluctuation

    def sigmoid(self, a):
        return 0.5 + 0.5 * np.tanh(0.5 * a)

    def reshape_col(self, X):
        if X.ndim == 1:
            X = X.reshape(len(X), 1)
        return X

    def fit(self, X:np.ndarray, t:np.ndarray):
        self.X_train = self.reshape_col(X)
        self.t_train = t
        Gram = self.kernel(X, X)
        self.cov = Gram + self.nu * np.eye(Gram.shape[0])
        self.cov_inv = np.linalg.inv(self.cov)

    def predict(self, x:np.ndarray):
        x = self.reshape_col(x)
        K = self.kernel(x, self.X_train)
        activation = K @ self.cov_inv @ self.t_train
        return self.sigmoid(activation)
