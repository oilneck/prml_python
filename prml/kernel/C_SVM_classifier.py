import numpy as np
import copy

class C_SVM(object):

    def __init__(self, kernel, C=np.inf):
        self.kernel = kernel
        self.C = C

    def vector2mat(self, X):
        if X.ndim == 1:
            X = X.reshape(len(X), 1)
        return X

    def fit(self, X:np.ndarray, t:np.ndarray, lr:float=0.05, n_iter=1000):
        assert t.ndim == 1
        self.X = X
        self.t = t
        N = len(t)
        a = np.zeros((N,1))
        PHI = self.kernel(X, X)
        t_train = self.vector2mat(t)
        np.seterr(divide='ignore', invalid='ignore')
        for _ in range(n_iter):
            for i in range(N):
                a[i] += lr * (1. - t[i] * np.dot(PHI, t_train * a)[i])
                np.clip(a, 0, self.C, out=a)


        self.alpha = a.ravel()
        mask = list(np.where(self.alpha > 1e-5)[0])
        rect = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
        PHI = self.kernel(X[rect], X[mask])

        self.alpha = self.alpha[mask]
        self.bias = np.mean(t_train[rect] - PHI @ (a[mask] * t_train[mask]))
        self.X = X[mask]
        self.t = t[mask]
        self.support_vector = {'x':X[mask], 't':t[mask]}


    def predict(self, x:np.ndarray):
        PHI = self.kernel(x, self.X)
        return np.sum(self.alpha * self.t * PHI, axis=-1) + self.bias
