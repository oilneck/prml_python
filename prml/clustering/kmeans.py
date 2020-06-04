import numpy as np
from deepL_module.base import *
from kernel import *

class Kmeans(object):

    def __init__(self, n_clusters:int):
        self.n_clusters = n_clusters
        self.kernel = Kernel()
        self.loss = []
        self.centers = {}
        self.cls_X = {}


    def cdist(self, x:np.ndarray, y:np.ndarray):
        X, Y = self.kernel.get_pairs(x, y)
        sum_sq = np.sum((X - Y) ** 2, axis=-1)
        return np.sqrt(sum_sq)


    def fit(self, X:np.ndarray, n_iter:int=100):
        self.means = np.random.normal(size=(self.n_clusters, X.shape[1]))
        for n in range(n_iter):
            old_means = np.copy(self.means)
            D = self.cdist(X, self.means)
            cls_num = np.argmin(D, axis=1)
            # E-step
            distortion = to_categorical(cls_num, self.n_clusters)
            self.cls_X['step' + str(n+1)] = self.predict(X)
            self.centers['step' + str(n+1)] = self.means.copy()
            # M-step
            denom = distortion.sum(axis=0)[:,None]
            denom[denom==0] = 1e-8
            self.means = (distortion.T @ X) / denom
            self.loss.append(np.trace(distortion.T @ D))


            if np.allclose(old_means, self.means):
                break


        self.loss = np.asarray(self.loss)


    def predict(self, x:np.ndarray):
        D = self.cdist(x, self.means)
        return np.argmin(D, axis=1)
