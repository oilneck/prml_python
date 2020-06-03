import numpy as np
from deepL_module.base import *
from kernel import *

class Kmeans(object):

    def __init__(self, n_clusters:int):
        self.n_clusters = n_clusters
        self.kernel = Kernel()
        self.loss = []


    def cdist(self, x:np.ndarray, y:np.ndarray):
        X, Y = self.kernel.get_pairs(x, y)
        sum_sq = np.sum((X - Y) ** 2, axis=-1)
        return np.sqrt(sum_sq)


    def fit(self, X_train:np.ndarray, n_iter:int=100):
        idx = np.random.choice(len(X_train), self.n_clusters, replace=False)
        means = X_train[idx]
        for _ in range(n_iter):
            old_means = means.copy()
            D = self.cdist(X_train, means)
            cls_num = np.argmin(D, axis=1)
            # E-step
            distortion = to_categorical(cls_num, self.n_clusters)
            # M-step
            means = (distortion.T @ X_train) / distortion.sum(axis=0)[:,None]
            self.loss.append(np.trace(distortion.T @ D))

            if np.allclose(old_means, means):
                break

        self.means = means
        self.loss = np.asarray(self.loss)


    def predict(self, x:np.ndarray):
        D = self.cdist(x, self.means)
        return np.argmin(D, axis=1)
