import numpy as np
from classifier import *

class Bayesian_Logistic_Regression(Logistic_Regression):

    def __init__(self,alpha:float=1e-4):
        self.alpha = alpha
        self.w_map = None
        self.w_cov = None

    def make_derivative_info(self,PHI,t,w_old):
        n_dim = np.size(PHI,1)
        y = self._sigmoid(PHI @ w_old)
        R = np.diag(y)
        gradE = PHI.T @ (y - t) + self.alpha * np.eye(n_dim,n_dim) @ w_old
        Hesse = PHI.T @ R @ PHI + self.alpha * np.eye(n_dim,n_dim)
        return gradE, Hesse

    def fit(self,train_PHI:np.ndarray,train_t:np.ndarray,n_iter:int=100):
        w = np.zeros(train_PHI.shape[1])
        for _ in range(n_iter): # Updating weight vector
            [gradE,hessian] = self.make_derivative_info(train_PHI,train_t,w)
            w_new = w - np.linalg.inv(hessian) @ gradE
            if np.allclose(w, w_new,rtol=0.01): break
            w = w_new
        self.w_map = w
        self.w_cov = np.linalg.inv(hessian)


    def predict(self,test_PHI:np.ndarray):
        mu_a = np.dot(test_PHI,self.w_map)
        var_a = np.sum(test_PHI * (self.w_cov @ test_PHI.T).T,axis=1)
        kappa_var = 1 / np.sqrt(1 + np.pi * var_a / 8)
        return self._sigmoid(mu_a * kappa_var)
