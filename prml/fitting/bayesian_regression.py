import numpy as np
from numpy.random import multivariate_normal

class Bayesian_Regression(object):

    def __init__(self, alpha:float=0.005, beta:float=11):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_cov = None


    def fit(self, PHI:np.ndarray, t:np.ndarray):
        self.X = PHI
        self.t = t
        S_inv = self.beta * (PHI.T @ PHI) + self.alpha * np.eye(np.size(PHI,1),np.size(PHI,1))
        S = np.linalg.inv(S_inv)
        self.w_mean = self.beta * S @ PHI.T @ t
        self.w_cov = S


    def predict(self, test_PHI:np.ndarray, get_std:bool=False):
        m_x = np.sum(test_PHI * np.array([self.w_mean] * test_PHI.shape[0]),axis=1)
        if get_std:
            s_x = np.sqrt(self.beta**(-1) + np.sum(test_PHI * (self.w_cov @ test_PHI.T).T,axis=1))
            return m_x,s_x
        return m_x


    def posterior(self, X_test:np.ndarray, n_sample:int=1):
        W = multivariate_normal(self.w_mean, self.w_cov, size=n_sample)
        return X_test @ W.T
